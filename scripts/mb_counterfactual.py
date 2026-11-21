"""Fit a public action-value residual from matched terminal-return branches."""

import argparse
import functools

import numpy as np
import torch
from mb_policy_iteration_eval import OUT, RUN, SEEDS
from mb_residual_preview import pilot, save
from mb_rollout_branches import PER_SHARD
from torch import nn
from vgc_posttrain import atomic_rows


class JointAdvantage(nn.Module):
    """Share action semantics across move positions, roster slots and active Pokémon."""

    def __init__(self, embedding_size):
        super().__init__()
        from vgc_bench.src.utils import chunk_obs_len, glob_obs_len, move_obs_len, side_obs_len

        self.start = glob_obs_len + side_obs_len
        self.move_width = move_obs_len
        # The first six entries following the side information are categorical IDs.
        self.mon = nn.Sequential(nn.Linear(chunk_obs_len - 6 + 6 * embedding_size, 64), nn.Tanh())
        self.move = nn.Sequential(nn.Linear(move_obs_len + embedding_size, 32), nn.Tanh())
        self.kind = nn.Embedding(7, 8)
        ids = torch.arange(107)
        self.register_buffer("move_index", ((ids - 7).clamp_min(0) % 20) // 5)
        self.register_buffer("kinds", torch.where(ids < 7, (ids > 0).long(), 2 + (ids - 7) // 20))
        self.register_buffer("target_index", (ids - 7) % 5)
        self.register_buffer("switch_index", (ids - 1).clamp(0, 5))
        self.single = nn.Sequential(
            nn.Linear(64 * 3 + 32 + 8, 80), nn.Tanh(), nn.Linear(80, 48), nn.Tanh()
        )
        self.joint = nn.Sequential(nn.Linear(48 * 2, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.zeros_(self.joint[-1].weight)
        nn.init.zeros_(self.joint[-1].bias)

    def forward(self, raw, pairs, ex):
        raw = raw.reshape(len(raw), 12, -1)
        pos = self.start
        ids = raw[:, :, pos : pos + 6].long()
        move_embedding = ex.move_embed(ids[:, :, 2:])
        categorical = (
            ex.ability_embed(ids[:, :, 0]),
            ex.item_embed(ids[:, :, 1]),
            move_embedding.flatten(2),
        )
        mons = self.mon(torch.cat((raw[:, :, :pos], *categorical, raw[:, :, pos + 6 :]), dim=2))
        moves_start = pos + 6
        move_props = raw[:, :, moves_start : moves_start + 4 * self.move_width].reshape(
            len(raw), 12, 4, -1
        )
        moves = self.move(torch.cat((move_embedding, move_props), dim=3))
        flags = raw[:, :, -9:-7].transpose(1, 2)
        own = flags[:, :, :6] @ mons[:, :6]
        other = flags[:, :, 6:] @ mons[:, 6:]
        active_moves = torch.einsum("bsi,bimw->bsmw", flags[:, :, :6], moves[:, :6])
        targets = torch.stack(
            (own[:, 1], own[:, 0], torch.zeros_like(own[:, 0]), other[:, 0], other[:, 1]), dim=1
        )
        b = torch.arange(len(raw))[:, None]
        encoded = []
        for slot in (0, 1):
            ids = pairs[:, :, slot]
            kind = self.kinds[ids]
            target = torch.where(
                (kind == 1)[:, :, None],
                mons[b, self.switch_index[ids]],
                targets[b, self.target_index[ids]],
            )
            m = active_moves[:, slot][b, self.move_index[ids]] * (kind >= 2)[:, :, None]
            count = ids.shape[1]
            features = torch.cat(
                (
                    own[:, slot, None].expand(-1, count, -1),
                    m,
                    target * (kind != 0)[:, :, None],
                    self.kind(kind),
                    mons.mean(1)[:, None].expand(-1, count, -1),
                ),
                dim=2,
            )
            encoded.append(self.single(features))
        # A doubles command is a pair. This head can value coordinated attacks differently.
        x, y = encoded
        return self.joint(torch.cat((x + y, x * y), dim=2)).squeeze(-1)


def folder(seed):
    return RUN / "fitted" / f"seed-{seed}"


def fit(seed, blocks):
    torch.set_num_threads(1)
    torch.manual_seed(632090 + seed)
    model = pilot(seed)
    ex = model.base.pi_features_extractor
    q = JointAdvantage(ex.embed_len)
    opt = torch.optim.AdamW(q.parameters(), lr=0.0008, weight_decay=0.001)
    data = {key: [] for key in ("obs", "actions", "valid", "returns")}
    sims = set()
    from mb_rollout_branches import pairing

    for block in range(blocks):
        f = RUN / "branches" / f"seed-{seed}" / f"{block:04}.npz"
        with np.load(f) as part:
            for k in range(PER_SHARD):
                i = block * PER_SHARD + k
                assert int(part["sim"][k]) == 410_000_000 + seed * 1_000_000 + i
                assert tuple(part[x][k] for x in ("own", "foe", "side")) == pairing(seed, i)
                sims.add(int(part["sim"][k]))
            for key, values in data.items():
                values.append(torch.from_numpy(part[key].copy()))
    obs, acts, mask, ret = (torch.cat(v) for v in data.values())
    assert len(sims) == len(obs) == blocks * PER_SHARD
    assert (mask.sum(1) >= 2).all() and (ret.abs() <= 1).all()
    target = ret.float().mean(2)

    def centered(z, valid):
        return z - (z * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True)

    target = centered(target, mask)
    # Within-root contrasts remove the nuisance matchup/position value. Keep a
    # fixed training-only diagnostic split to expose overfitting before deployment.
    perm = torch.randperm(len(obs), generator=torch.Generator().manual_seed(19299 + seed))
    val, train = perm[: len(obs) // 8], perm[len(obs) // 8 :]
    logs = []
    best_loss = float("inf")
    dest = folder(seed)
    dest.mkdir(parents=True, exist_ok=True)
    ck = dest / f"q-{blocks}.pt"
    for epoch in range(12):
        q.train()
        for group in train[torch.randperm(len(train))].split(32):
            pred = centered(q(obs[group], acts[group], ex), mask[group])
            loss = ((pred - target[group]).square() * mask[group]).sum() / mask[group].sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
        q.eval()
        with torch.no_grad():
            pred = centered(q(obs[val], acts[val], ex), mask[val])
            mse = ((pred - target[val]).square() * mask[val]).sum() / mask[val].sum()
            win = target[val].gather(1, pred.masked_fill(~mask[val], -torch.inf).argmax(1)[:, None])
            logs.append(
                dict(
                    seed=seed,
                    epoch=epoch + 1,
                    states=len(obs),
                    diagnostic_loss=mse.item(),
                    constant_loss=(target[val].square() * mask[val]).sum().item()
                    / mask[val].sum().item(),
                    predicted_choice_advantage=win.mean().item(),
                )
            )
        if mse < best_loss:
            best_loss = mse.item()
            save(
                ck,
                dict(
                    seed=seed, blocks=blocks, states=len(obs), epoch=epoch + 1, model=q.state_dict()
                ),
            )
    atomic_rows(OUT / "training" / f"fit-{blocks}-{seed}.csv", logs)
    print("fitted", seed, len(obs), "validation mse", best_loss, flush=True)


@functools.lru_cache(maxsize=8)
def head(seed, blocks):
    path = folder(seed) / f"q-{blocks}.pt"
    if not path.exists():
        path = OUT / "weights" / f"mcq-{blocks}-seed-{seed}.pt"
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert ck["seed"] == seed and ck["blocks"] == blocks
    q = JointAdvantage(pilot(seed).base.pi_features_extractor.embed_len)
    q.load_state_dict(ck["model"])
    return q.eval().requires_grad_(False)


def choose(model, obs, actions, logp, mode, seed):
    # Improvement over the frozen PPO prior, with a conservative penalty
    # against selecting a low-probability command by extrapolation.
    prefix, blocks, strength = mode.split("-")
    if prefix != "mcq" or float(strength) <= 0:
        raise ValueError(mode)
    q = head(seed, int(blocks))
    score = q(obs["observation"], actions[None], model.base.pi_features_extractor)[0]
    idx = (logp + float(strength) * score).argmax()
    return actions[idx].tolist()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blocks", type=int, default=40)
    args = p.parse_args()
    for s in SEEDS:
        fit(s, args.blocks)

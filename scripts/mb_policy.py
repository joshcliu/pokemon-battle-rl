"""Fine-tune the same imitation policy with PPO and two zero-initialized actor heads."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from vgc_posttrain import REPORT, RUN, Match, atomic_rows, batch_obs, players

from pokemon_rl.policy import advantages


class ActionResidual(nn.Module):
    def __init__(self, mode, extractor):
        super().__init__()
        from vgc_bench.src.utils import chunk_obs_len, glob_obs_len, side_obs_len

        self.mode = mode
        self.start = glob_obs_len + side_obs_len
        width = 32
        self.mon = nn.Sequential(
            nn.Linear(chunk_obs_len + 6 * (extractor.embed_len - 1), width), nn.Tanh()
        )
        if mode == "flat":
            self.first_id = nn.Embedding(107, width)
            self.head = nn.Sequential(
                nn.Linear(12 * width + width, 48), nn.Tanh(), nn.Linear(48, 214)
            )
        else:
            actions = torch.arange(107)
            self.register_buffer("move_idx", ((actions - 7).clamp_min(0) % 20) // 5)
            self.register_buffer("target", (actions - 7) % 5)
            self.register_buffer("switch_idx", (actions - 1).clamp(0, 5))
            # pass, switch, regular move, and four move modifiers.
            kind = torch.where(actions < 7, (actions > 0).long(), 2 + (actions - 7) // 20)
            self.register_buffer("kind", kind)
            self.modifier = nn.Embedding(7, width)
            self.candidate = nn.Sequential(nn.Linear(width * 5, 48), nn.Tanh())
            self.pair = nn.Linear(48, 48)
            self.head = nn.Sequential(nn.Tanh(), nn.Linear(48, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def encode(self, x, ex):
        raw = x.view(x.shape[0], 12, -1)
        start = self.start
        ids = raw[..., start : start + 6].long()
        moves = ex.move_embed(ids[..., 2:])
        rich = torch.cat(
            (
                raw[..., :start],
                ex.ability_embed(ids[..., 0]),
                ex.item_embed(ids[..., 1]),
                moves.flatten(-2),
                raw[..., start + 6 :],
            ),
            dim=-1,
        )
        mons = self.mon(rich)
        if self.mode == "flat":
            return mons.flatten(1)
        active = raw[..., -9:-7].transpose(1, 2)
        own_flags, foe_flags = active[:, :, :6], active[:, :, 6:]
        own = own_flags @ mons[:, :6]
        foes = foe_flags @ mons[:, 6:]
        move = torch.einsum("bsi,bimw->bsmw", own_flags, moves[:, :6])
        # Encoded target order is -2, -1, field, +1, +2.
        targets = torch.stack(
            (own[:, 1], own[:, 0], torch.zeros_like(own[:, 0]), foes[:, 0], foes[:, 1]), dim=1
        )
        target = torch.where(
            (self.kind == 1)[None, :, None], mons[:, self.switch_idx], targets[:, self.target]
        )
        target = target * (self.kind != 0)[None, :, None]
        move = move[:, :, self.move_idx] * (self.kind >= 2)[None, None, :, None]
        shape = (len(x), 2, 107, -1)
        candidates = torch.cat(
            (
                own[:, :, None].expand(shape),
                move,
                target[:, None].expand(shape),
                self.modifier(self.kind)[None, None].expand(shape),
                mons.mean(1)[:, None, None].expand(shape),
            ),
            dim=-1,
        )
        return self.candidate(candidates)

    def scores(self, encoded, first=None):
        if self.mode == "flat":
            partner = encoded.new_zeros(len(encoded), 32) if first is None else self.first_id(first)
            halves = self.head(torch.cat((encoded, partner), dim=1)).chunk(2, dim=1)
            return halves[int(first is not None)]
        if first is None:
            return self.head(encoded[:, 0]).squeeze(-1)
        partner = encoded[torch.arange(len(first)), 0, first]
        return self.head(encoded[:, 1] + self.pair(partner)[:, None]).squeeze(-1)


class FineTune(nn.Module):
    def __init__(self, mode):
        super().__init__()
        from train_vgcbench_mb import FORMAT

        from pokemon_rl.vgcbench import load_bc_player

        self.base = load_bc_player(FORMAT).policy
        self.mode = mode
        self.extra = (
            None if mode == "native" else ActionResidual(mode, self.base.pi_features_extractor)
        )

    def decision(self, obs, action=None, deterministic=False):
        logits, value = self.base.get_logits(obs, actor_grad=True)
        mask = obs["action_mask"]
        x = (
            self.extra.encode(obs["observation"], self.base.pi_features_extractor)
            if self.extra
            else None
        )
        first, second = logits.chunk(2, dim=1)
        if self.extra:
            first = first + self.extra.scores(x)
        dist1 = self.base.get_dist_from_logits(torch.cat((first, second), dim=1), mask)
        a = (
            action[:, 0]
            if action is not None
            else dist1.get_actions(deterministic=deterministic)[:, 0]
        )
        if self.extra:
            second = second + self.extra.scores(x, a)
        dist2 = self.base.get_dist_from_logits(torch.cat((first, second), dim=1), mask, a[:, None])
        b = (
            action[:, 1]
            if action is not None
            else dist2.get_actions(deterministic=deterministic)[:, 1]
        )
        d1, d2 = dist1.distribution[0], dist2.distribution[1]
        return (
            torch.stack((a, b), dim=1),
            value.flatten(),
            d1.log_prob(a) + d2.log_prob(b),
            d1.entropy() + d2.entropy(),
        )

    def forward(self, obs, deterministic=False):
        return self.decision(obs, deterministic=deterministic)[:3]


def load_trained(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = FineTune(ck["arm"])
    model.load_state_dict(ck["state"])
    return model.eval()


def update_ppo(model, optimizer, sampling, data, actions, logp, values, rewards, dones, bootstrap):
    obs = {k: torch.as_tensor(np.stack([o[k] for o in data])) for k in data[0]}
    actions = torch.stack(actions)
    old = torch.tensor(logp)
    v = torch.tensor(values)
    terminal = torch.tensor(dones)
    adv = advantages(
        torch.tensor(rewards, dtype=torch.float32),
        v,
        torch.tensor(values[1:] + [bootstrap]),
        terminal,
        terminal,
        0.99,
        0.95,
    )
    target = adv + v
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for _ in range(3):
        for idx in torch.randperm(256, generator=sampling).split(32):
            _, pred, lp, entropy = model.decision(
                {k: arr[idx] for k, arr in obs.items()}, action=actions[idx]
            )
            ratio = (lp - old[idx]).exp()
            pg = torch.minimum(adv[idx] * ratio, adv[idx] * ratio.clamp(0.8, 1.2)).mean()
            loss = -pg + 0.5 * (pred - target[idx]).square().mean() - 0.005 * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
    return loss.item()


def train(arm, seed, steps):
    if arm not in ("native", "flat", "structured") or steps % 256:
        raise ValueError("Expected a supported arm and a multiple of 256 steps")
    spec = json.loads((REPORT / "protocol.json").read_text())
    if steps != spec["steps"] or seed not in spec["training_seeds"]:
        raise ValueError("Unexpected training budget or seed")
    dest = RUN / f"{arm}-{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "policy.pt").exists():
        return
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    match = Match()
    model = FineTune(arm)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, eps=1e-5)
    rng = np.random.default_rng(390_000 + seed)
    sampling = torch.Generator().manual_seed(seed + 400)
    torch.manual_seed(seed + 900)
    episodes, logs = [], []

    def reset():
        own = str(rng.choice(spec["own_train"]))
        foe = str(rng.choice(spec["foe_train"]))
        battle_seed = 100_000_000 + seed * 1_000_000 + len(episodes)
        side = ("p1", "p2")[len(episodes) % 2]
        match.reset(own, foe, battle_seed, side)
        return dict(seed=battle_seed, own=Path(own).stem, foe=Path(foe).stem, side=side)

    episode = reset()
    # Every arm must initially reproduce the released player's joint distribution and sampled action.
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        ob = batch_obs(match.obs())
        torch.manual_seed(231)
        a0, v0, l0 = players()[0].policy(ob)
        torch.manual_seed(231)
        a1, v1, l1 = model(ob)
        assert (
            torch.equal(a0, a1)
            and torch.allclose(l0, l1, atol=1e-5)
            and torch.allclose(v0.flatten(), v1)
        )
    for end in range(256, steps + 1, 256):
        data, actions, logp, values, rewards, dones = [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(256):
                ob = match.obs()
                a, v, lp = model(batch_obs(ob))
                done, reward = match.step(a[0].tolist())
                data.append(ob)
                actions.append(a[0])
                logp.append(lp.item())
                values.append(v.item())
                rewards.append(reward)
                dones.append(done)
                if done:
                    episodes.append({**episode, "reward": reward, "turns": match.battle.turn})
                    episode = reset()
            bootstrap = model.base.predict_values(batch_obs(match.obs())).item()
        loss = update_ppo(
            model, optimizer, sampling, data, actions, logp, values, rewards, dones, bootstrap
        )
        logs.append(
            dict(
                steps=end,
                games=len(episodes),
                wins=sum(e["reward"] == 1 for e in episodes),
                loss=loss,
            )
        )
        atomic_rows(dest / "training.csv", logs)
        print(arm, seed, logs[-1], flush=True)
    atomic_rows(dest / "episodes.csv", episodes)
    tmp = dest / "policy.tmp"
    torch.save(
        dict(
            arm=arm,
            seed=seed,
            steps=steps,
            state=model.state_dict(),
            parameters=sum(p.numel() for p in model.parameters()),
        ),
        tmp,
    )
    tmp.replace(dest / "policy.pt")

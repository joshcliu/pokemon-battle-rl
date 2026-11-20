"""Post-train the two native preview choices around a frozen 16-roster battle pilot."""

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from eval_mb_long_ppo import FINAL, actor, read, schedule
from mb_roster_ppo import DIVERSE_OUT, DIVERSE_RUN, SEEDS, diversity_protocol, verify_assets
from poke_env.environment import DoublesEnv
from torch import nn
from torch.distributions import Categorical
from vgc_posttrain import ROOT, Match, atomic_rows, batch_obs, players

OUT = ROOT / "results/sketch5-mb-preview-ppo"
RUN = ROOT / "runs/sketch5-mb-preview-ppo"
SPEC = json.loads((OUT / "protocol.json").read_text())
ROSTERS = json.loads((DIVERSE_OUT / "protocol.json").read_text())
VARIANTS = ("native", "learned", "blind_native", "blind_learned")


class PreviewAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(256, 64), nn.Tanh())
        self.adjust = nn.Linear(64, 12)
        self.value = nn.Linear(64, 1)
        nn.init.zeros_(self.adjust.weight)
        nn.init.zeros_(self.adjust.bias)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    def forward(self, features):
        h = self.hidden(features)
        return self.adjust(h).reshape(*features.shape[:-1], 2, 6), self.value(h).squeeze(-1)


def likelihood(head, features, base_scores, masks, picks):
    # Batch x stage (lead/back) x position x legal team slot. The second
    # position is conditional on the first selection within the same stage.
    delta, values = head(features)
    valid = masks.clone()
    valid[:, :, 1].scatter_(-1, picks[:, :, :1], False)
    dist = Categorical(logits=(base_scores + delta).masked_fill(~valid, -torch.inf))
    return dist.log_prob(picks).sum((1, 2)), dist.entropy().sum((1, 2)), values[:, 0]


def preview_observation(raw, blind):
    if blind:
        raw.reshape(12, -1)[6:] = 0
    return raw


@torch.no_grad()
def choose(view, head, blind, rng_seed):
    player = players()[0]
    base = player.policy
    for switches in view.available_switches:
        switches[:] = view.team.values()
    mons = list(view.team.values())
    features, base_scores, masks, picks = [], [], [], []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(rng_seed)
        for _ in range(2):
            obs = batch_obs(
                dict(
                    observation=preview_observation(
                        player.embed_battle(view, fake_rating=2000), blind
                    ),
                    action_mask=np.asarray(DoublesEnv.get_action_mask(view), dtype=np.int8),
                )
            )
            latent = base.mlp_extractor.forward_actor(base.pi_features_extractor(obs))
            logits = base.action_net(latent)
            mask = obs["action_mask"]
            slot_logits = logits.reshape(2, -1)[:, 1:7]
            slot_mask = mask.reshape(2, -1)[:, 1:7]
            assert mask.sum() == slot_mask.sum()
            combined = logits.clone()
            if head is not None:
                delta, _ = head(latent)
                combined.reshape(2, -1)[:, 1:7] += delta[0]
            # Keep the exact native sampling calls (including discarded draws)
            # so identical probabilities also yield identical seeded lineups.
            first = base.get_dist_from_logits(combined, mask).get_actions(False)[:, :1]
            second = base.get_dist_from_logits(combined, mask, first).get_actions(False)[:, 1:]
            indices = torch.cat((first, second), 1)[0] - 1
            assert indices[0] != indices[1] and all(0 <= i < 6 for i in indices.tolist())
            for i in indices.tolist():
                mons[i]._selected_in_teampreview = True
            features.append(latent[0])
            base_scores.append(slot_logits)
            masks.append(slot_mask.bool())
            picks.append(indices)
    data = tuple(torch.stack(items).numpy() for items in (features, base_scores, masks, picks))
    lineup = "".join(str(i + 1) for i in data[-1].flatten())
    assert len(set(lineup)) == 4
    return lineup, data


@lru_cache(maxsize=4)
def pilot(seed):
    torch.set_num_threads(1)
    p = DIVERSE_RUN / f"moe-{seed}" / "policy.pt"
    expected = next(
        r["checkpoint_sha256"]
        for r in read(DIVERSE_OUT / "checkpoints.csv")
        if r["arm"] == "moe" and int(r["training_seed"]) == seed
    )
    assert hashlib.sha256(p.read_bytes()).hexdigest() == expected
    m = actor(f"diverse/moe-{seed}-{FINAL}-normal")
    m.collect_routes = False
    return m.eval().requires_grad_(False)


def folder(seed):
    return RUN / f"seed-{seed}"


@lru_cache(maxsize=4)
def acting_head(seed, update):
    state = torch.load(folder(seed) / "progress.pt", map_location="cpu", weights_only=True)
    assert state["update"] == update and state["seed"] == seed
    head = PreviewAdapter()
    head.load_state_dict(state["head"])
    return head.eval().requires_grad_(False)


@torch.no_grad()
def game(seed, own, foe, sim, side, variant, preview_seed, head=None):
    policy = pilot(seed)
    match = Match()
    data = None

    def picker(view):
        nonlocal data
        lineup, data = choose(view, head, variant.startswith("blind"), preview_seed)
        return lineup

    if variant == "native":
        match.reset(own, foe, sim, side, preview="native")
    else:
        match.reset(own, foe, sim, side, preview_picker=picker)
    torch.manual_seed(sim + 700_000_000)
    done = False
    while not done:
        a, _, _ = policy(batch_obs(match.obs()))
        done, reward = match.step(a[0].tolist())
    return dict(
        own=Path(own).stem,
        foe=Path(foe).stem,
        seed=sim,
        side=side,
        lineup=match.lineup,
        opponent_lineup=match.opponent.lineup,
        reward=reward,
        turns=match.battle.turn,
    ), data


def training_group(task):
    seed, update, group, own, foe, side = task
    head = acting_head(seed, update)
    sim = 310_000_000 + 10000 * seed + group
    rows, data = [], []
    for j in range(SPEC["lineups_per_group"]):
        # Vary only our preview sample within each quartet.
        preview_seed = 420_000_000 + 100000 * seed + 4 * group + j
        row, sample = game(seed, own, foe, sim, side, "learned", preview_seed, head)
        rows.append(dict(group=group, sample=j, preview_seed=preview_seed, **row))
        data.append(sample)
    assert len({r["opponent_lineup"] for r in rows}) == 1
    return rows, data


def training_schedule(seed):
    pairs = [(x, y) for x in ROSTERS["own_train"] for y in ROSTERS["foe_train"]]
    rng = np.random.default_rng([8759, seed])
    side_offset = rng.integers(2, size=len(pairs))
    for epoch in range(32):
        for pos, i in enumerate(rng.permutation(len(pairs))):
            group = epoch * len(pairs) + pos
            yield (
                seed,
                group // SPEC["groups_per_update"],
                group,
                *pairs[i],
                ("p1", "p2")[(epoch + side_offset[i]) % 2],
            )


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train(seed, workers):
    torch.set_num_threads(1)
    torch.manual_seed(64100 + seed)
    dest = folder(seed)
    dest.mkdir(parents=True, exist_ok=True)
    head = PreviewAdapter()
    opt = torch.optim.Adam(head.parameters(), lr=SPEC["learning_rate"], eps=1e-5)
    ck = dest / "progress.pt"
    start = 0
    logs = []
    if ck.exists():
        state = torch.load(ck, map_location="cpu", weights_only=True)
        assert state["seed"] == seed
        start, logs = state["update"], state["logs"]
        head.load_state_dict(state["head"])
        opt.load_state_dict(state["optimizer"])
    else:
        save(
            ck,
            dict(seed=seed, update=0, head=head.state_dict(), optimizer=opt.state_dict(), logs=[]),
        )
    if start == SPEC["updates"]:
        return
    tasks = list(training_schedule(seed))
    size = SPEC["groups_per_update"]
    assert len(tasks) * SPEC["lineups_per_group"] == SPEC["battles_per_seed"]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for u in range(start, SPEC["updates"]):
            games, samples = [], []
            for rows, data in pool.map(training_group, tasks[u * size : (u + 1) * size]):
                games.extend(rows)
                samples.extend(data)
            tensors = [torch.from_numpy(np.stack(col)) for col in zip(*samples, strict=True)]
            rewards = torch.tensor([r["reward"] for r in games], dtype=torch.float32)
            with torch.no_grad():
                old, _, value = likelihood(head, *tensors)
                grouped = rewards.view(size, -1)
                others = (grouped.sum(1, keepdim=True) - grouped) / (grouped.shape[1] - 1)
                adv = rewards - (value + others.flatten()) / 2
                adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
            rng = torch.Generator().manual_seed(713000 + seed * 1000 + u)
            for _ in range(SPEC["epochs"]):
                for idx in torch.randperm(len(games), generator=rng).split(SPEC["minibatch"]):
                    lp, ent, v = likelihood(head, *[t[idx] for t in tensors])
                    ratio = (lp - old[idx]).exp()
                    clipped = ratio.clamp(1 - SPEC["clip"], 1 + SPEC["clip"])
                    loss = (
                        -torch.minimum(ratio * adv[idx], clipped * adv[idx]).mean()
                        + 0.5 * (v - rewards[idx]).square().mean()
                        - SPEC["entropy"] * ent.mean()
                    )
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(head.parameters(), 0.5, error_if_nonfinite=True)
                    opt.step()
            with torch.no_grad():
                new, entropy, _ = likelihood(head, *tensors)
                lr = new - old
                logs.append(
                    dict(
                        update=u + 1,
                        battles=(u + 1) * len(games),
                        wins=int((rewards == 1).sum()),
                        mean_reward=float(rewards.mean()),
                        entropy=float(entropy.mean()),
                        approx_kl=float((lr.exp() - 1 - lr).mean()),
                    )
                )
            atomic_rows(dest / f"rollouts-{u:03}.csv", games)
            save(
                ck,
                dict(
                    seed=seed,
                    update=u + 1,
                    head=head.state_dict(),
                    optimizer=opt.state_dict(),
                    logs=logs,
                ),
            )
            print(seed, u + 1, logs[-1]["battles"], flush=True)


def evaluate_chunk(task):
    seed, variant, own, foe, games, cell = task
    head = acting_head(seed, SPEC["updates"]) if variant.endswith("learned") else None
    rows = []
    for sim, side in games:
        row, _ = game(seed, own, foe, sim, side, variant, sim + 500_000_000, head)
        rows.append(dict(training_seed=seed, variant=variant, cell=cell, **row))
    return rows


def evaluate(seed, workers):
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for variant in VARIANTS:
            path = folder(seed) / f"eval-{variant}.csv"
            if path.exists():
                continue
            tasks = [
                (seed, variant, own, foe, sims, cell)
                for _, _, own, foe, sims, _, cell in schedule(
                    "moe", seed, FINAL, "normal", diverse=True
                )
            ]
            rows = [r for chunk in pool.map(evaluate_chunk, tasks) for r in chunk]
            assert len(rows) == 1024
            atomic_rows(path, rows)
            print(seed, variant, len(rows), flush=True)


def check():

    head = PreviewAdapter().eval()
    match = Match()
    seed = SEEDS[0]
    tests = []
    blind_by_own = {}
    for own in ROSTERS["own_train"][:2]:
        for foe in ROSTERS["foe_train"][:2]:
            for side in ("p1", "p2"):
                sim = 385000000 + ROSTERS["own_train"].index(own)
                data = None

                def pick(view):
                    nonlocal data
                    lineup, data = choose(view, head, False, sim + 500_000_000)
                    return lineup

                match.reset(own, foe, sim, side, preview="native")
                expected = match.lineup
                match.reset(own, foe, sim, side, preview_picker=pick)
                actual = match.lineup
                assert actual == expected

                def hide(view):
                    lineup, masked = choose(view, head, True, 123456)
                    k = own, side
                    raw = players()[0].embed_battle(view, fake_rating=2000)
                    changed = raw.copy().reshape(12, -1)
                    changed[6:] = 17
                    assert np.array_equal(
                        preview_observation(raw, True), preview_observation(changed.flatten(), True)
                    )
                    # In the first seat Showdown initializes our roster before the
                    # opposing one, keeping our randomized attributes fixed as well.
                    if side == "p1" and k in blind_by_own:
                        assert all(
                            np.array_equal(x, y)
                            for x, y in zip(masked, blind_by_own[k], strict=True)
                        )
                    blind_by_own[k] = masked
                    return lineup

                match.reset(own, foe, sim, side, preview_picker=hide)
                feat, base, mask, actions = [torch.from_numpy(x)[None] for x in data]
                with torch.no_grad():
                    assert not head(feat)[0].any()
                    lp, _, _ = likelihood(head, feat, base, mask, actions)
                    prob = 1.0
                    for scores, valid, order in zip(base[0], mask[0], actions[0], strict=True):
                        p = scores[0].masked_fill(~valid[0], -torch.inf).softmax(-1)
                        qmask = valid[1].clone()
                        qmask[order[0]] = False
                        q = scores[1].masked_fill(~qmask, -torch.inf).softmax(-1)
                        prob *= float(p[order[0]] * q[order[1]])
                    assert np.isclose(np.log(prob), lp.item(), atol=2e-5)
                tests.append(
                    dict(
                        own=Path(own).stem,
                        foe=Path(foe).stem,
                        side=side,
                        seed=sim,
                        native=expected,
                        zero=actual,
                    )
                )
    # End-to-end equivalence includes identical samples, action RNG, and battle result.
    own, foe = ROSTERS["own_train"][0], ROSTERS["foe_train"][0]
    a, _ = game(seed, own, foe, 381000000, "p1", "native", 881000000)
    b, _ = game(seed, own, foe, 381000000, "p1", "learned", 881000000, head)
    assert a == b
    atomic_rows(OUT / "initialization.csv", tests)
    print(
        "Verified zero residual, native sampled lineups and full battle replay, log probabilities and blinded input invariance.",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "train", "evaluate"))
    p.add_argument("--seed", type=int, choices=SEEDS, default=2)
    p.add_argument("--jobs", type=int, default=6)
    args = p.parse_args()
    verify_assets()
    diversity_protocol()
    if args.stage == "check":
        check()
    elif args.stage == "train":
        train(args.seed, args.jobs)
    else:
        evaluate(args.seed, args.jobs)

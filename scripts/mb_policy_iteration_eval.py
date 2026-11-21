"""Matched development evaluation of decision rules for the same frozen battle PPO pilots."""

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from mb_residual_preview import pilot
from vgc_posttrain import ROOT, Match, atomic_rows, batch_obs

OUT = ROOT / "results/sketch5-mb-policy-iteration"
RUN = ROOT / "runs/sketch5-mb-policy-iteration"
SPEC = json.loads((OUT / "protocol.json").read_text())
SEEDS = SPEC["training_seeds"]
GAMES = 8


def verify():
    from mb_roster_ppo import verify_assets

    verify_assets()
    for file, expected in SPEC["hashes"].items():
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == expected, file


def distribution(model, obs):
    logits, _ = model.battle_logits(obs)
    mask = obs["action_mask"]
    first_ids = mask[0, :107].nonzero().flatten()
    first = model.base.get_dist_from_logits(logits, mask).distribution[0].logits[0, first_ids]
    second = (
        model.base.get_dist_from_logits(
            logits.expand(len(first), -1), mask.expand(len(first), -1), first_ids[:, None]
        )
        .distribution[1]
        .logits
    )
    logp = first[:, None] + second
    row, col = logp.isfinite().nonzero(as_tuple=True)
    return torch.stack((first_ids[row], col), dim=1), logp[row, col]


@torch.no_grad()
def decide(model, obs, mode, seed, view=None):
    if mode == "wide-belief-es":
        from mb_midgame_plan import successive_halving

        return successive_halving(model, obs, view, seed)[0]
    if mode == "multistep-es":
        from mb_multistep_value import deploy

        return deploy(model, obs, view, seed)
    if mode == "endgame-es":
        from mb_endgame_plan import decision

        return decision(model, obs, view, seed)[0]
    if mode == "belief-es":
        from mb_midgame_plan import choose

        return choose(model, obs, view, seed, check=False)[0]
    if mode.startswith(("joint-ppo-", "distilled-ppo-")):
        from mb_joint_mechanics_ppo import deployment

        return deployment(
            view, obs, seed, int(mode.rsplit("-", 1)[1]), mode.startswith("distilled-")
        )
    if mode == "context-ppo":
        from mb_context_ppo import deploy

        return deploy(model, obs, view, seed)
    if mode.startswith("fitted-"):
        from mb_mechanics_es import action
        from mb_td_improvement import weights

        return action(model, obs, view, 8.0 * weights(seed, mode.removeprefix("fitted-")))
    if mode in ("es-initial", "es-trained"):
        from mb_mechanics_es import START, action, fitted

        return action(model, obs, view, START if mode == "es-initial" else fitted(seed))
    if mode == "native":
        return model(obs)[0][0].tolist()
    actions, logp = distribution(model, obs)
    if mode == "map":
        idx = logp.argmax()
    elif mode == "cool":
        idx = (logp * 2).softmax(0).multinomial(1).item()
    else:
        from mb_counterfactual import choose

        return choose(model, obs, actions, logp, mode, seed)
    return actions[idx].tolist()


def schedule(split):
    if split == "validation":
        pairs = [
            (i, j, direction, x, y)
            for i, a in enumerate(SPEC["validation_a"])
            for j, b in enumerate(SPEC["validation_b"])
            for direction, x, y in ((0, a, b), (1, b, a))
        ]
    elif split == "custom-dev":
        pairs = [
            (0, j, 0, SPEC["custom"], b)
            for j, b in enumerate(SPEC["validation_a"] + SPEC["validation_b"])
        ]
    else:
        raise ValueError(split)
    for i, j, direction, x, y in pairs:
        start = (
            (610_000_000 if split == "validation" else 620_000_000)
            + 10000 * i
            + 100 * j
            + 20 * direction
        )
        yield i, j, direction, x, y, start


@torch.no_grad()
def score(task):
    mode, seed, cell = task
    i, j, direction, own, foe, start = cell
    match = Match()
    if mode.startswith("group-"):
        from mb_group_battle_ppo import load_actor

        _, update, deployment = mode.split("-")
        assert deployment in ("native", "map")
        policy = load_actor(seed, int(update))
    else:
        policy, deployment = pilot(seed), mode
    rows = []
    for k in range(GAMES):
        sim, side = start + k, ("p1", "p2")[k % 2]
        match.reset(own, foe, sim, side, preview="native")
        torch.manual_seed(sim + 700_000_000)
        done = False
        while not done:
            choice = decide(policy, batch_obs(match.obs()), deployment, seed, match.view)
            done, reward = match.step(choice)
        rows.append(
            dict(
                method=mode,
                training_seed=seed,
                i=i,
                j=j,
                direction=direction,
                k=k,
                own=Path(own).stem,
                foe=Path(foe).stem,
                sim=sim,
                side=side,
                lineup=match.lineup,
                opp_lineup=match.opponent.lineup,
                reward=reward,
                turns=match.battle.turn,
            )
        )
    return rows


def evaluate(modes, workers, split):
    verify()
    cells = list(schedule(split))
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for mode in modes:
            for seed in SEEDS:
                dest = OUT / "scores" / f"{split}-{mode}-{seed}.csv"
                if dest.exists():
                    continue
                rows = [
                    r for batch in pool.map(score, [(mode, seed, c) for c in cells]) for r in batch
                ]
                assert len(rows) == GAMES * len(cells)
                atomic_rows(dest, rows)
                print(split, mode, seed, len(rows), sum(r["reward"] == 1 for r in rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("modes", nargs="+")
    p.add_argument("--split", choices=("validation", "custom-dev"), default="validation")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    evaluate(args.modes, args.workers, args.split)

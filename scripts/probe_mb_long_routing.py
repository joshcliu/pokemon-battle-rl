"""Measure the effect of expert identity on joint commands on identical held-out opening boards."""

import torch
from eval_mb_long_ppo import EARLY, FINAL, actor, label, schedule
from mb_roster_ppo import LONG_OUT, SEEDS, verify_assets
from vgc_posttrain import Match, atomic_rows, batch_obs


def joint(model, obs):
    logits, _ = model.battle_logits(obs)
    mask = obs["action_mask"]
    p_first = model.base.get_dist_from_logits(logits, mask).distribution[0].probs[0]
    valid = mask[0, : len(p_first)].bool().nonzero().flatten()
    second = (
        model.base.get_dist_from_logits(
            logits.expand(len(valid), -1), mask.expand(len(valid), -1), valid[:, None]
        )
        .distribution[1]
        .probs
    )
    dist = p_first[valid, None] * second
    assert abs(float(dist.sum()) - 1) < 1e-5
    return dist.flatten()


@torch.no_grad()
def probe():
    match = Match()
    boards = []
    for _, _, own, foe, sims, _, cell in schedule("moe", SEEDS[0], EARLY, "rotate"):
        for sim, side in sims[:2]:
            match.reset(own, foe, sim, side, preview="native")
            boards.append((cell, sim, side, batch_obs(match.obs())))
    assert len(boards) == 128
    rows = []
    for t in (EARLY, FINAL):
        for arm in ("moe", "moe8", "moe1"):
            for seed in SEEDS:
                models = {v: actor(label(arm, seed, t, v)) for v in ("normal", "rotate", "zero")}
                for cell, sim, side, obs in boards:
                    normal = joint(models["normal"], obs)
                    for v in ("rotate", "zero"):
                        other = joint(models[v], obs)
                        rows.append(
                            dict(
                                arm=arm,
                                training_seed=seed,
                                steps=t,
                                variant=v,
                                cell=cell,
                                sim_seed=sim,
                                side=side,
                                joint_tv=float((normal - other).abs().sum() / 2),
                                different_argmax=int(normal.argmax() != other.argmax()),
                            )
                        )
    atomic_rows(LONG_OUT / "opening-actions.csv", rows)
    print(
        "Compared frozen expert interventions on", len(boards), "matched held-out opening boards."
    )


if __name__ == "__main__":
    verify_assets()
    probe()

"""Run the HP and opposing-survivor feedback factorial with BC-initialized PPO or AWAC."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from evaluate_vgcbench_mb import check_simulator
from mb_awac import AWAC
from mb_policy import FineTune, update_ppo
from vgc_posttrain import ROOT, Match, atomic_rows, batch_obs

from pokemon_rl.rewards import public_board, shaped_outcome

OUT = ROOT / "results/sketch5-mb-reward-factorial"
RUN = ROOT / "runs/sketch5-mb-reward-factorial"
SPEC = json.loads((OUT / "protocol.json").read_text())


def verify_assets():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("Set PYTHONHASHSEED=0")
    check_simulator()
    for path, expected in SPEC["hashes"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unexpected roster contents in {path}")


def train(algo, mode, seed):
    dest = RUN / f"{algo}-{mode}-{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "policy.pt").is_file():
        return
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    match = Match()
    model = FineTune("native")
    rng = np.random.default_rng(390_000 + seed)
    sampling = torch.Generator().manual_seed(seed + 400)
    if algo == "awac":
        awac = AWAC(model, SPEC["awac"], SPEC["steps"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, eps=1e-5)
    torch.manual_seed(seed + 900)
    hp_weight = SPEC["hp_coefficient"] if mode in ("hp", "both") else 0.0
    ko_weight = SPEC["alive_coefficient"] if mode in ("alive", "both") else 0.0
    episodes, logs = [], []

    def reset():
        own = str(rng.choice(SPEC["own_train"]))
        foe = str(rng.choice(SPEC["foe_train"]))
        battle_seed = 100_000_000 + seed * 1_000_000 + len(episodes)
        side = ("p1", "p2")[len(episodes) % 2]
        match.reset(own, foe, battle_seed, side)
        return dict(seed=battle_seed, own=Path(own).stem, foe=Path(foe).stem, side=side)

    episode = reset()
    current = public_board(match.view)
    initial = current
    episode_discount = 1.0
    discounted_shape = 0.0
    first_rollout = hashlib.sha256()
    for end in range(256, SPEC["steps"] + 1, 256):
        data, actions, logp, values, rewards, dones = [], [], [], [], [], []
        hp_abs = ko_abs = 0.0
        with torch.no_grad():
            for _ in range(256):
                ob = match.obs()
                a, v, lp = model(batch_obs(ob))
                done, outcome = match.step(a[0].tolist())
                after = public_board(match.view)
                reward, hp, ko = shaped_outcome(
                    outcome,
                    current,
                    after,
                    terminal=done,
                    gamma=SPEC["gamma"],
                    hp_weight=hp_weight,
                    ko_weight=ko_weight,
                )
                if end == 256:
                    first_rollout.update(a.numpy().tobytes())
                    first_rollout.update(bytes([outcome + 1, done]))
                hp_abs += abs(hp)
                ko_abs += abs(ko)
                discounted_shape += episode_discount * (hp + ko)
                episode_discount *= SPEC["gamma"]
                data.append(ob)
                actions.append(a[0])
                logp.append(lp.item())
                values.append(v.item())
                rewards.append(reward)
                dones.append(done)
                if done:
                    start_cost = hp_weight * initial[0] + ko_weight * initial[1]
                    assert abs(discounted_shape + start_cost) < 1e-6
                    episodes.append(
                        {
                            **episode,
                            "reward": outcome,
                            "turns": match.battle.turn,
                            "foe_alive": 4 - round(after[1] * 4),
                        }
                    )
                    episode = reset()
                    current = public_board(match.view)
                    initial = current
                    episode_discount = 1.0
                    discounted_shape = 0.0
                else:
                    current = after
            following = match.obs()
            bootstrap = (
                model.base.predict_values(batch_obs(following)).item() if algo == "ppo" else None
            )
        if algo == "ppo":
            metrics = dict(
                loss=update_ppo(
                    model,
                    optimizer,
                    sampling,
                    data,
                    actions,
                    logp,
                    values,
                    rewards,
                    dones,
                    bootstrap,
                )
            )
        else:
            awac.add(data, actions, rewards, dones, following)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed * 1_000_000 + end)
                metrics = awac.update(SPEC["gamma"])
        logs.append(
            dict(
                steps=end,
                games=len(episodes),
                wins=sum(e["reward"] == 1 for e in episodes),
                hp_abs=hp_abs,
                ko_abs=ko_abs,
                **metrics,
            )
        )
        atomic_rows(dest / "training.csv", logs)
        print(algo, mode, seed, logs[-1], flush=True)
    atomic_rows(dest / "episodes.csv", episodes)
    tmp = dest / "policy.tmp"
    torch.save(
        dict(
            arm="native",
            algorithm=algo,
            reward=mode,
            seed=seed,
            steps=SPEC["steps"],
            first_rollout=first_rollout.hexdigest(),
            state=model.state_dict(),
        ),
        tmp,
    )
    tmp.replace(dest / "policy.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("algorithm", choices=SPEC["optimizers"])
    ap.add_argument("reward", choices=SPEC["rewards"])
    ap.add_argument("seed", type=int, choices=SPEC["training_seeds"])
    args = ap.parse_args()
    verify_assets()
    train(args.algorithm, args.reward, args.seed)

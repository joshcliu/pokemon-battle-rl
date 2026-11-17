"""Swap only the first learner decision between the archived damage and mixed policies."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from analyze_long_rewards import load, write_csv
from analyze_rewards import records
from opponent_diversity import BATTLES, EVAL_SEED, SEEDS
from reconstruct_setup_games import archived_training

from pokemon_rl.battle import ROOT as ENGINE
from pokemon_rl.env import BattleEnv
from pokemon_rl.policy import Policy


def run_seed(seed, raw):
    torch.set_num_threads(1)
    results = []
    for recipient, donor in (("mixture", "damage"), ("damage", "mixture")):
        base = raw / f"seed-{seed}"
        archived = records(base / recipient / "final/setup.jsonl")
        saved = torch.load(
            archived_training(seed, recipient) / "checkpoint.pt",
            weights_only=True,
            map_location="cpu",
        )
        policy = Policy(saved["obs_size"], saved["hidden_size"])
        policy.load_state_dict(saved["policy"])
        policy.eval()
        env = BattleEnv(opponent="setup")
        try:
            for i in range(BATTLES):
                stems = [
                    base / c / "final/replays" / f"setup-{i:04}.input.json"
                    for c in (recipient, donor)
                ]
                native, replacement = [load(p)[2]["choices"]["p1"] for p in stems]
                torch.manual_seed(EVAL_SEED + i)
                obs, info = env.reset(seed=EVAL_SEED + i)
                assert info["simulator_seed"] == archived[i]["simulator_seed"]
                assert info["opponent_team_index"] == archived[i]["opponent_team_index"]
                first = True
                while True:
                    with torch.no_grad():
                        # Sample the recipient's opening even when replacing it, to preserve its RNG stream.
                        action = policy.sample(
                            torch.from_numpy(obs)[None], torch.from_numpy(env.joint_mask())[None]
                        )[0][0].numpy()
                    if first:
                        legal = {
                            a: o.message.removeprefix("/choose ") for a, o in env.legal().items()
                        }
                        assert legal[tuple(action)] == native
                        matches = [a for a, command in legal.items() if command == replacement]
                        assert len(matches) == 1
                        action = matches[0]
                        first = False
                    obs, _, done, cutoff, info = env.step(action)
                    if done or cutoff:
                        break
                if native == replacement:
                    assert (info["outcome"], info["turn"]) == (
                        archived[i]["outcome"],
                        archived[i]["turn"],
                    )
                results.append(
                    {
                        "training_seed": seed,
                        "eval_seed": EVAL_SEED + i,
                        "recipient": recipient,
                        "original_opening": native,
                        "substituted_opening": replacement,
                        "original_outcome": archived[i]["outcome"],
                        "swapped_outcome": info["outcome"],
                        "swapped_turns": info["turn"],
                    }
                )
        finally:
            env.close()
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    args = p.parse_args()
    assert ENGINE == args.engine.resolve(), "Run with the original engine's src first on PYTHONPATH"
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(run_seed, seed, args.runs.resolve()) for seed in SEEDS]
        rows = [r for job in jobs for r in job.result()]
    write_csv(args.runs / "opening-swaps.csv", rows)

"""Training/evaluation entry points. Large runs are always explicitly requested."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .battle import ROOT
from .env import BattleEnv
from .opponents import Heuristic
from .policy import Policy, advantages
from .rewards import RewardEnv


def make_env(args, opponent=None):
    options = dict(
        opponent=opponent or args.opponent,
        team=args.team,
        opponent_teams=args.opponent_teams,
        battle_format=args.format,
        max_turns=args.max_turns,
    )
    if args.command == "train" and getattr(args, "reward_mode", "win-loss") != "win-loss":
        return RewardEnv(mode=args.reward_mode, gamma=args.gamma, **options)
    return BattleEnv(**options)


def prepare_run(args):
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    files = [args.team, *args.opponent_teams]
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True
    ).stdout.strip()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "git_commit": git or None,
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True
            ).stdout.strip()
        ),
        "python": platform.python_version(),
        "host": platform.node(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "simulator": json.loads((ROOT / "simulator.lock.json").read_text()),
        "teams": {
            p: {
                "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest(),
                "text": (ROOT / p).read_text(),
            }
            for p in files
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def append_json(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def wilson(wins, count):
    if not count:
        return [None, None]
    z = 1.959963984540054
    proportion = wins / count
    denominator = 1 + z * z / count
    middle = (proportion + z * z / (2 * count)) / denominator
    radius = (
        z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count**2)) / denominator
    )
    return [middle - radius, middle + radius]


def summarize(records, seconds):
    counts = Counter(r["outcome"] for r in records)
    completed = len(records) - counts["truncated"]
    return {
        "battles": len(records),
        "completed": completed,
        "outcomes": {k: counts[k] for k in ("win", "loss", "tie", "truncated")},
        "win_rate_completed": counts["win"] / completed if completed else None,
        "win_rate_95pct_wilson": wilson(counts["win"], completed),
        "truncation_rate": counts["truncated"] / len(records) if records else 0,
        "mean_turns": float(np.mean([r["turn"] for r in records])) if records else None,
        "seconds": seconds,
        "battles_per_second": len(records) / max(seconds, 1e-6),
    }


def run_battles(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = prepare_run(args)
    model = None
    if args.command == "evaluate":
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved["format"] != args.format:
            raise ValueError("Evaluation format differs from checkpoint; use the training format")
        model = Policy(saved["obs_size"], saved["hidden_size"])
        model.load_state_dict(saved["policy"])
        model.eval()
    opponents = args.opponents if args.command == "evaluate" else [args.opponent]
    summaries = {}
    for opponent in opponents:
        records = []
        start = time.perf_counter()
        env = make_env(args, opponent)
        try:
            for i in range(args.battles):
                # Evaluation ordering must not change a policy's random action stream.
                torch.manual_seed(args.seed + i)
                obs, _ = env.reset(seed=args.seed + i)
                actor = Heuristic(
                    args.agent if model is None else "random",
                    np.random.default_rng(np.random.SeedSequence([args.seed, i, 7])),
                )
                while True:
                    if model is None:
                        action = actor.choose(env.battle.players["p1"], env.legal())
                    else:
                        with torch.no_grad():
                            action = model.sample(
                                torch.from_numpy(obs).unsqueeze(0),
                                torch.from_numpy(env.joint_mask()).unsqueeze(0),
                                deterministic=args.deterministic,
                            )[0][0].numpy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        record = {"episode": i, "seed": args.seed + i, "reward": reward, **info}
                        records.append(record)
                        append_json(out / f"{opponent}.jsonl", record)
                        if i < args.replays:
                            env.battle.save(out / "replays", f"{opponent}-{i:04}")
                        break
                if (i + 1) % 100 == 0:
                    print(f"{opponent}: {i + 1}/{args.battles} battles", flush=True)
        except Exception as error:
            append_json(out / "errors.jsonl", {"opponent": opponent, "error": str(error)})
            env.battle.save(out / "failure", opponent)
            raise
        finally:
            env.close()
        summaries[opponent] = summarize(records, time.perf_counter() - start)
        (out / "summary.json").write_text(json.dumps(summaries, indent=2))
        print(json.dumps({opponent: summaries[opponent]}, indent=2), flush=True)


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out = prepare_run(args)
    env = make_env(args)
    start = time.perf_counter()
    episodes = 0
    outcomes = Counter()
    total = 0
    try:
        obs, _ = env.reset(seed=args.seed)
        model = Policy(len(obs), args.hidden_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
        while total < args.steps:
            buffer = []
            for _ in range(min(args.rollout_steps, args.steps - total)):
                tensor = torch.as_tensor(obs, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, logp, value, masks = model.sample(
                        tensor, torch.as_tensor(env.joint_mask(), device=device).unsqueeze(0)
                    )
                next_obs, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
                with torch.no_grad():
                    next_value = model.value(torch.as_tensor(next_obs, device=device).unsqueeze(0))
                buffer.append(
                    (
                        obs.copy(),
                        action[0],
                        logp[0],
                        value[0],
                        masks[0],
                        reward,
                        next_value[0],
                        terminated,
                        terminated or truncated,
                    )
                )
                total += 1
                obs = next_obs
                if terminated or truncated:
                    outcomes[info["outcome"]] += 1
                    append_json(
                        out / "episodes.jsonl",
                        {"episode": episodes, "step": total, "reward": reward, **info},
                    )
                    if episodes < args.replays:
                        env.battle.save(out / "replays", f"train-{episodes:04}")
                    episodes += 1
                    obs, _ = env.reset()
            states = torch.as_tensor(np.stack([b[0] for b in buffer]), device=device)
            actions, old_logp, values, masks = [
                torch.stack([b[i] for b in buffer]) for i in range(1, 5)
            ]
            rewards = torch.tensor([b[5] for b in buffer], device=device)
            next_values = torch.stack([b[6] for b in buffer])
            terminal = torch.tensor([b[7] for b in buffer], device=device)
            ends = torch.tensor([b[8] for b in buffer], device=device)
            adv = advantages(
                rewards, values, next_values, terminal, ends, args.gamma, args.gae_lambda
            )
            returns = adv + values
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            losses = []
            for _ in range(args.epochs):
                permutation = torch.randperm(len(buffer), device=device)
                for indices in permutation.split(args.batch_size):
                    logp, entropy, prediction = model.evaluate(
                        states[indices], actions[indices], masks[indices]
                    )
                    ratio = (logp - old_logp[indices]).exp()
                    policy_loss = torch.maximum(
                        -adv[indices] * ratio,
                        -adv[indices] * ratio.clamp(1 - args.clip, 1 + args.clip),
                    ).mean()
                    value_loss = 0.5 * (prediction - returns[indices]).square().mean()
                    loss = (
                        policy_loss
                        + args.value_coef * value_loss
                        - args.entropy_coef * entropy.mean()
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite PPO loss")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(float(loss.detach()))
            elapsed = time.perf_counter() - start
            metrics = {
                "steps": total,
                "episodes": episodes,
                "loss": float(np.mean(losses)),
                "steps_per_second": total / elapsed,
                "outcomes": dict(outcomes),
            }
            append_json(out / "metrics.jsonl", metrics)
            print(json.dumps(metrics), flush=True)
            checkpoint = {
                "policy": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "obs_size": len(obs),
                "hidden_size": args.hidden_size,
                "steps": total,
                "seed": args.seed,
                "format": args.format,
                "feature_schema": 1,
                "args": vars(args),
            }
            temporary = out / "checkpoint.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(out / "checkpoint.pt")
        (out / "summary.json").write_text(json.dumps(metrics, indent=2))
    finally:
        env.close()

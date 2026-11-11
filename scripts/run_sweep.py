"""Explicit, sequential launch of the proposed nine-run final study."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=262144)
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.battles <= 0:
        parser.error("Steps and battles must be positive")
    output = args.output.resolve()
    commands = []
    for seed in (0, 1, 2):
        for condition in ("damage", "switch", "mixture"):
            run = output / f"{condition}-seed{seed}"
            base = [sys.executable, "-m", "pokemon_rl.cli"]
            commands.append(
                base
                + [
                    "train",
                    "--opponent",
                    condition,
                    "--seed",
                    str(seed),
                    "--steps",
                    str(args.steps),
                    "--device",
                    args.device,
                    "--run-dir",
                    str(run),
                ]
            )
            commands.append(
                base
                + [
                    "evaluate",
                    "--checkpoint",
                    str(run / "checkpoint.pt"),
                    "--opponents",
                    "random",
                    "damage",
                    "switch",
                    "setup",
                    "--seed",
                    "500000",
                    "--battles",
                    str(args.battles),
                    "--run-dir",
                    str(run) + "-eval",
                ]
            )
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "commands.json").write_text(json.dumps(commands, indent=2))
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

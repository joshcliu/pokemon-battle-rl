import argparse

from .battle import DEFAULT_FORMAT
from .opponents import STYLES


def main():
    parser = argparse.ArgumentParser(description="Pokemon doubles RL experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("battle", "train", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument(
            "--run-dir", required=True, help="New directory; existing runs are never overwritten"
        )
        p.add_argument("--team", default="teams/sun.txt")
        p.add_argument(
            "--opponent-teams", nargs="+", default=["teams/balance.txt", "teams/sun.txt"]
        )
        p.add_argument("--format", default=DEFAULT_FORMAT)
        p.add_argument("--max-turns", type=int, default=200)
        p.add_argument("--replays", type=int, default=3)
        p.add_argument("--threads", type=int, default=1)
        p.add_argument("--opponent", choices=STYLES, default="damage")
        if command != "train":
            p.add_argument("--battles", type=int, default=100)
            if command == "battle":
                p.add_argument("--agent", choices=STYLES, default="random")
            else:
                p.add_argument("--checkpoint", required=True)
                p.add_argument(
                    "--opponents",
                    nargs="+",
                    choices=STYLES,
                    default=["random", "damage", "switch", "setup"],
                )
                p.add_argument("--deterministic", action="store_true")
        else:
            p.add_argument("--steps", type=int, default=4096)
            p.add_argument("--rollout-steps", type=int, default=256)
            p.add_argument("--batch-size", type=int, default=64)
            p.add_argument("--epochs", type=int, default=4)
            p.add_argument("--hidden-size", type=int, default=128)
            p.add_argument("--learning-rate", type=float, default=3e-4)
            p.add_argument("--gamma", type=float, default=0.99)
            p.add_argument("--gae-lambda", type=float, default=0.95)
            p.add_argument("--clip", type=float, default=0.2)
            p.add_argument("--entropy-coef", type=float, default=0.01)
            p.add_argument("--value-coef", type=float, default=0.5)
            p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()
    for name in (
        "steps",
        "battles",
        "rollout_steps",
        "batch_size",
        "epochs",
        "threads",
        "max_turns",
    ):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    from .experiments import run_battles, train

    (train if args.command == "train" else run_battles)(args)


if __name__ == "__main__":
    main()

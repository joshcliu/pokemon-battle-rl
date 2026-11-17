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
            "--shuffle-move-slots",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Independently permute each learner Pokémon's moves once per battle",
        )
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
            p.add_argument("--learner-teams", nargs="+", help="Sample a learner roster per battle")
            p.add_argument(
                "--init-checkpoint",
                help="Start PPO from these actor and critic weights with a fresh optimizer",
            )
            p.add_argument("--save-initial", action="store_true", help="Keep the untrained policy")
            p.add_argument(
                "--reward-mode",
                choices=["win-loss", "damage-only", "win-plus-hp"],
                default="win-loss",
                help="Training reward; evaluation always scores wins",
            )
            p.add_argument("--steps", type=int, default=4096)
            p.add_argument(
                "--save-at",
                nargs="+",
                type=int,
                default=[],
                help="Retain checkpoints after these completed PPO updates (learner steps)",
            )
            p.add_argument("--rollout-steps", type=int, default=256)
            p.add_argument("--batch-size", type=int, default=64)
            p.add_argument("--epochs", type=int, default=4)
            p.add_argument("--hidden-size", type=int, default=128)
            p.add_argument(
                "--transformer-chunks",
                type=lambda value: "full" if value == "full" else int(value),
                help="Use the latest N learner decisions, or 'full' for the whole battle prefix",
            )
            p.add_argument("--learning-rate", type=float, default=3e-4)
            p.add_argument("--gamma", type=float, default=0.99)
            p.add_argument("--gae-lambda", type=float, default=0.95)
            p.add_argument("--clip", type=float, default=0.2)
            p.add_argument("--entropy-coef", type=float, default=0.01)
            p.add_argument("--value-coef", type=float, default=0.5)
            p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()
    if getattr(args, "transformer_chunks", None) is not None and (
        (args.transformer_chunks != "full" and args.transformer_chunks < 1)
        or args.hidden_size < 4
        or args.hidden_size % 4
    ):
        parser.error("Transformer policies need positive chunks and a width divisible by four")
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
    if args.command == "train" and any(
        s <= 0 or s > args.steps or (s != args.steps and s % args.rollout_steps)
        for s in args.save_at
    ):
        parser.error("--save-at requires a positive rollout boundary or the final step")
    from .experiments import run_battles, train

    (train if args.command == "train" else run_battles)(args)


if __name__ == "__main__":
    main()

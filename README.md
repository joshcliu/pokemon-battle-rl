# Pokémon battle RL

Joshua Liu and Carol Li · MIT 6.7920 final project

**Question:** Does training against a mixture of opponent styles produce a battle policy that generalizes better than training against one style?

This repository contains a local Pokémon Champions doubles environment, heuristic opponents, a masked PPO policy, and reproducible training/evaluation commands. It uses Pokémon Showdown's battle engine rather than reimplementing Pokémon mechanics. This is a research prototype; the starter teams and early runs are engineering fixtures, not evidence of competitive playing strength.

## Quick start

Requirements: Git, npm, and [uv](https://docs.astral.sh/uv/). Run commands from this repository. Setup downloads Python dependencies, a pinned Showdown revision, and a project-local Node runtime. macOS CPU has been tested; Linux/CUDA support is not yet verified.

```sh
uv sync --locked
uv run python scripts/setup_simulator.py
uv run pytest -q

# Run a random baseline and save three viewable replays.
uv run pokemon-rl battle --agent random --opponent mixture \
  --battles 1000 --seed 0 --run-dir runs/random-v-mixture

# First learning run; terminal win/loss reward only.
uv run pokemon-rl train --opponent mixture --steps 16384 \
  --seed 1 --run-dir runs/pilot

# Development evaluation on new battle seeds.
uv run pokemon-rl evaluate --checkpoint runs/pilot/checkpoint.pt \
  --opponents random damage switch --battles 100 --seed 10000 \
  --run-dir runs/pilot-eval
```

Each command requires a **new** output directory and never overwrites an existing run. Training defaults to CPU with one PyTorch thread. `--device cuda` is available for later GPU work; benchmark it before paying for a large machine, since simulator throughput may dominate.

## What is implemented

```text
Pinned Showdown engine ↔ JSON-lines bridge ↔ player-visible poke-env state
                                              ↓
                                1,973 features + legal joint-action mask
                                              ↓
                                 PPO policy → two Pokémon's commands
```

- Format: `gen9championsvgc2026regmc`, pinned in the source and validated by Showdown.
- Six-Pokémon rosters; fixed first four selected, first two lead. The policy chooses moves, targets, switches, and Mega Evolution. Team building and preview selection are outside this pilot.
- Separate observations for each player. Opponent moves are included only after revelation; the learner receives no opponent team file, private requests, identity label, or simulator internals. Optional open team sheets are not enabled.
- A two-layer 128-unit actor/critic. The joint distribution is `p(a₁ | observation) p(a₂ | observation, a₁)`, with masks that enforce compatible choices. Invalid actions raise errors; there is no fallback action.
- PPO with GAE, terminal rewards `+1 / −1 / 0`, zero intermediate rewards, and truncation bootstrapping. Turn/decision limits are logged separately from wins, losses, and ties.
- Seeded simulation, per-episode opponent sampling, replay export, checkpoints, raw episode records, and evaluation confidence intervals.

The `damage` heuristic scores base power, accuracy, STAB, and type effectiveness. It is **not** an exact damage calculator. `switch` sometimes switches and otherwise uses that score. `mixture` samples one of those two styles per battle. `setup` prioritizes self-boosting moves and is held out of training for the final comparison. These opponents are not strength-matched.

## Experiment workflow

Joshua runs training and evaluation on **Sketch5**. The project checkout there is
`/mnt/md0/jcliu/pokemon-battle-rl/project`. Run compute commands from a shell
on that host. The reward replication uses the server's CPU build of PyTorch.
The [four-seed reward comparison](results/sketch5-rewards/README.md) includes
the win rates, per-seed gains, and follow-up experiments.
In the [longer reward study](results/sketch5-long-rewards/README.md), the two
rewards scored within 0.38 percentage points at 65,536 decisions. Terminal
win/loss led by 5.44 points at 262,144 decisions and won all four seed comparisons.

The [opponent diversity study](results/sketch5-opponent-diversity/README.md)
compares damage-bot training, switching-bot training, and a 50/50 mixture on
four matched training seeds. Each final checkpoint plays the held-out setup bot
on the same 100 battle seeds. Damage-bot training won 74.75% against setup,
ahead of the 50/50 mix at 64.50% and switching-bot training at 58.75%.
The [experiment plan](docs/experiment-plan.md)
and [related work](docs/literature.md) describe the research question.

The [learner-roster transfer experiment](results/sketch5-roster-transfer/README.md)
gives each policy Sun and seven previously unpiloted tournament teams. It compares
Sun training with uniform sampling from 15 other rosters, and scores the untrained
weights on the same battles. Tournament training won 22.55% on held-out teams
versus 18.00% for Sun training and 4.95% for the untrained policies.

The [Sun fine-tuning study](results/sketch5-roster-finetuning/README.md) compares
those Sun checkpoints with scratch training after 16,384, 65,536 and 262,144
additional decisions on the 15 tournament teams. It tracks win rates on the
practice teams, the seven held-out teams and Sun.

The [Transformer PPO study](results/sketch5-transformer-rosters/README.md) compares
one- and eight-decision histories against the 65,536-step feedforward baseline
on those same seven held-out teams.

The [policy-size experiment](results/sketch5-policy-size/README.md) widens both
architectures to about 1.74 million parameters and measures held-out battle wins
and how targeting changes when the opposing Pokémon's type advantages switch sides.
Its full-battle Transformer uses `--transformer-chunks full`: each of four heads
can attend to every visible learner decision since the beginning of the current battle.

The [action and replay analysis](results/sketch5-action-diagnosis/README.md)
follows the full-battle policy through wins and losses, measures transfer at three
training checkpoints, and tests how targets, move order and available history
change the outcomes of the same 5,600 held-out games.

To repeat the twelve opponent-diversity policies from scratch on Sketch5:

```sh
.venv/bin/python scripts/opponent_diversity.py --output runs/diversity-repeat
.venv/bin/python scripts/analyze_opponent_diversity.py --runs runs/diversity-repeat \
  --output results/diversity-repeat
```

To plot a training run, install the optional analysis dependencies and use:

```sh
uv sync --locked --group analysis
uv run --group analysis python scripts/plot_training.py runs/pilot \
  --output runs/pilot/training-curve.png
```

## Outputs and reproducibility

`runs/<name>/` contains:

| File | Purpose |
| --- | --- |
| `manifest.json` | Arguments, source commit/dirty flag, runtime versions, simulator lock, full teams and hashes |
| `episodes.jsonl`, `metrics.jsonl` | Training outcomes and PPO update metrics |
| `<opponent>.jsonl`, `summary.json` | Per-battle evaluation records and aggregate results |
| `checkpoint.pt` | Policy, optimizer, model configuration, and training step |
| `replays/*.html`, `*.log`, `*.input.json` | Player-view HTML replay, spectator log, and seeded simulator commands |

Replay HTML uses upstream Showdown web assets and may require internet access to display. Battle execution itself is local and needs no network after setup. Checkpoints support evaluation. Using `train --init-checkpoint PATH` starts a new run from saved actor and critic weights with a fresh optimizer and step count. Only load checkpoints you trust.

Python dependencies are locked in `uv.lock`. Showdown and Node are pinned in `simulator.lock.json`. Do not update either in the middle of a comparison. Team stat points use the Showdown export field `EVs:` with Champions values (for example, 32/32/2), not standard 252-EV spreads.

## Scope and limitations

This is a Showdown simulation of a particular Champions ruleset, not integration with the commercial game client. Experiments use closed sheets, fixed team selection, and two opponent rosters. The learner uses Sun by default or samples from an explicit roster pool. The encoder omits abilities, items, explicit opponent preview features, and recurrent memory. Fixed slot positions can encourage team-specific shortcuts. Broader teams, open sheets, stronger opponents, and a recurrent policy are later extensions.

Tests exercise action legality, deterministic seeds, a hidden-information check, terminal/truncated handling, and a short train/save/load/evaluate cycle. They do not certify every Pokémon mechanic or equivalence to the official client. The pinned poke-env action mapper uses an upstream private helper, so upgrades require rerunning integration tests.

## Repository map

- `simulator/bridge.cjs`: seeded simulator transport and player-perspective messages.
- `src/pokemon_rl/`: environment, features, opponents, PPO, and CLI.
- `teams/`: legal starter fixtures, not optimized teams.
- `tests/`: unit and local simulator integration checks.
- `scripts/`: simulator setup and explicit study runner.
- `docs/`: reading notes and experiment design.

See [AI_USAGE.md](AI_USAGE.md) for AI assistance disclosure. Upstream Showdown, poke-env, and research projects retain their own licenses and attribution. This project is not affiliated with Pokémon, Nintendo, or Pokémon Showdown.

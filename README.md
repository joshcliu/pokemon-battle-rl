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

The `damage` heuristic scores base power, accuracy, STAB, and type effectiveness. It is **not** an exact damage calculator. `switch` sometimes switches and otherwise uses that score. `mixture` samples one of those two styles per battle. `setup` prioritizes self-boosting moves and is reserved for the planned final comparison. These opponents are not strength-matched.

## Experiment workflow

Start with [the experiment plan](docs/experiment-plan.md) and [related work](docs/literature.md). The first controlled comparison is damage-only, switch-only, and a 50/50 mixture, using identical architectures, teams, rewards, and training budgets.

```sh
# Full proposed study; run deliberately after freezing development choices.
# Nine training runs, then evaluation against four styles for every checkpoint.
uv run python scripts/run_sweep.py --steps 262144 --output runs/study-v1
```

The sweep is a runnable plan, not a completed experiment. Pilot results, when available, live in [results/pilot](results/pilot). Do not use a single pilot run to conclude that opponent diversity helps.

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

Replay HTML uses upstream Showdown web assets and may require internet access to display. Battle execution itself is local and needs no network after setup. Checkpoints support evaluation; continuing interrupted training is not implemented. Only load checkpoints you trust.

Python dependencies are locked in `uv.lock`. Showdown and Node are pinned in `simulator.lock.json`. Do not update either in the middle of a comparison. Team stat points use the Showdown export field `EVs:` with Champions values (for example, 32/32/2), not standard 252-EV spreads.

## Scope and limitations

This is a Showdown simulation of a particular Champions ruleset, not integration with the commercial game client. It is currently a closed-sheet, fixed-selection task with one learner roster and two opponent rosters. The encoder omits abilities, items, explicit opponent preview features, and recurrent memory. Fixed slot positions can encourage team-specific shortcuts. Broader teams, open sheets, stronger opponents, and a recurrent policy are later extensions.

Tests exercise action legality, deterministic seeds, a hidden-information check, terminal/truncated handling, and a short train/save/load/evaluate cycle. They do not certify every Pokémon mechanic or equivalence to the official client. The pinned poke-env action mapper uses an upstream private helper, so upgrades require rerunning integration tests.

## Repository map

- `simulator/bridge.cjs`: seeded simulator transport and player-perspective messages.
- `src/pokemon_rl/`: environment, features, opponents, PPO, and CLI.
- `teams/`: legal starter fixtures, not optimized teams.
- `tests/`: unit and local simulator integration checks.
- `scripts/`: simulator setup and explicit study runner.
- `docs/`: reading notes and experiment design.

See [AI_USAGE.md](AI_USAGE.md) for AI assistance disclosure. Upstream Showdown, poke-env, and research projects retain their own licenses and attribution. This project is not affiliated with Pokémon, Nintendo, or Pokémon Showdown.

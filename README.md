# Pokémon battle RL

Joshua Liu and Carol Li · MIT 6.7920 final project

How well can an RL policy choose four Pokémon from a team of six and play a doubles battle when both sides are given unfamiliar tournament teams? We started with PPO against scripted opponents, tested transfer across teams and opponent styles, and then added team preview and in-battle search on the VGC-Bench simulator.

## Final result on held-out teams

Our frozen three-turn planner won **6,969 of 8,192 games (85.07%, 95% interval 81.86–87.85%)** against the released VGC-Bench behavior-cloned (BC) player. We reserved two groups of eight Regulation M-B tournament teams, played every cross-group pairing in both directions, and alternated simulator seats. Each of four independently trained policies played all 16 assigned teams.

| Second held-out evaluation | Games per method | Original PPO | First planner | Three-turn planner |
| --- | ---: | ---: | ---: | ---: |
| Tournament teams vs. released BC, both roster assignments | 8,192 | 57.41% | 78.94% | **85.07%** |
| Our Sun Room team vs. released BC on each tournament team | 1,024 | 57.71% | 83.20% | **87.01%** |
| Tournament teams vs. independently trained PPO, both roster assignments | 8,192 | 52.14% | 73.75% | **80.00%** |

The bot first simulates lineups to choose its two leads and two reserves. During play, a frozen PPO policy and learned tactical preferences propose moves for both active Pokémon. The planner compares up to eight joint moves across eight possible hidden states, predicts the opponent’s first reply, and uses a learned value function to score the next three turns. The three-turn planner gained **6.13 percentage points** over the first planner on the same tournament games (paired 95% interval +3.45 to +9.13).

One opponent sheet, MB502, had an itemless Talonflame. We recorded a preview-reader fix before viewing the results. Excluding games against that sheet gives 84.77%. Counting every affected game as a loss gives 79.47%. [Final report and exact team matrix](results/sketch5-mb-opponent-search/README.md) · [All 52,224 game records](results/sketch5-mb-opponent-search/confirmation/games.csv.gz) · [Restore the policies and reproduce the evaluation](docs/vgcbench.md#reproduce-the-final-planner)

## Quick start for the Regulation M-C environment

Install Git, npm and [uv](https://docs.astral.sh/uv/). Run the commands from this repository. Setup downloads Python dependencies, a pinned Showdown revision and a project-local Node runtime. The starter environment runs on macOS CPU. The reported experiments ran on Sketch5.

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

## Starter environment and PPO policy (Regulation M-C)

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

The [move-slot training study](results/sketch5-move-slots/README.md) compares
fixed and independently shuffled move slots at 65,536 and 262,144 PPO decisions.
It reports unaided wins, move-order sensitivity and targeting on all 15 familiar
and seven held-out learner teams.

In the [fixed-team league experiment](results/sketch5-fixed-league/README.md),
a full-history PPO Transformer playing MC461 won 99.08% across twelve calibrated
bot–roster matchups. The report includes both playable policies, opponent weights,
and all 4,800 fresh-seed results for each policy.

The [counter-training experiment](results/sketch5-counterleague/README.md) fine-tunes that
Transformer on four new teams and trains PPO opponents against its frozen weights.
A single challenger won 94.65% in team mirrors. A second single network kept MC461
against every opponent and won 91.60%. Each result covers 4,000 fresh battles.

The [challenger transfer test](results/sketch5-challenger-transfer/README.md) evaluates
those policies against the previous bots on the original seven test teams.
On the five teams absent from counter-training, the mirror challenger won 29.58%,
its parent 17.76%, and the earlier feedforward PPO 40.72%.

The [MC461 comparison](results/sketch5-mc461-vs-seven/README.md) instead keeps every
learner on MC461 and gives the seven test rosters to the previous bots. Across
75,600 matched games, the original Transformer scored 82.42%, the mirror challenger
80.07%, and the MC461-trained challenger 80.96%.

The [checkpoint and opponent-mixture study](results/sketch5-mixture-pilot/README.md)
measures when those gains appeared and tests four new 8,192-step PPO continuations.
Mixing in earlier opponents improved broad performance on one training seed and
lowered it on the other. The weaker seed's regression persisted under sampled moves.

The [team-preview experiment](results/sketch5-preview/README.md) freezes the MC461 battle
Transformer and tries all 90 lead-pair and reserve-pair combinations against each of
22 opposing rosters. A separate preview head learns from the 15 training rosters and
faces the best training-selected fixed lineup on seven held-out rosters.

The [expanded preview experiment](results/sketch5-preview-expanded/README.md) adds 45
regional rosters with the same frozen MC461 battle policy. The new head recovered
64.84 percentage points against MC405. Across the seven held-out rosters it won
71.43%, behind the original four at 82.18%.

External cross-play uses the authors' published behavior-cloned VGC-Bench policy and
three benchmark heuristics. The [MC461 transfer test](results/sketch5-vgcbench/README.md)
covers 12 external rosters and seven previous rosters. The [M-B training study](results/sketch5-vgcbench-mb/README.md)
evaluates a legal MC461 variant with open sheets on the benchmark simulator.
The [BC-opponent training study](results/sketch5-vgcbench-bc/README.md) compares four
matched 65k PPO runs against the previous league-trained models on those 12 teams.
The [M-B post-training pilot](results/sketch5-mb-posttraining-pilot/README.md) compares battle PPO and two preview optimizers on familiar and unseen rosters, with a separate test that swaps roster assignments.
The [HP and knockout ablation](results/sketch5-mb-reward-factorial/README.md) crosses the two intermediate signals with PPO and AWAC and measures roster-balanced held-out wins and sweeps.
The [shared, per-team and top-2 MoE PPO comparison](results/sketch5-mb-roster-ppo/README.md) matches the training decisions per roster and uses the same frozen lineup selector in training and evaluation.
The [262k training and routing experiment](results/sketch5-mb-long-ppo/README.md) tests longer training, expert-bank size and the number of active experts.
The [team diversity experiment](results/sketch5-mb-team-diversity/README.md) compares shared PPO and four-expert top-2 routing after training on four or sixteen learner rosters with the same 262k budget.
The [team preview PPO experiment](results/sketch5-mb-preview-ppo/README.md) tests whether learning the leads and reserves improves on BC preview with the battle policy frozen.

The [Regulation M-B battle policy study](results/sketch5-mb-policy-iteration/README.md) compares terminal-reward learning, simulated lineup selection, and in-battle planning across reciprocal tournament matchups and a newly built Sun Room team. The [four-seed scorecard](results/sketch5-mb-policy-iteration/scorecard.md) also reports transfer to an independently trained opponent.
The first planner won 77.17% on its [first independent 16-roster evaluation](results/sketch5-mb-policy-iteration/confirmation/README.md). We retired those rosters and reserved a new set of 16 before selecting the three-turn planner on development games.
[Reproduce the VGC-Bench experiments](docs/vgcbench.md).

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

Every experiment report should give the exact team matrix for training and each evaluation. Identify the pilot and roster IDs on both sides, link the ordered six-Pokémon sheets, state who chose the four and the leads, and give games per pairing. Show any assignment swaps explicitly. [Example team matrix](results/sketch5-mb-posttraining-pilot/team-matrix.md)

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

The early M-C experiments used closed sheets and fixed lineup selection. The final M-B evaluation used open team sheets, selected its own leads and reserves, and modeled unrevealed information from the player’s perspective. “Held out” refers to our training and model selection. Overlap with the upstream BC model’s replay data is unknown. The win rates measure play against the named opponents on the pinned Showdown simulator. We have no win-rate estimate against human players in the commercial game.

Tests exercise action legality, deterministic seeds, a hidden-information check, terminal/truncated handling, and a short train/save/load/evaluate cycle. They do not certify every Pokémon mechanic or equivalence to the official client. The pinned poke-env action mapper uses an upstream private helper, so upgrades require rerunning integration tests.

## Repository map

- `simulator/bridge.cjs`: seeded simulator transport and player-perspective messages.
- `src/pokemon_rl/`: environment, features, opponents, PPO, and CLI.
- `teams/` contains starter teams, tournament exports and the original Sun Room team.
- `tests/`: unit and local simulator integration checks.
- `scripts/` contains simulator setup, training, evaluation and reporting code.
- [Results](results/) contains the reports, figures, game records and selected trained weights.
- [VGC-Bench reproduction guide](docs/vgcbench.md) covers benchmark setup and study commands.

See [AI_USAGE.md](AI_USAGE.md) for AI assistance disclosure. Upstream Showdown, poke-env, and research projects retain their own licenses and attribution. This project is not affiliated with Pokémon, Nintendo, or Pokémon Showdown.

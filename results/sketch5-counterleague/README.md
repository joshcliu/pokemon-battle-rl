# Training a counter to the multi-team Transformer

The subsequent test against the previous bots found partial transfer. On five teams absent from the challenger's training ancestry, the single mirror challenger won 29.58%, up from its parent's 17.76%. The earlier feedforward PPO won 40.72% on those same games. [Seven-team transfer evaluation](../sketch5-challenger-transfer/README.md)

**A single full-history PPO Transformer won 3,786/4,000 games (94.65%) when given the defender's team.** It won 3,055/3,200 against the adapted Transformer on four tournament teams (95.47%) and 731/800 against the original champion on MC461 (91.38%). Its weakest mirror was MC646 at 701/800 (87.63%, Wilson 95% interval 85.16–89.73%). One network played every matchup. [Game-level results](confirmation/games.csv.gz) · [Weights](weights/counter-12.pt)

A second single network used MC461 in every game, including when the defender changed teams. It won 3,664/4,000 (91.60%). Its weakest matchup was MC646 at 613/800 (76.63%, 95% interval 73.57–79.43%). Allowing a fixed-lineup bot to select a trained expert from the visible opposing leads raised its score to 3,929/4,000 (98.23%). [Fixed-lineup games](fixed-confirmation/games.csv.gz) · [Single-network weights](weights/counter-8.pt)

| Defender | One network, mirror | Expert selection, mirror | One network, always MC461 | Expert selection, always MC461 |
| --- | ---: | ---: | ---: | ---: |
| Adapted on MC463 | 98.50% | 100.00% | 97.25% | 98.88% |
| Adapted on MC646 | 87.63% | 99.00% | 76.63% | 96.75% |
| Adapted on MC751 | 100.00% | 100.00% | 99.88% | 99.00% |
| Adapted on MC781 | 95.75% | 99.63% | 86.13% | 99.88% |
| Original on MC461 | 91.38% | 99.00% | 98.13% | 96.63% |
| **All five** | **94.65%** | **99.53%** | **91.60%** | **98.23%** |

**When the defender sampled from the same weights**, the single mirror challenger won 1,077/1,200 (89.75%). The single fixed-MC461 challenger fell to 802/1,200 (66.83%), including 103/240 against MC646 and 86/240 against MC781. The mirror and fixed-lineup expert selectors scored 1,137/1,200 (94.75%) and 1,036/1,200 (86.33%). The challengers still chose their moves greedily. This diagnostic used 240 games per team and separate seeds. [Sampled mirrors](sampled-mirrors/scores.csv) · [Sampled fixed-lineup games](sampled-fixed/scores.csv)

In a fresh example with MC461 against MC646, the single network's Meowstic used Expanding Force and Kommo-o used Clanging Scales. They knocked out Pelipper on turn one. Farigiraf set Trick Room that turn, and the challenger went on to win on turn 11. [Battle log](example/71000001-p2.log) · [HTML replay](example/71000001-p2.html)

## How the opponent was trained

The starting point was the single 1.75M-parameter, full-history Transformer that won 4,756/4,800 games with MC461 in the previous league. We fine-tuned four copies for 65,536 PPO decisions each. Each copy sampled four tournament teams uniformly as its own roster. Its opponents were the previous twelve-bot field plus the original Transformer on MC461. We used terminal win/loss rewards. [Original champion](../sketch5-fixed-league/README.md) · [Opponent pool](adapt-pool.json)

We selected the strongest of eight checkpoints by its win total across the four new rosters and thirteen opponents on development seeds. Seed 297 at 65,536 steps won. On 4,992 separate games matched to the unadapted champion, the new weights won 3,121 games (62.5%), up from 746 (14.9%). On the original MC461 twelve-opponent benchmark, those new weights won 2,449/4,800 (51.0%). We kept the original weights as the incumbent on MC461 and froze the adapted weights on the four new rosters. [Checkpoint selection](defender-selection-scores.csv) · [Paired adaptation results](adaptation.csv)

| Defender's team | Original weights against the field | Adapted weights against the field |
| --- | ---: | ---: |
| MC463 | 19.2% | 58.2% |
| MC646 | 10.8% | 54.2% |
| MC751 | 0.2% | 50.7% |
| MC781 | 29.6% | 87.0% |

## How the challengers were trained

We tested two ways to face the five frozen team–policy combinations. In **mirrors**, the two players received identical teams. Each shared candidate sampled a new mirror each episode. We also trained four specialists per team and evaluated a policy that chose an expert once from the publicly visible opposing leads. In **fixed-team play**, the challenger always used MC461. Three networks trained against a uniformly sampled defender team. We also trained three specialists for each of the four adapted matchups and reused the MC461 mirror specialists for the original matchup. A second opening-based selector could therefore change its policy while keeping the same four Pokémon. Every neural candidate used masked PPO and the win/loss reward. [Training runs](training.csv) · [Roster exposure](exposure.csv)

In the five equal-weight mirrors, self-play of each defender against itself scored 419/800 (52.4%). The hand-coded damage bot scored 576/800 (72.0%) and the tactical bot 656/800 (82.0%). The tactical bot ranged from 45.6% against the original MC461 specialist to 100% against the adapted MC751 pilot. These baseline games used separate seeds from selection and confirmation. [Baseline scores by team](baseline-scores.csv)

All players brought the first four Pokémon from the published team and used its published move order. Each frozen neural player picked legal actions from its own observations. The primary test used greedy moves, matching the previous league. We screened on development seeds, selected checkpoints on separate validation seeds by their worst team score and then total wins, and froze the weights before final evaluation. Each final matchup had 800 fresh simulator seeds, with 400 starts on each side. Ties and cutoffs counted as non-wins. The intervals describe battle-seed uncertainty for these frozen players and five matchups. [Locked matchups and hashes](locked.json) · [Mirror validation](counter-validation-scores.csv) · [Fixed-team validation](fixed-validation-scores.csv)

## Using the policies

The exported weights and configurations can play a battle from the project checkout with the simulator installed. This example gives the challenger MC461 and the adapted Transformer MC646.

```sh
.venv/bin/python scripts/replay_counterleague.py --matchup fixed_team --team MC646 \
  --seed 71000001 --side p2 --out runs/counter-example
```

Use `--matchup single` for the shared network that plays any of the five mirrors. The replay command saves a text battle log, simulator inputs and an HTML replay. The opponent pool can also be supplied to the existing PPO trainer. [Packaged candidates](candidates.json) · [Frozen opponents](opponents.json) · [Weight provenance](weights.csv)

## Reproducing the training run

Sketch5 retains the full training trajectories and checkpoints in `runs/sketch5-counterleague-v1`. Training from the same initializations also uses the archived tournament checkpoints in `runs/sketch5-move-slots-v1`. The source replay check compares the exported policies' results with the original training checkpoints on every confirmation game. [Source replay verification](export-replay-check.json)

```sh
run=runs/counterleague-repeat
out=results/counterleague-repeat
.venv/bin/python scripts/train_counterleague.py start --runs "$run"
for stage in zero-shot defender calibrate retention baselines; do
  .venv/bin/python scripts/evaluate_counterleague.py "$stage" --runs "$run"
done
.venv/bin/python scripts/train_counterleague.py counter --runs "$run"
.venv/bin/python scripts/train_counterleague.py fixed-experts --runs "$run"
for stage in screen validate confirm sampled; do
  .venv/bin/python scripts/evaluate_counterleague.py "$stage" --runs "$run" --public "$out"
done
.venv/bin/python scripts/audit_counterleague.py --runs "$run" --public "$out"
```

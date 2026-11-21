# Learning to predict opponent replies for model-based RL

Our first policy won 80.71% on development and 77.17% on its independent 16-roster test. We retired that test, kept the original development matrix, and reserved another 16 teams before scoring any new method. Each reserved roster shares at most three species with any roster previously used in the project. [First confirmation](../sketch5-mb-policy-iteration/confirmation/README.md) · [Second-round protocol](protocol.json)

## Development

| Method | Wins out of 2,048 | Win rate (95% interval) | Difference vs. first-round planner (points) |
|---|---:|---:|---:|
| adaptive-belief-es | 1,653 | 80.71% (75.44–85.64) | +0.00 (+0.00 to +0.00) |
| reply | 1,676 | 81.84% (76.46–86.43) | +1.12 (-3.27 to +5.42) |
| reply-critic | 1,711 | 83.54% (78.52–87.94) | +2.83 (-1.76 to +7.18) |
| reply-wide | 1,699 | 82.96% (77.49–87.50) | +2.25 (-2.54 to +6.79) |

We fixed six candidate configurations before the first complete four-seed result. The highest development win count qualifies at 1,680 of 2,048 wins (82%). A fixed order resolves ties. Every development comparison uses the incumbent’s assigned rosters, simulator seats, seeds and both submitted lineups. Intervals resample training seeds, both team axes and game replicates within seat. [Selection rule](selection.json)

## What the policy learns and simulates

All arms use the same four PPO battle pilots, trained mechanics preferences, terminal win/loss/draw reward and 196-game public lineup planner as the first candidate. We changed how the bot evaluates the moves it could play next.

- **Neural opponent responses.** For each public matchup, simulate 48 BC team previews across four hypothetical private stat spreads. Condition possible reserves on the Pokémon already revealed. Construct an opponent-perspective embedding for each hypothetical world using guessed stats and selected reserves. Three of each four worlds sample the released actor's predicted moves. The fourth samples the earlier mechanics-based reply model. Each possible move is judged by the learned value after Showdown resolves the joint turn.
- **Search width and depth.** Compare four actions under four worlds, eight actions under eight worlds, and three-step lookahead. The first step uses the modeled BC response. Further steps use the mechanics-based continuation policy.
- **Factored joint-action search.** Hold one partner's move fixed while testing up to nine commands for the other. Combine the most promising commands for both partners, then compare up to eight joint finalists on eight fresh modeled worlds.
- **World value learning.** Fit a separate residual state-value network from final returns in 1,120 training battles per seed. The collector sees the fully specified training world, which matches the information available inside a hypothetical simulation. At deployment its input is the sampled world. The contest opponent's actual private state is inaccessible to this model.

Across 16 training fixtures, changing both the true hidden reserves and private stats with the public board fixed produced identical hypothesized reserves, response logits and selected commands in both seats. [Intervention records](public-input-audit.csv)

Median move-selection times on the same 58 decisions from eight training-roster games were incumbent 215 ms, reply-wide 522 ms, coordinate 992 ms. These timings used the shared Sketch5 host. [Median and 95th percentile by method](move-latency.csv) · [Cached and uncached preview timing](preview-latency.csv)

On 282 training-roster decisions, a sampled neural response matched the actual BC command, including its target and Mega choice, on 40.29% of Pokémon slots. The old response model matched 24.38%. Exact joint-action agreement was 14.89% and 5.50%, respectively. [Per-decision predictions](reply-predictions.csv)

On 32 opening positions, the modeled reserve probabilities had a Brier score of 0.256. Assigning 50% to each of the four unseen Pokémon scored 0.250. Lower is better. [Reserve forecasts and checked outcomes](reserve-predictions.csv)

## Exact team matrix

- Training learner pool (28) — MC461-MB, MB537, MB51, MB45, MB503, MB56, MB156, MB190, MB306, MB374, MB76, MB239, MB314, MB232, MB17, MB497, MB177, MB388, MB346, MB19, MB31, MB277, MB567, MB498, MB248, MB270, MB64, MB506.
- Training opponent pool (20) — MB501, MB245, MB269, MB387, MB106, MB90, MB40, MB102, MB319, MB365, MB13, MB434, MB429, MB551, MB205, MB539, MB300, MB165, MB466, MB342.
- Development A — MB566, MB482, MB561, MB296.
- Development B — MB252, MB475, MB380, MB140, MB330, MB291, MB549, MB463.
- Second confirmation A — MB324, MB542, MB236, MB147, MB119, MB564, MB241, MB178.
- Second confirmation B — MB288, MB502, MB234, MB226, MB276, MB280, MB87, MB436.
- The learner plays every A team against every B team, then the two sides exchange rosters. The new reciprocal test plays 16 games per direction per seed, alternating seats. The original Sun Room team separately plays all 16 tournament sheets. The independent opponent is shared PPO seed 2 at 262,144 steps with native BC preview. [Sun Room import](../../teams/reg-mb-original/sun-room.txt)

The roster exclusions cover our own training and model selection. The released behavior-cloning checkpoint provides the common starting point for the pilots and uses its upstream replay data. These results measure performance in the pinned Showdown simulator.

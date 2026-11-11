# Related work and implementation choices

These are initial reading notes, not a systematic review or reproduced benchmark. Paper results below apply to their own formats and evaluation populations; they are not directly comparable with this Champions pilot. Read VGC-Bench in depth first, then Metamon, then the older policy/search work.

| Work | What it contributes | What we take from it |
| --- | --- | --- |
| [VGC-Bench](https://arxiv.org/abs/2506.10326) · Angliss et al. · [code](https://github.com/cameronangliss/vgc-bench) | A doubles benchmark with population-based RL, behavior cloning, and evaluation across team matchups. Its restricted fixed-team success should not be read as unrestricted human-level play. | Closest comparison. Treat legal joint actions, opponent populations, and team generalization as central experimental choices. Inspect its pinned dependency stack before attempting reuse. |
| [Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers](https://arxiv.org/abs/2504.04395) · Grigsby et al. · [Metamon code](https://github.com/UT-Austin-RPL/metamon) | Offline RL from human battle replays with sequence models and additional self-play data. The paper focuses on older singles generations. | Motivation for history-dependent policies and careful reconstruction of what each player knew. A large replay/transformer pipeline is a possible extension, not our initial compute target. |
| [PokéChamp](https://arxiv.org/abs/2503.04094) · Karten, Nguyen, and Jin | LLM-guided minimax decision making using action proposals, opponent modeling, and value estimation. | Useful for categorizing search/planning baselines. This is not the Pokémon Champions game and is not our PPO training approach. |
| [PokéLLMon](https://arxiv.org/abs/2402.01118) · Hu, Huang, and Liu · [code](https://github.com/git-disl/PokeLLMon) | A language-model agent using textual feedback, external knowledge, and consistent action generation to mitigate failures such as repeated switching. | Useful failure categories for replay review. Its in-context feedback should not be described as PPO parameter training. |
| [A Self-Play Policy Optimization Approach to Battling Pokémon](https://ieee-cog.org/2020/papers2019/paper_175.pdf) · Huang and Lee · 2019 | Earlier policy optimization through self-play, with search-agent and ladder comparisons and investigation of transfer. | Historical anchor for direct policy learning. Opponent quality and evaluation population matter as much as the training algorithm label. |
| [Winning at Pokémon Random Battles Using Reinforcement Learning](https://hdl.handle.net/1721.1/153888) · Jett Wang · MIT thesis, 2024 | Combines PPO/self-play with actor-critic-informed search for Generation 4 random battles. | A relevant local thesis to read for implementation and evaluation detail. Search is a later extension; ladder rank in that format is not a baseline win rate for doubles. |
| [PokéAgent Challenge: battling](https://pokeagentchallenge.com/battling.html) | A shared evaluation effort with published battle formats, baselines, and datasets. | Borrow the discipline of explicit formats and opponent sets. Its leaderboard tasks do not automatically match our Champions ruleset. |

## Infrastructure

- [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) supplies mechanics, team validation, legal requests, seeded transitions, and battle logs. We pin a source revision rather than implement a new game engine.
- [poke-env documentation](https://poke-env.readthedocs.io/en/stable/) supplies player-perspective battle parsing and doubles order handling. The project pins `0.16.1` and isolates its version-specific action mapping in `battle.py`.
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) supplies the clipped policy objective; our small implementation makes the action mask and truncation behavior explicit.

## What we should claim

The intended contribution is a controlled opponent-diversity experiment in a bounded doubles task. Building another Pokémon bot, using PPO, or beating a weak heuristic alone is not the research finding. A useful result would show when mixed-opponent training improves or fails to improve performance on a reserved opponent, with uncertainty across independent training runs.

The current implementation is an original small harness around Showdown and poke-env. It does not reproduce VGC-Bench, Metamon, PokéChamp, or their published scores. No pretrained model or replay dataset from those projects is included.

## Reading checklist before extending the model

1. Extract VGC-Bench's observation, action, team-split, population, and evaluation choices into a comparison table.
2. Read the Metamon replay reconstruction and partial-observability sections before adding imitation learning.
3. Inspect failure replays from this project: unnecessary switches, ally targeting, ignored Protect, poor Mega timing, and inability to anticipate an unrevealed move.
4. Choose one extension only after the basic diversity comparison works: stronger opponent mixtures, recurrence, or behavior-cloning initialization.

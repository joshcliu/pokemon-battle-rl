# PPO for team preview with a frozen battle policy

All four seed-matched preview adapters reached 16,384 training battles on Sketch5, totaling 65,536 games. The last verified remote snapshot showed 12 of the 16 held-out evaluation batches written. Final win rates and paired intervals await retrieval. [Protocol](protocol.json) · [Native and zero-initialized preview checks](initialization.csv)

Each adapter changes the released BC policy's legal lineup scores at preview. Its two ordered choices produce the leads and then the reserves. During the ensuing game, the frozen 16-team MoE checkpoint from the same training seed chooses all battle actions. Training uses terminal win, draw and loss rewards and compares four independently sampled lineups on the same matchup, seat, simulator seed and opposing lineup.

## Team matrix

| Schedule | Our teams | Frozen BC opponent teams |
|---|---|---|
| Training | MC461-MB, MB537, MB51, MB45, MB503, MB56, MB156, MB190, MB306, MB374, MB76, MB239, MB314, MB232, MB17, MB497 | MB501, MB245, MB269, MB387, MB106, MB90, MB40, MB102 |
| Held-out forward | MB516, MB216, MB99, MB444 | MB481, MB335, MB259, MB558, MB163, MB562, MB149, MB337 |
| Held-out reverse | MB481, MB335, MB259, MB558, MB163, MB562, MB149, MB337 | MB516, MB216, MB99, MB444 |

All four conditions share the same battle policies and matched evaluation seeds. The main comparison is learned preview minus native BC preview. The controls hide the six opposing Pokémon from both the learned and native pickers during preview. [Previous battle policies and matchup results](../sketch5-mb-team-diversity/README.md)

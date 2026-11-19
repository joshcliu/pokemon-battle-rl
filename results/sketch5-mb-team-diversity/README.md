# Does a more diverse training roster make expert routing useful?

## Team matrix

| Schedule | PPO rosters | Frozen VGC-Bench BC rosters |
|---|---|---|
| Four-team training baseline | MC461-MB, MB537, MB51, MB45 | MB501, MB245, MB269, MB387, MB106, MB90, MB40, MB102 |
| Sixteen-team training | MC461-MB, MB537, MB51, MB45, MB503, MB56, MB156, MB190, MB306, MB374, MB76, MB239, MB314, MB232, MB17, MB497 | MB501, MB245, MB269, MB387, MB106, MB90, MB40, MB102 |
| Held-out forward | MB516, MB216, MB99, MB444 | MB481, MB335, MB259, MB558, MB163, MB562, MB149, MB337 |
| Held-out reverse | MB481, MB335, MB259, MB558, MB163, MB562, MB149, MB337 | MB516, MB216, MB99, MB444 |

The larger pool expands learner coverage from 20 to 52 distinct base species. Of the 40 base species on the held-out teams, 14 occur on the original four learner teams and 25 occur on the expanded pool. [All six Pokémon on every team](rosters.md)

The 12 additions come from the pinned VGC-Bench Regulation M-B roster bank. We selected them by a fixed hash ordering before playing any games. Each has a legal team sheet, six Pokémon with four moves each, and at most three base species in common with any other addition, original training roster, held-out roster or reserved development roster. The twelve held-out teams were excluded from both sides of PPO training. They were used in earlier evaluations and their exposure during upstream behavior cloning is unknown. [Roster file hashes and protocol](protocol.json) · [Roster selection](../../scripts/lock_mb_diversity.cjs)

## Training and evaluation

Both networks start from the released BC battle policy. Shared PPO fine-tunes its native actor and critic. The MoE fine-tunes the same backbone plus four residual action heads, each 256 → 128 → 214. The learned gate selects two experts per decision and normalizes their weights. The frozen BC preview chooses the four Pokémon from six for both sides in every training and evaluation game. Battle actions are sampled, team sheets are open, and rewards are +1/0/−1 for win/draw/loss.

Each architecture receives 262,144 decisions per seed with seeds 2, 3, 4 and 5. Each original learner team received 65,536 decisions per seed in the four-team baseline. In the new run, 16 learner teams each receive exactly 16,384. Each roster has its own live battle, opponent sampling stream and alternating seat. A seeded shuffle visits all 16 rosters in each block of four PPO updates. Each update still collects 64 consecutive decisions from each of four active battles. At the end of each rollout we bootstrap the value of its own next state. Adam, the PPO settings, rollout length, total interaction budget and opposing roster distribution match the baseline. [Previous four-team experiment](../sketch5-mb-long-ppo/README.md) · [Trainer](../../scripts/mb_roster_ppo.py)

For each checkpoint and routing condition, we play all 32 pairings in both directions. Each pairing has 16 games, eight per seat, for 1,024 games per training seed. The headline rate is the average of the two directions and counts draws as non-wins. The matchup, simulation seed, action-sampling seed, seat and both BC previews match the four-team archive. We resample training seeds, both roster axes and battle seeds within seats together across comparisons, using the original 6,000-draw paired percentile bootstrap.

The primary intervention holds the fine-tuned network and gate fixed and cyclically reassigns the four expert identities. Normal minus rotated estimates the contribution of the learned assignment relative to that permutation. We also remove all expert residuals to evaluate the fine-tuned backbone, and compare the MoE with independently trained shared PPO. The difference between the 16-team and four-team routing effects tests whether training diversity changed the value of routing. [Reproduction commands](../../docs/vgcbench.md)

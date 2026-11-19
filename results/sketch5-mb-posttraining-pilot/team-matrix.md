# Which teams each bot played

In battle PPO training, we independently sampled one learner roster and one frozen BC opponent roster per game. Our learner brought slots 1–4 in order. The BC opponent selected its leads and reserves from its six at team preview.

| Assignment | Our PPO roster pool | Frozen BC opponent roster pool |
|---|---|---|
| Training and familiar/familiar evaluation | MC461-MB, MB537, MB51, MB45 | MB501, MB245, MB269, MB387, MB106, MB90, MB40, MB102 |
| Familiar learner / unseen opponent | MC461-MB, MB537, MB51, MB45 | MB481, MB335, MB259, MB558, MB163, MB562, MB149, MB337 |
| Unseen learner / familiar opponent | MB516, MB216, MB99, MB444 | MB501, MB245, MB269, MB387, MB106, MB90, MB40, MB102 |
| Both unseen | MB516, MB216, MB99, MB444 | MB481, MB335, MB259, MB558, MB163, MB562, MB149, MB337 |
| Both unseen, assignments reversed | MB481, MB335, MB259, MB558, MB163, MB562, MB149, MB337 | MB516, MB216, MB99, MB444 |
| MC461 development | MC461-MB | MB381, MB383, MB392, MB103, MB361, MB427, MB528, MB191, MB512, MB375, MB507, MB115 |

Each evaluation row crosses every learner roster with every opposing roster. Matrix evaluation played eight games per pair per checkpoint, four in each simulator seat. The MC461 development evaluation played 64 per pair per checkpoint.

## Roster sheets in slot order

In the fixed-four runs, slots 1–2 led and slots 3–4 were in back. Adaptive previews could choose any four. The individual game records store both ordered selections as slot numbers. [Matrix games](matrix-games.csv.gz) · [Development games](development-games.csv.gz) · [Reciprocal games](reciprocal-games.csv.gz)

| Pool | Roster and full set sheet | Slot 1 | Slot 2 | Slot 3 | Slot 4 | Slot 5 | Slot 6 |
|---|---|---|---|---|---|---|---|
| PPO training | [MC461-MB](../../teams/reg-mb-variant/MC461-MB.txt) | Meowstic-F | Kommo-o | Farigiraf | Sneasler | Kingambit | Charizard-Mega-Y |
| PPO training | [MB537](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB537.txt) | Charizard | Whimsicott | Basculegion | Mawile | Farigiraf | Garchomp |
| PPO training | [MB51](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB51.txt) | Aerodactyl | Blaziken | Garchomp | Metagross | Rotom-Wash | Sinistcha |
| PPO training | [MB45](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB45.txt) | Swampert | Sableye | Basculegion-F | Venusaur | Archaludon | Pelipper |
| BC training opponent | [MB501](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB501.txt) | Slowking-Galar | Dragonite | Sinistcha | Farigiraf | Mawile | Incineroar |
| BC training opponent | [MB245](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB245.txt) | Pelipper | Swampert | Archaludon | Incineroar | Whimsicott | Staraptor |
| BC training opponent | [MB269](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB269.txt) | Camerupt | Abomasnow | Farigiraf | Staraptor | Sneasler | Kingambit |
| BC training opponent | [MB387](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB387.txt) | Meowstic | Sneasler | Overqwil | Delphox | Sableye | Klefki |
| BC training opponent | [MB106](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB106.txt) | Staraptor | Sinistcha | Grimmsnarl | Dragalge | Kingambit | Incineroar |
| BC training opponent | [MB90](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB90.txt) | Floette-Eternal | Gholdengo | Annihilape | Grimmsnarl | Incineroar | Milotic |
| BC training opponent | [MB40](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB40.txt) | Raichu | Sylveon | Garchomp | Whimsicott | Gyarados | Gholdengo |
| BC training opponent | [MB102](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB102.txt) | Staraptor | Grimmsnarl | Sinistcha | Basculegion | Garchomp | Delphox |
| Held-out learner | [MB516](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB516.txt) | Delphox | Raichu | Sneasler | Sinistcha | Gyarados | Gliscor |
| Held-out learner | [MB216](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB216.txt) | Pelipper | Swampert | Archaludon | Politoed | Floette-Eternal | Sinistcha |
| Held-out learner | [MB99](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB99.txt) | Corviknight | Sceptile | Tyranitar | Sneasler | Rotom-Heat | Garchomp |
| Held-out learner | [MB444](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB444.txt) | Charizard | Incineroar | Venusaur | Pelipper | Archaludon | Hydreigon |
| Held-out opponent | [MB481](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB481.txt) | Charizard | Kingambit | Basculegion | Sylveon | Garchomp | Dragonite |
| Held-out opponent | [MB335](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB335.txt) | Incineroar | Sneasler | Delphox | Kingambit | Sinistcha | Blastoise |
| Held-out opponent | [MB259](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB259.txt) | Staraptor | Sinistcha | Farigiraf | Barbaracle | Torkoal | Sylveon |
| Held-out opponent | [MB558](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB558.txt) | Venusaur | Houndoom | Annihilape | Torkoal | Kangaskhan | Farigiraf |
| Held-out opponent | [MB163](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB163.txt) | Raichu-Mega-X | Incineroar | Sinistcha | Kommo-o | Gholdengo | Whimsicott |
| Held-out opponent | [MB562](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB562.txt) | Tyranitar | Excadrill | Staraptor | Milotic | Gholdengo | Sinistcha-Masterpiece |
| Held-out opponent | [MB149](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB149.txt) | Whimsicott | Glimmora | Floette-Eternal | Basculegion | Malamar | Torkoal |
| Held-out opponent | [MB337](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/teams/reg_mb/MB337.txt) | Talonflame | Garchomp | Incineroar | Basculegion | Gholdengo | Floette-Mega |
| BC development opponent | [MB381](../../teams/vgcbench-mb-transfer/MB381.txt) | Scolipede | Raichu | Sinistcha | Weavile | Ceruledge | Milotic |
| BC development opponent | [MB383](../../teams/vgcbench-mb-transfer/MB383.txt) | Pyroar | Vileplume | Torkoal | Incineroar | Floette-Eternal | Farigiraf |
| BC development opponent | [MB392](../../teams/vgcbench-mb-transfer/MB392.txt) | Vivillon | Charizard | Maushold-Four | Garchomp | Whimsicott | Blastoise |
| BC development opponent | [MB103](../../teams/vgcbench-mb-transfer/MB103.txt) | Archaludon | Pelipper | Raichu | Basculegion | Sinistcha | Froslass |
| BC development opponent | [MB361](../../teams/vgcbench-mb-transfer/MB361.txt) | Charizard-Mega-Y | Glimmora-Mega | Kingambit | Mamoswine | Basculegion | Whimsicott |
| BC development opponent | [MB427](../../teams/vgcbench-mb-transfer/MB427.txt) | Pyroar | Aerodactyl | Torkoal | Venusaur | Farigiraf | Sylveon |
| BC development opponent | [MB528](../../teams/vgcbench-mb-transfer/MB528.txt) | Sinistcha | Vivillon | Floette-Eternal | Ceruledge | Incineroar | Scrafty |
| BC development opponent | [MB191](../../teams/vgcbench-mb-transfer/MB191.txt) | Pyroar-Mega | Whimsicott | Ninetales | Farigiraf | Basculegion | Floette-Eternal |
| BC development opponent | [MB512](../../teams/vgcbench-mb-transfer/MB512.txt) | Charizard | Eelektross | Whimsicott | Garchomp | Sylveon | Toxapex |
| BC development opponent | [MB375](../../teams/vgcbench-mb-transfer/MB375.txt) | Charizard | Metagross | Tsareena | Aerodactyl | Garchomp | Sylveon |
| BC development opponent | [MB507](../../teams/vgcbench-mb-transfer/MB507.txt) | Metagross | Incineroar | Sinistcha | Tyranitar | Excadrill | Dragonite |
| BC development opponent | [MB115](../../teams/vgcbench-mb-transfer/MB115.txt) | Staraptor | Sinistcha | Garchomp | Incineroar | Glimmora | Whimsicott |

## Wins and games for each roster pair

Rows name the team our PPO battle policy piloted. Columns name the frozen BC opponent’s team. Counts pool the four training seeds. The matrix and reversed games used the released BC preview on both sides. In the development games our PPO learner brought its fixed first four.

### Both familiar

| Our roster ↓ / BC roster → | MB501 | MB245 | MB269 | MB387 | MB106 | MB90 | MB40 | MB102 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MC461-MB | 24/32 | 17/32 | 24/32 | 26/32 | 28/32 | 19/32 | 15/32 | 13/32 |
| MB537 | 29/32 | 17/32 | 16/32 | 28/32 | 28/32 | 22/32 | 17/32 | 23/32 |
| MB51 | 19/32 | 23/32 | 19/32 | 26/32 | 15/32 | 10/32 | 13/32 | 13/32 |
| MB45 | 15/32 | 18/32 | 22/32 | 30/32 | 21/32 | 13/32 | 13/32 | 19/32 |

### Familiar learner / unseen opponent

| Our roster ↓ / BC roster → | MB481 | MB335 | MB259 | MB558 | MB163 | MB562 | MB149 | MB337 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MC461-MB | 10/32 | 22/32 | 19/32 | 17/32 | 23/32 | 12/32 | 7/32 | 17/32 |
| MB537 | 14/32 | 14/32 | 24/32 | 20/32 | 21/32 | 23/32 | 17/32 | 10/32 |
| MB51 | 12/32 | 8/32 | 11/32 | 17/32 | 20/32 | 12/32 | 21/32 | 21/32 |
| MB45 | 8/32 | 10/32 | 12/32 | 18/32 | 24/32 | 12/32 | 18/32 | 4/32 |

### Unseen learner / familiar opponent

| Our roster ↓ / BC roster → | MB501 | MB245 | MB269 | MB387 | MB106 | MB90 | MB40 | MB102 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MB516 | 17/32 | 20/32 | 24/32 | 21/32 | 16/32 | 15/32 | 19/32 | 18/32 |
| MB216 | 26/32 | 23/32 | 22/32 | 28/32 | 30/32 | 16/32 | 12/32 | 21/32 |
| MB99 | 18/32 | 13/32 | 13/32 | 18/32 | 15/32 | 12/32 | 8/32 | 23/32 |
| MB444 | 30/32 | 18/32 | 25/32 | 30/32 | 19/32 | 18/32 | 20/32 | 12/32 |

### Both unseen

| Our roster ↓ / BC roster → | MB481 | MB335 | MB259 | MB558 | MB163 | MB562 | MB149 | MB337 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MB516 | 16/32 | 6/32 | 11/32 | 11/32 | 14/32 | 13/32 | 18/32 | 8/32 |
| MB216 | 6/32 | 3/32 | 12/32 | 30/32 | 12/32 | 18/32 | 8/32 | 13/32 |
| MB99 | 1/32 | 10/32 | 15/32 | 21/32 | 11/32 | 10/32 | 23/32 | 15/32 |
| MB444 | 11/32 | 13/32 | 17/32 | 16/32 | 18/32 | 11/32 | 7/32 | 14/32 |

### Both unseen, assignments reversed

| Our roster ↓ / BC roster → | MB516 | MB216 | MB99 | MB444 |
|---|---:|---:|---:|---:|
| MB481 | 28/32 | 17/32 | 19/32 | 13/32 |
| MB335 | 21/32 | 27/32 | 18/32 | 20/32 |
| MB259 | 20/32 | 12/32 | 12/32 | 17/32 |
| MB558 | 19/32 | 5/32 | 19/32 | 10/32 |
| MB163 | 20/32 | 9/32 | 24/32 | 20/32 |
| MB562 | 25/32 | 24/32 | 17/32 | 25/32 |
| MB149 | 19/32 | 12/32 | 11/32 | 26/32 |
| MB337 | 10/32 | 21/32 | 8/32 | 26/32 |

### MC461 development

| Our roster ↓ / BC roster → | MB381 | MB383 | MB392 | MB103 | MB361 | MB427 | MB528 | MB191 | MB512 | MB375 | MB507 | MB115 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MC461-MB | 187/256 | 88/256 | 163/256 | 183/256 | 105/256 | 103/256 | 60/256 | 71/256 | 187/256 | 99/256 | 92/256 | 211/256 |

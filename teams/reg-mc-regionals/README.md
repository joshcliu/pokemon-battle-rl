# Tournament rosters

The 22 team files are published Pokémon Champions Regulation M-C regional top-eight pastes, frozen from the [MunchStats VGCPastes archive](https://github.com/PizzaTimeJoshua/munchstats/tree/12315f299202f168f5bd123454c1959a3a7f7fc9/site). The [split manifest](split.csv) credits the players and links to the individual pastes. Source order determines the selected four and the two leads.

From the frozen archive we took regional top-eight entries with archived pastes and stat spreads, excluding descriptions marked Juniors or Seniors. We required six complete sets with abilities, natures, nonzero stat spreads and four moves, and verified legality with the project’s pinned Showdown validator. We normalized each species to its Showdown base species for deduplication. In descending archive date and ID order, we kept one copy of each unordered six and each unordered playable four, also reserving the project’s Sun and Balance combinations. Of 25 candidates, three were duplicates.

We sorted the remaining IDs by the SHA-256 of the UTF-8 string `roster-transfer-v1/<team_id>` and reserved the first seven for evaluation. The remaining fifteen form the uniformly sampled training pool. Individual species and partial cores can recur across splits. Published rosters are a convenience sample of competitors who shared full sets.

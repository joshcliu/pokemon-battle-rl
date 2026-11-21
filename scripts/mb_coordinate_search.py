"""Factor the two active Pokémon's commands to propose and validate new joint actions."""

import numpy as np
from mb_midgame_plan import backed_up_values, value
from mb_opening_planner import text_choice
from mb_opponent_belief import public_beliefs


def coordinate_search(engine, view, pos, seed, prior, actions, logprior, mode):
    ranked = logprior.argsort()[::-1].tolist()
    anchor = ranked[0]
    center = actions[anchor]
    draws = public_beliefs(view, pos, prior, 12)

    def evaluate(ids, beliefs, salt):
        orders = [text_choice(view, actions[j].tolist()).removeprefix("/choose ") for j in ids]
        worlds = engine.forecast(pos, orders, beliefs, 1, salt, partial=True)
        q = np.full((len(worlds), len(ids)), np.nan)
        rows = []
        indices = []
        for w, leaves in enumerate(worlds):
            for j, leaf in enumerate(leaves):
                if leaf is not None:
                    indices.append((w, j))
                    rows.append(leaf)
        if rows:
            if mode == "coordinate-critic":
                from mb_world_value import leaf_values

                pred = leaf_values([rows], seed, 1)[:, 0]
            else:
                pred = backed_up_values([rows], value(seed), 1)[:, 0]
            for idx, v in zip(indices, pred, strict=True):
                q[idx] = v
        gains = {}
        for j, k in enumerate(ids):
            paired = np.isfinite(q[:, 0]) & np.isfinite(q[:, j])
            if paired.sum() >= 2:
                gains[k] = float((q[paired, j] - q[paired, 0]).mean())
        return gains, len(worlds)

    singles = []
    for slot in (0, 1):
        singles.append([j for j in ranked if actions[j, 1 - slot] == center[1 - slot]][:9])
    initial = list(dict.fromkeys([anchor, *ranked[:4], *singles[0], *singles[1]]))
    gains, n = evaluate(initial, draws[:4], "coordinate-screen-")
    if anchor not in gains:
        return center.tolist(), n
    # Estimate the gain from each slot independently with the other anchored.
    # The model then simulates actual joint commands to resolve interactions.
    marginals = [{actions[j, s]: gains[j] for j in singles[s] if j in gains} for s in (0, 1)]
    prior_delta = 0.02 * (logprior - logprior[anchor])
    proposed = []
    for j in ranked:
        x, y = actions[j]
        if x in marginals[0] and y in marginals[1]:
            proposed.append((marginals[0][x] + marginals[1][y] + prior_delta[j], j))
    predicted = [j for _, j in sorted(proposed, reverse=True)]
    best_observed = sorted(gains, key=lambda j: gains[j] + prior_delta[j], reverse=True)
    finalists = list(dict.fromkeys([anchor, *best_observed[:3], *predicted, *best_observed]))[:8]
    final, m = evaluate(finalists, draws[4:], "coordinate-final-")
    if anchor not in final:
        return actions[best_observed[0]].tolist(), n + m
    # Independent worlds choose the final combination. Use paired first-stage
    # information at one-third weight for combinations observed in both stages.
    best = max(final, key=lambda j: (2 * final[j] + gains.get(j, final[j])) / 3 + prior_delta[j])
    return actions[best].tolist(), n + m

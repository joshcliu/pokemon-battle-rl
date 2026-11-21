"""Hold the public board fixed and intervene on the opponent's actual private reserves."""

import copy
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import torch
from mb_midgame_plan import estimates, projection, successive_halving
from mb_policy_iteration_eval import OUT, RUN, SEEDS, SPEC, verify
from mb_residual_preview import pilot
from vgc_posttrain import Match, atomic_rows, batch_obs, team_text


def change_hidden_stats(sheet):
    sets = sheet.strip().split("\n\n")
    assert len(sets) == 6
    for j in range(2, 6):
        # Only change reserves, so lead speed order and entry effects are identical.
        replacement = "EVs: 32 HP / 32 Def / 2 SpD"
        sets[j], n = re.subn(r"(?m)^EVs: .+$", replacement, sets[j])
        assert n == 1
        sets[j], n = re.subn(r"(?m)^\w+ Nature$", "Relaxed Nature", sets[j])
        assert n == 1
    return "\n\n".join(sets) + "\n"


def state(match, own, foe, seed, side, line):
    def pick(view, sim):
        for i, m in enumerate(view.team.values(), 1):
            m._selected_in_teampreview = str(i) in line
        match.opponent.lineup = line
        match.opponent.generator.manual_seed(sim + 1_000_000_000)
        return tuple(map(int, line))

    match.opponent.preview = pick
    match.reset(own, foe, 957_000_000 + seed, side, lineup="1234")
    ob = match.obs()
    view = match.view
    pos = projection(view)
    assert pos["vacancies"] == 2 and len(pos["unknown"]) == 4
    assert all("stats" not in m for m in [*pos["foe"], *pos["unknown"]] if m)
    result = []
    for depth in (1, 3):
        a, v, prior, n = estimates(pilot(seed), batch_obs(ob), view, seed, False, depth)
        assert n >= 2
        result.extend([a, v.tolist(), prior.tolist(), n])
    result.append(successive_halving(pilot(seed), batch_obs(ob), view, seed, audit=True))
    return (
        {k: np.array(v, copy=True) for k, v in ob.items()},
        json.dumps(pos, sort_keys=True),
        result,
    )


@torch.no_grad()
def main():
    verify()
    match = Match()
    match.opponent = copy.copy(match.opponent)
    match.opponent.generator = torch.Generator()
    out = []
    with tempfile.TemporaryDirectory(dir=RUN) as tmp:
        for seed in SEEDS:
            own, foe = SPEC["own_train"][seed + 1], SPEC["foe_train"][seed]
            original = team_text(foe)
            modified = change_hidden_stats(original)
            assert original != modified
            changed = Path(tmp) / f"private-{seed}.txt"
            changed.write_text(modified)
            for seat in ("p1", "p2"):
                ref, pos, choices = state(match, own, foe, seed, seat, "1234")
                for sheet, line, label in [
                    (foe, "1256", "reserves"),
                    (str(changed), "1256", "reserves-and-stats"),
                ]:
                    actual, new_pos, new_choices = state(match, own, sheet, seed, seat, line)
                    assert all(np.array_equal(actual[k], ref[k]) for k in ref)
                    assert new_pos == pos and new_choices == choices
                    out.append(
                        dict(
                            seed=seed,
                            seat=seat,
                            own=Path(own).stem,
                            opponent=Path(foe).stem,
                            intervention=label,
                            original="1234",
                            alternate=line,
                            identical_observation=True,
                            identical_forecasts=True,
                            identical_command=True,
                        )
                    )
    atomic_rows(OUT / "public-planning-audit.csv", out)
    print(
        "Identical public inputs, matched forecasts and selected commands for",
        len(out),
        "private roster interventions",
        flush=True,
    )


if __name__ == "__main__":
    main()

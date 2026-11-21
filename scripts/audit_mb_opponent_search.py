"""Intervene on the true hidden four and private stats while keeping the learner view fixed."""

import copy
import hashlib
import json
import tempfile
from pathlib import Path

import torch
from audit_mb_public_planning import change_hidden_stats
from mb_midgame_plan import projection
from mb_opponent_belief import OUT, RUN, private_prior, public_beliefs
from mb_opponent_search import act
from mb_policy_iteration_eval import SEEDS, SPEC
from mb_public_preview_search import public_fixture
from mb_residual_preview import pilot
from vgc_posttrain import Match, atomic_rows, batch_obs, team_text


@torch.no_grad()
def test_fixture(match, own, foe, seed, side, line, prior):
    def preview(view, sim):
        for i, m in enumerate(view.team.values(), 1):
            m._selected_in_teampreview = str(i) in line
        match.opponent.lineup = line
        match.opponent.generator.manual_seed(sim + 1_000_000_000)
        return tuple(map(int, line))

    match.opponent.preview = preview
    match.reset(own, foe, 958_000_000 + seed, side, lineup="1234")
    view = match.view
    pos = projection(view)
    assert pos["vacancies"] == 2
    rng = match.opponent.generator.get_state()
    worlds = public_beliefs(view, pos, prior, 8)
    action, n = act(pilot(seed), batch_obs(match.obs()), view, seed, prior, "reply-wide")
    assert n >= 2 and torch.equal(rng, match.opponent.generator.get_state())
    digest = hashlib.sha256(json.dumps([worlds, action, n], sort_keys=True).encode()).hexdigest()
    return digest


def main():
    match = Match()
    match.opponent = copy.copy(match.opponent)
    match.opponent.generator = torch.Generator()
    out = []
    with tempfile.TemporaryDirectory(dir=RUN) as tmp:
        for seed in SEEDS:
            own, foe = SPEC["own_train"][seed + 1], SPEC["foe_train"][seed]
            prior = private_prior(own, public_fixture(own, foe), seed)
            modified = Path(tmp) / f"s{seed}.txt"
            modified.write_text(change_hidden_stats(team_text(foe)))
            assert public_fixture(own, str(modified)) == public_fixture(own, foe)
            for side in ("p1", "p2"):
                ref = test_fixture(match, own, foe, seed, side, "1234", prior)
                for path, label in ((foe, "reserves"), (str(modified), "reserves and spreads")):
                    assert test_fixture(match, own, path, seed, side, "1256", prior) == ref
                    out.append(
                        dict(
                            seed=seed,
                            seat=side,
                            own=Path(own).stem,
                            opponent=Path(foe).stem,
                            intervention=label,
                            identical_hypotheses_logits_and_action=True,
                            sha256=ref,
                        )
                    )
    atomic_rows(OUT / "public-input-audit.csv", out)
    print("Verified", len(out), "hidden-information interventions", flush=True)


if __name__ == "__main__":
    main()

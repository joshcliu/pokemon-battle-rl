"""Enumerate all 360 preview choices on matched boards to measure matchup sensitivity."""

import itertools
from pathlib import Path

import numpy as np
import torch
from mb_residual_preview import (
    OUT,
    ROSTERS,
    SEEDS,
    SPEC,
    VARIANTS,
    PreviewAdapter,
    acting_head,
    likelihood,
    preview_observation,
)
from poke_env.environment import DoublesEnv
from vgc_posttrain import Match, atomic_rows, players

ORDERS = torch.tensor(list(itertools.permutations(range(6), 4))).reshape(-1, 2, 2)
LEADS = list(itertools.combinations(range(6), 2))
STATE = torch.tensor([1 + LEADS.index(tuple(sorted(order[0].tolist()))) for order in ORDERS])


@torch.no_grad()
def distributions(match, own, foe, sim, heads, zero):
    records = {}

    def pick(view):
        player = players()[0]
        base = player.policy
        mons = list(view.team.values())
        for switches in view.available_switches:
            switches[:] = mons
        observations = []
        masks = []
        for chosen in [(), *LEADS]:
            for i, mon in enumerate(mons):
                mon._selected_in_teampreview = i in chosen
            observations.append(player.embed_battle(view, fake_rating=2000))
            masks.append(DoublesEnv.get_action_mask(view))
        for mon in mons:
            mon._selected_in_teampreview = False
        for blind in (False, True):
            raw = np.stack([preview_observation(x.copy(), blind) for x in observations])
            mask = torch.tensor(np.stack(masks)).bool()
            feat = base.mlp_extractor.forward_actor(
                base.pi_features_extractor(
                    dict(observation=torch.from_numpy(raw), action_mask=mask)
                )
            )
            scores = base.action_net(feat).reshape(16, 2, -1)[:, :, 1:7]
            mask = mask.reshape(16, 2, -1)[:, :, 1:7]
            states = torch.stack((torch.zeros_like(STATE), STATE), dim=1)

            def probabilities(net):
                lp, _, _ = likelihood(net, feat[states], scores[states], mask[states], ORDERS)
                probs = lp.double().exp().numpy()
                assert abs(probs.sum() - 1) < 2e-5
                return probs / probs.sum()

            prefix = "blind_" if blind else ""
            native = probabilities(zero)
            for seed, head in heads.items():
                records[seed, prefix + "native"] = native
                records[seed, prefix + "learned"] = probabilities(head)
        return "1234"

    match.reset(own, foe, sim, "p1", preview_picker=pick)
    return records


def probe():
    torch.set_num_threads(1)
    heads = {seed: acting_head(seed, SPEC["updates"]) for seed in SEEDS}
    zero = PreviewAdapter().eval()
    match = Match()
    rows = []
    summaries = []
    for direction, owns, foes in (
        ("forward", ROSTERS["own_test"], ROSTERS["foe_test"]),
        ("reverse", ROSTERS["foe_test"], ROSTERS["own_test"]),
    ):
        for i, own in enumerate(owns):
            boards = [distributions(match, own, foe, 390000000 + i, heads, zero) for foe in foes]
            for seed in SEEDS:
                for v in VARIANTS:
                    dist = np.stack([board[seed, v] for board in boards])
                    native = np.stack(
                        [
                            board[seed, ("blind_native" if v.startswith("blind") else "native")]
                            for board in boards
                        ]
                    )
                    mean = dist.mean(0)
                    sensitivity = np.abs(dist - mean).sum(1) / 2
                    change = np.abs(dist - native).sum(1) / 2
                    if v.startswith("blind"):
                        assert np.max(sensitivity) < 1e-5
                    modes = [
                        "".join(str(slot + 1) for slot in ORDERS[j].flatten().tolist())
                        for j in dist.argmax(1)
                    ]
                    for j, foe in enumerate(foes):
                        rows.append(
                            dict(
                                training_seed=seed,
                                cell=direction,
                                own=Path(own).stem,
                                foe=Path(foe).stem,
                                variant=v,
                                modal_lineup=modes[j],
                                tv_from_own_average=sensitivity[j],
                                tv_from_native=change[j],
                            )
                        )
                    summaries.append(
                        dict(
                            training_seed=seed,
                            cell=direction,
                            own=Path(own).stem,
                            variant=v,
                            foes=len(foes),
                            distinct_modal_lineups=len(set(modes)),
                            mean_tv_from_own_average=sensitivity.mean(),
                            mean_tv_from_native=change.mean(),
                        )
                    )
    atomic_rows(OUT / "preview-distributions.csv", rows)
    atomic_rows(OUT / "matchup-sensitivity.csv", summaries)
    print(
        "Enumerated 360 ordered lineups on each of 64 held-out matchups. Blinded distributions are invariant within own roster.",
        flush=True,
    )


if __name__ == "__main__":
    probe()

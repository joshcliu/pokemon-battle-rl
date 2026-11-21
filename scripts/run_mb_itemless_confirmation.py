"""Resume the frozen test with a recorded open-sheet adapter for legal empty held-item slots."""

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import confirm_mb_opponent_search as confirm
import mb_opponent_search
import mb_policy_transfer
import mb_public_policy_search
import mb_public_preview_search
from mb_opponent_belief import RUN
from vgc_posttrain import Match, team_text

ORIGINAL = mb_public_preview_search.public_sets
ADDENDUM = confirm.LOCK.parent / "itemless-input.json"


def public_sets(view):
    if all(m.item is not None for m in view.opponent_team.values()):
        return ORIGINAL(view)
    # Both existing sheet generators omit the item line for this sentinel. Keep
    # the real battle view intact, including the None passed to the original PPO.
    mons = {
        k: SimpleNamespace(
            species=m.species,
            ability=m.ability,
            moves=m.moves,
            item="unknown_item" if m.item is None else m.item,
        )
        for k, m in view.opponent_team.items()
    }
    return ORIGINAL(SimpleNamespace(opponent_team=mons))


def install():
    for module in (
        mb_public_preview_search,
        mb_public_policy_search,
        mb_opponent_search,
        mb_policy_transfer,
    ):
        assert module.public_sets in (ORIGINAL, public_sets)
        module.public_sets = public_sets


def fixture(mode):
    own, foe = confirm.SPEC["own_train"][0], confirm.SPEC["foe_train"][0]
    original = mb_public_preview_search.public_fixture(own, foe)
    assert len(original) == 6
    before = Match()

    def parity(v):
        assert public_sets(v) == ORIGINAL(v)
        return "1234"

    before.reset(own, foe, 568000000, "p1", preview_picker=parity)
    sets = team_text(foe).strip().split("\n\n")
    for i, s in enumerate(sets):
        header = s.splitlines()[0]
        # Training fixture with one ordinary held item removed.
        if " @ " in header and not re.search(r"ite(?: [XY])?$|mask$|core$", header, re.I):
            sets[i] = s.replace(header, header.split(" @ ")[0], 1)
            break
    else:
        raise ValueError("No suitable held item on training roster")
    with tempfile.TemporaryDirectory(dir=RUN) as t:
        path = Path(t) / "itemless-training.txt"
        path.write_text("\n\n".join(sets) + "\n")
        seen = []

        def check(v):
            before = [m.item for m in v.opponent_team.values()]
            assert before.count(None) == 1
            try:
                ORIGINAL(v)
            except AssertionError:
                pass
            else:
                raise AssertionError("Training fixture did not reproduce the original issue")
            corrected = public_sets(v)
            assert sum(m[2] == "unknown_item" for m in corrected) == 1
            assert before == [m.item for m in v.opponent_team.values()]
            seen.extend(corrected)
            return "1234"

        before.reset(own, str(path), 568000000, "p1", preview_picker=check)
        install()
        cell = (0, 0, 0, own, str(path), 668000000)
        candidates = mb_opponent_search.score((2, cell, mode, 2))
        first = mb_public_policy_search.play((2, cell), method=confirm.OLD)
        for x, y in zip(candidates, first[:2], strict=True):
            for k in ("sim", "side", "lineup", "opp_lineup"):
                assert x[k] == y[k]
        assert len(seen) == 6 and len(candidates) == 2
        print(
            "Training-only empty-item fixture passed with both frozen planners and matched lineups",
            flush=True,
        )


def inspect_open_sheets():
    affected = {}
    for foe in confirm.SPEC["confirmation_a"] + confirm.SPEC["confirmation_b"]:

        def inspect(v):
            empty = [m.species for m in v.opponent_team.values() if m.item is None]
            if empty:
                affected[Path(foe).stem] = empty
            else:
                assert ORIGINAL(v) == public_sets(v)
            assert len(public_sets(v)) == 6
            return "1234"

        Match().reset(confirm.SPEC["own_train"][0], foe, 570033815, "p1", preview_picker=inspect)
    return affected


def lock():
    def h(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    expected = dict(
        candidate_sha256=h(confirm.LOCK),
        adapter_sha256=h(Path(__file__)),
        affected_public_sheets=inspect_open_sheets(),
        order="Filed after original parser exception and before observing any second-round win rates.",
        intervention="When the original open-sheet extractor encounters a legal empty held-item slot (None), map that field of the public lineup-planning tuple to the frozen generator's existing no-item sentinel. It emits no item in the hypothetical team. Leave native observations, real rosters, opponent/pilot weights, RNG, legal mask, seeds and scores unchanged.",
        audit="Retain every scheduled battle. Report results for affected directions and all other directions. Also report the conservative win fraction counting every affected game as a forfeit by an unwrapped planner.",
    )
    if not ADDENDUM.exists():
        with tempfile.NamedTemporaryFile("w", dir=ADDENDUM.parent, delete=False) as f:
            json.dump(expected, f, indent=2, sort_keys=True)
            f.write("\n")
        try:
            os.link(f.name, ADDENDUM)
        except FileExistsError:
            pass
        finally:
            Path(f.name).unlink()
    assert json.loads(ADDENDUM.read_text()) == expected
    print("Verified empty-item adapter", expected["affected_public_sheets"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "score"))
    p.add_argument("--split", choices=confirm.SPLITS)
    p.add_argument("--seeds", nargs="+", type=int, default=confirm.SEEDS)
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    mode = json.loads(confirm.LOCK.read_text())["method"]
    confirm.freeze(mode)
    if args.stage == "check":
        fixture(mode)
        lock()
    else:
        assert ADDENDUM.exists() and args.split and set(args.seeds) <= set(confirm.SEEDS)
        lock()
        tasks = [(args.split, mode, s, c) for s in args.seeds for c in confirm.cells(args.split)]
        with ProcessPoolExecutor(
            args.jobs, mp_context=multiprocessing.get_context("spawn"), initializer=install
        ) as pool:
            for k, _ in enumerate(pool.map(confirm.matched, tasks)):
                if (k + 1) % 32 == 0:
                    print(args.split, k + 1, "of", len(tasks), "matched groups", flush=True)

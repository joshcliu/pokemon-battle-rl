"""Measure first-slot target preferences on identical, untouched M-B opening boards."""

import json
from pathlib import Path

import numpy as np
import torch
from trace_battle_actions import matchup, ordinary
from vgc_posttrain import REPORT, Match, actor, atomic_rows


def opening_cases(spec):
    match = Match()
    boards, cases = [], []
    for i, own in enumerate(spec["own_test"]):
        for j, foe in enumerate(spec["foe_test"]):
            for k in range(4):
                seed = 885_000_000 + i * 10_000 + j * 100 + k
                side = ("p1", "p2")[k % 2]
                match.reset(own, foe, seed, side)
                targets = match.view.opponent_active_pokemon
                legal = match.battle.legal_actions(side)
                first = {a: joint.first_order for (a, _), joint in legal.items()}
                index = len(boards)
                boards.append(match.obs())
                for a, order in first.items():
                    b = a + 1
                    if not (
                        order.move_target == 1
                        and b in first
                        and first[b].move_target == 2
                        and first[b].order == order.order
                        and not order.mega
                        and ordinary(order.order)
                    ):
                        continue
                    mult = [matchup(order.order, t) for t in targets]
                    if mult[0] == mult[1]:
                        continue
                    high, low = (a, b) if mult[0] > mult[1] else (b, a)
                    cases.append(
                        dict(
                            board=index,
                            own=Path(own).stem,
                            foe=Path(foe).stem,
                            seed=seed,
                            move=order.order.id,
                            hi=high,
                            lo=low,
                        )
                    )
    if not cases:
        raise ValueError("No eligible target pairs on opening boards")
    return boards, cases


@torch.inference_mode()
def main():
    spec = json.loads((REPORT / "protocol.json").read_text())
    boards, cases = opening_cases(spec)
    rows = []
    for label in [
        "bc",
        *(f"{arm}-{s}" for arm in ("native", "flat", "structured") for s in spec["training_seeds"]),
    ]:
        model = actor(label)
        base = model if label == "bc" else model.base
        first = []
        for start in range(0, len(boards), 16):
            batch = boards[start : start + 16]
            obs = {k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in batch[0]}
            logits = base.get_logits(obs, actor_grad=False)[0][:, :107]
            if label != "bc" and model.extra:
                encoded = model.extra.encode(obs["observation"], base.pi_features_extractor)
                logits = logits + model.extra.scores(encoded)
            first.extend(logits)
        for c in cases:
            logits = first[c["board"]]
            p = (logits[c["hi"]] - logits[c["lo"]]).sigmoid().item()
            rows.append(dict(policy=label, **c, p_higher_multiplier=p))
    atomic_rows(REPORT / "target-preferences.csv", rows)
    print("Probe cases", len(cases), "from", len(boards), "opening boards", flush=True)


if __name__ == "__main__":
    main()

"""Evaluate the prespecified four transfer conditions with matched sides and random streams."""

import json

from vgc_posttrain import REPORT, evaluate


def matrix_tasks(label):
    spec = json.loads((REPORT / "protocol.json").read_text())
    if label in ("bc", "bc-native") or label.startswith("preview-"):
        policy = "bc"
        selector = "fixed" if label == "bc" else "native" if label == "bc-native" else label
    elif label.startswith("native-preview-"):
        policy = "native-" + label.rsplit("-", 1)[1]
        selector = "native"
    else:
        policy, selector = label, "fixed"
    tasks = []
    for i, own_key in enumerate(("own_train", "own_test")):
        for j, foe_key in enumerate(("foe_train", "foe_test")):
            for a, own in enumerate(spec[own_key]):
                for b, foe in enumerate(spec[foe_key]):
                    base = 800_000_000 + i * 1_000_000 + j * 100_000 + a * 10_000 + b * 100
                    seeds = [
                        (base + k, ("p1", "p2")[k % 2]) for k in range(spec["eval_games_per_pair"])
                    ]
                    tasks.append(
                        (policy, selector, own, foe, seeds, "sample", own_key + "_" + foe_key)
                    )
    return tasks


def test_matrix(jobs, label):
    evaluate("matrix-" + label, jobs, matrix_tasks(label))

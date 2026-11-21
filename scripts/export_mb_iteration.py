"""Package trained residuals and their exact parent checkpoint identities."""

import csv
import hashlib
import json
import shutil

import numpy as np
import torch
from mb_policy_iteration_eval import OUT, RUN, SEEDS, verify
from mb_residual_preview import DIVERSE_OUT, DIVERSE_RUN
from vgc_posttrain import atomic_rows


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def main():
    verify()
    with (DIVERSE_OUT / "checkpoints.csv").open() as f:
        parents = {
            int(r["training_seed"]): r["checkpoint_sha256"]
            for r in csv.DictReader(f)
            if r["arm"] == "moe"
        }
    rows = []
    for s in SEEDS:
        assert digest(DIVERSE_RUN / f"moe-{s}" / "policy.pt") == parents[s]
        for name, file in (
            ("es", RUN / "es" / f"seed-{s}" / "weights.json"),
            ("mcq-40", RUN / "fitted" / f"seed-{s}" / "q-40.pt"),
            ("lineup-q", RUN / "lineup-value" / f"seed-{s}.npz"),
            ("context-ppo", RUN / "context-ppo" / f"seed-{s}" / "progress.pt"),
            ("endgame-value", RUN / "endgame" / f"seed-{s}" / "value.npy"),
            ("midgame-value", RUN / "midgame" / f"seed-{s}" / "value.npy"),
        ):
            if name in ("context-ppo", "endgame-value", "midgame-value") and not file.exists():
                continue
            if name == "context-ppo":
                ck = torch.load(file, weights_only=True, map_location="cpu")
                if ck["update"] < 64:
                    continue
                assert ck["seed"] == s and ck["update"] == 64
            if file.suffix == ".npy":
                values = np.load(file)
                assert values.shape == ((14,) if name == "endgame-value" else (23,))
                assert np.isfinite(values).all()
            if name == "es":
                ck = json.loads(file.read_text())
                assert ck["seed"] == s and ck["update"] == 24 and len(ck["logs"]) == 24
            dest = OUT / "weights" / f"{name}-seed-{s}{file.suffix}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or digest(dest) != digest(file):
                tmp = dest.with_suffix(".tmp")
                shutil.copyfile(file, tmp)
                tmp.replace(dest)
            rows.append(
                dict(
                    method=name,
                    seed=s,
                    checkpoint=str(dest.relative_to(OUT)),
                    sha256=digest(dest),
                    parent_sha256=parents[s],
                    protocol_sha256=digest(OUT / "protocol.json"),
                )
            )
    atomic_rows(OUT / "checkpoints.csv", rows)
    print("Exported", len(rows), "seed-specific residual weights", flush=True)


if __name__ == "__main__":
    main()

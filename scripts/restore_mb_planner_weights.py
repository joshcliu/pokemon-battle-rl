"""Restore the frozen planner's inference weights to the paths used on Sketch5."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def check_hash(path, expected):
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, found {actual}")


def restore(root):
    final = root / "results/sketch5-mb-opponent-search"
    earlier = root / "results/sketch5-mb-policy-iteration/weights"
    manifest = json.loads((final / "confirmation/candidate.json").read_text())
    sources = {
        "runs/sketch5-mb-long-ppo/shared-2/policy.pt": final / "weights/shared-ppo-seed-2.pt",
    }
    for seed in manifest["training_seeds"]:
        prefix = "runs/sketch5-mb-policy-iteration"
        sources[f"runs/sketch5-mb-team-diversity/moe-{seed}/policy.pt"] = (
            final / f"weights/moe-seed-{seed}.pt"
        )
        sources[f"{prefix}/es/seed-{seed}/weights.json"] = earlier / f"es-seed-{seed}.json"
        sources[f"{prefix}/lineup-value/seed-{seed}.npz"] = earlier / f"lineup-q-seed-{seed}.npz"
        sources[f"{prefix}/midgame/seed-{seed}/value.npy"] = (
            earlier / f"midgame-value-seed-{seed}.npy"
        )

    hashes = {k: v for k, v in manifest["frozen_assets"].items() if k.startswith("runs/")}
    if sources.keys() != hashes.keys():
        raise ValueError("Published weights and frozen run paths differ")
    missing = []
    for relative, source in sources.items():
        target = root / relative
        check_hash(source, hashes[relative])
        if target.exists():
            check_hash(target, hashes[relative])
        else:
            missing.append((source, target, hashes[relative]))

    for source, target, expected in missing:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent) as tmp:
            with source.open("rb") as handle:
                shutil.copyfileobj(handle, tmp)
            tmp.flush()
            check_hash(Path(tmp.name), expected)
            try:
                os.link(tmp.name, target)
            except FileExistsError:
                check_hash(target, expected)
    print(f"Verified {len(sources)} frozen weight files under {root / 'runs'}")


if __name__ == "__main__":
    restore(Path(__file__).resolve().parents[1])

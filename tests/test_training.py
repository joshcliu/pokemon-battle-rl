import subprocess
import sys

import pytest
import torch

from pokemon_rl.battle import ROOT


@pytest.mark.integration
@pytest.mark.skipif(
    not (ROOT / ".cache/pokemon-showdown/dist/sim/index.js").exists(),
    reason="Run scripts/setup_simulator.py first",
)
def test_train_checkpoint_and_reload_evaluation(tmp_path):
    common = [sys.executable, "-m", "pokemon_rl.cli"]
    subprocess.run(
        [
            *common,
            "train",
            "--steps",
            "64",
            "--rollout-steps",
            "64",
            "--epochs",
            "1",
            "--replays",
            "0",
            "--run-dir",
            str(tmp_path / "train"),
        ],
        cwd=ROOT,
        check=True,
        timeout=60,
        capture_output=True,
        text=True,
    )
    checkpoint = tmp_path / "train/checkpoint.pt"
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["steps"] == 64
    assert all(torch.isfinite(t).all() for t in saved["policy"].values())
    subprocess.run(
        [
            *common,
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--battles",
            "2",
            "--opponents",
            "random",
            "damage",
            "--replays",
            "0",
            "--run-dir",
            str(tmp_path / "eval"),
        ],
        cwd=ROOT,
        check=True,
        timeout=60,
        capture_output=True,
        text=True,
    )
    assert (tmp_path / "eval/summary.json").exists()
    warm = tmp_path / "warm"
    subprocess.run(
        [
            *common,
            "train",
            "--init-checkpoint",
            str(checkpoint),
            "--save-initial",
            "--seed",
            "17",
            "--steps",
            "32",
            "--epochs",
            "1",
            "--replays",
            "0",
            "--run-dir",
            str(warm),
        ],
        cwd=ROOT,
        check=True,
        timeout=60,
        capture_output=True,
        text=True,
    )
    initial = torch.load(warm / "checkpoint-0.pt", weights_only=True)
    assert initial["steps"] == 0 and initial["seed"] == 17
    assert all(torch.equal(initial["policy"][k], v) for k, v in saved["policy"].items())
    updated = torch.load(warm / "checkpoint.pt", weights_only=True)
    assert updated["steps"] == 32
    assert any(not torch.equal(updated["policy"][k], v) for k, v in saved["policy"].items())
    assert all(s["step"].item() == 1 for s in updated["optimizer"]["state"].values())


def test_sampled_bc_opponent_has_an_independent_rng(monkeypatch):
    from pokemon_rl import vgcbench

    # Constructing the actual actor initializes a neural network before loading its weights.
    monkeypatch.setattr(vgcbench, "load_bc_player", lambda fmt: torch.rand(16))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(112)
        before = torch.get_rng_state()
        opponent = vgcbench.TrainingBC("gen9championsvgc2026regmb")
        assert torch.equal(before, torch.get_rng_state())
        reference = torch.Generator().manual_seed(826002288)
        opponent.generator.manual_seed(826002288)
        for _ in range(3):
            torch.rand(4)
            before = torch.get_rng_state()
            sampled = opponent.sample(torch.rand, 5)
            assert torch.equal(sampled, torch.rand(5, generator=reference))
            assert torch.equal(before, torch.get_rng_state())

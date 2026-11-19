"""Restart PPO with the exact optimizer, random streams and in-flight Showdown battles."""

import csv
import hashlib
import json

import torch


def check_reference(original, dest, model):
    file = original / dest.name / "policy.pt"
    with (original.parent.parent / "results" / original.name / "checkpoints.csv").open() as f:
        expected = next(
            row["sha256"] for row in csv.DictReader(f) if row["checkpoint"] == dest.name
        )
    assert hashlib.sha256(file.read_bytes()).hexdigest() == expected
    state = torch.load(file, weights_only=True, map_location="cpu")["state"]
    actual = model.state_dict()
    assert set(state) == set(actual) and all(torch.equal(v, actual[k]) for k, v in state.items()), (
        dest.name
    )
    row = dict(reference_sha256=expected, identical_tensors=len(actual), steps=65536)
    (dest / "reference.json").write_text(json.dumps(row, indent=2) + "\n")


def observation_hash(stream):
    obs = stream.match.obs()
    return hashlib.sha256(obs["observation"].tobytes() + obs["action_mask"].tobytes()).hexdigest()


def save_progress(dest, model, optimizer, sampling, streams, logs):
    states = [
        dict(
            setup=s.setup,
            history=s.history,
            episodes=s.episodes,
            record=s.record,
            team_rng=s.rng.bit_generator.state,
            opponent_rng=s.match.opponent.generator.get_state(),
            obs_hash=observation_hash(s),
        )
        for s in streams
    ]
    tmp = dest / "progress.tmp"
    torch.save(
        dict(
            run=dest.name,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            learner_rng=torch.get_rng_state(),
            sampling_rng=sampling.get_state(),
            streams=states,
            logs=logs,
        ),
        tmp,
    )
    tmp.replace(dest / "progress.pt")


def load_progress(dest, model, optimizer, sampling, streams):
    state = torch.load(dest / "progress.pt", weights_only=True, map_location="cpu")
    assert state["run"] == dest.name
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    for s, d in zip(streams, state["streams"], strict=True):
        s.setup, s.history, s.record, s.episodes = (
            d["setup"],
            d["history"],
            d["record"],
            d["episodes"],
        )
        s.match.reset(*s.setup, preview="native")
        for choice in s.history:
            done, _ = s.match.step(choice)
            assert not done
        assert observation_hash(s) == d["obs_hash"]
        assert torch.equal(s.match.opponent.generator.get_state(), d["opponent_rng"])
        s.rng.bit_generator.state = d["team_rng"]
    sampling.set_state(state["sampling_rng"])
    torch.set_rng_state(state["learner_rng"])
    return state["logs"]

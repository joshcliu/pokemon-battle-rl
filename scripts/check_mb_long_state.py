"""Test exact resumption of an active battle and the PPO optimizer on Sketch5."""

import tempfile
from pathlib import Path

import torch
from mb_long_state import load_progress, save_progress
from mb_roster_ppo import Stream, batch, create_model, verify_assets


def check():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = create_model("moe1")
    streams = [Stream(2, i) for i in (0, 1)]
    opt = torch.optim.Adam(model.parameters(), lr=3e-5)
    sample = torch.Generator().manual_seed(231)

    def update():
        obs = batch([s.match.obs() for s in streams])
        a, v, lp = model(obs)
        opt.zero_grad()
        (v.square().mean() - lp.mean()).backward()
        opt.step()
        for s, acts in zip(streams, a.tolist(), strict=True):
            s.step(acts)
        return torch.randperm(8, generator=sample), a.detach().clone()

    for _ in range(5):
        update()
    assert max(len(s.history) for s in streams) > 0
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        save_progress(p, model, opt, sample, streams, [{"steps": 5}])
        expected = [update() for _ in range(3)]
        params = {k: t.clone() for k, t in model.state_dict().items()}
        # Construct new simulators, model and optimizer. Rebuild in-flight games from moves.
        model = create_model("moe1")
        opt = torch.optim.Adam(model.parameters(), lr=3e-5)
        streams = [Stream(2, i) for i in (0, 1)]
        sample = torch.Generator()
        assert load_progress(p, model, opt, sample, streams) == [{"steps": 5}]
        for perm, action in expected:
            actual_perm, actual_action = update()
            assert torch.equal(perm, actual_perm) and torch.equal(action, actual_action)
        assert all(torch.equal(v, params[k]) for k, v in model.state_dict().items())
        print(
            "Independent restored battle streams, optimizer updates and model tensors reproduced."
        )


if __name__ == "__main__":
    verify_assets()
    check()

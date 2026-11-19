"""The frozen VGC-Bench behavior-cloned opponent and its native preview selection."""

import hashlib
import io
import zipfile
from contextlib import chdir

import numpy as np
import torch
from gymnasium import spaces

from .battle import ROOT

CACHE = ROOT / ".cache/vgcbench-external"
ZIP = CACHE / "vgcbench-bc100-mc461-20261118.zip"
BC_HASH = "57f5edcab415cf6ccc1b6231923c8b66d3b1b7249b6b2562b97531471e4ca60b"
UPSTREAM_SHA256 = {
    "vgc_bench/src/policy.py": "ab8969ebe18f88be955b19d1b9f77157a65ee23f38f9e8d1d5d601ae12f822a4",
    "vgc_bench/src/policy_player.py": "3c8cefbdea8d68a6504d9dd5fbf97a2d689b80cecb3e7522d67138b8b129d86f",
    "vgc_bench/src/utils.py": "534b93ea18806b84a52f21998b8485b56915794751ab77497e882734caaf0313",
    "data/abilities.json": "c6110d7963365d44d5741b204df8a92aa543db1d2a20a6e96233815c96d09d68",
    "data/items.json": "0bdc97131a61608de77cc1de8e4142d39561895d9cfc796d9f6a1fb576624c9b",
    "data/moves.json": "f0cb56065ec93cdf028f4ca187df0b9943a25a75812cd8a89d2c90e2ce24373e",
}


def load_bc_player(fmt):
    for path, expected in [
        (ZIP, BC_HASH),
        *((CACHE / "source" / p, h) for p, h in UPSTREAM_SHA256.items()),
    ]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"VGC-Bench asset differs from the pinned release: {path}")
    kwargs = dict(battle_format=fmt, start_listening=False)
    with chdir(CACHE / "source"):
        from vgc_bench.src.policy_player import MaskedActorCriticPolicy, PolicyPlayer
        from vgc_bench.src.utils import act_len, chunk_obs_len
    obs = spaces.Dict(
        {
            "observation": spaces.Box(-1, np.inf, shape=(12 * chunk_obs_len,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(2 * act_len),
        }
    )
    model = MaskedActorCriticPolicy(
        obs,
        spaces.MultiDiscrete([act_len] * 2),
        lambda _: 0.0,
        d_model=256,
        choose_on_teampreview=True,
    )
    with zipfile.ZipFile(ZIP) as z:
        weights = torch.load(
            io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True
        )
    model.load_state_dict(weights, strict=True)
    model.eval()
    return PolicyPlayer(policy=model, deterministic=False, **kwargs)


def learned_preview(player, view):
    # The converter checks available_switches while decoding its two preview
    # actions. Before selection, all six are legal, including the default leads.
    for switches in view.available_switches:
        switches[:] = view.team.values()
    return player.teampreview(view).removeprefix("/team ")


class TrainingBC:
    style = "vgcbench-bc"

    def __init__(self, fmt):
        with torch.random.fork_rng(devices=[]):
            self.player = load_bc_player(fmt)
        self.generator = torch.Generator(device="cpu")

    def sample(self, fn, *args):
        # SB3 samples through the global CPU generator. Preserve the learner's
        # action and minibatch RNG while advancing the opponent's own stream.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.generator.get_state())
            result = fn(*args)
            self.generator.set_state(torch.get_rng_state())
        return result

    def preview(self, view, seed):
        self.generator.manual_seed(seed)
        self.lineup = self.sample(learned_preview, self.player, view)
        self.generator.manual_seed(seed + 1_000_000_000)
        return tuple(map(int, self.lineup))

    def order(self, view):
        message = self.sample(self.player.choose_move, view).message
        if message == "/choose default":
            raise ValueError("BC returned a default order for an active battle request")
        return message.removeprefix("/choose ")

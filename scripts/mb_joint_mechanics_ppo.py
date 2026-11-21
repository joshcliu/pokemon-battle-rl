"""End-to-end joint-action PPO initialized from the trained tactical policy."""

import argparse
import copy
import functools

import numpy as np
import torch
from mb_long_state import load_progress, save_progress
from mb_mechanics_es import fitted, mechanics
from mb_policy_iteration_eval import OUT, RUN, SEEDS, distribution, verify
from mb_residual_preview import pilot, save
from mb_rollout_branches import pairing
from mb_roster_ppo import batch, create_model
from torch import nn
from torch.distributions import Categorical
from vgc_posttrain import Match, atomic_rows, batch_obs

STREAMS = 4
BLOCK = 512
TEMP = 0.70
BUDGET = 65536


class Stream:
    def __init__(self, seed, index):
        self.seed, self.index = seed, index
        self.match = Match(independent=True)
        self.match.opponent = copy.copy(self.match.opponent)
        self.match.opponent.generator = torch.Generator()
        self.episodes = []
        self.rng = np.random.default_rng([6285, seed, index])
        self.reset()

    def reset(self):
        g = STREAMS * len(self.episodes) + self.index
        own, foe, side = pairing(self.seed, g)
        if (g // 2) % 2:
            own, foe = foe, own
        sim = 929_000_000 + 1_000_000 * self.seed + g
        self.setup = (own, foe, sim, side)
        self.history = []
        self.match.reset(*self.setup, preview="native")
        from pathlib import Path

        self.record = dict(own=Path(own).stem, foe=Path(foe).stem, sim=sim, side=side)

    def step(self, a):
        done, r = self.match.step(a)
        if done:
            self.episodes.append(dict(**self.record, reward=r, turns=self.match.battle.turn))
            self.reset()
        else:
            self.history.append(a)
        return done, r


def options(base, raw, view, seed):
    mask = torch.as_tensor(raw["action_mask"])[None]
    first = mask[0, :107].nonzero().flatten()
    second = base._update_mask(mask.expand(len(first), -1), first[:, None])[:, 107:]
    i, j = second.nonzero(as_tuple=True)
    pairs = torch.stack((first[i], j), dim=1)
    bias = torch.tensor(mechanics(view, pairs) @ fitted(seed), dtype=torch.float32)
    return raw, pairs, bias


def pack(frames):
    raw, pairs, bias = zip(*frames, strict=True)
    ids = nn.utils.rnn.pad_sequence(pairs, batch_first=True, padding_value=0)
    bias = nn.utils.rnn.pad_sequence(bias, batch_first=True, padding_value=-torch.inf)
    return batch(raw), ids, bias


def joint(model, frames, temperature=TEMP):
    obs, ids, bias = pack(frames)
    logits, value = model.battle_logits(obs)
    first, second = logits.chunk(2, dim=1)
    a, b = ids.unbind(2)
    z = second.gather(1, b)
    valid = bias.isfinite()
    # The published joint probability normalizes the second move over choices
    # compatible with the first. The first-move normalizer cancels in the final softmax.
    high = first.new_full(first.shape, -1e30).scatter_reduce(
        1, a, z.masked_fill(~valid, -1e30), reduce="amax", include_self=True
    )
    exps = torch.exp((z - high.gather(1, a)).masked_fill(~valid, -torch.inf))
    denom = first.new_zeros(first.shape).scatter_add(1, a, exps)
    norm = (high + torch.log(denom.clamp_min(1e-24))).gather(1, a)
    scores = (first.gather(1, a) + z - norm + bias) / temperature
    return Categorical(logits=scores), value.flatten(), ids


def folder(s):
    return RUN / "joint-ppo" / f"seed-{s}"


@functools.lru_cache(maxsize=4)
def load(seed, steps):
    ck = torch.load(folder(seed) / f"policy-{steps}.pt", map_location="cpu", weights_only=True)
    assert (ck["seed"], ck["steps"]) == (seed, steps)
    m = create_model("moe")
    m.load_state_dict(ck["model"])
    return m.eval().requires_grad_(False)


def deployment(view, obs, seed, steps):
    model = load(seed, steps)
    raw = {k: v[0].numpy() for k, v in obs.items()}
    frame = options(model.base, raw, view, seed)
    dist, _, ids = joint(model, [frame])
    return ids[0, dist.logits[0].argmax()].tolist()


def check(model, match, seed):
    ob = match.obs()
    frame = options(model.base, ob, match.view, seed)
    choices, logp = distribution(model, batch_obs(ob))
    assert torch.equal(choices, frame[1])
    dist, _, _ = joint(model, [(ob, choices, torch.zeros_like(frame[2]))], 1.0)
    torch.testing.assert_close(dist.logits[0], logp, atol=1e-5, rtol=1e-5)
    dist, _, ids = joint(model, [frame])
    assert torch.equal(choices[(logp + frame[2]).argmax()], ids[0, dist.logits[0].argmax()])


def train(seed, limit):
    dest = folder(seed)
    dest.mkdir(parents=True, exist_ok=True)
    model = copy.deepcopy(pilot(seed)).eval().requires_grad_(True)
    opt = torch.optim.Adam(model.parameters(), lr=2e-5, eps=1e-5)
    streams = [Stream(seed, i) for i in range(STREAMS)]
    sampler = torch.Generator().manual_seed(442 + seed)
    torch.manual_seed(1378 + seed)
    logs = []
    if (dest / "progress.pt").exists():
        logs = load_progress(dest, model, opt, sampler, streams)
    start = logs[-1]["steps"] if logs else 0
    try:
        if start == 0:
            with torch.no_grad():
                for s in streams:
                    check(model, s.match, seed)
        for end in range(start + BLOCK, limit + 1, BLOCK):
            frames = []
            actions = []
            old = []
            vals = []
            dones = []
            rewards = []
            with torch.no_grad():
                for _ in range(BLOCK // STREAMS):
                    f = [options(model.base, s.match.obs(), s.match.view, seed) for s in streams]
                    dist, v, ids = joint(model, f)
                    idx = dist.sample()
                    a = ids[torch.arange(STREAMS), idx]
                    frames.extend(f)
                    actions.extend(idx.tolist())
                    old.extend(dist.log_prob(idx).tolist())
                    vals.append(v)
                    results = [
                        s.step(choice) for s, choice in zip(streams, a.tolist(), strict=True)
                    ]
                    di, ri = zip(*results, strict=True)
                    dones.append(di)
                    rewards.append(ri)
                tail = model.base.predict_values(batch([s.match.obs() for s in streams])).flatten()
                v = torch.stack(vals)
                adv = torch.zeros_like(v)
                running = torch.zeros(STREAMS)
                for t in reversed(range(len(v))):
                    cont = 1 - torch.as_tensor(dones[t], dtype=torch.float32)
                    nxt = v[t + 1] if t + 1 < len(v) else tail
                    delta = torch.tensor(rewards[t]) + 0.99 * cont * nxt - v[t]
                    running = delta + 0.99 * 0.95 * cont * running
                    adv[t] = running
                y = (adv + v).flatten()
                adv = adv.flatten()
                adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
                actions = torch.tensor(actions)
                old = torch.tensor(old)
            for _ in range(3):
                for ix in torch.randperm(BLOCK, generator=sampler).split(64):
                    dist, pred, _ = joint(model, [frames[i] for i in ix.tolist()])
                    ratio = (dist.log_prob(actions[ix]) - old[ix]).exp()
                    pg = torch.minimum(ratio * adv[ix], ratio.clamp(0.8, 1.2) * adv[ix]).mean()
                    loss = (
                        -pg + 0.5 * (pred - y[ix]).square().mean() - 0.002 * dist.entropy().mean()
                    )
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
                    opt.step()
            logs.append(
                dict(
                    steps=end,
                    games=sum(len(s.episodes) for s in streams),
                    wins_in_batch=sum(r == 1 for group in rewards for r in group),
                    loss=float(loss.detach()),
                )
            )
            atomic_rows(OUT / "training" / f"joint-ppo-{seed}.csv", logs)
            if end in (16384, BUDGET):
                save(
                    dest / f"policy-{end}.pt", dict(seed=seed, steps=end, model=model.state_dict())
                )
            if end % 4096 == 0 or end == limit:
                save_progress(dest, model, opt, sampler, streams, logs)
                print(seed, logs[-1], flush=True)
        if limit == BUDGET:
            atomic_rows(dest / "episodes.csv", [r for s in streams for r in s.episodes])
    finally:
        for s in streams:
            s.match.battle.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    p.add_argument("--steps", type=int, choices=(512, 16384, BUDGET), default=BUDGET)
    args = p.parse_args()
    verify()
    train(args.seed, args.steps)

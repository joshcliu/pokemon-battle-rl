"""Matched one-decision preview PPO and group-relative policy gradients."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from poke_env.teambuilder import Teambuilder
from preview_head import describe
from preview_sweep import LINEUPS
from torch import nn
from torch.distributions import Categorical
from vgc_posttrain import REPORT, RUN, Match, atomic_rows, batch_obs, players, team_text


@lru_cache(maxsize=128)
def descriptors(team):
    mons = Teambuilder.parse_showdown_team(team_text(team).strip())
    return describe(tuple(m.species or m.nickname for m in mons), ())


class LineupPolicy(nn.Module):
    """Shared species encoder; only public species, base stats and types are used."""

    def __init__(self):
        super().__init__()
        self.project = nn.Linear(24, 32)
        self.role = nn.Parameter(torch.zeros(2, 32))
        self.encoder = nn.TransformerEncoderLayer(
            32, 2, 64, dropout=0, batch_first=True, activation="gelu"
        )
        self.score = nn.Sequential(nn.Linear(96, 32), nn.Tanh(), nn.Linear(32, 1))
        self.value = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.register_buffer("slots", torch.tensor([[int(c) - 1 for c in s] for s in LINEUPS]))

    def forward(self, own, foe):
        tokens = torch.cat((self.project(own) + self.role[0], self.project(foe) + self.role[1]))
        ctx = self.encoder(tokens[None])[0]
        lead = ctx[self.slots[:, :2]].mean(1)
        back = ctx[self.slots[:, 2:]].mean(1)
        pooled = ctx[6:].mean(0)
        logits = self.score(torch.cat((lead, back, pooled.expand_as(lead)), dim=1)).flatten()
        val = self.value(torch.cat((ctx[:6].mean(0), pooled))).squeeze()
        return Categorical(logits=logits), val


@lru_cache(maxsize=12)
def load_head(path):
    head = LineupPolicy()
    head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state"])
    return head.eval()


@torch.inference_mode()
def choose_lineup(path, own, foe):
    dist, _ = load_head(path)(descriptors(own), descriptors(foe))
    return LINEUPS[dist.logits.argmax().item()]


@torch.inference_mode()
def battle_reward(match, own, foe, seed, side, lineup):
    match.reset(own, foe, seed, side, lineup=lineup)
    torch.manual_seed(seed + 700_000_000)
    done = False
    while not done:
        a, _, _ = match.player.policy(batch_obs(match.obs()))
        done, rew = match.step(a[0].tolist())
    return rew, match.opponent.lineup


def train_preview(arm, seed):
    if arm not in ("grpo", "ppo"):
        raise ValueError("Unknown preview optimizer")
    spec = json.loads((REPORT / "protocol.json").read_text())
    if seed not in spec["training_seeds"]:
        raise ValueError("Seed is outside the locked pilot protocol")
    dest = RUN / f"preview-{arm}-{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "head.pt").exists():
        return
    match = Match()
    players()[0].policy.eval()
    torch.manual_seed(400 + seed)
    head = LineupPolicy()
    opt = torch.optim.Adam(head.parameters(), lr=0.001)
    pairs = [(x, y) for x in spec["own_train"] for y in spec["foe_train"]]
    rng = np.random.default_rng(385_000 + seed)
    table = []
    for g, idx in enumerate(rng.permutation(len(pairs))[: spec["preview_groups"]]):
        own, foe = pairs[idx]
        x, y = descriptors(own), descriptors(foe)
        with torch.no_grad():
            dist, value = head(x, y)
            torch.manual_seed(348_000 + seed * 1000 + g)
            acts = dist.sample((spec["group_size"],))
            old = dist.log_prob(acts)
        outcomes = []
        for a in acts:
            rs = []
            for k in range(spec["battle_seeds_per_group"]):
                sim_seed = 250_000_000 + seed * 1_000_000 + g * 100 + k
                side = ("p1", "p2")[k % 2]
                label = LINEUPS[a.item()]
                reward, opp = battle_reward(match, own, foe, sim_seed, side, label)
                table.append(
                    dict(
                        group=g,
                        own=Path(own).stem,
                        foe=Path(foe).stem,
                        seed=sim_seed,
                        side=side,
                        lineup=label,
                        opponent_lineup=opp,
                        reward=reward,
                    )
                )
                rs.append(reward)
            outcomes.append(np.mean(rs))
        rewards = torch.tensor(outcomes, dtype=torch.float32)
        if arm == "grpo":
            adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
        else:
            adv = rewards - value
        for _ in range(3):
            dist, pred = head(x, y)
            ratio = (dist.log_prob(acts) - old).exp()
            loss = (
                -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
                - 0.005 * dist.entropy()
            )
            if arm == "ppo":
                loss = loss + 0.5 * (pred - rewards).square().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 0.5, error_if_nonfinite=True)
            opt.step()
        if (g + 1) % 4 == 0:
            atomic_rows(dest / "rollouts.csv", table)
            print(arm, seed, "groups", g + 1, "battles", len(table), flush=True)
    assert (
        len(table) == spec["preview_groups"] * spec["group_size"] * spec["battle_seeds_per_group"]
    )
    tmp = dest / "head.tmp"
    torch.save(dict(state=head.state_dict(), arm=arm, seed=seed), tmp)
    tmp.replace(dest / "head.pt")

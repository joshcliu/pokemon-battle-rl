"""Autoregressive joint-action policy: p(a1|obs) p(a2|obs,a1)."""

import torch
from torch import nn
from torch.distributions import Categorical

from .battle import ACTION_SIZE


class Policy(nn.Module):
    def __init__(self, obs_size, hidden_size=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.first = nn.Linear(hidden_size, ACTION_SIZE)
        self.first_embedding = nn.Embedding(ACTION_SIZE, 32)
        self.second = nn.Sequential(
            nn.Linear(hidden_size + 32, hidden_size), nn.Tanh(), nn.Linear(hidden_size, ACTION_SIZE)
        )
        self.critic = nn.Linear(hidden_size, 1)

    @staticmethod
    def distribution(logits, mask):
        if not mask.any(dim=-1).all():
            raise ValueError("Empty legal-action mask")
        return Categorical(logits=logits.masked_fill(~mask, -1e9))

    def value(self, obs):
        return self.critic(self.encoder(obs)).squeeze(-1)

    def sample(self, obs, joint_mask, deterministic=False):
        features = self.encoder(obs)
        mask1 = joint_mask.any(dim=-1)
        dist1 = self.distribution(self.first(features), mask1)
        a1 = dist1.logits.argmax(-1) if deterministic else dist1.sample()
        rows = torch.arange(len(obs), device=obs.device)
        mask2 = joint_mask[rows, a1]
        logits2 = self.second(torch.cat([features, self.first_embedding(a1)], dim=-1))
        dist2 = self.distribution(logits2, mask2)
        a2 = dist2.logits.argmax(-1) if deterministic else dist2.sample()
        return (
            torch.stack([a1, a2], dim=-1),
            dist1.log_prob(a1) + dist2.log_prob(a2),
            self.critic(features).squeeze(-1),
            torch.stack([mask1, mask2], dim=1),
        )

    def evaluate(self, obs, actions, masks):
        features = self.encoder(obs)
        a1, a2 = actions.unbind(-1)
        first = self.distribution(self.first(features), masks[:, 0])
        second = self.distribution(
            self.second(torch.cat([features, self.first_embedding(a1)], dim=-1)), masks[:, 1]
        )
        return (
            first.log_prob(a1) + second.log_prob(a2),
            first.entropy() + second.entropy(),
            self.critic(features).squeeze(-1),
        )


def advantages(rewards, values, next_values, terminated, episode_ends, gamma, lam):
    """Bootstrap truncations, but never carry GAE across an episode boundary."""
    result = torch.zeros_like(rewards)
    last = torch.zeros((), device=rewards.device)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        last = delta + gamma * lam * (~episode_ends[t]) * last
        result[t] = last
    return result

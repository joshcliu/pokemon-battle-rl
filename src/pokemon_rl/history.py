"""Encode an ordered window ending at the learner's current decision."""

import math

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .battle import ACTION_SIZE
from .policy import Policy


class DecisionWindow(gym.Wrapper):
    """Keep visible boards and the learner action immediately preceding each board."""

    def __init__(self, env, chunks):
        super().__init__(env)
        width = env.observation_space.shape[0] + 2
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (chunks, width), np.float32)
        self.window = np.empty((chunks, width), np.float32)
        self.length = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.window.fill(0)
        self.window[:, -2:] = -1  # Right padding; valid action IDs are nonnegative.
        self.length = 0
        return self.append(obs, (ACTION_SIZE, ACTION_SIZE)), info

    def append(self, obs, previous_action):
        if self.length == len(self.window):
            self.window[:-1] = self.window[1:]
        else:
            self.length += 1
        row = self.window[self.length - 1]
        row[:-2] = obs
        row[-2:] = previous_action
        return self.window.copy()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.append(obs, action), reward, terminated, truncated, info


class TurnEncoder(nn.Module):
    def __init__(self, obs_size, width, chunks):
        super().__init__()
        self.board = nn.Linear(obs_size, width)
        self.previous_first = nn.Embedding(ACTION_SIZE + 1, width)
        self.previous_second = nn.Embedding(ACTION_SIZE + 1, width)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                width,
                nhead=4,
                dim_feedforward=2 * width,
                dropout=0,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            for _ in range(2)
        )
        self.norm = nn.LayerNorm(width)
        positions = torch.arange(chunks)
        frequencies = torch.exp(torch.arange(0, width, 2) * (-math.log(10000) / width))
        angles = positions[:, None] * frequencies
        self.register_buffer("positions", positions, persistent=False)
        self.register_buffer(
            "age_encoding",
            torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1),
            persistent=False,
        )
        self.register_buffer("future", positions[:, None] < positions[None, :], persistent=False)

    def forward(self, window):
        actions = window[:, :, -2:].long()
        last = (actions[:, :, 0] >= 0).sum(1) - 1
        first, second = actions.clamp_min(0).unbind(-1)
        age = (last[:, None] - self.positions).clamp_min(0)
        tokens = (
            self.board(window[:, :, :-2])
            + self.previous_first(first)
            + self.previous_second(second)
            + self.age_encoding[age]
        )
        # Right padding is always in the future of every valid token.
        for block in self.blocks:
            tokens = block(tokens, src_mask=self.future)
        return self.norm(tokens[torch.arange(len(window), device=window.device), last])


def make_policy(obs_size, hidden_size, chunks=None):
    encoder = TurnEncoder(obs_size, hidden_size, chunks) if chunks else None
    return Policy(obs_size, hidden_size, encoder=encoder)

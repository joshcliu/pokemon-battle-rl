"""Encode an ordered window ending at the learner's current decision."""

import math

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .battle import ACTION_SIZE
from .policy import Policy


class FullBattleHistory(gym.Wrapper):
    """Return the entire player-visible decision prefix, including the current board."""

    def __init__(self, env):
        super().__init__(env)
        width = env.observation_space.shape[0] + 2
        token = gym.spaces.Box(-np.inf, np.inf, (width,), np.float32)
        self.observation_space = gym.spaces.Sequence(token, stack=True)
        self.tokens = []

    def append(self, obs, previous_action):
        token = np.concatenate((obs, np.asarray(previous_action, dtype=np.float32)))
        self.tokens.append(token)
        return np.stack(self.tokens)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.tokens.clear()
        return self.append(obs, (ACTION_SIZE, ACTION_SIZE)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        prefix = self.append(obs, action)
        return prefix, reward, terminated, truncated, info | {"learner_decisions": len(prefix) - 1}


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
        frequencies = torch.exp(torch.arange(0, width, 2) * (-math.log(10000) / width))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.set_positions(1 if chunks == "full" else chunks)

    @torch.inference_mode(False)
    def set_positions(self, length):
        positions = torch.arange(length, device=self.frequencies.device)
        angles = positions[:, None] * self.frequencies
        self.register_buffer("positions", positions, persistent=False)
        self.register_buffer(
            "age_encoding",
            torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1),
            persistent=False,
        )
        self.register_buffer("future", positions[:, None] < positions[None, :], persistent=False)

    def forward(self, window):
        length = window.shape[1]
        if length > len(self.positions):
            self.set_positions(length)
        positions = self.positions[:length]
        actions = window[:, :, -2:].long()
        last = (actions[:, :, 0] >= 0).sum(1) - 1
        first, second = actions.clamp_min(0).unbind(-1)
        age = (last[:, None] - positions).clamp_min(0)
        tokens = (
            self.board(window[:, :, :-2])
            + self.previous_first(first)
            + self.previous_second(second)
            + self.age_encoding[age]
        )
        # Right padding is always in the future of every valid token.
        for block in self.blocks:
            tokens = block(tokens, src_mask=self.future[:length, :length])
        return self.norm(tokens[torch.arange(len(window), device=window.device), last])


def make_policy(obs_size, hidden_size, chunks=None):
    encoder = TurnEncoder(obs_size, hidden_size, chunks) if chunks else None
    return Policy(obs_size, hidden_size, encoder=encoder)

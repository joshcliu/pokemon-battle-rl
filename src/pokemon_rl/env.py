"""Gymnasium wrapper for a seeded, partially observed doubles battle."""

from __future__ import annotations

from itertools import groupby

import gymnasium as gym
import numpy as np

from .battle import ACTION_SIZE, DEFAULT_FORMAT, ROOT, LocalBattle
from .features import OBSERVATION_SIZE, observation
from .opponents import Heuristic


def shuffled_moves(team: str, episode_seed: int) -> str:
    rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 31]))
    lines = []
    for is_moves, block in groupby(team.splitlines(), key=lambda line: line.startswith("- ")):
        group = list(block)
        if is_moves:
            rng.shuffle(group)
        lines.extend(group)
    return "\n".join(lines) + "\n"


class BattleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent="damage",
        team="teams/sun.txt",
        opponent_teams=None,
        battle_format=DEFAULT_FORMAT,
        max_turns=200,
        max_decisions=600,
        learner_teams=None,
        shuffle_move_slots=False,
        opponent_pool=None,
        mirror_teams=False,
    ):
        super().__init__()
        self.style = opponent
        self.shuffle_move_slots = shuffle_move_slots
        self.learner_paths = list(learner_teams or [team])
        self.learner_teams = [(ROOT / p).read_text() for p in self.learner_paths]
        self.opponent_paths = list(opponent_teams or ["teams/balance.txt", "teams/sun.txt"])
        self.mirror_teams = mirror_teams
        if mirror_teams and self.learner_paths != self.opponent_paths:
            raise ValueError(
                "Mirror training requires the same ordered learner and opponent rosters"
            )
        self.opponent_teams = [(ROOT / p).read_text() for p in self.opponent_paths]
        self.pool = None
        if opponent_pool:
            from .league import OpponentPool

            self.pool = OpponentPool(opponent_pool, self.opponent_paths)
        self.battle = LocalBattle(battle_format)
        self.max_turns, self.max_decisions = max_turns, max_decisions
        self.action_space = gym.spaces.MultiDiscrete([ACTION_SIZE, ACTION_SIZE])
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self.decisions = 0
        self.finished = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**32))
        self.team_index = int(self.np_random.integers(len(self.opponent_teams)))
        # Opponent decisions must not change the next episode's team/seed schedule.
        opponent_rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 19]))
        self.opponent = (
            self.pool.draw(self.opponent_paths[self.team_index], opponent_rng)
            if self.pool
            else Heuristic(self.style, opponent_rng)
        )
        self.episode_seed = episode_seed
        self.learner_index = 0
        if self.mirror_teams:
            self.learner_index = self.team_index
        elif len(self.learner_teams) > 1:
            learner_rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 23]))
            self.learner_index = int(learner_rng.integers(len(self.learner_teams)))
        learner = self.learner_teams[self.learner_index]
        if self.shuffle_move_slots:
            learner = shuffled_moves(learner, episode_seed)
        self.battle.reset(episode_seed, learner, self.opponent_teams[self.team_index])
        self.decisions, self.finished = 0, False
        obs = observation(self.battle.players["p1"])
        return obs, self._info()

    def _info(self):
        return {
            "turn": self.battle.turn,
            "simulator_seed": self.episode_seed,
            "opponent": self.opponent.style,
            "opponent_team_index": self.team_index,
            "learner_team": str(self.learner_paths[self.learner_index]),
        }

    def legal(self):
        return self.battle.legal_actions("p1")

    def joint_mask(self):
        mask = np.zeros((ACTION_SIZE, ACTION_SIZE), dtype=bool)
        for a, b in self.legal():
            mask[a, b] = True
        return mask

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset before stepping a completed episode")
        legal = self.legal()
        array = np.asarray(action)
        if array.shape != (2,) or not np.issubdtype(array.dtype, np.integer):
            raise ValueError("An action must contain exactly two integer choices")
        pair = tuple(int(x) for x in action)
        if pair not in legal:
            raise ValueError(f"Illegal joint action {pair}; no fallback was executed")
        choices = {"p1": legal[pair].message.removeprefix("/choose ")}
        while True:
            if self.battle.needs_action("p2"):
                opponent_legal = self.battle.legal_actions("p2")
                other = self.opponent.choose(self.battle.players["p2"], opponent_legal)
                choices["p2"] = opponent_legal[other].message.removeprefix("/choose ")
            self.battle.step(choices)
            self.decisions += 1
            truncated = not self.battle.ended and (
                self.battle.turn >= self.max_turns or self.decisions >= self.max_decisions
            )
            if self.battle.ended or truncated or self.battle.needs_action("p1"):
                break
            choices = {}
        terminated = self.battle.ended
        reward = float(self.battle.winner == "Learner") - float(self.battle.winner == "Opponent")
        self.finished = terminated or truncated
        info = self._info()
        if self.finished:
            info["outcome"] = (
                "truncated"
                if truncated
                else "win"
                if reward > 0
                else "loss"
                if reward < 0
                else "tie"
            )
        return observation(self.battle.players["p1"]), reward, terminated, truncated, info

    def close(self):
        self.battle.close()

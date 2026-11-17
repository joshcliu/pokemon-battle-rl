import gymnasium as gym
import numpy as np
import torch

from pokemon_rl.battle import ACTION_SIZE
from pokemon_rl.history import DecisionWindow, FullBattleHistory, make_policy


class Board(gym.Env):
    observation_space = gym.spaces.Box(0, 100, (3,), np.float32)

    def reset(self, **kwargs):
        self.decision = 0
        return np.full(3, self.decision, np.float32), {}

    def step(self, action):
        self.decision += 1
        return np.full(3, self.decision, np.float32), 0.0, False, False, {}


def test_window_order_reset_and_rollover():
    env = DecisionWindow(Board(), 2)
    start, _ = env.reset()
    first = env.step((4, 5))[0]
    second = env.step((6, 7))[0]
    assert first.tolist() == [[0, 0, 0, ACTION_SIZE, ACTION_SIZE], [1, 1, 1, 4, 5]]
    assert second.tolist() == [[1, 1, 1, 4, 5], [2, 2, 2, 6, 7]]
    assert np.array_equal(env.reset()[0], start)
    assert start[1, -2:].tolist() == [-1, -1]


def test_whole_battle_prefix_and_padded_ppo_replay():
    env = FullBattleHistory(Board())
    start, _ = env.reset()
    prefixes = [start]
    for i in range(12):
        state, _, _, _, info = env.step((i, 3))
        assert info["learner_decisions"] == i + 1
        prefixes.append(state)
    assert state[:, 0].tolist() == list(range(13))
    assert np.array_equal(state[:1], start)
    assert np.array_equal(env.reset()[0], start)
    torch.manual_seed(42)
    model = make_policy(3, 16, "full")
    tensors = [torch.from_numpy(prefixes[i]) for i in (12, 0, 8, 3)]
    mask = torch.ones((1, ACTION_SIZE, ACTION_SIZE), dtype=torch.bool)
    with torch.no_grad():
        roll = [model.sample(p[None], mask) for p in tensors]
        changed = tensors[0].clone()
        changed[0, 0] += 20
        assert not torch.allclose(model.encoder(changed[None]), model.encoder(tensors[0][None]))
    batch = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=-1)
    actions, logp, values, masks = [torch.cat(items) for items in zip(*roll, strict=True)]
    replay, _, estimates = model.evaluate(batch, actions, masks)
    torch.testing.assert_close(replay, logp)
    torch.testing.assert_close(estimates, values)
    (replay + estimates).sum().backward()


def test_context_control_padding_and_ppo_replay():
    torch.manual_seed(4)
    short = make_policy(3, 16, 1)
    torch.manual_seed(4)
    long = make_policy(3, 16, 8)
    assert all(torch.equal(v, long.state_dict()[k]) for k, v in short.state_dict().items())

    env = DecisionWindow(Board(), 8)
    first = torch.from_numpy(env.reset()[0])[None]
    second = torch.from_numpy(env.step((1, 2))[0])[None]
    altered = second.clone()
    altered[0, 0, 0] += 20
    padded = second.clone()
    padded[:, 2:, :-2] = 250
    for training in (True, False):
        long.train(training)
        short.train(training)
        with torch.no_grad():
            torch.testing.assert_close(short.encoder(first[:, :1]), long.encoder(first))
            torch.testing.assert_close(long.encoder(padded), long.encoder(second))
            assert not torch.allclose(long.encoder(altered), long.encoder(second))
            rollout = torch.cat([first, second])
            mask = torch.zeros((2, ACTION_SIZE, ACTION_SIZE), dtype=torch.bool)
            mask[:, 1:5, 2:4] = True
            actions, old_logp, values, masks = long.sample(rollout, mask)
        logp, _, new_values = long.evaluate(rollout.flip(0), actions.flip(0), masks.flip(0))
        torch.testing.assert_close(logp.flip(0), old_logp)
        torch.testing.assert_close(new_values.flip(0), values)

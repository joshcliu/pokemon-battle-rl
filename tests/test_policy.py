import numpy as np
import pytest
import torch

from pokemon_rl.battle import ACTION_SIZE
from pokemon_rl.experiments import summarize, wilson
from pokemon_rl.policy import Policy, advantages


def test_joint_policy_never_samples_incompatible_choices():
    torch.manual_seed(3)
    model = Policy(8, 16)
    states = torch.zeros(300, 8)
    masks = torch.zeros(300, ACTION_SIZE, ACTION_SIZE, dtype=torch.bool)
    masks[:, 1, 2] = True
    masks[:, 2, 1] = True
    actions, logp, values, used_masks = model.sample(states, masks)
    assert all(tuple(a) in {(1, 2), (2, 1)} for a in actions.tolist())
    recomputed, _, prediction = model.evaluate(states, actions, used_masks)
    torch.testing.assert_close(logp, recomputed)
    torch.testing.assert_close(values, prediction)


def test_empty_mask_fails_instead_of_random_fallback():
    with pytest.raises(ValueError, match="Empty"):
        Policy(8).sample(
            torch.zeros(1, 8), torch.zeros(1, ACTION_SIZE, ACTION_SIZE, dtype=torch.bool)
        )


def test_gae_distinguishes_terminal_and_time_limit_and_cuts_episodes():
    result = advantages(
        torch.tensor([1.0, 0.0, 100.0]),
        torch.tensor([2.0, 2.0, 0.0]),
        torch.tensor([10.0, 10.0, 0.0]),
        torch.tensor([True, False, True]),
        torch.tensor([True, True, True]),
        gamma=1.0,
        lam=1.0,
    )
    torch.testing.assert_close(result, torch.tensor([-1.0, 8.0, 100.0]))


def test_evaluation_does_not_count_timeouts_as_ties():
    result = summarize([{"outcome": v, "turn": 5} for v in ["win", "loss", "truncated"]], 1.0)
    assert result["completed"] == 2
    assert result["outcomes"]["tie"] == 0
    assert result["win_rate_completed"] == 0.5
    assert result["truncation_rate"] == pytest.approx(1 / 3)
    assert wilson(0, 0) == [None, None]
    lo, hi = wilson(50, 100)
    assert 0 < lo < 0.5 < hi < 1


def test_gae_carries_nonterminal_advantages():
    result = advantages(
        torch.tensor([0.0, 1.0]),
        torch.zeros(2),
        torch.zeros(2),
        torch.tensor([False, True]),
        torch.tensor([False, True]),
        1.0,
        1.0,
    )
    np.testing.assert_array_equal(result.numpy(), [1.0, 1.0])

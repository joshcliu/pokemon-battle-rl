"""Advantage-weighted actor critic for the benchmark's masked two-command policy."""

import copy

import numpy as np
import torch
from torch import nn


class JointQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Embedding(107, 32)
        self.second = nn.Embedding(107, 32)
        self.head = nn.Sequential(
            nn.Linear(320, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state, action):
        features = torch.cat((state, self.first(action[:, 0]), self.second(action[:, 1])), dim=-1)
        return self.head(features).flatten()


def joint_distribution(policy, logits, mask, action=None):
    first = policy.get_dist_from_logits(logits, mask).distribution[0]
    a = first.sample() if action is None else action[:, 0]
    second = policy.get_dist_from_logits(logits, mask, a[:, None]).distribution[1]
    b = second.sample() if action is None else action[:, 1]
    return torch.stack((a, b), dim=1), first.log_prob(a) + second.log_prob(b)


class AWAC:
    def __init__(self, model, settings, capacity):
        self.base = model.base
        self.settings = settings
        self.critics = nn.ModuleList([JointQ(), JointQ()])
        self.targets = copy.deepcopy(self.critics).requires_grad_(False)
        self.q_optimizer = torch.optim.Adam(self.critics.parameters(), lr=settings["critic_lr"])
        self.actor = list(self.base.pi_features_extractor.parameters()) + list(
            self.base.action_net.parameters()
        )
        self.pi_optimizer = torch.optim.Adam(self.actor, lr=settings["actor_lr"], eps=1e-5)
        self.buffer = {}
        self.size = 0
        self.capacity = capacity

    def logits(self, obs):
        return self.base.action_net(
            self.base.mlp_extractor.forward_actor(self.base.pi_features_extractor(obs))
        )

    @torch.no_grad()
    def add(self, data, actions, rewards, dones, following):
        # following is a reset observation for terminal rows; their bootstrap is masked out.
        obs = {k: torch.from_numpy(np.stack([o[k] for o in [*data, following]])) for k in data[0]}
        state = torch.cat(
            [
                self.base.vf_features_extractor({"observation": chunk})
                for chunk in obs["observation"].split(32)
            ]
        )
        entries = {
            "obs": obs["observation"][:-1],
            "mask": obs["action_mask"][:-1],
            "next_obs": obs["observation"][1:],
            "next_mask": obs["action_mask"][1:],
            "state": state[:-1],
            "next_state": state[1:],
            "action": torch.stack(actions),
            "reward": torch.tensor(rewards, dtype=torch.float32),
            "done": torch.tensor(dones),
        }
        stop = self.size + len(actions)
        for key, value in entries.items():
            if key not in self.buffer:
                self.buffer[key] = torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
            self.buffer[key][self.size : stop] = value
        self.size = stop

    def update(self, gamma):
        config = self.settings
        metrics = []
        for _ in range(config["updates_per_256_steps"]):
            idx = torch.randint(self.size, (config["minibatch"],))
            batch = {k: v[idx] for k, v in self.buffer.items()}
            with torch.no_grad():
                next_logits = self.logits({"observation": batch["next_obs"]})
                next_action, _ = joint_distribution(self.base, next_logits, batch["next_mask"])
                q_next = torch.stack(
                    [q(batch["next_state"], next_action) for q in self.targets]
                ).amin(0)
                target = batch["reward"] + gamma * (~batch["done"]) * q_next
            estimates = torch.stack([q(batch["state"], batch["action"]) for q in self.critics])
            q_loss = (estimates - target).square().mean()
            self.q_optimizer.zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(self.critics.parameters(), 1.0, error_if_nonfinite=True)
            self.q_optimizer.step()
            logits = self.logits({"observation": batch["obs"]})
            with torch.no_grad():
                k = config["advantage_samples"]
                candidate, _ = joint_distribution(
                    self.base, logits.repeat(k, 1), batch["mask"].repeat(k, 1)
                )
                state = batch["state"].repeat(k, 1)
                baseline = (
                    torch.stack([q(state, candidate) for q in self.critics])
                    .amin(0)
                    .reshape(k, -1)
                    .mean(0)
                )
                q_data = torch.stack(
                    [q(batch["state"], batch["action"]) for q in self.critics]
                ).amin(0)
                weight = (
                    ((q_data - baseline) / config["temperature"])
                    .exp()
                    .clamp_max(config["max_weight"])
                )
                weight = weight / weight.mean().clamp_min(1e-8)
            _, logp = joint_distribution(self.base, logits, batch["mask"], batch["action"])
            pi_loss = -(weight * logp).mean()
            self.pi_optimizer.zero_grad()
            pi_loss.backward()
            nn.utils.clip_grad_norm_(self.actor, 0.5, error_if_nonfinite=True)
            self.pi_optimizer.step()
            with torch.no_grad():
                for target_param, param in zip(
                    self.targets.parameters(), self.critics.parameters(), strict=True
                ):
                    target_param.lerp_(param, config["polyak_tau"])
            metrics.append(
                (
                    pi_loss.item(),
                    q_loss.item(),
                    weight.std().item(),
                    (idx < self.size - 256).float().mean().item(),
                )
            )
        return dict(
            zip(
                ("loss", "q_loss", "weight_sd", "old_replay_fraction"),
                np.mean(metrics, axis=0).tolist(),
                strict=True,
            )
        )

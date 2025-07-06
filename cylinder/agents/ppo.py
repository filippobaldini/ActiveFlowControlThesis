# agents/ppo.py

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    def __init__(self, observation_space, action_space, h_dim=512):
        super().__init__()
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, h_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(h_dim, h_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(h_dim, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, h_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(h_dim, h_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(h_dim, act_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.get_value(obs)
        return action, logprob, entropy, value


def scale_action(action, action_space):
    low = action_space.low
    high = action_space.high
    scaled = low + 0.5 * (action + 1.0) * (high - low)
    return np.clip(scaled, low, high)

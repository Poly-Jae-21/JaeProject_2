import torch
import torch.nn as nn
from pandas import Categorical
from sympy.physics.units import action
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 5000),
            nn.Tanh(),
            nn.Linear(5000, 1000),
            nn.Tanh(),
            nn.Linear(1000, 100),
            nn.Tanh(),
            nn.Linear(100, action_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 5000),
            nn.Tanh(),
            nn.Linear(5000, 1000),
            nn.Tanh(),
            nn.Linear(1000, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )

    def act(self, state, memory):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_probs = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.action_probs.append(action_probs)

        return action.item()
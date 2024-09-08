import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical, Normal

from utils.misc import mlp, Actor

class CategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()

        self.logits_networks =  mlp([obs_dim] + list(hidden_sizes) + [act_dim], [np.sqrt(2)] * (len(hidden_sizes)) + [0.01], nn.ReLU, nn.Identity)

    def _distribution(self, obs):
        logits = self.logits_networks(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class GaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()

        self.mu_network = mlp([obs_dim] + list(hidden_sizes) + [act_dim], [np.sqrt(2)] * (len(hidden_sizes)) + [0.01], activation, output_activation)
        self.log_std = nn.Parameter(-0.5 * torch.ones(1, act_dim))

    def _distribution(self, obs):
        mu = self.mu_network(obs)
        std = torch.exp(self.log_std)

        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        self.value_network = mlp([obs_dim] + list(hidden_sizes) + [1],
                          [np.sqrt(2)] * (len(hidden_sizes)) + [1.00], activation, output_activation)

    def forward(self, obs):
        return self.value_network(obs).squeeze(-1)
class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space, actor_hidden_sizess, critic_hidden_sizess, device):
        super().__init__()

        self.obs_dim = obs_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.actor = GaussianActor(self.obs_dim, self.action_dim, actor_hidden_sizess).to(device)

        self.critic = Critic(self.obs_dim, critic_hidden_sizess).to(device)

    def step(self, obs):
        obs = torch.tensor(obs)
        pi = self.actor._distribution(obs)
        action = pi.sample()

        logp_a = self.actor._log_prob_from_distribution(pi, action)
        value = self.critic(obs).view(-1)

        return action, logp_a, value

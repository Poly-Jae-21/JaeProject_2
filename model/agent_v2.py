import math
import os
from os import times

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
from env.utils.action import Action
from torch.autograd import Variable
from collections import OrderedDict
# Basic PPO policy and value network
class PolicyNetwork(nn.Module):
    def __init__(self, obs_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_space, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.action_mean = nn.Linear(64, action_space)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        action_mean = self.action_mean(x)
        value = self.value_head(x)

        return action_mean, value

# PPO implementation
class PPO:
    def __init__(self, config, policy_net):
        self.policy_net = policy_net
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr) # lr = 1e-4
        self.gamma = config.gamma # 0.99
        self.clip_epsilon = config.clip_epsilon # = 0.2
        self.value_coeff = config.value_coeff # = 0.5
        self.entropy_coeff = config.entropy_coeff #0.01
        self.max_steps = config.max_steps
        self.batch_size = config.batch_size
        self.device = config.device

    def select_action(self, env, state):

        device = self.device

        state = Variable(torch.Tensor(state))
        action_mean, value = self.policy_net(state)
        action_dist = torch.distributions.Normal(action_mean, 1.0)
        action = action_dist.sample()
        action = torch.clamp(action, np.min(env.action_space.low[0], np.max(env.action_space.high[0])))
        log_prob = action_dist.log_prob(action)
        action = action.data.numpy()

        return action, log_prob, value

    def compute_gae(self, rewards, values, dones, next_value, gamma, lam=0.95):
        gae = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * (1 - dones[i]) * next_value - values[i]
            gae = delta + gamma * lam * gae * (1 - dones[i])
            returns.insert(0, gae + values[i])
            next_value = values[i]
        return returns

    def update(self, trajectories, old_log_probs, advantages, returns):
        states = torch.tensor(np.vstack([t[0] for t in trajectories]), dtype=torch.float32)
        actions = torch.tensor(np.vstack([t[1] for t in trajectories]), dtype=torch.float32)
        log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        dataset_size = states.size(0)
        batch_iter = int(math.ceil(dataset_size / self.batch_size))

        for _ in range(self.max_steps):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm)

            states, actions, returns, advantages, log_probs = states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), log_probs[perm].clone()
            for i in range(batch_iter):
                ind = slice(i * self.batch_size, min((i+1) * self.batch_size, states.shape[0]))
                batch_states, batch_actions, batch_advantages, batch_returns, batch_log_probs = states[ind], actions[ind], advantages[ind], returns[ind], log_probs[ind]

                new_action_mean, new_value = self.policy_net(batch_states)
                action_dist = torch.distributions.Normal(new_action_mean, 1.0)
                new_log_probs = action_dist.log_prob(actions).sum(dim=-1)

                ratio = torch.exp(new_log_probs - batch_log_probs)

                # Policy loss
                surrogate_1 = ratio * batch_advantages
                surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                # Value loss
                value_loss = self.value_coeff * ((batch_returns - new_value).pow(2)).mean()

                # Entropy loss for exploration
                entropy_loss = -self.entropy_coeff * action_dist.entropy().mean()

                # Total loss
                loss = policy_loss + value_loss + entropy_loss

                # Gradient descent step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

class MetaPPO(PPO):
    def __init__(self, global_policy_net, local_policy_nets, env, config, batch_size):
        self.env = env
        self.config = config
        self.global_policy_net = global_policy_net
        self.local_policy_nets = local_policy_nets
        self.lr = self.config.lr
        self.gamma = self.config.gamma
        self.clip_epsilon = self.config.clip_epsilon
        self.value_coeff = self.config.value_coeff
        self.entropy_coeff = self.config.entropy_coeff
        self.batch_size = batch_size


    def adapt_to_task(self, local_policy_net, env, factors, inner_steps=1, timesteps = 100):
        """
        Perform inner-loop adaptation on a specific criteria from three criteria (e.g., environment, economic, urbanity) using a local learner.

        """

        ppo = PPO(self.config, local_policy_net)

        # Gather experience and train on the task
        trajectories, old_log_probs, rewards, values, dones = self.rollout(local_policy_net, env, factors, timesteps)
        returns = self.compute_gae(rewards, values, dones, values[-1], self.gamma)
        advantages = np.array(returns) - np.array(values)

        for _ in range(inner_steps):
            ppo.update(trajectories, old_log_probs, advantages, returns)

        return local_policy_net

    def meta_train(self, env, meta_steps=3, inner_steps=1, timesteps=100):
        """
        Meta-training loop over three local policies. Aggregate gradients local learners into the global model

        """
        adapted_policies = []
        for i in range(meta_steps):
            # meta steps = number of factors = environment, economic, and urbanity
            criteria = ['environment', 'economic', 'urbanity']
            local_policy_net = self.local_policy_nets[i]
            adapted_policy = self.adapt_to_task(local_policy_net, env, criteria[i], inner_steps, timesteps)
            adapted_policies.append(adapted_policy)

        # Meta-updates: aggregate parameters from all local models to update the global model
        self.aggregat_local_to_meta_global(adapted_policies)

    def aggregat_local_to_meta_global(self, adapted_policies):
        """
        Aggregate parameters from local learners to update the global meta-learner using weighted MAML.
        """
        # Initialize parameter aggregation
        global_params = OrderedDict(self.global_policy_net.named_parameters())
        for name, param in global_params.items():
            param.data.copy_(torch.zeros_like(param.data))

        # Aggregate parameters from local adapted policies
        for adapted_policy in adapted_policies:
            local_params = adapted_policy.named_parameters()
            for (name, param), (_, adapted_param) in zip(global_params.items(), local_params):
                param.data += adapted_param.data

        # Normalize aggregated parameters by the number of local networks
        for name, param in global_params.items():
            param.data /= len(adapted_policies)

        # Now synchronize the parameters across all workers
        for param in self.global_policy_net.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= dist.get_world_size()

        # Broadcast the updated global model to all workers
        for param in self.global_policy_net.parameters():
            dist.broadcast(param.data, src=0) # Assume rank 0 is the parameter server(main worker)

    def reduce_and_broadcast(self, adapted_global_policies):

        global_params = OrderedDict(self.global_policy_net.named_parameters())


    def rollout(self, policy_net, env, factors, timesteps):
        """
        Rollout one time to collect data for each factor using the given policy network
        """

        trajectories = []
        rewards = []
        values = []
        dones = []
        old_log_probs = []
        state = env.reset_free()
        for _ in range(timesteps):
            action, log_prob = PPO.select_action(env, state)
            converted_action = Action.action_converter(action)
            next_state, reward, done, _ = env.step(converted_action, state, factors)
            _, value = policy_net(torch.tensor(state), dtype=torch.float32)
            trajectories.append((state, action))
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            old_log_probs.append(log_prob)
            if not done:
                state = next_state

        return trajectories, old_log_probs, rewards, values, dones



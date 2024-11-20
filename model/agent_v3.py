import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

class PolicyNetwork(nn.Module):
    def __init__(self, obs_space, action_space):
        super(PolicyNetwork, self).__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(obs_space, 5000),
            nn.Tanh(),
            nn.Linear(5000, 1000),
            nn.Tanh(),
            nn.Linear(1000, 100),
            nn.Tanh(),
        )

        self.policy_mean_net = nn.Sequential(
            nn.Linear(100, action_space)
        )

        self.policy_stddev_net = nn.Sequential(
            nn.Linear(100, action_space)
        )

        self.value_net = nn.Sequential(
            nn.Linear(100, 1)
        )


    def forward(self, state):
        shared_features = self.shared_net(state)
        mean = self.policy_mean_net(shared_features)
        log_std = self.policy_stddev_net(shared_features)
        std = torch.log(1 + torch.exp(log_std))
        value = self.value_net(shared_features)
        return mean, std, value

class PPO:
    def __init__(self, args, device, local_policy_net):
        self.local_policy_net = local_policy_net
        self.optimizer = optim.Adam(self.local_policy_net.parameters(), lr=args.lr)
        self.gamma = args.gamma
        self.clip_epsilon = args.clip_epsilon
        self.value_coeff = args.value_coeff
        self.entropy_coeff = args.entropy_coeff
        self.inner_steps = args.update_timesteps
        self.batch_size = args.batch_size
        self.device = device

    def select_action(self, state):

        state = np.reshape(state, [1, 10000])
        state = torch.Tensor(state).to(self.device)
        actions_mean, actions_std, value = self.local_policy_net(state)
        dist_ = torch.distributions.Normal(actions_mean, actions_std)
        sampled_actions = dist_.sample()
        log_probs = dist_.log_prob(sampled_actions)

        return sampled_actions.cpu().detach(), log_probs.cpu().detach(), value.cpu().detach()

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
        states = torch.tensor(np.vstack([t[0] for t in trajectories])).to(self.device)
        actions = torch.tensor(np.vstack([t[1] for t in trajectories]))
        log_probs = torch.tensor(old_log_probs)
        advantages = torch.tensor(advantages)
        returns = torch.tensor(returns)

        dataset_size = states.size(0)
        batch_iter = int(math.ceil(dataset_size / self.batch_size))

        for _ in range(self.inner_steps):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm)

            states, actions, returns, advantages, log_probs = states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), log_probs[perm].clone()
            for i in range(batch_iter):
                ind = slice(i * self.batch_size, min((i+1) * self.batch_size, states.shape[0]))
                batch_states, batch_actions, batch_advantages, batch_returns, batch_log_probs = states[ind], actions[ind], advantages[ind], returns[ind], log_probs[ind]

                new_action_mean, new_action_stddev, new_value = self.local_policy_net(batch_states)

                dist = torch.distributions.Normal(new_action_mean, new_action_stddev)
                sampled_actions = dist.sample()
                new_log_probs = dist.log_prob(sampled_actions).item()

                ratio = torch.exp(new_log_probs - batch_log_probs)

                # Policy loss
                surrogate_1 = ratio * batch_advantages
                surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                # Value loss
                value_loss = self.value_coeff * nn.MSELoss()(returns, new_value).mean()

                # Entropy loss for exploration
                entropy_loss = -self.entropy_coeff * dist.entropy().mean()

                # Total loss
                loss = policy_loss + value_loss + entropy_loss

                # Gradient descent step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

class MetaPPO(PPO):
    def __init__(self, global_policy_net, local_policy_net, device, env, args, batch_size=32):
        self.env = env
        self.args = args
        self.local_policy_net = local_policy_net
        self.lr = args.lr
        self.gamma = args.gamma
        self.clip_epsilon = args.clip_epsilon
        self.value_coeff = args.value_coeff
        self.entropy_coeff = args.entropy_coeff
        self.batch_size = batch_size
        self.lam = self.args.lambda_
        self.device = device
        self.terminate = False

        self.local_policy_nets = [[] for _ in range(3)]

    def adapt_to_task(self, args, local_policy_net, env, initial_observation, factor):
        inner_steps = args.update_timesteps
        timesteps = args.max_timesteps


        ppo = PPO(args, self.device, local_policy_net)

        trajectories, old_log_probs, rewards, values, dones = self.rollout(local_policy_net, env, initial_observation, ppo, factor, timesteps)
        returns = ppo.compute_gae(rewards, values, dones, values[-1], self.gamma, self.lam)
        advantages = np.array(returns) - np.array(values)

        for _ in range(inner_steps):
            ppo.update(trajectories, old_log_probs, advantages, returns)

        return local_policy_net

    def aggregate_local_to_global(self, episode, global_policy_net):
        global_params = OrderedDict(global_policy_net.named_parameters())
        if episode == 0:
            for name, param in global_params.items():
                param.data.copy_(torch.zeros_like(param.data))

        for local_policy in self.local_policy_nets:
            local_params = OrderedDict(local_policy.named_parameters())
            for (name, param), (_, local_param) in zip(global_params.items(), local_params.items()):
                param.data += local_param.data

        for name, param in global_params.items():
            param.data /= len(self.local_policy_nets)

        return global_policy_net.load_state_dict(global_params)

    def rollout(self, local_policy_net, env, initial_observation, ppo, factor, timesteps):
        trajectories = []
        rewards = []
        values = []
        dones = []
        old_log_probs = []
        infos = {}
        state = initial_observation
        for _ in range(timesteps):
            action, log_prob, value = ppo.select_action(state)
            action_with_factor = (action.numpy().ravel(), factor)
            next_state, reward, done, terminate, info = env.step(action_with_factor)
            self.terminate = terminate
            trajectories.append((state, action.numpy().ravel()))
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)
            old_log_probs.append(log_prob.numpy().ravel())
            infos.update(info)
            if not done:
                state = next_state

        return trajectories, old_log_probs, rewards, values, dones

    def plot(self, epi_rewards_list, average_rewards_list, episode):
        plt.figure()
        plt.plot(epi_rewards_list, label='Episode total Rewards')
        plt.plot(average_rewards_list, label='Average Rewards')
        plt.title('Episode {} Rewards'.format(episode))
        plt.xlabel('Episode')
        plt.ylabel('Episode Rewards')
        plt.legend()
        plt.show()
        plt.pause(1)
        plt.close()









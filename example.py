import torch
import torch.distributed as dist
from sympy.strategies.branch import do_one
from torch import nn, optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import os

class PolicyNet(nn.Module):
    def __init__(self,  state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.action_mean = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        action_mean = self.action_mean(x)
        value = self.value_head(x)
        return action_mean, value

class PPO:
    def __init__(self, policy_net, device, lr=3e-4, gamma=0.99, clip_epsilon=0.2, value_coeff=0.5, entropy_coeff=0.01, batch_size=32):
        self.policy_net = policy_net
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.batch_size = batch_size
        self.device = device

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_mean, _ = self.policy_net(state)
        action_prob = nn.functional.softmax(action_mean, dim=-1)
        action_dist = torch.distributions.Categorical(action_prob)
        action = action_dist.sample()
        log_prob_action = action_dist.log_prob(action)
        return action.cpu().detach(), log_prob_action

        #action_dist = torch.distributions.Normal(action_mean, 1.0)
        #action = action_dist.sample().item()
        #return action.cpu().detach().numpy(), action_dist.log_prob(action_mean[action].item())

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
        states = torch.tensor(np.vstack([t[0] for t in trajectories]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.vstack([t[1] for t in trajectories]), dtype=torch.float32)
        log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        dataset_size = states.size(0)
        for _ in range(10):
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_states = states[start:end]
                batch_actions = actions[start:end]
                batch_log_probs = log_probs[start:end]
                batch_advantages = advantages[start:end]
                batch_returns = returns[start:end]

                new_action_mean, new_value = self.policy_net(batch_states)
                new_value = new_value.cpu().detach()
                action_prob = nn.functional.softmax(new_action_mean, dim=-1)
                action_dist = torch.distributions.Categorical(action_prob)
                new_log_probs = action_dist.log_prob(batch_actions.to(self.device)).cpu().detach()

                ratio = torch.exp(new_log_probs - batch_log_probs)

                surrogate_1 = ratio * batch_advantages
                surrogate_2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                value_loss = self.value_coeff * ((batch_returns - new_value).pow(2)).mean()

                entropy_loss = -self.entropy_coeff * action_dist.entropy().mean()

                loss = policy_loss + value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

class MetaPPO(PPO):
    def __init__(self, meta_global_policy_nets, global_of_global_policy_net, device, lr=3e-4, gamma=0.99, clip_epsilon=0.2, value_coeff=0.5, entropy_coeff=0.01, batch_size=32):
        self.meta_global_policy_nets = meta_global_policy_nets
        self.global_of_global_policy_net = global_of_global_policy_net
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.batch_size = batch_size
        self.device = device

    def adapt_to_task(self, meta_global_policy_net, env, inner_steps=1, timesteps=100):
        ppo = PPO(meta_global_policy_net, self.device, lr=self.lr, batch_size=self.batch_size)
        trajectories, old_log_probs, rewards, values, dones, total_rewards = self.rollout(meta_global_policy_net, env, timesteps)
        returns = self.compute_gae(rewards, values, dones, values[-1], self.gamma)
        advantages = np.array(returns) - np.array(values)
        for _ in range(inner_steps):
            ppo.update(trajectories, old_log_probs, advantages, returns)
        return total_rewards

    def reduce_and_broadcast(self):
        global_params = OrderedDict(self.global_of_global_policy_net.named_parameters())
        for name, param in global_params.items():
            param.data.copy_(torch.zeros_like(param.data))

        for meta_global_policy_net in self.meta_global_policy_nets:
            meta_params = meta_global_policy_net.named_parameters()
            for (name, param), (_, meta_params) in zip(global_params.items(), meta_params):
                param.data += meta_params.data

        for name, param in global_params.items():
            param.data /= len(self.meta_global_policy_nets)

        self.global_of_global_policy_net.load_state_dict(global_params)

        for meta_global_policy_net in self.meta_global_policy_nets:
            meta_global_policy_net.load_state_dict(global_params)

    def rollout(self, policy_net, env, timesteps, render=False):
        trajectories = []
        rewards = []
        values = []
        dones = []
        old_log_probs = []
        total_reward = 0
        state, _ = env.reset()
        for _ in range(timesteps):
            action, log_prob = self.select_action_(policy_net, state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            _, value = policy_net(torch.tensor(state, dtype=torch.float32).to(self.device))
            trajectories.append((state, action))
            rewards.append(reward)
            values.append(value.item())
            done = terminated or truncated
            dones.append(done)
            old_log_probs.append(log_prob.item())
            state = next_state
            total_reward += reward
            if done:
                state, _ = env.reset()
        return trajectories, old_log_probs, rewards, values, dones, total_reward

    def select_action_(self, policy_net, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_mean, _ = policy_net(state)
        action_prob = nn.functional.softmax(action_mean, dim=-1)
        action_dist = torch.distributions.Categorical(action_prob)
        action = action_dist.sample()
        log_prob_action = action_dist.log_prob(action)
        return action.cpu().detach(), log_prob_action


def plot_rewards(individual_rewards, average_rewards, global_rewards, episode):
    plt.figure()
    for i in range(3):
        plt.plot(individual_rewards[i], label=f'Worker {i+1}')
    plt.plot(average_rewards, label='Average Reward of meta policies', color='red', linestyle='--')
    plt.plot(global_rewards, label='Global reward', color='purple', linestyle='--')
    plt.title(f'Rewards up to Episode {episode}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()

def train_meta_worker(meta_global_policy_nets, global_of_global_policy_net, device, rank, size):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['MASTER_ADDR'] = 'fe80::5af6:2525:196e:5f43%17'
    os.environ['MASTER_PORT'] = '9999'
    dist.init_process_group('gloo', init_method="env://?use_libuv=False", rank=rank, world_size=size)

    env = gym.make('CartPole-v1')
    meta_ppo = MetaPPO(meta_global_policy_nets, global_of_global_policy_net, device)

    rewards_log = [[] for _ in range(size)]
    average_rewards_log = []
    global_rewards_log = []

    for episode in range(100):
        episode_rewards = []
        for i in range(size):
            meta_global_policy_net = meta_global_policy_nets[i]
            reward = meta_ppo.adapt_to_task(meta_global_policy_net, env, inner_steps=1, timesteps=100)
            episode_rewards.append(reward)

        for i, reward in enumerate(episode_rewards):
            rewards_log[i].append(reward)

        average_reward = sum(episode_rewards) / 3
        average_rewards_log.append(average_reward)

        meta_ppo.reduce_and_broadcast()

        _, _, _, _, _, global_reward = meta_ppo.rollout(meta_ppo.global_of_global_policy_net, env, timesteps=100)
        global_rewards_log.append(global_reward)


        if episode % 10 == 0:
            print(f"Episode {episode}: Rewards = {episode_rewards}, Global of Global Reward = {global_reward}")
            plot_rewards([rewards_log[i] for i in range(3)], average_rewards_log, global_rewards_log, episode)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    mp.set_start_method('spawn')

    meta_global_policy_nets = [
        PolicyNet(4, 2).to(device).share_memory()
        for _ in range(3)
    ]

    global_of_global_policy_net = PolicyNet(4, 2).to(device)
    global_of_global_policy_net.share_memory()

    size = 3
    processes = []
    for rank in range(size):
        p = mp.Process(target=train_meta_worker, args=(meta_global_policy_nets, global_of_global_policy_net, device, rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    torch.save(global_of_global_policy_net.state_dict(), f'global_of_global_policy_net.pt')

import copy
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

from collections import OrderedDict

from torch.distributions import Beta, Normal

class CNNPolicyNetwork(nn.Module):
    def __init__(self, obs_space, action_space):
        super(CNNPolicyNetwork, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Conv2d(obs_space, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, 64)

        self.policy_mean_net = nn.Sequential(
            nn.Linear(64, action_space),
            nn.Tanh()
        )

        self.value_net = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, state):
        shared_features = self.shared_net(state)
        shared_features = torch.flatten(shared_features, start_dim=1)

        shared_net = F.relu(self.fc1(shared_features))
        shared_net = F.relu(self.fc2(shared_net))

        mean = self.policy_mean_net(shared_net)
        value = self.value_net(shared_net)
        return mean, value

class BetaPolicyNetwork(nn.Module):
    def __init__(self, obs_space, action_space):
        super(BetaPolicyNetwork, self).__init__()

        self.l1 = nn.Linear(obs_space, 150)
        self.l2 = nn.Linear(150, 150)
        self.alpha_head = nn.Linear(150, action_space)
        self.beta_head = nn.Linear(150, action_space)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))

        alpha = F.softplus(self.alpha_head(a)) + 1.0
        beta = F.softplus(self.beta_head(a)) + 1.0

        return alpha, beta

    def get_dist(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist

    def deterministic_action(self, state):
        alpha, beta = self.forward(state)
        mode = (alpha) / (alpha + beta)
        return mode

class GaussianActor_musigma(nn.Module):
    def __init__(self, obs_space, action_space):
        super(GaussianActor_musigma, self).__init__()

        self.l1 = nn.Linear(obs_space, 150)
        self.l2 = nn.Linear(150, 150)
        self.mu_head = nn.Linear(150, action_space)
        self.sigma_head = nn.Linear(150, action_space)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        mu = torch.sigmoid(self.mu_head(state))
        sigma = F.softplus(self.sigma_head(state))

        return mu, sigma

    def get_dist(self, state):
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)
        return dist

    def deterministic_action(self, state):
        mu, sigma = self.forward(state)
        return mu

class Critic(nn.Module):
    def __init__(self, obs_space):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(obs_space, 150)
        self.C2 = nn.Linear(150, 150)
        self.C3 = nn.Linear(150, 1)

    def forward(self, state):
        v = torch.tanh(self.C1(state))
        v = torch.tanh(self.C2(v))
        v = self.C3(v)
        return v

class PolicyNetwork(nn.Module):
    def __init__(self, obs_space, action_space):
        super(PolicyNetwork, self).__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(obs_space, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
        )

        self.policy_mean_net = nn.Sequential(
            nn.Linear(64, action_space),
            nn.Softsign()
        )

        self.value_net = nn.Sequential(
            nn.Linear(64, 1)
        )


    def forward(self, state):
        shared_features = self.shared_net(state)
        mean = self.policy_mean_net(shared_features)
        value = self.value_net(shared_features)
        return mean, value

class PPO:
    def __init__(self, args, device, local_policy_net):

        self.gamma = args.gamma
        self.clip_epsilon = args.clip_epsilon #clip rate
        self.value_coeff = args.value_coeff # critic
        self.entropy_coeff = args.entropy_coeff # entropy coefficient
        self.entropy_coeff_decay = args.entropy_coeff_decay
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.l2_reg = args.l2_reg

        self.inner_steps = args.update_timesteps
        self.batch_size = args.batch_size
        self.device = device
        self.lam = args.lambda_

        self.cov_var = torch.full(size=(1, 3), fill_value=0.5).to(device)
        self.cov_mat = torch.diag(self.cov_var)

        self.max_grad_norm = 1.0

        self.buffer = RolloutBuffer()

        self.local_policy_net = local_policy_net
        self.optimizer = optim.Adam(self.local_policy_net.parameters(), lr=args.lr)

        self.loss = nn.MSELoss()

        self.loss_dict = {
            'policy_loss': [],
            'critic_loss': [],
            'entropy_loss': []
        }

    def select_action(self, state):
        state = torch.Tensor(state).to(self.device)
        self.buffer.states.append(state)

        actions_mean, value = self.local_policy_net(state)
        dist_ = torch.distributions.Normal(actions_mean, self.cov_mat)

        sampled_actions = dist_.rsample()
        log_prob_ = dist_.log_prob(sampled_actions)

        self.buffer.actions.append(sampled_actions.cpu().detach())
        self.buffer.logprobs.append(log_prob_.cpu().detach().sum(dim=-1))
        self.buffer.values.append(value.cpu().detach())

        return sampled_actions.cpu().detach()

    def compute_gae(self, gamma=0.99, lamda=0.95):

        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones

        gamma = self.gamma
        lam = self.lam

        gae = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * (1 - dones[i]) * values[i+1] - values[i]
            gae = delta + gamma * lam * gae * (1 - dones[i])
            returns.insert(0, gae + values[i])

        return returns

    def update(self, returns):

        old_states = torch.stack(self.buffer.states, dim=0).to(self.device)
        old_actions = torch.stack(self.buffer.actions, dim=0).to(self.device)
        old_log_probs = torch.stack(self.buffer.logprobs, dim=0).to(self.device)
        old_values = torch.stack(self.buffer.values, dim=0).to(self.device)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        advantages = returns.detach() - old_values.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # adjustment of epsilon (1e-10 -> 1e-8)

        dataset_size = old_states.size(0)
        batch_iter = int(math.ceil(dataset_size / self.batch_size))

        policy_losses, critic_losses, entropy_losses = [], [], []

        for _ in range(self.inner_steps):
            '''
            perm = torch.randperm(dataset_size).to(self.device)
            shuffled_states = old_states.index_select(0, perm)
            shuffled_actions = old_actions.index_select(0, perm)
            shuffled_log_probs = old_log_probs.index_select(0, perm)
            shuffled_returns = returns.index_select(0, perm)
            shuffled_advantages = advantages.index_select(0, perm)
            '''

            batch_sequence = []
            for i in range(batch_iter):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, dataset_size)
                batch_sequence.append(list(range(start_idx, end_idx)))

            np.random.shuffle(batch_sequence)

            for i in range(batch_iter):
                batch_indices = batch_sequence[i]
                '''
                start_idx = i * self.batch_size
                end_idx = min((i+1)*self.batch_size, dataset_size)
                batch_indices = slice(start_idx, end_idx)

                batch_old_states = shuffled_states[batch_indices]
                batch_old_actions = shuffled_actions[batch_indices]
                batch_advantages = shuffled_advantages[batch_indices]
                batch_returns = shuffled_returns[batch_indices]
                batch_old_log_probs = shuffled_log_probs[batch_indices]
                '''
                batch_old_states = old_states[batch_indices]
                batch_old_actions = old_actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                new_action_mean, new_value = self.local_policy_net(batch_old_states)

                new_value = new_value.squeeze()

                dist = torch.distributions.Normal(new_action_mean, self.cov_mat)

                sampled_actions = dist.rsample()

                new_log_probs = dist.log_prob(sampled_actions).sum(dim=-1)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                ratio = ratio.view(-1,1)

                # Policy loss
                surrogate_1 = ratio * batch_advantages
                surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                # Value loss
                value_loss = self.value_coeff * nn.MSELoss()(batch_returns.squeeze(), new_value)

                # Entropy loss for exploration
                entropy_loss = -self.entropy_coeff * entropy

                # Total loss
                loss = policy_loss + value_loss + entropy_loss

                policy_losses.append(policy_loss)
                critic_losses.append(value_loss)
                entropy_losses.append(entropy_loss)

                # Gradient descent step

                self.optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.local_policy_net.parameters(), max_norm=0.5)

                self.optimizer.step()
                self.loss = loss


        self.buffer.clear()

        return policy_losses[-1], critic_losses[-1], entropy_losses[-1]

    def loss_value(self):
        return self.loss

def adapt_to_task(state, ppo, factor, env, episode, device):

    env_update = False
    done = False

    while not done:

        action = ppo.select_action(state)
        action_with_factor = (action.numpy().ravel(), factor, env_update, False)

        next_state, reward, done, terminate, info = env.step(action_with_factor)

        ppo.buffer.rewards.append(reward)
        ppo.buffer.dones.append(done)

        state = next_state

        if done:
            _, next_value = ppo.local_policy_net(torch.Tensor(state).to(device))
            ppo.buffer.values.append(next_value.cpu().detach())
            returns = ppo.compute_gae(gamma=0.99, lamda=0.95)

            policy_loss, critic_loss, entropy_loss = ppo.update(returns)

            ppo.loss_dict['policy_loss'].append(policy_loss)
            ppo.loss_dict['critic_loss'].append(critic_loss)
            ppo.loss_dict['entropy_loss'].append(entropy_loss)

            return ppo.local_policy_net

def moving_average(data, window_size=10):
    if len(data) < window_size:
        return np.array(data)
    return np.convolve(data, np.ones(window_size) / window_size, 'valid')

def result_plot(args, epi_rewards_list, average_rewards_list, episode, factor):

    moving_rewards = moving_average(epi_rewards_list, window_size=10)

    episodes = np.arange(1, len(epi_rewards_list) + 1)

    plt.figure()
    plt.plot(episodes, epi_rewards_list, label='Episode Reward', alpha=0.4)
    plt.plot(episodes[:len(moving_rewards)], moving_rewards, label='Moving average', linewidth=2)
    plt.title('Episode_{} {} Rewards'.format(episode, factor))
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.reward_folder + '/test/{}_rewards.png'.format(factor))
    plt.close()

    moving_average_rewards = moving_average(average_rewards_list, window_size=10)
    plt.figure()
    plt.plot(episodes, average_rewards_list, label='Episode average Reward', alpha=0.4)
    plt.plot(episodes[:len(moving_average_rewards)], moving_average_rewards, label='Moving average', linewidth=2)
    plt.title('Episode_{} {} average Rewards'.format(episode, factor))
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.reward_folder + '/test/{}_average_rewards.png'.format(factor))
    plt.close()

def loss_plot(args, loss_dict, episode, factor):

    policy_loss = [t.cpu().item() if t.numel() == 1 else t.cpu().numpy() for t in loss_dict['policy_loss']]
    critic_loss = [t.cpu().item() if t.numel() == 1 else t.cpu().numpy() for t in loss_dict['critic_loss']]
    entropy_loss = [t.cpu().item() if t.numel() == 1 else t.cpu().numpy() for t in loss_dict['entropy_loss']]

    plt.figure()
    plt.plot(policy_loss, label='Episode Policy Loss')
    plt.title('{} policy loss in  Episode_{}'.format(factor, episode))
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.savefig(args.reward_folder + '/test/{}_loss.png'.format(factor))
    plt.close()

    plt.figure()
    plt.plot(critic_loss, label='Episode Critic Loss')
    plt.title('{} critic loss in  Episode_{}'.format(factor, episode))
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.savefig(args.reward_folder + '/test/{}_critic_loss.png'.format(factor))
    plt.close()

    plt.figure()
    plt.plot(entropy_loss, label='Episode Entropy Loss')
    plt.title('{} entropy loss in  Episode_{}'.format(factor, episode))
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.savefig(args.reward_folder + '/test/{}_entropy_loss.png'.format(factor))
    plt.close()

def aggregate_local_to_global(local_policy_nets, global_policy_net):
    global_parms = OrderedDict()

    for name, param in global_policy_net.named_parameters():
        global_parms[name] = param.clone().detach()

    for param in global_parms.values():
        param.zero_()

    with torch.no_grad():
        for local_net in local_policy_nets:
            for (name, local_param) in local_net.named_parameters():
                global_parms[name] += local_param.detach()

        num_nets = len(local_policy_nets)
        for name in global_parms:
            global_parms[name] /= num_nets


    global_policy_net.load_state_dict(global_parms)

    return global_policy_net

def test(state, ppo, network, local_ppos, factor, env, episode, args, average_reward_list, system=False):
    done = False

    ppo.local_policy_net = aggregate_local_to_global(local_ppos, network)

    with torch.no_grad():
        while not done:

            action = ppo.select_action(state)
            action_with_factor = (action.numpy().ravel(), factor, system, False)

            next_state, reward, done, terminate, info = env.step(action_with_factor)

            ppo.buffer.states.append(next_state)
            ppo.buffer.rewards.append(reward)
            ppo.buffer.dones.append(env.converted_action)

            for k, v in info.items():
                ppo.buffer.infos.setdefault(k, []).append(v)

            state = next_state

            if done:

                total_rewards = sum(ppo.buffer.rewards)
                average_rewards = total_rewards / len(ppo.buffer.rewards)

                baseline = sum(average_reward_list) / len(average_reward_list) if len(average_reward_list) > 0 else 0

                if average_rewards >= baseline:

                    folder_path = args.reward_folder
                    os.makedirs(folder_path, exist_ok=True)

                    file_name = '_{}_trajectory_{}_episode.csv'.format(factor, episode)
                    file_path = os.path.join(folder_path, file_name)

                    with open(file_path, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['step reward', 'x_extent', 'y_extent', 'capacity'])
                        for total_reward, actions in zip(ppo.buffer.rewards, ppo.buffer.dones):
                            row = [total_reward] + actions
                            writer.writerow(row)

                    print("Find new solution in the system model")

        if system:
            last_info = [v[-1] for v in ppo.buffer.infos.values()]
        else:
            last_info = 0

        ppo.buffer.clear()

        return reward, total_rewards, average_rewards, last_info


def evaluate(state, ppo, factor, env, episode, args, average_reward_list, env_update=False):

    done = False

    with torch.no_grad():
        while not done:

            action = ppo.select_action(state)
            action_with_factor = (action.numpy().ravel(), factor, env_update, False)

            next_state, reward, done, terminate, info = env.step(action_with_factor)

            ppo.buffer.states.append(next_state)
            ppo.buffer.rewards.append(reward)
            ppo.buffer.dones.append(env.converted_action)

            for k, v in info.items():
                ppo.buffer.infos.setdefault(k, []).append(v)

            state = next_state

            if done:

                total_rewards = sum(ppo.buffer.rewards)
                average_rewards = total_rewards / len(ppo.buffer.rewards)
                baseline = sum(average_reward_list) / len(average_reward_list) if len(average_reward_list) > 0 else 0

                if average_rewards >= baseline:

                    folder_path = args.reward_folder
                    os.makedirs(folder_path, exist_ok=True)

                    file_name = '_{}_trajectory_{}_episode.csv'.format(factor, episode)
                    file_path = os.path.join(folder_path, file_name)

                    with open(file_path, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['step reward', 'x_extent', 'y_extent', 'capacity'])
                        for total_reward, actions in zip(ppo.buffer.rewards, ppo.buffer.dones):
                            row = [total_reward] + actions
                            writer.writerow(row)

                    print("Find new solution in the {} model".format(factor))

        if env_update or factor == "system":
            last_info = [v[-1] for v in ppo.buffer.infos.values()]
        else:
            last_info = 0

        ppo.buffer.clear()

        return reward, total_rewards, average_rewards, last_info

class MetaPPO(PPO):
    def __init__(self, device, env, args, local_nets, local_policy_net, batch_size=32):
        super().__init__(args, device, local_policy_net)
        self.env = env
        self.args = args

        self.lr = args.lr
        self.gamma = args.gamma
        self.clip_epsilon = args.clip_epsilon
        self.value_coeff = args.value_coeff
        self.entropy_coeff = args.entropy_coeff
        self.batch_size = batch_size
        self.lam = self.args.lambda_
        self.device = device
        self.terminate = False

        self.loss_dict = {
            'policy_loss': [],
            'critic_loss': [],
            'entropy_loss': []
        }

    def adapt_to_task(self, args, local_policy_net, env, initial_observation, factor, episode):

        env_update = False
        done = False

        ppo = PPO(args, self.device, local_policy_net)

        while not done:
            action = ppo.select_action(initial_observation)
            action_with_factor = (action.numpy().ravel(), factor, env_update, False)

            next_state, reward, done, terminate, info = env.step(action_with_factor)

            ppo.buffer.states.append(next_state)
            ppo.buffer.rewards.append(reward)

            self.terminate = terminate

            state = next_state

            mask = 1 if done else float(not done)
            ppo.buffer.masks.append(mask)

            if done:
                if sum(ppo.buffer.rewards) > -200:
                    policy_loss, critic_loss, entropy_loss = ppo.update(episode)

                    self.local_policy_nets = {
                        'policy': ppo.policy,
                        'critic': ppo.critic
                    }

                    self.loss_dict = {
                        'policy_loss': policy_loss,
                        'critic_loss': critic_loss,
                        'entropy_loss': entropy_loss
                    }
                    return self.local_policy_nets, self.loss_dict

                else:
                    return self.local_policy_nets, self.loss_dict

    def global_evaluate(self, global_policy_net, env, state, env_update = True, factor=None, timesteps=100):

        env_update = env_update
        total_rewards = 0
        done = False
        global_infos = []
        dones = 1
        while not done:
            with torch.no_grad():
                state = torch.Tensor(state).to(self.device)
                actions_mean, _ = global_policy_net(state)
                dist_ = torch.distributions.Normal(actions_mean, self.cov_mat)
                sampled_actions = dist_.sample()
                action = sampled_actions.cpu().detach()
                action_with_factor = (action.numpy().ravel(), factor, env_update, False)
                next_state, reward, done, terminate, info = env.step(action_with_factor)

                total_rewards += reward
                global_infos.append(info)
                state = next_state
                if done is False:
                    dones += 1
        average_reward = total_rewards / dones
        return total_rewards, average_reward, global_infos

    def aggregate_local_to_global(self, episode, local_policy_nets, global_policy_net):
        global_parms = OrderedDict(global_policy_net.named_parameters())

        with torch.no_grad():
            if episode == 0:
                for name, param in global_parms.items():
                    param.copy_(torch.zeros_like(param))

                for local_policy in local_policy_nets:
                    local_params = OrderedDict(local_policy.named_parameters())
                    for (name, param), (_, local_param) in zip(global_parms.items(), local_params.items()):
                        param.add_(local_param)

                divisor = len(local_policy_nets) + (1 if episode > 0 else 0)
                for name, param in global_parms.items():
                    param.div_(divisor)

            if episode > 0:
                for local_policy in local_policy_nets:
                    local_params = OrderedDict(local_policy.named_parameters())
                    for (name, param), (_, local_param) in zip(global_parms.items(), local_params.items()):
                        param.add_(local_param)

                divisor = len(local_policy_nets) + (1 if episode > 0 else 0)
                for name, param in global_parms.items():
                    param.div_(divisor)

        global_policy_net.load_state_dict(global_parms)

        return global_policy_net

    def rollout(self, env, initial_observation, ppo, factor, timesteps):

        state = initial_observation
        done = False
        env_update = False
        while not done:

            action = ppo.select_action(state)
            action_with_factor = (action.numpy().ravel(), factor, env_update, False)
            next_state, reward, done, terminate, info = env.step(action_with_factor)
            self.terminate = terminate

            ppo.buffer.rewards.append(reward)
            ppo.buffer.dones.append(done)
            ppo.buffer.infos.update(info)
            state = next_state

            if done:
                _, next_value = ppo.local_policy_net(torch.Tensor(next_state).to(self.device))
                ppo.buffer.values.append(next_value.cpu().detach())

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.dones = []
        self.rewards = []
        self.next_states = []
        self.infos = {}

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.values[:]
        del self.dones[:]
        del self.rewards[:]
        del self.next_states[:]
        self.infos.clear()




"""
    def aggregate_local_to_global(self, episode, local_policy_nets, global_policy_net):
        global_params = OrderedDict(global_policy_net.named_parameters())
        if episode == 0:
            for name, param in global_params.items():
                param.data.copy_(torch.zeros_like(param.data))

        for local_policy in local_policy_nets:
            local_params = OrderedDict(local_policy.named_parameters())
            for (name, param), (_, local_param) in zip(global_params.items(), local_params.items()):
                param.data += local_param.data

        for name, param in global_params.items():
            param.data /= len(local_policy_nets)

        return global_policy_net.load_state_dict(global_params)
"""



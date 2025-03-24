import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from collections import OrderedDict

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
    def __init__(self, args, device, local_policy_net, cov_mat):
        self.local_policy_net = local_policy_net
        #self.optimizer = optim.Adam(self.local_policy_net.parameters(), lr=args.lr)
        self.optimizer = optim.RMSprop(self.local_policy_net.parameters(), lr=args.lr)
        self.gamma = args.gamma
        self.clip_epsilon = args.clip_epsilon
        self.value_coeff = args.value_coeff
        self.entropy_coeff = args.entropy_coeff
        self.inner_steps = args.update_timesteps
        self.batch_size = args.batch_size
        self.device = device
        self.cov_mat = cov_mat
        self.max_grad_norm = 1.0

        self.buffer = RolloutBuffer()

        self.loss = None

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

    def compute_gae(self, gamma, lam=0.95):

        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones

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

                assert not torch.isnan(new_action_mean).any(), "mean contains NaN"

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

                assert not torch.isnan(loss).any(), "Loss contains Nan"

                # Gradient descent step

                self.optimizer.zero_grad()

                loss.backward()

                #torch.nn.utils.clip_grad_norm_(self.local_policy_net.parameters(), self.max_grad_norm)

                self.optimizer.step()
                self.loss = loss


        self.buffer.clear()

    def loss_value(self):
        return self.loss

class MetaPPO(PPO):
    def __init__(self, device, env, args, batch_size=32):
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

        self.cov_var = torch.full(size=(1,3), fill_value=0.1).to(device)
        self.cov_mat = torch.diag(self.cov_var)

    def adapt_to_task(self, args, local_policy_net, env, initial_observation, factor):

        timesteps = args.max_timesteps


        ppo = PPO(args, self.device, local_policy_net, self.cov_mat)

        self.rollout(env, initial_observation, ppo, factor, timesteps)

        returns = ppo.compute_gae(self.gamma, self.lam)

        ppo.update(returns)

        return local_policy_net

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


    def plot(self, epi_rewards_list, average_rewards_list, episode, factor):
        plt.figure()
        #plt.plot(epi_rewards_list, label='Episode total Rewards')
        plt.plot(average_rewards_list, label='Average {} Rewards'.format(factor))
        plt.title('Episode_{} {} Rewards'.format(episode, factor))
        plt.xlabel('Episode')
        plt.ylabel('Episode average Rewards')
        plt.legend()
        plt.savefig(self.args.reward_folder + '/test/{}_rewards.png'.format(factor))
        plt.close()


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.dones = []
        self.rewards = []
        self.infos = {}

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.values[:]
        del self.dones[:]
        del self.rewards[:]
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



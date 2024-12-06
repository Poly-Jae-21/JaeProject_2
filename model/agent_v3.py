import math
import torch
import torch.nn as nn
import torch.optim as optim
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

        self.buffer = RolloutBuffer()

    def select_action(self, state):
        state = torch.Tensor(state).to(self.device)
        actions_mean, actions_std, value = self.local_policy_net(state)
        dist_ = torch.distributions.Normal(actions_mean, actions_std)
        sampled_actions = dist_.sample()
        log_probs = dist_.log_prob(sampled_actions)

        self.buffer.states.append(state)
        self.buffer.actions.append(sampled_actions.cpu().detach())
        self.buffer.logprobs.append(log_probs.cpu().detach().sum(dim=-1))
        self.buffer.values.append(value.cpu().detach())

        return sampled_actions.cpu().detach()

    def compute_gae(self, gamma, lam=0.95):

        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        next_value = values[-1]

        gae = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * (1 - dones[i]) * next_value - values[i]
            gae = delta + gamma * lam * gae * (1 - dones[i])
            returns.insert(0, gae + values[i])
            next_value = values[i]

        return returns

    def update(self, returns):

        old_states = torch.stack(self.buffer.states, dim=0).to(self.device)
        old_actions = torch.stack(self.buffer.actions, dim=0).to(self.device)
        old_log_probs = torch.stack(self.buffer.logprobs, dim=0).to(self.device)
        old_values = torch.stack(self.buffer.values, dim=0).to(self.device)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        advantages = returns.detach() - old_values.detach()

        dataset_size = old_states.size(0)
        batch_iter = int(math.ceil(dataset_size / self.batch_size))

        for _ in range(self.inner_steps):

            perm = torch.randperm(dataset_size).to(self.device)
            shuffled_states = old_states.index_select(0, perm)
            shuffled_actions = old_actions.index_select(0, perm)
            shuffled_log_probs = old_log_probs.index_select(0, perm)
            shuffled_returns = returns.index_select(0, perm)
            shuffled_advantages = advantages.index_select(0, perm)

            for i in range(batch_iter):
                start_idx = i * self.batch_size
                end_idx = min((i+1)*self.batch_size, dataset_size)
                batch_indices = slice(start_idx, end_idx)

                batch_old_states = shuffled_states[batch_indices]
                batch_old_actions = shuffled_actions[batch_indices]
                batch_advantages = shuffled_advantages[batch_indices]
                batch_returns = shuffled_returns[batch_indices]
                batch_old_log_probs = shuffled_log_probs[batch_indices]

                new_action_mean, new_action_stddev, new_value = self.local_policy_net(batch_old_states)

                new_value = new_value.squeeze()

                dist = torch.distributions.Normal(new_action_mean, new_action_stddev)
                sampled_actions = dist.sample()
                new_log_probs = dist.log_prob(sampled_actions).sum(dim=-1)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Policy loss
                surrogate_1 = ratio * batch_advantages
                surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                # Value loss
                value_loss = self.value_coeff * nn.MSELoss()(batch_returns, new_value)

                # Entropy loss for exploration
                entropy_loss = -self.entropy_coeff * entropy

                # Total loss
                loss = policy_loss + value_loss + entropy_loss

                # Gradient descent step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.buffer.clear()

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

    def adapt_to_task(self, args, local_policy_net, env, initial_observation, factor):

        timesteps = args.max_timesteps


        ppo = PPO(args, self.device, local_policy_net)

        self.rollout(env, initial_observation, ppo, factor, timesteps)

        returns = ppo.compute_gae(self.gamma, self.lam)

        ppo.update(returns)

        return local_policy_net

    def global_evaluate(self, global_policy_net, env, initial_observation, factor=None, timesteps=100):
        total_rewards = 0
        total_dones = 0
        global_infos = []
        for t in range(timesteps):
            with torch.no_grad():
                state = torch.Tensor(initial_observation).to(self.device)
                state = torch.Tensor(state).to(self.device)
                actions_mean, actions_std, _ = global_policy_net(state)
                dist_ = torch.distributions.Normal(actions_mean, actions_std)
                sampled_actions = dist_.sample()
                action = sampled_actions.cpu().detach()
                action_with_factor = (action.numpy().ravel(), factor)

            next_state, reward, done, terminate, info = env.step(action_with_factor)

            total_rewards += reward
            total_dones += done
            global_infos.append(info)

        average_reward = total_rewards / total_dones if total_dones > 0 else 0
        return total_rewards, average_reward, global_infos


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

    def aggregate_local_to_global(self, episode, local_policy_nets, global_policy_net):

    def rollout(self, env, initial_observation, ppo, factor, timesteps):

        state = initial_observation

        for _ in range(timesteps):

            action = ppo.select_action(state)
            action_with_factor = (action.numpy().ravel(), factor)
            next_state, reward, done, terminate, info = env.step(action_with_factor)
            self.terminate = terminate

            ppo.buffer.rewards.append(reward)
            ppo.buffer.dones.append(done)
            ppo.buffer.infos.update(info)

            if not done:
                state = next_state

    def plot(self, epi_rewards_list, average_rewards_list, episode):
        plt.figure()
        plt.plot(epi_rewards_list, label='Episode total Rewards')
        plt.plot(average_rewards_list, label='Average Rewards')
        plt.title('Episode {} Rewards'.format(episode))
        plt.xlabel('Episode')
        plt.ylabel('Episode Rewards')
        plt.legend()
        plt.savefig(self.args.reward_folder + '/test/rewards.png')


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








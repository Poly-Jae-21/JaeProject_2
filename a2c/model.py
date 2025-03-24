
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    def __init__(self, obs_space, hidden_dim, action_space):
        super(PolicyNetwork, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_space, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_space),
            nn.Softsign()
        )

        self.value = nn.Sequential(
            nn.Linear(obs_space, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        policy = self.policy(state)
        value = self.value(state)
        return policy, value

class A2C(object):
    def __init__(self, args, device, local_policy_net, cov_mat):
        self.local_policy_net = local_policy_net
        #self.optimizer = optim.Adam(local_policy_net.parameters(), lr=args.lr)
        self.optimizer = optim.RMSprop(local_policy_net.parameters(), lr=args.lr)

        self.gamma = args.gamma
        self.steps = args.update_timesteps
        self.device = device
        self.cov_mat = cov_mat

        self.loss = 0

        self.buffer = RolloutBuffer()

        self.args = args

    def select_action(self, state):
        state = torch.Tensor(state).to(self.device)
        self.buffer.states.append(state)

        log_prob, value = self.local_policy_net(state)

        dist = torch.distributions.Normal(log_prob, self.cov_mat)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        action = torch.clamp(action, np.min(-1), np.max(+1))
        entropy = dist.entropy().sum(dim=-1)

        self.buffer.actions.append(action.cpu().detach())
        self.buffer.logprobs.append(log_prob.sum(dim=-1))
        self.buffer.values.append(value)
        self.buffer.entropies.append(entropy)

        return action.cpu().detach(), log_prob, value, entropy

    def compute_returns(self, gamma):
        returns = []

        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones

        R = values[-1]

        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + gamma * R * (1.0 - done)
            returns.insert(0, R)
        return returns

    def update(self, returns):

        values = torch.cat(self.buffer.values, dim=0).to(self.device)
        log_probs = torch.stack(self.buffer.logprobs, dim=0).to(self.device)
        entropies = torch.stack(self.buffer.entropies, dim=0).to(self.device)

        returns = torch.FloatTensor(returns).to(self.device)

        advantages = returns - values

        policy_loss = -(log_probs * advantages.detach()).mean()

        value_loss = advantages.pow(2).mean()

        entropy_loss = entropies.mean()

        loss = policy_loss + 0.5 * value_loss - entropy_loss * self.args.entropy_beta

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()
        self.loss = loss

        self.buffer.clear()

class MetaA2C(A2C):
    def __init__(self, device, env, args):
        self.env = env
        self.args = args
        self.gamma = args.gamma
        self.steps = args.update_timesteps
        self.device = device
        self.terminate = False

        self.cov_var = torch.full(size=(1,3), fill_value=0.1).to(self.device)
        self.cov_mat = torch.diag(self.cov_var)

    def adapt_to_task(self, args, local_policy_net, env, initial_observation, factor):

        timesteps = args.max_timesteps

        a2c = A2C(self.args, self.device, local_policy_net, self.cov_mat)

        self.rollout(env, initial_observation, a2c, factor, timesteps)

        returns = a2c.compute_returns(self.gamma)

        a2c.update(returns)

        return local_policy_net

    def global_evaluate(self, global_policy_net, env, state, env_update=True, factor=None, timesteps=100):

        env_update = env_update
        total_rewards = 0
        done = False
        global_infos = []
        dones = 1
        while not done:
            with torch.no_grad():
                state = torch.Tensor(state).to(self.device)
                action, _ = global_policy_net(state)
                dist = torch.distributions.Normal(action, self.cov_mat)
                sampled_action = dist.sample()
                action = sampled_action.cpu().detach()
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

    def rollout(self, env, initial_observation, a2c, factor, timesteps):

        state = initial_observation
        done = False
        env_update = False

        values = []
        log_probs = []
        rewards = []
        infos = []

        while not done:
            action, log_prob, value, entropy = a2c.select_action(state)
            action_with_factor = (action.numpy().ravel(), factor, env_update, False)
            next_state, reward, done, terminate, info = env.step(action_with_factor)

            self.terminate = terminate

            state = next_state

            a2c.buffer.rewards.append(reward)
            a2c.buffer.dones.append(done)
            a2c.buffer.infos.update(info)

            if done:
                break

    def plot(self, epi_rewards_list, average_rewards_list, episode, factor):
        plt.figure()
        #plt.plot(epi_rewards_list, label='{} Total Rewards'.format(factor))
        plt.plot(average_rewards_list, label='{} Average Rewards'.format(factor))
        plt.title('{} Rewards in Episode_{} '.format(episode, factor))
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
        self.entropies = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.values[:]
        del self.dones[:]
        del self.rewards[:]
        del self.entropies[:]
        self.infos.clear()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
eps = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        #Q2 architecture
        self.linear4 = nn.Linear(num_inputs+num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear3(x1))

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = F.relu(self.linear6(x2))

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + eps)
        if len(log_prob) == 3:
            log_prob = log_prob.sum(-1, keepdim=True)
        else:
            log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)

import os
from torch.optim import Adam, RMSprop
from update import soft_update, hard_update
import math
import numpy as np
from collections import OrderedDict

class SAC:
    def __init__(self, args, device, local_critic, local_target_critic, local_policy):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.batch_size = args.batch_size

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.buffer = RolloutBuffer()

        self.device = device

        self.critic = local_critic
        self.critic_optim = RMSprop(self.critic.parameters(), lr=args.lr)

        self.critic_target = local_target_critic
        hard_update(self.critic_target, self.critic)

        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(3).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = RMSprop([self.log_alpha], lr=args.lr)

        self.policy = local_policy
        self.policy_optim = RMSprop(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, eval=False):
        state = torch.Tensor(state).to(self.device)
        self.buffer.states.append(state)

        if eval is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)

        self.buffer.actions.append(action)
        return action.cpu().detach()

    def update(self, updates):
        old_state = torch.stack(self.buffer.states, dim=0).to(self.device)
        old_next_state = torch.stack([torch.tensor(s, dtype=torch.float32) for s in self.buffer.next_states], dim=0).to(self.device)
        old_action = torch.stack(self.buffer.actions, dim=0).to(self.device)
        old_reward = torch.stack([torch.tensor(s, dtype=torch.float32) for s in self.buffer.rewards], dim=0).to(self.device)
        old_mask = torch.stack([torch.tensor(s, dtype=torch.float32) for s in self.buffer.masks], dim=0).to(self.device)

        dataset_size = old_state.size(0)
        batch_iter = int(math.ceil(dataset_size / self.batch_size))

        batch_sequences = []

        for i in range(batch_iter):
            start_idx = i * self.batch_size
            end_idx = min((i+1) * self.batch_size, dataset_size)
            batch_sequences.append(list(range(start_idx, end_idx)))

        np.random.shuffle(batch_sequences)


        batch_indices = batch_sequences[0]

        state_batch = old_state[batch_indices]
        next_state_batch = old_next_state[batch_indices]
        action_batch = old_action[batch_indices]
        reward_batch = old_reward[batch_indices].unsqueeze(1)
        mask_batch = old_mask[batch_indices].unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            q1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf2_next_target, q1_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf2_pi, qf1_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        self.buffer.clear()

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists("checkpoints/"):
            os.makedirs("checkpoints/")
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print("Saving checkpoint to {}".format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

        def load_checkpoint(ckpt_path, evaluate=False):
            print("Loading checkpoint from {}".format(ckpt_path))
            if ckpt_path is None:
                checkpoint = torch.load(ckpt_path)
                self.policy.load_state_dict(checkpoint['policy_state_dict'])
                self.critic.load_state_dict(checkpoint['critic_state_dict'])
                self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
                self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

                if evaluate:
                    self.policy.eval()
                    self.critic.eval()
                    self.critic_target.eval()

                else:
                    self.policy.train()
                    self.critic.train()
                    self.critic_target.train()

import matplotlib.pyplot as plt

class MetaSAC(SAC):
    def __init__(self, device, env, args, local_nets, batch_size=50):
        self.env = env
        self.args = args
        self.device = device

        self.lr = args.lr
        self.batch_size = batch_size

        self.terminate = False

        self.local_policy_nets = {}
        self.loss_dict = {}

        self.local_policy_nets = {
            'policy': local_nets['policy'],
            'critic': local_nets['critic'],
            'target_critic': local_nets['target_critic']
        }
        self.loss_dict = {
            'critic_loss': None,
            'critic_target_loss': None,
            'policy_loss': None,
            'ent_loss': None,
            'alpha': None,
        }

    def adapt_to_task(self, args, local_nets, env, state, factor, episode):

        env_update = False
        done = False

        sac = SAC(args, self.device, local_nets['critic'], local_nets['target_critic'], local_nets['policy'])

        while not done:

            action = sac.select_action(state)
            action_with_factor = (action.numpy().ravel(), factor, env_update, False)

            next_state, reward, done, terminate, info = env.step(action_with_factor)

            sac.buffer.next_states.append(next_state)
            sac.buffer.rewards.append(reward)

            self.terminate = terminate

            state = next_state

            mask = 1 if done else float(not done)
            sac.buffer.masks.append(mask)

            if done:
                if sum(sac.buffer.rewards) > -200:
                    critic1_loss, critic2_loss, policy_loss, ent_loss, alpha = sac.update(episode)

                    self.local_policy_nets = {
                        'policy': sac.policy,
                        'critic': sac.critic,
                        'target_critic': sac.critic_target
                    }

                    self.loss_dict = {
                        'critic_loss': critic1_loss,
                        'critic_target_loss': critic2_loss,
                        'policy_loss': policy_loss,
                        'ent_loss': ent_loss,
                        'alpha': alpha,
                    }
                    return self.local_policy_nets, self.loss_dict
                else:
                    return self.local_policy_nets, self.loss_dict

    def global_evaluate(self, global_nets, env, state, env_update=True, factor=None, timesteps=100):

        env_update = env_update
        total_rewards = 0
        done = False
        global_infos = []
        dones = 1
        while not done:
            with torch.no_grad():
                state = torch.Tensor(state).to(self.device)
                _, _, action = global_nets['policy'].sample(state)
                action = action.cpu().detach()
                action_with_factor = (action.numpy().ravel(), factor, env_update, False)
                next_state, reward, done, terminate, info = env.step(action_with_factor)

                total_rewards += reward
                global_infos.append(info)
                state = next_state
                if done is False:
                    dones += 1

        average_reward = total_rewards / dones
        return total_rewards, average_reward, global_infos

    def aggregate_local_to_global(self, episode, local_nets, global_nets):
        global_policy_net = global_nets['policy']
        global_parms = OrderedDict(global_policy_net.named_parameters())

        with torch.no_grad():
            if episode == 0:
                for name, param in global_parms.items():
                    param.copy_(torch.zeros_like(param))

                for local_net in local_nets:
                    local_policy = local_net['policy']
                    local_params = OrderedDict(local_policy.named_parameters())
                    for (name, param), (_, local_param) in zip(global_parms.items(), local_params.items()):
                        param.add_(local_param)

                divisor = len(local_nets) + (1 if episode > 0 else 0)
                for name, param in global_parms.items():
                    param.div_(divisor)

            if episode > 0:
                for local_net in local_nets:
                    local_policy = local_net['policy']
                    local_params = OrderedDict(local_policy.named_parameters())
                    for (name, param), (_, local_param) in zip(global_parms.items(), local_params.items()):
                        param.add_(local_param)

                divisor = len(local_nets) + (1 if episode > 0 else 0)
                for name, param in global_parms.items():
                    param.div_(divisor)

        global_policy_net.load_state_dict(global_parms)
        global_policy_net = {'policy': global_policy_net}
        return global_policy_net

    def plot(self, epi_rewards_list, average_rewards_list, loss_list, episode, factor):
        if factor in ["environment", "economic", "urbanity"]:
            plt.figure()
            plt.plot(epi_rewards_list)
            plt.title("{} Rewards in Episode {}".format(factor, episode))
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.savefig(self.args.reward_folder + '/test/{}_total_rewards.png'.format(factor))

            plt.figure()
            plt.plot(average_rewards_list)
            plt.title("{} Average Rewards in Episode {}".format(factor, episode))
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.savefig(self.args.reward_folder + '/test/{}_average_rewards.png'.format(factor))

            plt.plot(loss_list['policy_loss'])
            plt.title("{} Policy Loss in Episode {}".format(factor, episode))
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.savefig(self.args.reward_folder + '/test/{}_policy_loss.png'.format(factor))

            plt.plot(loss_list['critic_loss'])
            plt.title("{} Critic Loss in Episode {}".format(factor, episode))
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.savefig(self.args.reward_folder + '/test/{}_critic_loss.png'.format(factor))

            plt.plot(loss_list['critic_target_loss'])
            plt.title("{} Critic Target Loss in Episode {}".format(factor, episode))
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.savefig(self.args.reward_folder + '/test/{}_critic_target_loss.png'.format(factor))

            plt.close()

        elif factor == "average":
            plt.figure()
            plt.plot(epi_rewards_list)
            plt.title("Entire Average Rewards in Episode {}".format(episode))
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.savefig(self.args.reward_folder + '/test/entire_average_rewards.png')

            plt.figure()
            plt.plot(average_rewards_list)
            plt.title("Entire Total Rewards in Episode {}".format(episode))
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.savefig(self.args.reward_folder + '/test/entire_total_rewards.png')

            plt.close()

        else:
            plt.figure()
            plt.plot(epi_rewards_list)
            plt.title("{} Rewards in Episode {}".format(factor, episode))
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.savefig(self.args.reward_folder + '/test/{}_total_rewards.png'.format(factor))

            plt.figure()
            plt.plot(average_rewards_list)
            plt.title("{} Average Rewards in Episode {}".format(factor, episode))
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.savefig(self.args.reward_folder + '/test/{}_average_rewards.png'.format(factor))

            plt.close()

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.masks = []

    def clear(self):
        del self.states[:]
        del self.next_states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.masks[:]
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


class PPO_agent:
    def __init__(self, args, device, env):

        self.batch_size = args.batch_size
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.gamma = args.gamma
        self.lam = args.lambda_
        self.clip_epsilon = args.clip_epsilon  # clip rate
        self.value_coeff = args.value_coeff  # critic
        self.entropy_coeff = args.entropy_coeff  # entropy coefficient
        self.entropy_coeff_decay = args.entropy_coeff_decay
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.l2_reg = args.l2_reg

        self.device = device
        self.inner_steps = args.update_timesteps
        self.batch_size = args.batch_size
        self.Distribution = args.Distribution

        self.loss = nn.MSELoss()

        self.reward_scaling = True
        if self.reward_scaling:
            self.running_reward_mean = 0
            self.running_reward_std = 1
            self.reward_count = 0

        self.loss_dict = {
            'policy_loss': [],
            'critic_loss': [],
            'entropy_loss': []
        }

        self.buffer = RolloutBuffer()

        # Actor Distribution
        if self.Distribution == "Beta":
            self.actor = BetaPolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        elif self.Distribution == "Gamma_mustd":
            self.actor = GaussianActor_musigma(self.state_dim, self.action_dim).to(self.device)
        else:
            print('Nonimplemented Network distribution')
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Critic
        self.critic = Critic(self.state_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def select_action(self, state, diterministic):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            self.buffer.states.append(state.cpu().numpy())
            if diterministic:
                a = self.actor.deterministic_action(state)
                return a.cpu().numpy(), None
            else:
                dist = self.actor.get_dist(state)
                a = dist.sample()
                a = torch.clamp(a, 0, 1)
                logprob_a = dist.log_prob(a).cpu().numpy()

                self.buffer.actions.append(a.cpu().numpy())
                self.buffer.logprobs.append(logprob_a)

                return a.cpu().numpy(), logprob_a

    def update(self):

        self.entropy_coeff *= self.entropy_coeff_decay
        s = torch.from_numpy(np.array(self.buffer.states)).to(self.device)
        a = torch.from_numpy(np.array(self.buffer.actions)).to(self.device)
        r = torch.from_numpy(np.array(self.buffer.rewards)).to(self.device)
        s_next = torch.from_numpy(np.array(self.buffer.next_states)).to(self.device)
        logprob_a = torch.from_numpy(np.array(self.buffer.logprobs)).to(self.device)
        dones = torch.from_numpy(np.array(self.buffer.dones)).to(self.device)

        if self.reward_scaling:
            self.running_reward_mean, self.running_reward_std, self.reward_count = self.update_running_stats(r.float(), self.running_reward_mean, self.running_reward_std, self.reward_count)

            r = (r - self.running_reward_mean) / (self.running_reward_std + 1e-8)
        with torch.no_grad():

            vs = self.critic(s)
            vs_ = self.critic(s_next)

            deltas = r + self.gamma * vs_ * (~dones) - vs
            deltas = deltas.cpu().flatten().numpy()

            adv = [0]

            for dlt, mask in zip(deltas[::-1], dones.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lam * adv[-1] * (~mask)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.device)
            td_target = adv + vs
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        a_optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))
        c_optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))

        a_losses, c_losses, e_losses = [], [], []

        for i in range(self.inner_steps):

            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)
            s, a, td_target, adv, logprob_a = s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[
                perm].clone(), logprob_a[perm].clone()

            for i in range(a_optim_iter_num):
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0]))
                distribution = self.actor.get_dist(s[index])
                dist_entropy = distribution.entropy().sum(1, keepdim=True)
                logprob_a_next = distribution.log_prob(a[index])
                ratio = torch.exp(logprob_a_next.sum(1, keepdims=True) - logprob_a[index].sum(1, keepdim=True))

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv[index]
                e_loss = - self.entropy_coeff * dist_entropy
                a_loss = -torch.min(surr1, surr2) + e_loss

                a_losses.append(a_loss.mean().detach().cpu().item())
                e_losses.append(e_loss.mean().detach().cpu().item())

                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

            for i in range(c_optim_iter_num):
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0]))
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                c_losses.append(c_loss.detach().cpu().item())

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()

        self.buffer.clear()

        return np.array(a_losses).mean(), np.array(c_losses).mean(), np.array(e_losses).mean()

    def save(self, episode):
        torch.save(self.actor.state_dict(), "./out/{}_actor.pth".format(episode))
        torch.save(self.critic.state_dict(), "./out/{}_critic.pth".format(episode))

    def load(self, episode):
        self.actor.load_state_dict(torch.load("./out/{}_actor.pth".format(episode)))
        self.critic.load_state_dict(torch.load("./out/{}_critic.pth".format(episode)))

    def update_running_stats(self, x, running_mean, running_std, count):
        """Calculates running mean and std."""
        batch_mean = torch.mean(x)
        batch_var = torch.var(x)
        batch_size = x.shape[0]

        delta = batch_mean - running_mean
        total_count = count + batch_size
        new_mean = running_mean + delta * batch_size / total_count

        m_a = running_std ** 2 * count
        m_b = batch_var * batch_size
        M2 = m_a + m_b + delta ** 2 * count * batch_size / total_count
        new_std = torch.sqrt(M2 / total_count)

        return new_mean, new_std, total_count

def adapt_to_task(state, ppo, factor, env, device, episode):
    env_update = False
    done = False

    while not done:

        action, _ = ppo.select_action(state, diterministic=False)
        action_with_factor = (action, factor, env_update, False)

        next_state, reward, done, terminate, info = env.step(action_with_factor)

        ppo.buffer.rewards.append(reward)
        ppo.buffer.dones.append(done)
        ppo.buffer.next_states.append(next_state)

        state = next_state

        if done:
            if episode ==0:
                policy_loss, critic_loss, entropy_loss = ppo.update()
                ppo.loss_dict['policy_loss'].append(policy_loss)
                ppo.loss_dict['critic_loss'].append(critic_loss)
                ppo.loss_dict['entropy_loss'].append(entropy_loss)
                return ppo.actor
            if sum(ppo.buffer.rewards) > -1000:
                policy_loss, critic_loss, entropy_loss = ppo.update()
                ppo.loss_dict['policy_loss'].append(policy_loss)
                ppo.loss_dict['critic_loss'].append(critic_loss)
                ppo.loss_dict['entropy_loss'].append(entropy_loss)
                return ppo.actor
            else:
                ppo.loss_dict['policy_loss'].append(ppo.loss_dict['policy_loss'][-1])
                ppo.loss_dict['critic_loss'].append(ppo.loss_dict['critic_loss'][-1])
                ppo.loss_dict['entropy_loss'].append(ppo.loss_dict['entropy_loss'][-1])
                ppo.buffer.clear()
                return ppo.actor



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
    policy_loss = loss_dict['policy_loss']
    critic_loss = loss_dict['critic_loss']
    entropy_loss = loss_dict['entropy_loss']

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

            action, _ = ppo.select_action(state, diterministic=False)
            action_with_factor = (action, factor, system, False)

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

                    file_name = 'system_{}_trajectory_{}_episode.csv'.format(factor, episode)
                    file_path = os.path.join(folder_path, file_name)

                    with open(file_path, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['step reward', 'x_extent', 'y_extent', 'capacity'])
                        for total_reward, actions in zip(ppo.buffer.rewards, ppo.buffer.dones):
                            row = [total_reward] + actions.tolist()
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

            action, _ = ppo.select_action(state, diterministic=False)
            action_with_factor = (action, factor, env_update, False)

            next_state, reward, done, terminate, info = env.step(action_with_factor)

            ppo.buffer.states.append(next_state)
            ppo.buffer.rewards.append(reward)
            ppo.buffer.actions.append(env.converted_action)

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

                    file_name = 'overall_{}_trajectory_{}_episode.csv'.format(factor, episode)
                    file_path = os.path.join(folder_path, file_name)

                    with open(file_path, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['step reward', 'x_extent', 'y_extent', 'capacity'])
                        for total_reward, actions in zip(ppo.buffer.rewards, ppo.buffer.actions):
                            row = [total_reward] + actions.tolist()
                            writer.writerow(row)

                    print("Find new solution in the {} model".format(factor))

        if env_update or factor == "system":
            last_info = [v[-1] for v in ppo.buffer.infos.values()]
        else:
            last_info = 0

        ppo.buffer.clear()

        return reward, total_rewards, average_rewards, last_info


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


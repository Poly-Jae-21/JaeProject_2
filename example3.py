import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp



from model.agent_v2 import PolicyNetwork

parser = argparse.ArgumentParser(description='Pytorch PPO Example')
parser.add_argument('--gpus', default=1, type=int, help='number of GPUs')
parser.add_argument('--env', type=str, default='LunarLander-v2')
parser.add_argument('--render', action='store_false', help='render the environment')
parser.add_argument('--solved_reward', type=float, default=200, help='stop training if avg_reward > solved_reward')
parser.add_argument('--print_interval', type=int, default=10, help='how many episodes to print the results out')
parser.add_argument('--save_interval', type=int, default=100, help='how many episodes to save a checkpoint')
parser.add_argument('--max_episodes', type=int, default=200)
parser.add_argument('--max_timesteps', type=int, default=300, help='maximum timesteps in one episode')
parser.add_argument('--update_timestep', type=int, default=200, help='how many timesteps to update the policy')
parser.add_argument('--K_epochs', type=int, default=4, help='number of policy updates per PPO step')
parser.add_argument('--eps_clip', type=float, default=0.2, help='epsilon for p/q clipping')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--seed', type=int, default=123, help='random seed to use')
parser.add_argument('--ckpt_folder', default='./checkpoints', help='location to save checkpoint models')
parser.add_argument('--tb', action='store_true', help='use Tensorboard?')
parser.add_argument('--log_folder', default='./logs', help='location to save logs')
parser.add_argument('--mode', default='train', help='choose train or test')
parser.add_argument('--restore', action='store_true', help='restore and continue training?')
parser.add_argument('--test', action='store_true', help='choose train or test')

parser.add_argument('--size', type=int, default=3, help='size of workers')

opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.logprobs.clear()

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def act(self, state, memory):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_log_prob)

        return action.item()

    def evaluate(self, state, action):
        state_value = self.critic(state)

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_log_probs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_size, action_size, lr, gamma, K_epochs, eps_clip, restore=False, ckpt=None):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_size, action_size).to(device)
        if restore and ckpt:
            try:
                self.policy.load_state_dict(torch.load(ckpt, map_location=device))
                print(f"Restored model from {ckpt}")
            except Exception as e:
                print(f"Failed to restore model: {e}")
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.old_policy = ActorCritic(state_size, action_size).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        return self.old_policy.act(state, memory)

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(memory.rewards), reversed(memory.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.cat(memory.states).detach().to(device)
        old_actions = torch.cat(memory.actions).detach().to(device)
        old_logprobs = torch.cat(memory.logprobs).detach().to(device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)

            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = 0.5 * self.MSE_loss(state_values, rewards)
            entropy_loss = -0.01 * dist_entropy.mean()

            loss = actor_loss + critic_loss + entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())

def train(env_name, env, state_dim, action_dim, render, solved_reward, max_episodes, max_timesteps, update_timestep, K_epochs, eps_clip, gamma, lr, ckpt_folder, restore, tb=False, print_interval=10, save_interval=100):
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    ckpt = os.path.join(ckpt_folder, f'PPO_discrete_{env_name}.pth')
    memory = Memory()
    ppo = PPO(state_dim, action_dim, lr, gamma, K_epochs, eps_clip, restore, ckpt=False)

    running_reward, avg_length, time_step = 0, 0, 0

    if tb:
        writer = SummaryWriter()

    for i_episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        for t in range(max_timesteps):
            time_step += 1
            action = ppo.select_action(state, memory)
            state, reward, done, _, _ = env.step(action)
            memory.rewards.append(reward)
            memory.dones.append(done)

            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0

            running_reward += reward

            if done:
                break
        avg_length += t

        if running_reward > (print_interval * solved_reward):
            print(f"########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), ckpt)
            break

        if i_episode % save_interval == 0:
            torch.save(ppo.policy.state_dict(), ckpt)

        if i_episode % print_interval == 0:
            avg_length = int(avg_length / print_interval)
            running_reward = int(running_reward / print_interval)

            print(f"Episode {i_episode} \t Avg length: {avg_length} \t Avg reward: {running_reward}")

            if tb:
                writer.add_scalar('Reward', running_reward, i_episode)
                writer.add_scalar('Length', avg_length, i_episode)

            running_reward, avg_length = 0, 0

    if tb:
        writer.close()

def test(env_name, state_dim, action_dim, render, K_epochs, eps_clip, gamma, lr, ckpt_folder, test_episodes):

    env = gym.make(env_name, render_mode='human')
    ckpt = ckpt_folder+'/PPO_discrete_'+env_name+'.pth'
    print('Load checkpoint from {}'.format(ckpt))

    memory = Memory()

    ppo = PPO(state_dim, action_dim, lr, gamma, K_epochs, eps_clip, restore=True, ckpt=ckpt)

    episode_reward, time_step = 0, 0
    avg_episode_reward, avg_length = 0, 0

    # test
    for i_episode in range(1, test_episodes+1):
        state, _ = env.reset()
        while True:
            time_step += 1

            # Run old policy
            action = ppo.select_action(state, memory)

            state, reward, done, _, _ = env.step(action)

            episode_reward += reward


            if done:
                print('Episode {} \t Length: {} \t Reward: {}'.format(i_episode, time_step, episode_reward))
                avg_episode_reward += episode_reward
                avg_length += time_step
                memory.clear_memory()
                time_step, episode_reward = 0, 0
                break

    print('Test {} episodes DONE!'.format(test_episodes))
    print('Avg episode reward: {} | Avg length: {}'.format(avg_episode_reward/test_episodes, avg_length/test_episodes))


if __name__ == '__main__':

    env = gym.make(opt.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if opt.mode == 'train':
        train(opt.env, env, state_dim, action_dim, opt.render, opt.solved_reward,
              opt.max_episodes, opt.max_timesteps, opt.update_timestep, opt.K_epochs,
              opt.eps_clip, opt.gamma, opt.lr, opt.ckpt_folder, opt.restore, opt.tb,
              opt.print_interval, opt.save_interval)
    else:
        raise ValueError("Invalid mode. Use 'train'.")
    if opt.mode_ == 'test':
        test(opt.env, state_dim, action_dim,
             render=opt.render, K_epochs=opt.K_epochs, eps_clip=opt.eps_clip,
             gamma=opt.gamma, lr=opt.lr, ckpt_folder=opt.ckpt_folder, test_episodes=100)



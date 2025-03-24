import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime

from utils.build import argument

gym.register('UrbanEnvChicago-v2', entry_point='template.env_name.envs.multi_policies:ChicagoMultiPolicyMap_v2')
env = gym.make('UrbanEnvChicago-v2', render_mode='human')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = argument()

agent = SAC(env.observation_space.shape[0], env.action_space.n, args)

writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), args.env_name, args.policy, "autotune" if args.automatic_entropy_tuning else ""))

memory = ReplayMemory(args.replay_size, args.seed)

total_numsteps = 0
updates = 0

for episode in range(args.num_episodes):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        if len(memory) > args.batch_size:
            for i in range(args.updates_per_step):
                critic_1_loss, critic_2_loss, policy_loss, entropy_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', entropy_loss, updates)
                writer.add_scalar('entropy_temperature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        mask = 1 if episode_steps == env._max_episode_steps else float (not done)

        memory.push(state, action, reward, next_state, mask)

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(episode, total_numsteps, episode_steps, episode_reward))

    if episode % 10 == 0 and args.eval is True:
        avg_reward = 0.0
        episodes = 10
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                state = next_state

            avg_reward += episode_reward

        avg_reward /= episodes

        writer.add_scalar('avg_reward/test', avg_reward, episode)

        print("--------------------------------------------")
        print("Test Episode: {}, Avg. Reward: {}".format(episodes, avg_reward))

env.close()


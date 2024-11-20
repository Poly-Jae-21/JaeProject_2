import os.path

from torch.utils.tensorboard import SummaryWriter



class Memory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.terminates = []
        self.logprobs = []

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.terminates.clear()
        self.logprobs.clear()

def train(args, env, state_dim, action_dim):
    if not os.path.exists(args.ckpt_folder):
        os.makedirs(args.ckpt_folder)

    ckpt = os.path.join(args.ckpt_folder, f'PPO_model_{args.env_name}.pth')
    memory = Memory()
    ppo = PPO(state_dim, action_dim, args, ckpt=False)

    running_reward , avg_length, time_step = 0, 0, 0

    if args.tb:
        writer = SummaryWriter()

    for i_episode in range(1, args.num_episodes+1):
        state_starting_point, observation = env.reset()
        for t in range(args.max_timesteps):
            time_step += 1
            action = ppo.select_action(observation, memory)
            next_observation, reward, done, terminate, info = env.step(action)
            memory.rewards.append(reward)
            memory.dones.append(done)
            memory.terminates.append(terminate)

            running_reward += reward

            if not done:
                observation = next_observation

from ppo.train_v2 import train

from utils.build import argument
import torch
import gymnasium as gym

gym.register('UrbanEnvChicago-v2', entry_point='template.env_name.envs.multi_policies:ChicagoMultiPolicyMapv2')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create one common environment for whole workers
    env = gym.make('UrbanEnvChicago-v2', render_mode='human')

    args = argument()


    world_size = 5

    train_ = train()
    train_.train(device, world_size, args, env)

    print("finished")


if __name__ == "__main__":
    main()
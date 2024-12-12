import argparse
import torch
import numpy as np
import random

import yaml
from sympy.physics.units import action

from utils.logger import configure_logger

def argument():
    parser = argparse.ArgumentParser(description="Using PPO for solving multiple cities planning for EVFCS placement problems")
    parser.add_argument('--name', default='logging', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--test', action='store_false')
    parser.add_argument('--ckpt_folder', default='./checkpoints', help='location to save checkpoint models')
    parser.add_argument('--tb', action='store_true', help='whether to save tensorboard logs')
    parser.add_argument('--log_folder', default='./logs', help='location to save tensorboard logs')
    parser.add_argument('--reward_folder', default='out/result/reward', help='location to save reward logs')
    parser.add_argument('--restore', action='store_true', help='whether to restore checkpoint')
    parser.add_argument('--env_name', type=str, default='UrbanEnvChicago-v1')

    parser.add_argument('--print_interval', type=int, default=10, help='how many episodes to print the result out')
    parser.add_argument('--save_interval', type=int, default=100, help='how many episodes to save the result')
    parser.add_argument('--max_episodes', type=int, default=10000, help='maximum number of episodes')
    parser.add_argument('--test_episodes', type=int, default=100, help='number of episodes to test')
    parser.add_argument('--max_timesteps', type=int, default=100, help='maximum timesteps in one episode')
    parser.add_argument('--update_timesteps', type=int, default=10, help='update timesteps in one episode (inner update)')

    parser.add_argument('--batch_size', type=int, default=10, help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing (default: 1000)')

    parser.add_argument('--max_steps', type=int, default=100, help='maximum number of steps (default: 100)')
    parser.add_argument('--inner_steps', type=int, default=2, help='number of steps to train (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor (default: 0.95)')
    parser.add_argument('--lambda_', type=float, default=0.95, help='lambda parameter (default: 0.95)')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='clipping parameter (default: 0.2)')
    parser.add_argument('--value_coeff', type=float, default=0.5, help='value coefficient (default: 0.5)')
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='entropy coefficient (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')

    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    return args

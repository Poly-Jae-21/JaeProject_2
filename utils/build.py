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
    parser.add_argument('--ckpt_folder', default='out/result/checkpoints', help='location to save checkpoint models')
    parser.add_argument('--tb', action='store_true', help='whether to save tensorboard logs')
    parser.add_argument('--log_folder', default='./logs', help='location to save tensorboard logs')
    parser.add_argument('--reward_folder', default='out/result/reward', help='location to save reward logs')
    parser.add_argument('--restore', action='store_true', help='whether to restore checkpoint')
    parser.add_argument('--env_name', type=str, default='UrbanEnvChicago-v1')

    parser.add_argument('--print_interval', type=int, default=10, help='how many episodes to print the result out')
    parser.add_argument('--save_interval', type=int, default=100, help='how many episodes to save the result')
    parser.add_argument('--max_episodes', type=int, default=20000, help='maximum number of episodes')
    parser.add_argument('--test_episodes', type=int, default=100, help='number of episodes to test')
    parser.add_argument('--max_timesteps', type=int, default=200, help='maximum timesteps in one episode')
    parser.add_argument('--update_timesteps', type=int, default=10, help='update timesteps in one episode (inner update)')

    parser.add_argument('--batch_size', type=int, default=50, help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing (default: 1000)')

    parser.add_argument('--max_steps', type=int, default=100, help='maximum number of steps (default: 100)')
    parser.add_argument('--inner_steps', type=int, default=2, help='number of steps to train (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.95)')
    parser.add_argument('--tau', type=float, default=0.005, help='target smoothing coefficient (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, help='Temperature parameter determines the relative importance of the entropy (default: 0.2)')
    parser.add_argument('--lambda_', type=float, default=0.95, help='lambda parameter (default: 0.95)')

    parser.add_argument('--target_update_interval', type=int, default=1, help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, help='Automaically adjust α (default: False)')

    parser.add_argument('--model_name', type=str, default='a2c', help='ppo name')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')

    parser.add_argument('--entropy_beta', type=float, default=1e-3, help='entropy beta')

    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (default: 1e-4)')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='clipping parameter (default: 0.2)')
    parser.add_argument('--value_coeff', type=float, default=0.5, help='value coefficient (default: 0.5)')
    parser.add_argument('--entropy_coeff', type=float, default=0.05, help='entropy coefficient (default: 0.01)')
    parser.add_argument("--entropy_coeff_decay", type=float, default=0.99, help='Decay rate of entropy_coef')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--actor_lr', type=float, default=2e-4, help='learning rate of actor (default: 1e-4)')
    parser.add_argument('--critic_lr', type=float, default=5e-5, help='learning rate of critic (default: 1e-4)')
    parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regularization coefficient for critic')
    parser.add_argument('--Distribution', type=str, default='Beta', help='Distribution type (Beta, Gamma_mustd, Gama_mu)')

    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    return args

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
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs to train (default: 5000)')
    parser.add_argument('--max_steps', type=int, default=100, help='maximum number of steps (default: 100)')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor (default: 0.95)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='clipping parameter (default: 0.2)')
    parser.add_argument('--value_coeff', type=float, default=0.5, help='value coefficient (default: 0.5)')
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='entropy coefficient (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--num-processes', type=int, default=3, help='how many training processes to use (default: 3)')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('mps', action='store_true', default=False, help='enables MPS training on appleslicon MAC')
    parser.add_argument('--save_model', action='store_true', default=False, help='enables saving model to state_dict')
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')

    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    return args

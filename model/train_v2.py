import os
import random
import time
import logging

import torch
import torch.nn as nn
import torchopt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model.agent import PPO

import gym.vector

def train_v2(args, writer: SummaryWriter = None):
    env = gym.vector.make('chicago-v1', num_envs=1)

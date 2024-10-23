import os
import random
import time
import logging

import torch
import torch.nn as nn
import torchopt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model.agent_v2 import PolicyNetwork, MetaPPO

import gym.vector
import torch.distributed as dist

# Initialize the distributed process group
def init_distributed(rank, world_size, master_addr='', master_port=''):
    # Initialize the process group for distributed communication
    dist.init_process_group(
        backend='nccl', # Use nccl for GPU communication
        init_method= f'tcp://{master_addr}:{master_port}',
        rank=rank, # unique rank for each process
        world_size=world_size # Total number of workers (across all machines)
    )

# Training function for each worker
def train_meta_worker(global_policy_net, rank, world_size, config):
    # Initialize distributed communication
    init_distributed(rank, world_size)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % torch.cuda.device_count())
    env = gym.make('chicago-v1')

    local_policy_nets = [PolicyNetwork(env.)]

def train_meta_worker(rank, args, writer: SummaryWriter = None):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % torch.cuda.device_count())

    # Create one common environment for whole workers
    env = gym.make('chicago-v1')

    # Create policy and meta-Learning PPO
    obs_space =env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    policy_net = PolicyNetwork(obs_space, action_space).to(f'cuda:{rank % torch.cuda.device_count()}')
    meta_ppo = MetaPPO(policy_net, env, args, batch_size=args.batch_size)

    # Meta-training loop
    meta_ppo.meta_train(env, meta_steps=3, inner_steps=1, timesteps=args.epochs)

    # Save model
    torch.save(meta_ppo.policy_net.state_dict(), f'meta_ppo_worker_{rank}'))

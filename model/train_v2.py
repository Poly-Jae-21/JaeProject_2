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
def init_distributed(rank, world_size, master_addr='10.165.96.29', master_port='1000'):
    # Initialize the process group for distributed communication
    dist.init_process_group(
        backend='nccl', # Use gloo for GPU & CPU communication
        init_method= f'tcp://{master_addr}:{master_port}',
        rank=rank, # unique rank for each process
        world_size=world_size # Total number of workers (across all machines)
    )

# Training function for each worker
def train_meta_worker(meta_global_policy_net, global_global_policy_net, rank, world_size, args, env):

    # Initialize distributed communication
    init_distributed(rank, world_size)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % torch.cuda.device_count())

    # Create three local policy nets
    local_policy_nets = [PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(f'cuda:{rank % torch.cuda.device_count()}') for _ in range(args.n_local)]

    #Meta-Learning with global and local policy nets
    meta_ppo = MetaPPO(meta_global_policy_net, local_policy_nets, env, args, batch_size=args.batch_size)

    factors = ["environment", "economic", "urbanity"]

    for episode in range(args.episodes):
        for i in range(args.n_local):
            local_policy_net = meta_ppo.local_policy_nets[i]
            meta_ppo.adapt_to_task(local_policy_net, env, factors[i], inner_steps=args.inner_steps, timesteps=args.max_steps)

        meta_ppo.aggregat_local_to_meta_global()

        meta_ppo.reduce_and_broadcast(global_global_policy_net)

        print(f"Worker {rank}: Completed Episode {episode}")




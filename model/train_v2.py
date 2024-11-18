import os
import random
import time
import logging

import torch
import torch.nn as nn
import torchopt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from example2 import rewards_log
from model.agent_v2 import PolicyNetwork, MetaPPO

import gym.vector
import torch.distributed as dist

# Initialize the distributed process group
def init_distributed(rank, world_size, master_addr='fe80::5af6:2525:196e:5f43%1', master_port='9999'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # single GPU
    ###os.environ['CUDA_VISIBLE_DEVICES'] ='0,2' # triple GPUs
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group('gloo', init_method="env://?use_libuv=False", rank=rank, world_size=world_size)

# Training function for each worker
def train_meta_worker(global_policy_nets, local_policy_nets, device, rank, world_size, args, env):

    # Initialize distributed communication
    init_distributed(rank, world_size)

    # Create three local policy nets
    local_policy_nets = [PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(device) for _ in range(args.n_local)]

    #Meta-Learning with global and local policy nets
    meta_ppo = MetaPPO(meta_global_policy_nets, local_policy_nets, env, args, batch_size=args.batch_size)

    rewards_log = [[] for _ in range(world_size)] ### each meta policy net
    average_rewards_log = [] ## for average of three meta policy nets
    global_rewards_log = [] ### for global policy net

    factors = ["environment", "economic", "urbanity"]

    for episode in range(args.episodes):
        initial_position, initial_observation = env.reset()

        for j in range(world_size):
            for i in range(args.n_local):
                local_policy_net = meta_ppo.local_policy_nets[i]
                meta_ppo.local_policy_nets[i] = meta_ppo.adapt_to_task(local_policy_net, env, initial_observation, factors[i], inner_steps=args.inner_steps, timesteps=args.max_steps)

            meta_ppo.aggregat_local_to_meta_global(meta_global_policy_nets[j])
            meta_global_policy_nets[j], reward, next_initial_position, next_initial_observation = meta_ppo.meta_adapt_to_task(meta_global_policy_nets[j], env, initial_observation, factors=None, inner_steps=1, timesteps = 100)
            rewards_log[j].append(reward)
            initial_observation = next_initial_observation
            initial_position = next_initial_position

        average_reward = sum([lst[-1] for lst in rewards_log]) / 3
        average_rewards_log.append(average_reward)

        meta_ppo.reduce_and_broadcast(global_global_policy_net)

        if episode % 100 ==0 and rank == 2:
            print(f)

        print(f"Worker {rank}: Completed Episode {episode}")




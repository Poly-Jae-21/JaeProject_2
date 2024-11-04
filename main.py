import os

from model.agent_v2 import PolicyNetwork
from model.train_v2 import train_meta_worker
from model.build import argument
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.logger import configure_logger
import torch.multiprocessing as mp
import gym



def main():
    mp.set_start_method('spawn')

    # Create one common environment for whole workers
    env = gym.make('chicago-v1')

    # Global shared policy network
    global_global_policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to('cuda:0')

    meta_global_policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to('cuda:0')
    meta_global_policy_net.share_memory()

    world_size = 3
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=train_meta_worker, args=(meta_global_policy_net, global_global_policy_net, rank, world_size, env))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Save the final global policy after meta-training
    torch.save(global_global_policy_net.state_dict(), 'out/result/model', "global_global_policy_net.pt")
    #torch.save(meta_global_policy_net.state_dict(), 'out/result/model', 'meta_global_policy_net.pt')

if __name__ == "__main__":
    main()
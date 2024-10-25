import os

from model.agent_v2 import PolicyNetwork
from model.train_v2 import train_meta_worker
from model.build import argument
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.logger import configure_logger
import torch.multiprocessing as mp
import gym



def main(args):
    args = args
    path = f"runs/{args.name}"
    configure_logger(args.name, path)
    writer = SummaryWriter(os.path.join(args.path, "tb"))

    use_cuda = args.cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    mp.set_start_method('spawn', True)

    # Create one common environment for whole workers
    env = gym.make('chicago-v1')

    # Global shared policy network
    global_policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

    world_size = args.n_workers
    processes = []
    for rank in range(args.n_workers):
        p = mp.Process(target=train_meta_worker, args=(global_policy_net, rank, world_size, args, env, writer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Save the final global policy after meta-training
    torch.save(global_policy_net.state_dict(), args.path, "global_policy_net.pt")

if __name__ == "__main__":
    config = argument()
    main(config)
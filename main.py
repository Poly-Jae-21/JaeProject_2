from ppo.agent_v2 import PolicyNetwork
from ppo.train_v2 import train_meta_worker
from utils.build import argument
import torch
import torch.multiprocessing as mp
import gym



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mp.set_start_method('spawn')

    # Create one common environment for whole workers
    env = gym.make('chicago-v1')

    # Global shared policy network
    global_global_policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n).to(device)

    meta_global_policy_net = [
        PolicyNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
        for _ in range(3)
    ]
    world_size = 3
    processes = []
    args = argument()
    for rank in range(world_size):
        p = mp.Process(target=train_meta_worker, args=(meta_global_policy_net, global_global_policy_net, device, rank, world_size, args, env))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Save the final global policy after meta-training
    torch.save(global_global_policy_net.state_dict(), 'out/result/model', "global_global_policy_net.pt")
    #torch.save(meta_global_policy_net.state_dict(), 'out/result/ppo', 'meta_global_policy_net.pt')

if __name__ == "__main__":
    main()
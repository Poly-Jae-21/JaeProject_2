from model.agent_v3 import PolicyNetwork, CNNPolicyNetwork
from model.train_v3 import train
from model.build import argument
import torch
import gymnasium as gym

gym.register('UrbanEnvChicago-v1', entry_point='template.env_name.envs.multi_policies:ChicagoMultiPolicyMap')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create one common environment for whole workers
    env = gym.make('UrbanEnvChicago-v1', render_mode='human')

    # Global shared policy network
    #global_policy_net = CNNPolicyNetwork(env.observation_space.shape, env.action_space.n).to(device)
    global_policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    local_policy_nets = [
        PolicyNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
        for _ in range(3)
    ]

    world_size = 3
    args = argument()
    train_ = train()
    train_.train(global_policy_net, local_policy_nets, device, world_size, args, env)

    '''
        world_size = 3
    processes = []
    args = argument()
    for rank in range(world_size):
        p = mp.Process(target=train, args=(global_policy_net, local_policy_nets[rank], device, rank, world_size, args, env))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    '''

    '''
    if args.test:
        test(args, env, env.observation_space.shape[0], env.action_space.n)   
    '''


    print("finished")


if __name__ == "__main__":
    main()
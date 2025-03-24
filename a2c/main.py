from a2c.model import PolicyNetwork
from a2c.train import train
from utils.build import argument
import torch
import gymnasium as gym

gym.register('UrbanEnvChicago-v2', entry_point='template.env_name.envs.multi_policies:ChicagoMultiPolicyMapv2')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = argument()

    # Create one common environment for whole workers
    env = gym.make('UrbanEnvChicago-v2', render_mode='human')

    # Global shared policy network
    global_policy_net = PolicyNetwork(env.observation_space.shape[0], args.hidden_dim, env.action_space.shape[0]).to(device)
    local_policy_nets = [
        PolicyNetwork(env.observation_space.shape[0], args.hidden_dim, env.action_space.shape[0]).to(device)
        for _ in range(3)
    ]

    world_size = 3

    train_ = train()
    train_.train(global_policy_net, local_policy_nets, device, world_size, args, env)

    print("finished")


if __name__ == "__main__":
    main()
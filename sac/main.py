
from sac.model import QNetwork, GaussianPolicy
from sac.train import train
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

    global_policy_nets = []
    for _ in range(1):
        global_policy = GaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0], args.hidden_dim).to(device)
        global_critic = QNetwork(env.observation_space.shape[0], env.action_space.shape[0], args.hidden_dim).to(device)
        global_target_critic = QNetwork(env.observation_space.shape[0], env.action_space.shape[0], args.hidden_dim).to(device)
        global_policy_nets.append({
            'policy': global_policy,
            'critic': global_critic,
            'target_critic': global_target_critic,
        })

    local_policy_nets = []
    for _ in range(3):
        local_policy = GaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0], args.hidden_dim).to(device)
        local_critic = QNetwork(env.observation_space.shape[0], env.action_space.shape[0], args.hidden_dim).to(device)
        local_target_critic = QNetwork(env.observation_space.shape[0], env.action_space.shape[0], args.hidden_dim).to(device)
        local_policy_nets.append({
            'policy': local_policy,
            'critic': local_critic,
            'target_critic': local_target_critic
        })

    world_size = 3

    train_ = train()
    train_.train(global_policy_nets, local_policy_nets, device, world_size, args, env)

    print("finished")


if __name__ == "__main__":
    main()
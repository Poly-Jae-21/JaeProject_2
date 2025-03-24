
import gymnasium as gym
from ppo.train_v4 import train, test
from utils.build import argument

gym.register('UrbanEnvChicago-v1', entry_point='template.env_name.envs.multi_policies:ChicagoMultiPolicyMap')
if __name__ == '__main__':

    env = gym.make('UrbanEnvChicago-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    args = argument()
    if args.mode == 'train':
        train(args, env, state_dim, action_dim)

    else:
        raise ValueError("Invalid mode. Use 'train'")

    if args.test:
        test(args, env, state_dim, action_dim)

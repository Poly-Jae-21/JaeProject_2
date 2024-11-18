import gymnasium as gym

gym.register('UrbanEnvChicago-v1', entry_point='template.env_name.envs.multi_policies:ChicagoMultiPolicyMap')
env = gym.make("UrbanEnvChicago-v1")



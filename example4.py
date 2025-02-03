import gymnasium as gym
env = gym.make('CartPole-v0')

action = env.action_space.sample()
print(action)

state, _ = env.reset()

next_state, reward, done, terminate, info = env.step(action)
print(next_state, reward)
next_state, reward, done, terminate, info = env.step(action)
print(next_state, reward)

import torch
from gymnasium.spaces import Box, Discrete

def discount_cumsum(x, discount):
    result = torch.zeros_like(x)
    running_add = 0
    for t in reversed(range(len(x))):
        running_add = x[t] + discount * running_add
        result[t] = running_add
    return result

class RolloutBuffer:
    def __init__(self, size, obs_space, action_space, gamma=0.99, gae_lambda=0.95, device=None):
        self.device = device
        obs_dim = obs_space.shape[0]

        if isinstance(action_space, Box):
            action_dim = action_space.shape[0]
            action_dim = (size, action_dim)
        elif isinstance(action_space, Discrete):
            action_dim = action_space.n
            action_dim = action_dim

        self.data = {
            "obs": torch.zeros((size, obs_dim)).to(device),
            "reward": torch.zeros((size)).to(device),
            "value": torch.zeros((size)).to(device),
            "action_log": torch.zeros((size)).to(device),
            "return": torch.zeros((size)).to(device),
            "advantage": torch.zeros((size)).to(device),
            "termination": torch.zeros((size)).to(device),
        }
        self.episodes = []
        self.gamma, self.gae_lambda = gamma, gae_lambda

        self.max_size, self.ptr, self.start = size, 0, 0

    def store(self, obs, action_log_prob, reward, value):
        assert self.ptr < self.max_size
        self.data["value"][self.ptr] = value
        self.data["action_log"][self.ptr] = action_log_prob
        self.data["obs"][self.ptr] = torch.tensor(obs).to(self.device)
        self.data["reward"][self.ptr] = torch.tensor(reward).to(self.device)
        self.ptr += 1

    def reset(self):
        for key, val in self.data.items():
            self.data[key] = torch.zeros_like(val)
        self.ptr, self.start = 0, 0

    def calculate_discounted_rewards(self, last_value=None, gamma=0.99, lam=0.95):
        if last_value is None:
            last_value = torch.tensor([0.0]).to(self.device)

        rews = torch.cat((self.data["reward"][self.start:self.ptr], last_value))

        vals = torch.cat((self.data["value"][self.start:self.ptr], last_value))

        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
        self.data["advantages"][self.start:self.ptr] = discount_cumsum(deltas, gamma * lam)
        self.data["return"][self.start:self.ptr] = discount_cumsum(rews, gamma)[:-1]

        self.start = self.ptr

    def get(self) -> dict:
        assert self.ptr == self.max_size
        return self.data



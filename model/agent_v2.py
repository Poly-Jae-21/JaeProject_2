import torch
import torch.nn as nn
from model.core_v2 import PolicyNetwork
import torch.optim as optim
import numpy as np
class PPO(nn.Module):
    def __init__(self, config, policy_net):
        super(PPO, self).__init__()
        self.policy_net = policy_net
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr) # lr = 1e-4
        self.gamma = config.gamma # 0.99
        self.clip_epsilon = config.clip_epsilon # = 0.2
        self.value_coeff = config.value_coeff # = 0.5
        self.entropy_coeff = config.entropy_coeff #0.01
        self.max_steps = config.max_steps
        self.batch_size = config.batch_size

    def select_action(self, state):
        state = torch.tensor(state)

    def update(self, trajectories, old_log_probs, advantages, returns):
        states = torch.tensor(np.vstack([t[0] for t in trajectories]), dtype=torch.float32)
        actions = torch.tensor(np.vstack([t[1] for t in trajectories]), dtype=torch.float32)
        log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        dataset_size = states.size(0)

        for _ in range(self.max_steps):
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_states = states[start:end]
                batch_actions = actions[start:end]
                batch_log_probs = log_probs[start:end]
                batch_advantages = advantages[start:end]
                batch_returns = returns[start:end]

                new_action_mean, new_value = self.policy_net(batch_states)
                action_dist = torch.distributions.Normal(new_action_mean, 1.0)
                new_log_probs = action_dist.log_prob(actions).sum(dim=-1)

                ratio = torch.exp(new_log_probs - batch_log_probs)

                # Policy loss
                surrogate_1 = ratio * batch_advantages
                surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                # Value loss
                value_loss = self.value_coeff * ((batch_returns - new_value).pow(2)).mean()

                # Entropy loss for exploration
                entropy_loss = -self.entropy_coeff * action_dist.entropy().mean()

                # Total loss
                loss = policy_loss + value_loss + entropy_loss

                # Gradient descent step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

import torch
import torch.nn as nn
import numpy as np
# Define the Actor network to predict both means and log_std for Diagonal Gaussian
class ProbabilisticActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=3):
        super(ProbabilisticActor, self).__init__()
        # Network for predicting means of actions
        self.mean_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Network for predicting log standard deviation (log_std)
        self.log_std_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.value_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        means = self.mean_layer(state)
        log_stds = self.log_std_layer(state)
        stds = torch.exp(log_stds)  # Ensure standard deviations are positive
        value = self.value_layer(state)
        return means, stds, value

# Function to convert predicted means to actual range
def convert_to_actual_range(means):
    x = torch.tanh(means[:, 0]) * 50  # Scale x and y to range -50 to 50
    y = torch.tanh(means[:, 1]) * 50
    capacity = torch.tanh(means[:, 2]) * 5 + 7  # Scale capacity to range 2 to 12
    return torch.stack([x, y, capacity], dim=1)

# Example usage
input_dim = 5  # Example input dimension
hidden_dim = 64
output_dim = 3  # Predict x, y, capacity

# Instantiate the Probabilistic Actor ppo
actor = ProbabilisticActor(input_dim, hidden_dim, output_dim)

# Example state input
state = torch.randn(1, input_dim)  # Batch of 4 states

# Get predicted means and standard deviations
means, stds, value = actor(state)

# Create the normal distribution with predicted means and std deviations
dist = torch.distributions.Normal(means, stds)

# Sample actions
sampled_actions = dist.sample()

# Convert sampled actions to actual range
mapped_actions = convert_to_actual_range(sampled_actions)

# Compute log probabilities of the sampled actions
log_probs = dist.log_prob(sampled_actions)  # Summing over the action dimension

print(sampled_actions.numpy().ravel())
print(value.cpu().detach())
print(value.cpu().detach().item())


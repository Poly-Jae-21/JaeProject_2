import torch
from torch.autograd import Variable
import numpy as np
import torch.distributions as distributions

class Action(object):
    def __init__(self, env, actor, critic, observation, state):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.observation = observation
        self.state = state

    def select_action(self, state):
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        elif torch.backends.mps.is_available():
            device = torch.device("mps", 0)
        else:
            device = torch.device("cpu", 0)

        state = Variable(torch.Tensor(state))
        log_probs = self.actor(state).to(device)
        value = self.critic(state).to(device)

        dist = distributions.Normal(log_probs, scale=0.1)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = torch.clamp(action, np.min(self.env.action_space.low[0]), np.max(self.env.action_space.high[0]))
        action = action.cpu().data.numpy()

        return action, log_prob, value

    def action_converter(self, action, position_record, action_record):
        boundary_x, boundary_y = self.env.boundary_x, self.env.boundary_y
        raw_action_x = (action[0][0] + 1) * 100 / 2 - 50
        raw_action_y = (action[0][1] + 1) * 100 / 2 - 50
        raw_action = np.array([-raw_action_y, raw_action_x])
        real_action_position = (position_record[-1] + raw_action).astype('int32')

        if (real_action_position[0] >= boundary_x) or (real_action_position[0] <= 0) or (real_action_position[1] >= boundary_y) or (real_action_position[1] <= 0):
            reward = -10
            real_action_position = real_action_position

        raw_capacity = (action[0][2] + 1) * (30000 - 0) / 2 + 0 # (value + 1) * ( max capacity in action - min capacity in action) / 2  + min capacity in action

        raw_capacity, real_action_position = np.matrix(raw_capacity.astype('int32')), np.matrix(real_action_position).astype('int32')
        position_record = np.append(position_record, real_action_position, axis=0)
        Action_record = np.concatenate((position_record[-1], raw_capacity), axis=1)
        action_record_ = np.append(action_record, Action_record, axis=0)

        position_record, action_record_ = np.squeeze(np.asarray(position_record)), np.squeeze(np.asarray(action_record_))

        return position_record, action_record_


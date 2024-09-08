import os
import logging

import torch
import numpy as np
import torch.nn as nn
from moviepy.editor import ImageSequenceClip

class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def mlp(dims, stds, hidden_activation=nn.Tanh, output_activation=nn.Identity):

    assert len(dims) == len(stds) + 1

    layers = []
    for i in range(len(dims) - 2):
        layers.extend([layer_init(nn.Linear(dims[i], dims[i+1]), stds[i]), hidden_activation()])
    layers.extend([layer_init(nn.Linear(dims[-2], dims[-1]), stds[-1]), output_activation()])
    return nn.Sequential(*layers)

def make_gif(agent, env, episode, config):
    obs, _ = env.reset()
    terminated, truncated = False, False

    prev_action = (torch.tensor(env.action_space.sample()).to(config["device"]).view(-1))
    prev_rew = torch.tensor(0).to(config["device"]).view(1,1)

    steps = []
    rewards = []

    while not (terminated or truncated):
        steps.append(env.render())

        obs = (torch.tensor(obs).float().to(config["device"]).float().unsqueeze(0))
        act, _ = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(act.cpu().numpy())

        obs = next_obs

        prev_action = act.detach()
        prev_rew = torch.tensor(reward).to(config["device"]).view(1,1)

        rewards.append(reward)

    clip = ImageSequenceClip(steps, fps=30)
    save_dir = os.path.join(config["path"], "gifs")
    gif_name = f"{save_dir}/cityplanning_epoch_{str(episode)}.gif"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    clip.write_gif(gif_name, fps=30, verbose=False, logger=None)

    logging.info(f"Generating GIF {gif_name}")

def save_state(state_dict, path, epoch=None, job_id=None):
    model_file = (os.path.join(path, f"e{epoch}_state") if epoch is not None else path)

    model_file_tmp = model_file if job_id is None else model_file + f"_{job_id}"
    torch.save(state_dict, model_file_tmp)
    if model_file_tmp != model_file:
        os.rename(model_file_tmp, model_file)


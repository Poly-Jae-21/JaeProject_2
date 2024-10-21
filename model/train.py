import os
import random
import time
import logging

import torch
import torch.nn as nn
import torchopt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model.agent import PPO

import gym.vector

from model.build import argument
from env.MultiCity.Chicago.chicago import ChicagoEnv

def train(config, writer: SummaryWriter = None):

    env = gym.vector.make("Chicago-v0", num_envs=3)
    test_env = gym.vector.make("Chicago-v0", num_envs=3)

    # Set the meta-parameter
    gamma = nn.Parameter(-torch.log(( 1 / torch.tensor(config["gamma"])) - 1), requires_grad=True)

    for meta_epoch in range(config["meta_epochs"]):

        agent = PPO(obs_space=env.observation_space, action_space=env.action_space, value_coeff=config["ppo"]["value_coeff"], writer=writer, ac_kwargs=config["actor_critic"], max_episode_steps=config["max_episode_steps"], device=config["device"])
        meta_optim = torchopt.SGD([gamma], lr=config["outer_lr"])
        inner_optim = torchopt.MetaSGD(agent.actor_critic, lr=config["inner_lr"])

        global_step = 0

        for _ in range(config["inner_steps"]):
            data = agent.collect_rollouts(env, torch.sigmoid(gamma))
            loss = agent.optimize(data, config["update_epochs"], global_step)

            inner_optim.step(loss)
        # Outer-loop
        data = agent.collect_rollouts(env, torch.sigmoid(gamma))
        meta_loss = agent.optimize(data)

        meta_optim.zero_grad()
        meta_loss.backward()

        writer.add_scalar("PPO/grad_gamma", gamma.grad, agent.global_step)
        meta_optim.step()

        # Detach the graph
        torchopt.stop_gradient(agent.ac)
        torchopt.stop_gradient(inner_optim)





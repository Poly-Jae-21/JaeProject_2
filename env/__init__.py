import logging
import gymnasium

from gymnasium.envs.registration import register, registry

logger = logging.getLogger(__name__)

version = 3.0

def register_envs():

    # Register new versions
    register(id=f'Chicago-v{version}',
             entry_point='env.Chicago.chicago:ChicagoEnv',
             max_episode_steps=100,
             reward_threshold=10000)

register_envs()
import logging
import gymnasium
from gym import envs
from gym.envs.registration import register, registry

logger = logging.getLogger(__name__)

version = 1
def unregister_env(env_id):
    if env_id in registry.env_specs:
        del registry.env_specs[env_id]
def register_envs():

    # Unregister old versions if they exist
    existing_version = version - 1
    if existing_version >= 1:
        unregister_env(f'Chicago-v{existing_version}')
        unregister_env(f'LA-v{existing_version}')
        unregister_env(f'Huston-v{existing_version}')

    # Register new versions
    register(id=f'Chicago-v{version}',
             entry_point='env.Chicago.chicago:ChicagoEnv',
             max_episode_steps=100,
             reward_threshold=1000.0)

    register(id=f'LA-v{version}',
             entry_point='env.LA.la:LAEnv',
             max_episode_steps=100,
             reward_threshold=1000.0)

    register(id=f'Huston-v{version}',
             entry_point='env.Huston.huston:HustonEnv',
             max_episode_steps=100,
             reward_threshold=1000.0)

register_envs()
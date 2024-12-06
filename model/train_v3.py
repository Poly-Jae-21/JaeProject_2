import os
import torch
from model.agent_v3 import MetaPPO

import torch.distributed as dist

class train():
    def __init__(self):
        self.initial_position, self.initial_observation = None, None
        self.terminate = False

    def train(self, global_policy_net, local_policy_net, device, world_size, args, env):
        reward_log = []
        average_rewards_log = []

        factor = ["environment", "economic", "urbanity"]
        meta_ppo = MetaPPO(device, env, args, batch_size=args.batch_size)
        for episode in range(args.max_episodes):
            torch.cuda.empty_cache()
            print("episode:" + str(episode+1) + "in train")
            self.initial_observation, _ = env.reset(seed=None, options=episode)
            for rank in range(3):
                if rank == 0:
                    local_policy_net[rank] = meta_ppo.adapt_to_task(args, local_policy_net[rank], env, self.initial_observation, factor[rank])
                    print("0")
                elif rank == 1:
                    local_policy_net[rank] = meta_ppo.adapt_to_task(args, local_policy_net[rank], env, self.initial_observation, factor[rank])
                    print("1")

                if rank == world_size -1:
                    local_policy_net[rank] = meta_ppo.adapt_to_task(args, local_policy_net[rank], env, self.initial_observation, factor[rank])
                    print("2")

                    global_policy_net = meta_ppo.aggregate_local_to_global(episode, local_policy_net, global_policy_net)
                    print("3")
                    total_global_reward, global_average_reward, global_infos = meta_ppo.global_evaluate(global_policy_net, env, self.initial_observation, factor=None, timesteps=args.max_timesteps)
                    reward_log.append(total_global_reward)
                    average_rewards_log.append(global_average_reward)
                    print("Train Episode {} | each reward: {}, meta total rewards: {}, meta average reward: {}".format(episode, global_infos, reward_log[-1], average_rewards_log[-1]))

                    if (episode+1) % args.print_interval == 0:
                        meta_ppo.plot(reward_log, average_rewards_log, episode)
                        print("x100")

            if self.terminate:
                torch.save(global_policy_net.state_dict(), args.ckpt_folder + '/PPO_{}_result_episode{}.pth'.format(args.env_name, episode))
                print("Save a global policy network")
                break



'''
def test(args, saved_env, observation_space, action_space):
    ckpt = args.ckpt_folder+'/PPO_discrete_'+args.env_name+'.pth'
    print('Loading model...')

    test_ppo = test_MetaPPO(state_dim, action_dim, args, restore=True, ckpt=ckpt)

    episode_reward = []
    avg_reward = []

    for i_episode in range(1, args.test_episodes+1):
        initial_position, initial_observation = saved_env.reset()
        total_rewards, step_done = test_ppo.test_rollout()
        episode_reward.append(total_rewards)
        avg_reward.append(total_rewards/(step_done+1))
        print('Test Episode {} | total rewards: {}, average reward: {}'.format((args.test_episodes + 1), episode_reward[-1], avg_reward[-1]))

    test_ppo.plot(episode_reward, avg_reward)
    saved_env.render(mode='human')
    save_result_gif()
'''








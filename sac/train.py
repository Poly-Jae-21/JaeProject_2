import torch
from sac.model import MetaSAC
import numpy as np
class train():
    def __init__(self):
        self.initial_position, self.initial_observation = None, None
        self.terminate = False

    def train(self, global_policy_net, local_policy_net, device, world_size, args, env):
        reward_log = []
        average_rewards_log = []

        reward_environment_log = []
        average_rewards_environment_log = []

        reward_economy_log = []
        average_rewards_economy_log = []

        reward_urbanity_log = []
        average_rewards_urbanity_log = []

        factor = ["environment", "economic", "urbanity"]
        meta_sac_environment = MetaSAC(device, env, args, local_policy_net[0], args.batch_size)
        meta_sac_economy = MetaSAC(device, env, args, local_policy_net[1], args.batch_size)
        meta_sac_urbanity = MetaSAC(device, env, args, local_policy_net[2], args.batch_size)

        meta_sac_meta = MetaSAC(device, env, args, global_policy_net, args.batch_size)

        for episode in range(args.max_episodes):
            torch.cuda.empty_cache()
            print("episode:" + str(episode+1) + "in train")
            self.initial_observation, _ = env.reset(seed=None, options=episode)
            for rank in range(world_size):
                if rank == 0:
                    local_policy_net[rank], loss_log1 = meta_sac_environment.adapt_to_task(args, local_policy_net[rank], env, self.initial_observation, factor[rank], episode)
                    total_local_reward, local_average_reward, _ = meta_sac_environment.global_evaluate(local_policy_net[rank], env, self.initial_observation, env_update = True, factor=factor[rank], timesteps=args.max_timesteps)
                    reward_environment_log.append(total_local_reward)
                    average_rewards_environment_log.append(local_average_reward)
                    if (episode + 1) % args.print_interval == 0:
                        env.plot(args, rank, meta=False)
                        meta_sac_environment.plot(reward_environment_log, average_rewards_environment_log, loss_log1, episode, factor[rank])

                elif rank == 1:
                    local_policy_net[rank], loss_log2 = meta_sac_economy.adapt_to_task(args, local_policy_net[rank], env, self.initial_observation, factor[rank], episode)
                    total_local_reward, local_average_reward, _ = meta_sac_economy.global_evaluate(local_policy_net[rank], env, self.initial_observation, env_update = True, factor=factor[rank], timesteps=args.max_timesteps)
                    reward_economy_log.append(total_local_reward)
                    average_rewards_economy_log.append(local_average_reward)
                    if (episode + 1) % args.print_interval == 0:
                        env.plot(args, rank, meta=False)
                        meta_sac_economy.plot(reward_economy_log, average_rewards_economy_log, loss_log2, episode, factor[rank])

                elif rank == 2:
                    local_policy_net[rank], loss_log3 = meta_sac_urbanity.adapt_to_task(args, local_policy_net[rank], env, self.initial_observation, factor[rank], episode)
                    total_local_reward, local_average_reward, _ = meta_sac_urbanity.global_evaluate(local_policy_net[rank], env, self.initial_observation, env_update = True, factor=factor[rank], timesteps=args.max_timesteps)
                    reward_urbanity_log.append(total_local_reward)
                    average_rewards_urbanity_log.append(local_average_reward)
                    if (episode + 1) % args.print_interval == 0:
                        env.plot(args, rank, meta=False)
                        meta_sac_urbanity.plot(reward_urbanity_log, average_rewards_urbanity_log, loss_log3, episode, factor[rank])

                        average_average_rewards = (np.array(reward_environment_log) + np.array(reward_economy_log) + np.array(reward_urbanity_log)) / 3
                        total_average_rewards = np.array(reward_environment_log) + np.array(reward_economy_log) + np.array(reward_urbanity_log)

                        meta_sac_meta.plot(average_average_rewards, total_average_rewards, _, episode, "average")

                    global_policy_net = meta_sac_meta.aggregate_local_to_global(episode, local_policy_net, global_policy_net)

                    total_global_reward, global_average_reward, global_infos = meta_sac_meta.global_evaluate(global_policy_net, env, self.initial_observation, env_update = True, factor=None, timesteps=args.max_timesteps)
                    reward_log.append(total_global_reward)
                    average_rewards_log.append(global_average_reward)
                    print("Train Episode {} | each reward: {}, meta total rewards: {}, meta average reward: {}".format(episode+1, global_infos[-1], reward_log[-1], average_rewards_log[-1]))
                    loss = None
                    if (episode+1) % args.print_interval == 0:
                        meta_sac_meta.plot(reward_log, average_rewards_log, loss, episode, factor="meta")
                        env.plot(args, rank, meta=True)
                        print("episode done")
                    if (episode+1) % args.save_interval == 0:
                        torch.save(global_policy_net['policy'].state_dict(),f"{args.ckpt_folder}/sac_{args.env_name}_result_episode{episode}.pth")
            if self.terminate:
                torch.save(global_policy_net['policy'].state_dict(),f"{args.ckpt_folder}/sac_{args.env_name}_result_episode{episode}.pth")
                print("Save a global policy network")
                break









import os
import torch
from ppo.agent_v3 import MetaPPO, PPO, adapt_to_task, evaluate, result_plot, loss_plot, test

import torch.distributed as dist

class train():
    def __init__(self):
        self.initial_position, self.initial_observation = None, None
        self.terminate = False

        self.env_reward = 0
        self.eco_reward = 0
        self.urb_reward = 0

    def train(self, system_policy_net, global_policy_net, local_policy_net, device, world_size, args, env):
        reward_log = []
        average_rewards_log = []

        reward_environment_log = []
        average_rewards_environment_log = []

        reward_economy_log = []
        average_rewards_economy_log = []

        reward_urbanity_log = []
        average_rewards_urbanity_log = []

        system_reward_log = []
        system_average_rewards_log = []

        factor = ["environment", "economic", "urbanity", "overall", "system"]

        local_policy_nets = []

        ppo_envivronment = PPO(args, device, local_policy_net[0])
        ppo_economy = PPO(args, device, local_policy_net[1])
        ppo_urbanity = PPO(args, device, local_policy_net[2])
        overall_ppo = PPO(args, device, global_policy_net)
        system_ppo = PPO(args, device, system_policy_net)

        for episode in range(args.max_episodes):

            if episode == args.max_episodes - 1:
                self.terminate = True
            torch.cuda.empty_cache()
            print("episode.:" + "" + str(episode+1) + "in train")
            self.initial_state, _ = env.reset(seed=None, options=episode)

            for rank in range(world_size):
                if rank == 0:
                    local_policy_net[rank]= adapt_to_task(self.initial_state, ppo_envivronment, factor[rank], env, episode, device)
                    self.env_reward, total_env_reward, average_env_reward, _ = evaluate(self.initial_state, ppo_envivronment, factor[rank], env, episode, args, average_rewards_environment_log)
                    reward_environment_log.append(total_env_reward)
                    average_rewards_environment_log.append(average_env_reward)
                    local_policy_nets.append(local_policy_net[rank])
                    if (episode+1) % args.print_interval == 0:
                        env.plot(args, rank, meta=False)
                        result_plot(args, reward_environment_log, average_rewards_environment_log, episode, factor[rank])
                        loss_plot(args, ppo_envivronment.loss_dict, episode, factor[rank])
                elif rank == 1:
                    local_policy_net[rank] = adapt_to_task(self.initial_state, ppo_economy, factor[rank], env, episode, device)
                    self.eco_reward, total_eco_reward, average_eco_reward, _ = evaluate(self.initial_state, ppo_economy, factor[rank], env, episode, args, average_rewards_economy_log)
                    reward_economy_log.append(total_eco_reward)
                    average_rewards_economy_log.append(average_eco_reward)
                    local_policy_nets.append(local_policy_net[rank])
                    if (episode+1) % args.print_interval == 0:
                        env.plot(args, rank, meta=False)
                        result_plot(args, reward_economy_log, average_rewards_economy_log, episode, factor[rank])
                        loss_plot(args, ppo_economy.loss_dict, episode, factor[rank])
                elif rank == 2:
                    local_policy_net[rank] = adapt_to_task(self.initial_state, ppo_urbanity, factor[rank], env, episode, device)
                    self.urb_reward, total_urb_reward, average_urb_reward, _ = evaluate(self.initial_state, ppo_urbanity, factor[rank], env, episode, args, average_rewards_urbanity_log)
                    reward_urbanity_log.append(total_urb_reward)
                    average_rewards_urbanity_log.append(average_urb_reward)
                    local_policy_nets.append(local_policy_net[rank])
                    if (episode+1) % args.print_interval == 0:
                        env.plot(args, rank, meta=False)
                        result_plot(args, reward_urbanity_log, average_rewards_urbanity_log, episode, factor[rank])
                        loss_plot(args, ppo_urbanity.loss_dict, episode, factor[rank])
                elif rank == 3:
                    global_policy_net = adapt_to_task(self.initial_state, overall_ppo, factor[rank], env, episode, device)
                    _, total_overall_log, average_overall_reward, infos = evaluate(self.initial_state, overall_ppo, factor[rank], env, episode, args, average_rewards_log, env_update=True)
                    reward_log.append(total_overall_log)
                    average_rewards_log.append(average_overall_reward)

                    print("Train Episode {} | last Env reward: {}, last Eco reward: {}, last Urb reward: {}, overall last Env rewards: {}, overall last Eco rewards: {}, overall last Urb rewards: {}, overall last rewards: {}, overall total rewards: {}, overall average reward: {}".format(episode + 1, self.env_reward, self.eco_reward, self.urb_reward, infos[0], infos[1],infos[2], infos[3], reward_log[-1], average_rewards_log[-1]))

                    if (episode+1) % args.print_interval == 0:
                        env.plot(args, rank, meta=True)
                        result_plot(args, reward_log, average_rewards_log, episode, factor[rank])
                        loss_plot(args, overall_ppo.loss_dict, episode, factor[rank])

                elif rank == 4:
                    system_policy_net, total_system_log, average_system_reward, infos = test(self.initial_state, system_ppo, global_policy_net, local_policy_nets, factor[rank], env, episode, args, system_average_rewards_log, system=True)

                    local_policy_nets = []

                    system_reward_log.append(total_system_log)
                    system_average_rewards_log.append(average_system_reward)

                    if (episode+1) % args.print_interval == 0:
                        env.plot(args, rank, meta=False)
                        result_plot(args, system_reward_log, average_rewards_log, episode, factor[rank])

                    if (episode+1) % args.save_interval == 0:
                        torch.save(local_policy_net[0],args.ckpt_folder + '/Env_PPO_{}_result_episode{}.pth'.format(args.env_name, episode))
                        torch.save(local_policy_net[1],args.ckpt_folder + '/Eco_PPO_{}_result_episode{}.pth'.format(args.env_name, episode))
                        torch.save(local_policy_net[2],args.ckpt_folder + '/Urb_PPO_{}_result_episode{}.pth'.format(args.env_name, episode))
                        torch.save(global_policy_net,args.ckpt_folder + '/overall_PPO_{}_result_episode{}.pth'.format(args.env_name,episode))
                        torch.save(system_policy_net,args.ckpt_folder + '/system_policy_{}_result_episode{}.pth'.format(args.env_name,episode))

            if self.terminate:
                torch.save(local_policy_net[0],args.ckpt_folder + '/Env_PPO_{}_result_episode{}.pth'.format(args.env_name, episode))
                torch.save(local_policy_net[1],args.ckpt_folder + '/Eco_PPO_{}_result_episode{}.pth'.format(args.env_name, episode))
                torch.save(local_policy_net[2],args.ckpt_folder + '/Urb_PPO_{}_result_episode{}.pth'.format(args.env_name, episode))
                torch.save(global_policy_net, args.ckpt_folder + '/overall_PPO_{}_result_episode{}.pth'.format(args.env_name, episode))
                torch.save(system_policy_net,args.ckpt_folder + '/system_policy_{}_result_episode{}.pth'.format(args.env_name, episode))
                break

'''
def test(args, saved_env, observation_space, action_space):
    ckpt = args.ckpt_folder+'/PPO_discrete_'+args.env_name+'.pth'
    print('Loading ppo...')

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








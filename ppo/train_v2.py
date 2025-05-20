import os
import torch
from ppo.agent_v2 import PPO_agent, adapt_to_task, evaluate, result_plot, loss_plot, test


class train():
    def __init__(self):
        self.initial_position, self.initial_observation = None, None
        self.terminate = False

        self.env_reward = 0
        self.eco_reward = 0
        self.urb_reward = 0

    def train(self, device, world_size, args, env):
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

        ppo_envivronment = PPO_agent(args, device, env)
        ppo_economy = PPO_agent(args, device, env)
        ppo_urbanity = PPO_agent(args, device, env)
        overall_ppo = PPO_agent(args, device, env)
        system_ppo = PPO_agent(args, device, env)

        for episode in range(args.max_episodes):

            if episode == args.max_episodes - 1:
                self.terminate = True
            torch.cuda.empty_cache()
            print("episode.:" + "" + str(episode+1) + "in train")
            self.initial_state, info = env.reset(seed=None, options=episode)

            for rank in range(world_size):
                if rank == 0:
                    env_policy_net = adapt_to_task(self.initial_state, ppo_envivronment, factor[rank], env, device, episode)
                    self.env_reward, total_env_reward, average_env_reward, _ = evaluate(self.initial_state, ppo_envivronment, factor[rank], env, episode, args, average_rewards_environment_log)
                    reward_environment_log.append(total_env_reward)
                    average_rewards_environment_log.append(average_env_reward)
                    local_policy_nets.append(env_policy_net)
                    if (episode+1) % args.print_interval == 0:
                        env.plot(args, rank, meta=False)
                        result_plot(args, reward_environment_log, average_rewards_environment_log, episode, factor[rank])
                        loss_plot(args, ppo_envivronment.loss_dict, episode, factor[rank])
                elif rank == 1:
                    eco_policy_net = adapt_to_task(self.initial_state, ppo_economy, factor[rank], env, device, episode)
                    self.eco_reward, total_eco_reward, average_eco_reward, _ = evaluate(self.initial_state, ppo_economy, factor[rank], env, episode, args, average_rewards_economy_log)
                    reward_economy_log.append(total_eco_reward)
                    average_rewards_economy_log.append(average_eco_reward)
                    local_policy_nets.append(eco_policy_net)
                    if (episode+1) % args.print_interval == 0:
                        env.plot(args, rank, meta=False)
                        result_plot(args, reward_economy_log, average_rewards_economy_log, episode, factor[rank])
                        loss_plot(args, ppo_economy.loss_dict, episode, factor[rank])
                elif rank == 2:
                    urb_policy_net = adapt_to_task(self.initial_state, ppo_urbanity, factor[rank], env, device, episode)
                    self.urb_reward, total_urb_reward, average_urb_reward, _ = evaluate(self.initial_state, ppo_urbanity, factor[rank], env, episode, args, average_rewards_urbanity_log)
                    reward_urbanity_log.append(total_urb_reward)
                    average_rewards_urbanity_log.append(average_urb_reward)
                    local_policy_nets.append(urb_policy_net)
                    if (episode+1) % args.print_interval == 0:
                        env.plot(args, rank, meta=False)
                        result_plot(args, reward_urbanity_log, average_rewards_urbanity_log, episode, factor[rank])
                        loss_plot(args, ppo_urbanity.loss_dict, episode, factor[rank])
                elif rank == 3:
                    overall_policy_net = adapt_to_task(self.initial_state, overall_ppo, factor[rank], env, device, episode)
                    _, total_overall_log, average_overall_reward, infos = evaluate(self.initial_state, overall_ppo, factor[rank], env, episode, args, average_rewards_log, env_update=True)
                    reward_log.append(total_overall_log)
                    average_rewards_log.append(average_overall_reward)

                    print("Train Episode {} in community {} | last Env reward: {}, last Eco reward: {}, last Urb reward: {}, overall last Env rewards: {}, overall last Eco rewards: {}, overall last Urb rewards: {}, overall last rewards: {}, overall total rewards: {}, overall average reward: {}".format(episode + 1, info["community"], self.env_reward, self.eco_reward, self.urb_reward, infos[0], infos[1],infos[2], infos[3], reward_log[-1], average_rewards_log[-1]))

                    if (episode+1) % args.print_interval == 0:
                        env.plot(args, rank, meta=True)
                        result_plot(args, reward_log, average_rewards_log, episode, factor[rank])
                        loss_plot(args, overall_ppo.loss_dict, episode, factor[rank])

                elif rank == 4:
                    system_policy_net, total_system_log, average_system_reward, infos = test(self.initial_state, system_ppo, overall_policy_net, local_policy_nets, factor[rank], env, episode, args, system_average_rewards_log, system=True)

                    local_policy_nets = []

                    system_reward_log.append(total_system_log)
                    system_average_rewards_log.append(average_system_reward)

                    if (episode+1) % args.print_interval == 0:
                        env.plot(args, rank, meta=False)
                        result_plot(args, system_reward_log, average_rewards_log, episode, factor[rank])

                    if (episode+1) % args.save_interval == 0:
                        ppo_envivronment.save(episode)
                        ppo_economy.save(episode)
                        ppo_urbanity.save(episode)
                        overall_ppo.save(episode)
                        system_ppo.save(episode)

            if self.terminate:
                ppo_envivronment.save(episode)
                ppo_economy.save(episode)
                ppo_urbanity.save(episode)
                overall_ppo.save(episode)
                system_ppo.save(episode)
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








import torch
import numpy as np
import torch.nn as nn
from ppo.buffer import RolloutBuffer
from ppo.core import ActorCritic
import utils.misc as misc
class PPO(nn.Module):
    def __init__(self, obs_space, action_space, ac_kwargs, writer, device, seed=42, lr=3e-4, clip_ratio=0.2, value_coeff=0.5, entropy_coeff=0.01, max_grad_norm=0.5, **kwargs):
        super().__init__()
        torch.manual_seed(seed)

        self.writer = writer
        self.clip_ratio = clip_ratio
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm

        self.actor_critic = ActorCritic(obs_space, action_space, device=device, **ac_kwargs)

        self.value_loss = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)

        self.buffer = RolloutBuffer(self.max_episode_steps, obs_space, device=device)

    def act(self, obs):
        return self.actor_critic.step(obs)

    def collect_rollouts(self, env, gamma):
        termination, truncated = False, False
        self.buffer.reset()

        obs, _ = env.reset()
        for _ in range(self.max_episode_steps):
            action, action_log_prob, value = self.actor_critic.step(obs)
            next_obs, rew, termination, truncated, info = env.step(action.item())

            obs = next_obs

            self.buffer.store(obs, action_log_prob, rew, value)

            if termination or truncated:
                last_value = None
                if truncated:
                    with torch.no_grad():
                        _, _, last_value = self.actor_critic.step(obs)
                        self.buffer.calculate_discounted_rewards(last_value, gamma=gamma)
                self.writer.add_scalar('env/ep_return', info["episode"]['r'], self.global_step)
                self.writer.add_scalar('env/ep_length', info["episode"]['l'], self.global_step)
                self.writer.add_scalar('PPO/gamma', gamma.cpu(), self.global_step)

                obs, _ = env.reset()

        last_value = None
        if truncated:
            with torch.no_grad():
                _, _, last_value = self.actor_critic.step(obs)

        self.buffer.calculate_discounted_rewards(last_value, gamma=gamma)
        return self.buffer.get()

    def save_weights(self, path, epoch):
        misc.save_state(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
            epoch,
        )

    def optimize(self, batch, update_epochs, global_step, target_kl = None):
        info = dict(kl=torch.tensor(0), ent=torch.tensor(0), cf=torch.tensor(0))

        for _ in range(update_epochs):
            pi_loss, pi, info = self._compute_policy_loss(batch)
            v_loss, v = self._compute_value_loss(batch)

            loss = (
                pi_loss
                - self.entropy_coeff * info["ent"]
                + self.value_coeff * v_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if target_kl and info["kl"] > 1.5 * target_kl:
                break

        y_pred, y_true = (
            batch["value"].reshape(-1).cpu().numpy(),
            batch["return"].reshape(-1).cpu().numpy(),
        )
        print(y_pred[:5], y_true[:5])
        var_y = np.var(y_true)
        exp_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        self.writer.add_histogram("PPO/value_hist", batch["value"], global_step)
        self.writer.add_histogram("PPO/policy_hist", pi.sample(), global_step)
        self.writer.add_histogram("PPO/explained_variance", exp_var, global_step)
        self.writer.add_scalar("PPO/entropy", info["ent"].item(), global_step)
        self.writer.add_scalar("PPO/approx_kl", info["kl"].item(), global_step)
        self.writer.add_scalar("PPO/clip_frac", info["cf"].item(), global_step)
        self.writer.add_scalar("PPO/value_loss", v_loss.item(), global_step)
        self.writer.add_scalar("PPO/policy_loss", pi_loss.item(), global_step)


    def _compute_policy_loss(self, batch):
        b_obs, b_act, b_advantage, b_log_prob = (
            batch["obs"],
            batch["action"],
            batch["advantage"],
            batch["log_prob"],
        )

        norm_adv = (b_advantage - b_advantage.mean()) / (
            b_advantage.std() + 1e-8
        )

        pi, log_prob, _ = self.actor_critic.step(b_obs)

        ratio = torch.exp(log_prob - b_log_prob)

        clip_advantage = (
            torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            * norm_adv
        )
        loss_pi = - torch.mean(ratio * norm_adv, clip_advantage).mean()

        with torch.no_grad():
            approx_kl = (b_log_prob - log_prob).mean()
            clipped = ratio.gt(1.0 +self.clip_ratio) | ratio.lt(
                1.0 - self.clip_ratio
            )
            clip_frac = clipped.float().mean()
            ent = pi.entropy().mean()

        return loss_pi, pi, dict(kl=approx_kl, ent=ent, cf=clip_frac)

    def _compute_value_loss(self, batch):
        b_obs, b_return, b_value = (
            batch["obs"],
            batch["return"],
            batch["value"]
        )
        _, _, v = self.actor_critic.step(b_obs)

        loss_v_unclipped = (v - b_return) ** 2
        v_clipped = b_value + torch.clamp(
            v - b_value, -self.clip_ratio, self.clip_ratio
        )
        loss_v_clipped = (v_clipped - b_return) ** 2
        loss_v_max = torch.max(loss_v_unclipped, loss_v_clipped)
        loss_v = loss_v_max.mean()

        return loss_v, v
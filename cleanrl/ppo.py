import os, time, random
import numpy as np
from dataclasses import dataclass
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

# Import your custom AEROEnv
# from aero_env import AEROEnv   # <--- use your implementation
# For now we assume df is loaded globally
# env = AEROEnv(df, num_shards=16, max_migrations=3)

# -------------------
# Args
# -------------------
@dataclass
class Args:
    exp_name: str = "aero_ppo"
    seed: int = 1
    cuda: bool = True
    total_timesteps: int = 200000
    learning_rate: float = 1e-5
    num_envs: int = 1              # AEROEnv is expensive, start with 1
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # Runtime filled
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# -------------------
# Attention-based Agent
# -------------------
class AttentionPolicy(nn.Module):
    def __init__(self, obs_dim, action_nvec, hidden_dim=256, n_heads=6):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_nvec = action_nvec
        self.hidden_dim = hidden_dim

        self.state_proj = nn.Linear(obs_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim*2, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.policy_head = nn.Linear(hidden_dim, sum(action_nvec))
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action_history=None):
        # obs: (batch, obs_dim)
        x = self.state_proj(obs)
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
        if action_history is not None:
            x = torch.cat([action_history, x], dim=1)
        x = self.transformer(x)
        context = x[:, -1, :]  # last token
        logits = self.policy_head(context)
        value = self.value_head(context).squeeze(-1)
        return logits, value


class Agent(nn.Module):
    def __init__(self, obs_dim, action_nvec):
        super().__init__()
        self.action_nvec = action_nvec
        self.policy = AttentionPolicy(obs_dim, action_nvec)

    def get_value(self, obs, history=None):
        _, value = self.policy(obs, history)
        return value

    def get_action_and_value(self, obs, action=None, history=None):
        logits, value = self.policy(obs, history)
        splits = torch.split(logits, self.action_nvec, dim=-1)
        dists = [Categorical(logits=s) for s in splits]

        if action is None:
            action = torch.stack([dist.sample() for dist in dists], dim=-1)

        logprobs = torch.stack([dist.log_prob(a) for dist, a in zip(dists, action.T)], dim=-1).sum(-1)
        entropy = torch.stack([dist.entropy() for dist in dists], dim=-1).sum(-1)
        return action, logprobs, entropy, value


# -------------------
# Training Loop (simplified CleanRL PPO)
# -------------------
def train_aero(env, args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    obs_dim = env.observation_space.shape[0]
    action_nvec = env.action_space.nvec

    agent = Agent(obs_dim, action_nvec).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs_buf = torch.zeros((args.num_steps, obs_dim)).to(device)
    act_buf = torch.zeros((args.num_steps, len(action_nvec))).to(device)
    logp_buf = torch.zeros((args.num_steps,)).to(device)
    rew_buf = torch.zeros((args.num_steps,)).to(device)
    val_buf = torch.zeros((args.num_steps,)).to(device)
    done_buf = torch.zeros((args.num_steps,)).to(device)

    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    done = False

    for update in range(1, args.total_timesteps // args.num_steps + 1):
        for step in range(args.num_steps):
            with torch.no_grad():
                action, logprob, entropy, value = agent.get_action_and_value(obs.unsqueeze(0))
            next_obs, reward, done, _ = env.step(action.squeeze().cpu().numpy())

            obs_buf[step] = obs
            act_buf[step] = action.squeeze().cpu()
            logp_buf[step] = logprob
            rew_buf[step] = reward
            val_buf[step] = value
            done_buf[step] = float(done)

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            if done:
                obs = torch.tensor(env.reset(), dtype=torch.float32, device=device)

        # Compute returns + advantage (GAE)
        with torch.no_grad():
            last_val = agent.get_value(obs.unsqueeze(0))
        adv_buf = torch.zeros_like(rew_buf).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - done_buf[t]
                nextvalue = last_val
            else:
                nextnonterminal = 1.0 - done_buf[t + 1]
                nextvalue = val_buf[t + 1]
            delta = rew_buf[t] + args.gamma * nextvalue * nextnonterminal - val_buf[t]
            adv_buf[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        ret_buf = adv_buf + val_buf

        # Flatten
        b_obs = obs_buf
        b_act = act_buf.long()
        b_logp = logp_buf
        b_adv = adv_buf
        b_ret = ret_buf
        b_val = val_buf

        # PPO Update
        inds = np.arange(args.num_steps)
        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, args.num_steps, args.num_steps // args.num_minibatches):
                end = start + args.num_steps // args.num_minibatches
                mb_inds = inds[start:end]

                _, newlogp, entropy, newval = agent.get_action_and_value(b_obs[mb_inds], b_act[mb_inds])
                logratio = newlogp - b_logp[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_adv[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newval.squeeze() - b_ret[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        print(f"Update {update}, mean reward: {rew_buf.mean().item():.4f}")
    return agent

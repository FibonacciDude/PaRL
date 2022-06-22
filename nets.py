import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, categorical, hidden_dim=128, device="cuda"):
        super(MLPActor, self).__init__()
        self.actor_mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                )
        if categorical:
            self.dist = dist.Categorical
            self.head = nn.Sequential(
                    nn.Linear(hidden_dim, action_dim),
                    nn.Softmax()
                    )
        else:
            self.dist = dist.Normal
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.std = nn.Linear(hidden_dim, action_dim)

            def head(x):
                return (self.mean(x), self.std(x))

    def pi(self, obs):
        scores = self.actor_mlp(obs)
        xx = self.dist(*self.head(state_dim))
        action = xx.sample()
        logp = xx.log_prob(action)
        return action, logp

    def logprob(self, obs, act):
        scores = self.actor_mlp(obs)
        xx = self.dist(*self.head(state_dim))
        logp = xx.log_prob(action)
        return logp

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, categorical, hidden_dim=128, device="cuda"):
        super(MLPActor, self).__init__()
        self.critic_mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, 1),
                )

    def predict(self, traj):
        obs = torch.from_numpy(traj['obs'], device=self.device)
        v_t = self.critic_mlp(obs)
        return v_t

class MLPAC(nn.Module):
    def __init__(self, state_dim, action_dim, categorical, hidden_dim=128, device="cuda"):
        super(MLPActor, self).__init__()
        self.actor = Actor(state_dim, action_dim, categorical, hidden_dim=hidden_dim, device=device)
        self.critic = Critic(state_dim, action_dim, categorical, hidden_dim=hidden_dim, device=device)
        self.device = torch.device(device)

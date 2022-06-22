import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, categorical, hidden_dim=128, device="cuda"):
        super(Actor, self).__init__()
        self.actor_mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                )
        if categorical:
            self.dist = dist.Categorical
            self.head = nn.Sequential(
                    nn.Linear(hidden_dim, action_dim),
                    nn.Softmax(dim=-1)
                    )
        else:
            # TODO: Make this unpack correctly for continuous
            self.dist = dist.Normal
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.std = nn.Linear(hidden_dim, action_dim)

            def head(x):
                return (self.mean(x), self.std(x))

    def pi(self, obs):
        scores = self.actor_mlp(obs)
        xx = self.dist(self.head(scores))
        act = xx.sample()
        logp = xx.log_prob(act)
        return act, logp

    def logprob(self, obs, act):
        scores = self.actor_mlp(obs)
        xx = self.dist(self.head(scores))
        logp = xx.log_prob(act)
        return logp

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, categorical, hidden_dim=128, device="cuda"):
        super(Critic, self).__init__()
        self.critic_mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, 1),
                )

    def predict(self, obs):
        v_t = self.critic_mlp(obs)
        return v_t

class MLPAC(nn.Module):
    def __init__(self, state_dim, action_dim, categorical, hidden_dim=128, device="cuda"):
        super(MLPAC, self).__init__()
        self.actor = Actor(state_dim, action_dim, categorical, hidden_dim=hidden_dim, device=device)
        self.critic = Critic(state_dim, action_dim, categorical, hidden_dim=hidden_dim, device=device)
        self.device = torch.device(device)
        self.to(self.device)

    def step(self, obs):
        obs = torch.from_numpy(obs).to(self.device)
        action, logp = self.actor.pi(obs)
        v = self.critic.predict(obs)
        return action, v.detach().cpu().numpy(), logp

    def logprob(self, obs, act):
        obs = torch.from_numpy(obs).to(self.device)
        act = torch.tensor(act).to(self.device)
        return self.actor.logprob(obs, act)
    
    def predict(self, obs):
        obs = torch.from_numpy(obs).to(self.device)
        return self.critic.predict(obs)

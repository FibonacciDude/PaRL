import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, categorical, hidden_dim=128):
        super(Actor, self).__init__()
        self.actor_mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                )
        self.categorical = categorical
        if categorical:
            self.head = nn.Sequential(
                    nn.Linear(hidden_dim, action_dim),
                    nn.Softmax(dim=-1))
        else:
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.logvar = nn.Linear(hidden_dim, action_dim)

    def _get_distr(self, obs):
      scores = self.actor_mlp(obs)
      if self.categorical:
        xx = dist.Categorical(self.head(scores))
      else:
        xx = dist.Normal(self.mean(scores), self.logvar(scores).mul(.5).exp())
      return xx

    def pi(self, obs):
        xx = self._get_distr(obs)
        act = xx.sample()
        logp = xx.log_prob(act)
        return act, logp

    def logprob(self, obs, act):
        xx = self._get_distr(obs)
        logp = xx.log_prob(act)
        return logp

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, categorical, hidden_dim=128):
        super(Critic, self).__init__()
        self.critic_mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, 1),
                )

    def predict(self, obs, detach=False):
        v_t = self.critic_mlp(obs)
        if detach:
          v_t=v_t.detach().cpu().numpy()
        return v_t

class MLPAC(nn.Module):
    def __init__(self, state_dim, action_dim, categorical, hidden_dim=128, device="cuda"):
        super(MLPAC, self).__init__()
        self.actor = Actor(state_dim, action_dim, categorical, hidden_dim=hidden_dim)
        self.critic = Critic(state_dim, action_dim, categorical, hidden_dim=hidden_dim)
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
    
    def predict(self, obs, detach=False):
        obs = torch.from_numpy(obs).to(self.device)
        return self.critic.predict(obs, detach=detach)

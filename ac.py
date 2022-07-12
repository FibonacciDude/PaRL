import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, categorical, hidden_dim=128):
        super(Actor, self).__init__()
        self.actor_mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                )
        self.categorical = categorical
        if categorical:
            self.head = nn.Sequential(
                    nn.Linear(hidden_dim, action_dim),
                    nn.Softmax(dim=-1))
        else:
            self.mean = nn.Linear(hidden_dim, action_dim)
            # as parameter, not based on input
            self.log_std = torch.nn.Parameter(  torch.as_tensor( -.5*np.ones(action_dim, dtype=np.float32) )  )
            #self.sqrt_std = torch.nn.Parameter(  torch.as_tensor( -np.ones(action_dim, dtype=np.float32) )  )

    def _get_distr(self, obs):
      scores = self.actor_mlp(obs)
      if self.categorical:
        xx = dist.Categorical(self.head(scores))
      else:
        xx = dist.Normal(self.mean(scores), self.log_std.exp())
        #xx = dist.Normal(self.mean(scores), .5 * self.sqrt_std.square())
      return xx

    def pi(self, obs):
        xx = self._get_distr(obs)
        act = xx.sample().detach()
        logp = xx.log_prob(act).detach()
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
                nn.ReLU(),
                #nn.Linear(hidden_dim, hidden_dim),
                #nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                )

    def predict(self, obs, detach=False):
        v_t = self.critic_mlp(obs).squeeze()
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
        self.to(torch.float32)

    def to_torch(self, obs):
      return torch.from_numpy(obs).to(self.device).to(torch.float32)

    def step(self, obs):
        obs = self.to_torch(obs)
        action, logp = self.actor.pi(obs)
        v = self.critic.predict(obs, detach=True)
        return action, v, logp

    def logprob(self, obs, act):
        obs = self.to_torch(obs)
        act = torch.stack(act).to(self.device) # list
        return self.actor.logprob(obs, act)
    
    def predict(self, obs, detach=False):
        obs = self.to_torch(obs)
        return self.critic.predict(obs, detach=detach)

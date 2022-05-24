import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import time

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        # discrete action space only

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.buffer = []

    def act(self, state):  # for action
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def eval(self, state, action):  # for update
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        entropy = dist.entropy()
        state_vals = self.critic(state)

        return action_logprob, entropy, state_vals

class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, epochs, eps_clip):
	self.state_dim = state_dim
	self.action_dim = action_dim
	self.lr_actor = lr_actor
	self.lr_critic = lr_critic
	self.gamma = gamma
	self.epochs = epochs
	self.eps_clip = eps_clip

	self.buffer = RolloutBuffer()

	self.old_pi = ActorCritic(state_dim, action_dim, hidden_dim)
	self.pi = ActorCritic(state_dim, action_dim, hidden_dim)
	
	self.pi.load_state_dict(self.old_pi.state_dict())
	self.mse_loss = nn.MSELoss()


    def action(self, state):
	with torch.no_grad():
		state = torch.FloatTensor(state)
		action, action_logprob = self.policy_old.act(state)

		self.buffer.states.append(state)
		self.buffer.actions.append(action)
		self.buffer.logprobs.append(action_logprob)

	return action.item()


https://github.com/nikhilbarhate99/PPO-PyTorch/blob/d5c883783ac6406c4d58a1e1e9eb6f08a6462d89/PPO.py#L20

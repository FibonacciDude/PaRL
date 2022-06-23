import gym
import numpy as np
import ppo, rollout
import ac as models 
import core
import torch
from gym import envs

all_envs = envs.registry.all()
envs_ = sorted([env_spec.id for env_spec in all_envs])
print(envs_)
# TODO
# weird bug, always getting 1 reward and the model's random behavior wins

np.random.seed(42)
torch.manual_seed(42)

def run(epochs=100, env_idx=0):
    env = gym.make(envs_[env_idx])
    categorical = isinstance(env.action_space, gym.spaces.Discrete)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if categorical else env.action_space.shape[0]
    print("Categorical action space:", categorical)
    ac = models.MLPAC(state_dim, action_dim, categorical)

    pi_lr = 4e-4
    vf_lr = 1e-3
    pi_optim = torch.optim.Adam(ac.actor.parameters(), lr=pi_lr)
    vf_optim = torch.optim.Adam(ac.critic.parameters(), lr=vf_lr)

    for e in range(epochs):
        print("Epoch %d" % e)
        # get trajectory
        traj = rollout.rollout(ac, env, max_len=1000)
        # update
        ppo.ppo_clip(ac, traj, pi_optim, vf_optim, targ_kl=0.01)

        if (e!=0 and e % 10 == 0) or e==epochs-1:
            print("Validating...")
            rewards = core.validate(ac, env)
            print("Reward sum: {:.4f}".format(rewards.sum()))

if __name__ == "__main__":
    run(epochs=40, env_idx=envs_.index("Ant-v3"))

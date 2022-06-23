import gym
from gym import envs
import numpy as np
import torch
import ppo, rollout
import ac as models 
import core
import argparse

all_envs = envs.registry.all()
envs_ = sorted([env_spec.id for env_spec in all_envs])
print(envs_)

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
        trajs = []
        for i in range(10):
          traj = rollout.rollout(ac, env, 
          max_len=args.max_len, steps=args.steps)
          trajs.append(traj)

        # update
        ppo.ppo_clip(
        ac, trajs,
        pi_optim, vf_optim,
        targ_kl=args.targ_kl,
        eps=args.eps,
        gamma=args.gamma,
        lam=args.lam
        )

        if (e!=0 and e % 10 == 0) or e==epochs-1:
            print("Validating...")
            rewards = core.validate(ac, env, render=args.render)
            print("Reward sum: {:.4f}".format(rewards.sum()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--pi_iters", type=int, default=80)
    parser.add_argument("--vf_iters", type=int, default=80)
    parser.add_argument("--targ_kl", type=float, default=.003)
    parser.add_argument("--eps", type=float, default=.2)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--lam", type=float, default=.95)
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--render", type=bool, default=True)
    args = parser.parse_args()

    run(epochs=args.epochs, env_idx=envs_.index(args.env))


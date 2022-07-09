import gym
from gym import envs
import numpy as np
import torch
import ppo, rollout
import ac as models 
import core
import argparse
from mpi_tools import *

all_envs = envs.registry.all()
envs_ = sorted([env_spec.id for env_spec in all_envs])
logger = core.Logger()
#if proc_id()==0:
#  print(envs_)

def run(epochs=100, env_idx=0):
    setup_pytorch_for_mpi()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_name = envs_[env_idx]
    env = gym.make(env_name)
    env.reset(seed=args.seed)

    categorical = isinstance(env.action_space, gym.spaces.Discrete)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if categorical else env.action_space.shape[0]
    ac = models.MLPAC(state_dim, action_dim, categorical, hidden_dim=args.hidden_dim, device="cpu")

    pi_lr = args.pi_lr
    vf_lr = args.vf_lr
    pi_optim = torch.optim.Adam(ac.actor.parameters(), lr=pi_lr, eps=1e-5)
    vf_optim = torch.optim.Adam(ac.critic.parameters(), lr=vf_lr, eps=1e-5)

    total_steps = 0
    for e in range(epochs):
        # get trajectory
        traj, steps_taken = rollout.rollout(ac, env, 
        max_len=args.max_len, steps=args.steps)
        total_steps += steps_taken

        # update
        ppo.ppo_clip(
        ac, traj,
        pi_optim, vf_optim,
        targ_kl=args.targ_kl,
        eps=args.eps,
        gamma=args.gamma,
        lam=args.lam
        )
        logger.store(epoch=e, ret_traj=traj["rew"].sum(), total_steps=mpi_sum(total_steps))
        logger.log("epoch", mean=False)
        logger.log("total_steps", mean=False)
        logger.log("ret_traj", with_min_max=True)
        logger.dump()

        if (e % args.val_every == 0 or e==epochs-1) and e!=0: 
            reward = core.validate(ac, env_name, render=args.render if proc_id()==0 else False).sum()
            logger.store(ret_traj=reward)
            logger.log("ret_traj", with_min_max=True)
            logger.dump()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--pi_lr", type=float, default=1e-3)
    parser.add_argument("--vf_lr", type=float, default=1e-3)
    parser.add_argument("--pi_iters", type=int, default=80)
    parser.add_argument("--vf_iters", type=int, default=80)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--targ_kl", type=float, default=.01)
    parser.add_argument("--eps", type=float, default=.2)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--lam", type=float, default=.95)
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--val_rollouts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    mpi_fork(args.cpu)

    run(epochs=args.epochs, env_idx=envs_.index(args.env))


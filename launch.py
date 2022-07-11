import gym
import torch
import core
import argparse, warnings, time
import ppo, rollout
import numpy as np
import ac as models 
from gym import envs
from mpi_tools import *
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

all_envs = envs.registry.all()
envs_ = sorted([env_spec.id for env_spec in all_envs])
logger = core.Logger()

def run_ppo(epochs, env_idx=0):
    prefix = f'({args.env})-s{args.seed}-cpu{args.cpu}-com{args.comm}{"-avg" if args.avg_params else ""}'
    writer = SummaryWriter(log_dir=f'runs/{prefix}-{time.monotonic()//60}')
    setup_pytorch_for_mpi()
    seed = args.seed + 1000 * proc_id()

    np.random.seed(seed)
    torch.manual_seed(seed)

    env_name = envs_[env_idx]
    env = gym.make(env_name)
    env.reset(seed=seed)

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

        # update with trajectory
        ppo.ppo_clip(
        ac, traj,
        pi_optim, vf_optim,
        targ_kl=args.targ_kl,
        eps=args.eps,
        gamma=args.gamma,
        lam=args.lam,
        avg_grad=not args.avg_params,
        )

        if args.avg_params and (e+1) % args.comm == 0:
          # reached communication point -> avg parameters (after one ppo update)
          mpi_avg_params(ac)

        if e % args.val_freq == 0:
          returns = mpi_avg(core.validate( ac, env_name, timeout=args.max_len, render=False, seed=args.seed+2000*proc_id() ).sum())
          # rewards per steps
          writer.add_scalar('returns', returns, mpi_sum(total_steps)) 

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=750)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--max_len", type=int, default=1000)

    parser.add_argument("--pi_lr", type=float, default=3e-4)
    parser.add_argument("--vf_lr", type=float, default=1e-3)
    parser.add_argument("--pi_iters", type=int, default=80)
    parser.add_argument("--vf_iters", type=int, default=80)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--targ_kl", type=float, default=.01)
    parser.add_argument("--eps", type=float, default=.2)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--lam", type=float, default=.97)

    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--val_freq", type=int, default=2)
    # parser.add_argument("--val_rollouts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--comm", type=int, default=1)
    parser.add_argument("--avg_params", action="store_true")

    # parser.add_argument("--render", action="store_true")
    parser.add_argument("--env_list", action="store_true")
    args = parser.parse_args()

    if args.env_list:
      print(envs_)

    mpi_fork(args.cpu)
    run_ppo(epochs=args.epochs, env_idx=envs_.index(args.env))



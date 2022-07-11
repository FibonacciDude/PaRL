import gym
import torch
import argparse, warnings, time, copy
import core, ppo, rollout
import numpy as np
import ac as models 
from mpi_tools import *
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

all_envs = gym.envs.registry.all()
envs_ = sorted([env_spec.id for env_spec in all_envs])
logger = core.Logger()

def run_ppo(epochs, env_idx=0):
    prefix = f'({args.env})-s{args.seed}-cpu{args.cpu}-com{args.comm}{"-avg" if args.avg_params else ""}{"-ea" if args.easgd else ""}'
    writer = SummaryWriter(log_dir=f'runs/{prefix}-{time.monotonic()}')
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
    ac_latent = copy.deepcopy(ac)
    # make all the initializations the same
    sync_params(ac)
    sync_params(ac_latent)

    pi_lr = args.pi_lr
    vf_lr = args.vf_lr
    # this does make a difference
    pi_optim = torch.optim.Adam(ac.actor.parameters(), lr=pi_lr, eps=1e-5)
    vf_optim = torch.optim.Adam(ac.critic.parameters(), lr=vf_lr, eps=1e-5)
    #pi_optim = torch.optim.SGD(ac.actor.parameters(), lr=pi_lr)
    #vf_optim = torch.optim.SGD(ac.critic.parameters(), lr=vf_lr)

    total_steps = clk_pi = clk_vf = 0
    for e in range(epochs):
        traj, steps_taken = rollout.rollout(ac, env, 
        max_len=args.max_len, steps=args.steps)
        total_steps += steps_taken

        # we average/merge per gradient not per update
        #   - this can be seen as a form of stability against divergence with large 'comm'
        #     - the policy is updated with advantage (which uses value fn); thus, it will only change due to 
        #     variability (across processes) in its trajectory not baseline/value function
        #     *(not necessarily true, pi always thinks it's  optimal, so why force average of v)
        #     if we average the policies often, then we get little difference in trajectories
        #   - variability in the value function might be due to different visited states (novelty); however,
        #   this is only a residual effect of a varied policy. Thus, we want the policy to have more variability
        #   since a poor value function could lead to low reward

        clk_pi, clk_vf = ppo.ppo_clip(
        ac, traj,
        pi_optim, vf_optim,
        targ_kl=args.targ_kl,
        eps=args.eps,
        gamma=args.gamma,
        lam=args.lam,

        avg_grad=not args.avg_params,
        easgd=args.easgd,
        t=args.comm,
        clk=(clk_pi, clk_vf),

        # possible easgd hyperparams (kwargs)
        alpha=args.alpha,
        beta=args.beta,
        ac_latent=ac_latent,
        )

        if e % args.val_freq == 0:
          module = ac_latent if args.easgd else ac
          returns = mpi_avg( core.validate( module, env_name, timeout=args.max_len, render=False, seed=args.seed+2000*proc_id() ).sum() )
          # rewards per steps
          sum_total_steps = mpi_sum(total_steps)
          writer.add_scalar('returns', returns, sum_total_steps)
          #returns = mpi_avg(traj['rets'].sum())
          if proc_id()==0: print(f'{sum_total_steps}: {returns}')

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

    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=.9**8)

    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--comm", type=int, default=1)
    parser.add_argument("--avg_params", action="store_true")
    parser.add_argument("--env_list", action="store_true")
    parser.add_argument("--easgd", action="store_true")
    args = parser.parse_args()

    if args.env_list:
      print(envs_)

    mpi_fork(args.cpu)
    run_ppo(epochs=args.epochs, env_idx=envs_.index(args.env))



import core
import numpy as np
import torch
from mpi_tools import *
import torchviz as tv

def pi_loss(ac, traj, eps):
    obs, act, adv, logp_old = traj['obs'], traj['act'], traj['adv'], traj['logp']
    adv = torch.tensor(adv.copy(), device=ac.device)

    logp = ac.logprob(obs, act)

    log_ratio = logp - logp_old
    ratio = torch.exp(log_ratio)
    clip_adv = (ratio*adv.reshape(-1,1)).clamp(min=1-eps, max=1+eps)
    loss = -torch.min(ratio*adv.reshape(-1,1), clip_adv).mean()
    approx_kl = ((ratio-1) - log_ratio).mean().item()
    return loss, approx_kl

def vf_loss(ac, traj):
    ret = torch.tensor(traj['ret'][:-1].copy(), device=ac.device)
    loss = .5 * (ac.predict(traj['obs']) - ret)**2
    return loss.mean()

def ppo_clip(ac, traj, pi_optim, vf_optim, clk, ac_latent, pi_iters=80, vf_iters=80, targ_kl=.01, eps=.2, gamma=.99, lam=.95, t=1, alpha=.001, beta=.4, avg_grad=True, easgd=False):
    clk_pi, clk_vf = clk

    core.gae(traj, gamma, lam, avg_grad)
    if avg_grad:
      sync_params(ac)

    pi_old, kl_old = pi_loss(ac, traj, eps)
    for _ in range(pi_iters):
        pi_optim.zero_grad()
        loss_a, pi_kl = pi_loss(ac, traj, eps)
        # we need avg kl to avoid stopping earlier than other processes
        pi_kl = mpi_avg(pi_kl)
        if abs(pi_kl) > 1.5 * targ_kl:
          break 
        loss_a.backward()
        if avg_grad:
          mpi_avg_grads(ac.actor)
        pi_optim.step()
        clk_pi+=1

        if clk_pi % t == 0 and not avg_grad:
          if not easgd: mpi_avg_params(ac.actor)
          else:
            # implementing asynchronous version due to clk variability (because of kl stopping)
            # however, one must be careful as the kl stopping might not be randomly distributed (wrt processes)
            # ...actually. let's make it synchronous because shared memory in MPI is weird
            avg_params = mpi_avg_params_ac(ac.actor) # TODO: this is not memory efficient at all

            # the weighted moving average = to the weighted 'step' towards the difference
            # x_bar - beta(avg - x_bar) = (1-beta)*x_bar + beta*x_avg
            # order matters
            mpi_inplace_add(ac.actor, ac_latent.actor, consts=((1-alpha),alpha))
            mpi_inplace_add(ac_latent.actor, avg_params, consts=((1-beta),beta))

    vf_old = vf_loss(ac, traj)
    for _ in range(vf_iters):
        vf_optim.zero_grad()
        loss_c = vf_loss(ac, traj)
        loss_c.backward()
        if avg_grad:
          mpi_avg_grads(ac.critic)
        vf_optim.step()
        clk_vf+=1
        if clk_vf % t == 0 and not avg_grad:
          if not easgd: mpi_avg_params(ac.critic)
          else:
            avg_params = mpi_avg_params_ac(ac.critic)
            mpi_inplace_add(ac.critic, ac_latent.critic, consts=((1-alpha),alpha))
            mpi_inplace_add(ac_latent.critic, avg_params, consts=((1-beta),beta))
    # if not easgd: mpi_avg_params(ac)
    return (clk_pi, clk_vf)

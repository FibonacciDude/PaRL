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


def ppo_clip(ac, traj, pi_optim, vf_optim, pi_iters=80, vf_iters=80, targ_kl=.01, eps=.2, gamma=.99, lam=.95, avg_grad=True):

    core.gae(traj, gamma, lam, avg_grad)
    if avg_grad:
      sync_params(ac)

    pi_old, kl_old = pi_loss(ac, traj, eps)
    for _ in range(pi_iters):
        pi_optim.zero_grad()
        loss_a, pi_kl = pi_loss(ac, traj, eps)
        if avg_grad:
          pi_kl = mpi_avg(pi_kl)
        if pi_kl > 1.5 * targ_kl:
            # print("Finished due to too great of a KL %d" % _) # put this in log
            break
        loss_a.backward()
        if avg_grad:
          mpi_avg_grads(ac.actor)
        pi_optim.step()

    vf_old = vf_loss(ac, traj)
    for _ in range(vf_iters):
        vf_optim.zero_grad()
        loss_c = vf_loss(ac, traj)
        loss_c.backward()
        if avg_grad:
          mpi_avg_grads(ac.critic)
        vf_optim.step()

def ppo_clip_ea(ac, traj, pi_optim, vf_optim, pi_iters=80, vf_iters=80, targ_kl=.01, eps=.2, gamma=.99, lam=.95, avg_grad=True):

    core.gae(traj, gamma, lam, avg_grad)
    if avg_grad:
      sync_params(ac)

    pi_old, kl_old = pi_loss(ac, traj)
    for _ in range(pi_iters):
        pi_optim.zero_grad()
        loss_a, pi_kl = pi_loss(ac, traj)
        if avg_grad:
          pi_kl = mpi_avg(pi_kl)
        if pi_kl > 1.5 * targ_kl:
            # print("Finished due to too great of a KL %d" % _) # put this in log
            break
        loss_a.backward()
        if avg_grad:
          mpi_avg_grads(ac.actor)
        pi_optim.step()

    vf_old = vf_loss(ac, traj)
    for _ in range(vf_iters):
        vf_optim.zero_grad()
        loss_c = vf_loss(ac, traj)
        loss_c.backward()
        if avg_grad:
          mpi_avg_grads(ac.critic)
        vf_optim.step()

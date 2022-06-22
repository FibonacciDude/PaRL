import core
import numpy as np
import torch

# TODO
# add logger
# make adv/ret be tensors
# single out which DONT have to be tensors

def ppo_clip(ac, traj, pi_optim, vf_optim, pi_iters=80, vf_iters=80, targ_kl=.003, eps=.2, gamma=.99, lam=.95):

    core.gae(traj, gamma, lam)

    def pi_loss(ac, traj):
        obs, act, adv, logp_old = traj['obs'], traj['act'], traj['adv'], traj['logp']
        logp = ac.logprob(obs, act)
        ratio = torch.exp(logp - logp_old)
        adv = torch.tensor(adv, device=ac.device)
        clip_adv = (ratio*adv).clamp(min=1-eps, max=1+eps)
        loss = -torch.min(ratio*adv, clip_adv).mean()
        kl = (logp_old - logp).mean().item()
        return loss, kl

    def vf_loss(ac, traj):
        adv = torch.tensor(traj['adv'], device=ac.device)
        loss = (ac.predict(traj['obs']) - adv)**2
        return loss.mean()

    for _ in range(pi_iters):
        pi_optim.zero_grad()
        loss, pi_kl = pi_loss(ac, traj)
        if pi_kl > 1.5 * targ_kl:
            break
        loss.backward()
        pi_optim.step()

    for _ in range(vf_iters):
        vf_optim.zero_grad()
        loss = vf_loss(ac, traj)
        loss.backward()
        vf_optim.step()

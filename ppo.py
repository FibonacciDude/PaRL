import core
import numpy as np
import torch

# TODO
# add logger
# add mpi (for gradients) - parallel rather than sequential

def ppo_clip(ac, trajs, pi_optim, vf_optim, pi_iters=80, vf_iters=80, targ_kl=.003, eps=.2, gamma=.99, lam=.95):

    core.gae(trajs, gamma, lam)
    n = len(trajs)

    # TODO: why .copy() error (negative stride)
    def pi_loss(ac, traj):
        obs, act, adv, logp_old = traj['obs'], traj['act'], traj['adv'], traj['logp']
        adv = torch.tensor(adv.copy(), device=ac.device)

        logp = ac.logprob(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = (ratio*adv.reshape(-1,1)).clamp(min=1-eps, max=1+eps)
        loss = -torch.min(ratio*adv.reshape(-1,1), clip_adv).mean()
        kl = (logp_old - logp).mean().item()
        return loss, kl

    def vf_loss(ac, traj):
        ret = torch.tensor(traj['ret'].copy(), device=ac.device)
        loss = (ac.predict(traj['obs']) - ret)**2
        return loss.mean()

    for _ in range(pi_iters):
        pi_optim.zero_grad()
        # average losses on trajectories
        loss, pi_kl = pi_loss(ac, trajs[0])
        for traj in trajs[1:]:
          loss_t, pi_kl_t = pi_loss(ac, traj)
          loss += loss_t; pi_kl += pi_kl_t
        loss = loss / n
        if pi_kl/n > 1.5 * targ_kl:
            print("Finished due to too great of a KL %d" % _) # put this in log
            break
        loss.backward()
        pi_optim.step()

    for _ in range(vf_iters):
        vf_optim.zero_grad()
        loss = vf_loss(ac, trajs[0])
        for traj in trajs[1:]:
          loss += vf_loss(ac, traj)
        loss = loss / n
        loss.backward()
        vf_optim.step()

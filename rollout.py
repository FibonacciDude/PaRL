import torch
import numpy as np
import gym
from mpi_tools import *

# code similar to that of spinningup
def rollout(ac, env, steps=5000, max_len=1000):
    o = env.reset()
    obs, act, rew, val, logps = [], [], [], [], []
    ep_len = 0
    local_steps = int(steps / num_procs())
    for t in range(local_steps):
        a, v, logp = ac.step(o)

        next_o, r, d, _ = env.step(a.cpu().numpy())
        ep_len += 1

        # save
        obs.append(o)
        rew.append(r)
        act.append(a)
        val.append(v)
        logps.append(logp)
    
        # upd obs
        o = next_o

        timeout = (ep_len == max_len)
        terminal = d or timeout
        epoch_ended = t==(local_steps-1)
        if terminal or epoch_ended:
            if epoch_ended and not(terminal):
                #print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                pass
            # boostrap
            v = ac.predict(o,detach=True) if (timeout or epoch_ended) else 0

            obs = np.array(obs)
            rew = np.append(rew, v)
            val = np.append(val, v)
            # logp only one that isn't numpy (tensor)
            logps = torch.stack(logps)
            return dict(obs=obs, act=act, rew=rew, value=val, logp=logps), t

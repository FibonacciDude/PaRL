import numpy as np
import scipy.signal as signal

def discount(x, disc):
    y = signal.lfilter([1], [1, -disc], x=x[::-1])
    return y[::-1]

# modular rl
def gae(path, gamma, lam):
    # ret, value, adv
    path["ret"] = discount(path["rew"], gamma)

    bln = path["value"]
    td = path["rew"][:-1] + gamma*bln[1:] - bln[:-1]
    path["adv"] = discount(td, gamma*lam)

    # normalize
    mean, std = path["adv"].mean(), path["adv"].std()
    path["adv"] = (path["adv"] - mean) / std

def validate(ac, env, timeout=500):
    o = env.reset(seed=42)
    rews = []
    for _ in range(timeout):
        env.render()
        act, v, logp = ac.step(o)
        o, r, done, info = env.step(act.detach().cpu().numpy())
        rews.append(r)
        if done:
            break
    env.reset(seed=42)
    return np.array(rews)

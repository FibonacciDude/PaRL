import numpy as np
import torch
import scipy.signal as signal
import time
import glfw, gym, mujoco_py
from mpi_tools import *

# -------------- logger ----------------

class Logger:
  def __init__(self, CONST=80):
    self.store_dict = dict()
    self.string = ""
    self.CONST = CONST

  def add_string(self, s):
    self.string += "\n%s\n" % s

  def store(self, **kwargs):
    for k,v in kwargs.items():
      self.store_dict[k] = v

  def log(self, key, val=None, mean=False, with_min_max=False):
    if val is not None:
      self.store_dict[key] = val
    v = self.store_dict[key]
    if with_min_max:
        mean, std, min_, max_ = mpi_statistics_scalar([v], with_min_and_max=True)
        self.add_string("|%s -> min:%.2e | max:%.2e | mean:%.3e | std:%.3e" \
        % (key, min_, max_, mean, std) + "|")
    else:
        if mean:
          mean, std = mpi_statistics_scalar([v], with_min_and_max=False)
          self.add_string("|%s -> %.1f" % (key, mean) + "|")
        else:
          self.add_string("|%s -> %s" % (key, str(v)) + "|")
          

  def dump(self):
    if proc_id()==0:
      print("-"*self.CONST)
      print(self.string)
      print("-"*self.CONST)
    self.string = ""
    self.store_dict = dict()

# -------------- rl functions -----------------

def discount(x, disc):
    y = signal.lfilter([1], [1, -float(disc)], x=x[::-1])
    return y[::-1]

def gae(path, gamma, lam):
    path["ret"] = discount(path["rew"], gamma)
    bln = path["value"]
    td = path["rew"][:-1] + gamma*bln[1:] - bln[:-1]
    adv = path["adv"] = discount(td, gamma*lam)
    # TODO: check/understand effect of normalization (on only one trajectory)
    mean, std = mpi_statistics_scalar(adv)
    path["adv"] = (path["adv"] - mean) / std

def validate(ac, env_name, timeout=500, render=True):
    env = gym.make(env_name)
    o = env.reset()
    rews = []
    done = False
    for _ in range(timeout):
        act, v, logp = ac.step(o)
        o, r, done, info = env.step(act.detach().cpu().numpy())
        rews.append(r)
        if render:
          env.render()
        if done:
          break
    env.reset()
    env.close()
    return np.array(rews)

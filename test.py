from mpi_tools import *
from mpi4py import MPI
import argparse
import time
import numpy as np

if __name__ == "__main__":
  cpu = 4
  alpha = .01
  # beta = .9 ** 8
  mpi_fork(cpu)

  x_bar = (cpu-1) / 2 # mean
  x = proc_id()

  parser = argparse.ArgumentParser()
  parser.add_argument("--iters", type=int, default=100)
  args = parser.parse_args()

  MPI.COMM_WORLD.bcast(x_bar) # sync
  for i in range(args.iters):
    diff = x - x_bar
    x = x - alpha*(diff)
    x_bar = x_bar + alpha*(diff)
    MPI.COMM_WORLD.bcast(x_bar) #sync
    print("iter:", i, "my xvar", x_bar, f"(${proc_id()})") 
    # long thing
    for _ in range(100):
      np.sin(_)
    print("xbar I see=", x_bar)

  if proc_id()==0:
    print("final xbar=", x_bar)

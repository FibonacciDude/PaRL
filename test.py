from mpi_tools import *
from mpi4py import MPI
import argparse
if __name__ == "__main__":
  mpi_fork(cpu)

  cpu = 4

  alpha = .01
  # beta = .9 ** 8
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
    MPI.COMM_WORLD.bcast(x_bar)
    print(i, "--", x_bar) 

  if proc_id()==0:
    print("x-", x_bar)
  print(proc_id(), x)

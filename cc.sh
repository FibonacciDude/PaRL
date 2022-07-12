#!/bin/bash

python launch.py --env "Hopper-v3" --cpu=4 --epochs=750 # --avg_params

# gold standrt - avg grad
# python launch.py --env "HalfCheetah-v3" --cpu=8 --epochs=750
# python launch.py --env "HalfCheetah-v3" --cpu=4 --epochs=750
# python launch.py --env "HalfCheetah-v3" --cpu=2 --epochs=750

# avg params
# python launch.py --env "HalfCheetah-v3" --cpu=4 --comm=4 --avg_params --epochs=750
# python launch.py --env "HalfCheetah-v3" --cpu=2 --comm=4 --avg_params --epochs=750
# python launch.py --env "HalfCheetah-v3" --cpu=4 --comm=1 --avg_params --epochs=750

## Research project

Inspired by OpenAI’s request for research 2.0, I decided to explore the effect of averaging the parameters of
multiple parallel workers in reinforcement learning. To test this, I coded a PPO implementation (with
elements from John Schulman’s code and OpenAI’s SpinningUp code) that instead of averaging the gradients
at each step, took multiple steps and averaged the parameters of the models in different reinforcement
learning environments.

The model converged faster to optimal behavior in the environment in reward per communication, but
equally as the average gradient (instead of parameter) model in reward per step. It reduced the communication
of parallel workers while still keeping the same performance as the baseline (from my analysis).

Link to request for research 2.0: https://openai.com/index/requests-for-research-2/

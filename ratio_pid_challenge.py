import gymnasium as gym
import mujoco_envs
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

env = gym.make("elbow_angle-v0")
obs, _ = env.reset()
truncated = False

# NOTE: It is recommended to sample a target uniformly as: np.random.uniform(np.deg2rad(5), np.deg2rad(77.0))
# the full range of possible elbow angles may not be a feasible target, but the environment will sample the entire
# range for a target.

while not truncated:
    env.unwrapped.mj_render()
    obs, _, _, truncated, _ = env.step(env.action_space.sample())

env.close()

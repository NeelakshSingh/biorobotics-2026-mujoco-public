import gymnasium as gym
import mujoco_envs

env = gym.make("custom_env-v0")
obs = env.reset()
truncated = False

while not truncated:
  env.unwrapped.mj_render()
  obs, _, _, truncated, _ = env.step(env.action_space.sample())

env.close()
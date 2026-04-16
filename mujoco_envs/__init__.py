from gymnasium.envs.registration import register
import numpy as np

register(
  id="myosuite_default_env-v0",
  entry_point="myosuite.envs.myo.myobase.reach_v0:ReachEnvV0",
  max_episode_steps=100_000,
  kwargs={
      "model_path": "./mujoco_envs/xml/simple_arm/elbow.xml",
      "target_reach_range": {
          "ee_site": ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20)),
      },
      #"normalize_act": True,
      "frame_skip": 5,
  },  
)

register(
  id="elbow_angle-v0",
  entry_point="mujoco_envs.custom_env:ElbowAngleEnv",
  max_episode_steps=500,
  kwargs={
      "model_path": "./mujoco_envs/xml/simple_arm/elbow.xml",
      # target joint angle range for elbow_flexion (radians); ~5°–130°
      "target_jnt_range": {
          "elbow_flexion": (np.deg2rad(5), np.deg2rad(130)),
      },
      "target_type": "generate",   # randomise target each episode
      "reset_type": "random",      # start at a random angle each episode
      "frame_skip": 1,
  },
)
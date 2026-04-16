
from myosuite.envs.myo.myobase.reach_v0 import ReachEnvV0
from myosuite.envs.myo.myobase.pose_v0 import PoseEnvV0

class ElbowAngleEnv(PoseEnvV0):
    """
    Objective: drive elbow_flexion to a target joint angle.

    obs vector: [qpos(1), qvel(1), pose_err(1), act(2)]  →  5 values
      - qpos     : elbow_flexion angle (radians)
      - qvel     : angular velocity * dt
      - pose_err : target_angle - qpos  (radians)
      - act      : muscle activation states [flexor, extensor]

    reward: dense = -|pose_err| + bonus if close, penalty if too far
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mujoco_render_frames = True
        self.viewer_setup(azimuth=90,
                          elevation=-90,
                          distance=-1,
                          render_actuator=True,
                          render_tendon=True,
                          )
import gymnasium as gym
import mujoco_envs
import numpy as np
import matplotlib
matplotlib.use("agg")  # must use a non-interactive backend; see README for why
import matplotlib.pyplot as plt

env = gym.make("elbow_angle-v0")

obs, _ = env.reset()
# obs vector: [qpos, qvel, pose_err, act1, act2]
#   qpos     : elbow_flexion angle (radians)
#   qvel     : elbow_flexion angular velocity * dt
#   pose_err : target_angle - qpos  (radians)
#   act1     : flexor  muscle activation state
#   act2     : extensor muscle activation state
truncated = False

# The environment randomly draws a new target angle each episode.
# You can read it from here, but note that pose_err (obs[2]) already
# gives you an error at every step, so you may not need it.
# Be careful about the error's convention, and the signs for the gains,
# because it will affect the sign of your control signal and thus the direction of movement.
env.unwrapped.target_jnt_value[0] = np.random.uniform(np.deg2rad(5), np.deg2rad(77.0))
ELBOW_SETPOINT = env.unwrapped.target_jnt_value[0]
ELBOW_ANGLE_RANGE = env.unwrapped.target_jnt_range[0]  # [min_rad, max_rad]

# The action space maps [-1, 1] to actual muscle activation.
# Maximum useful activation is 1.0.
# Note that despite taking inputs from -1 to 1; myosim internally
# clips the activation to [0, 1] making [-1, 0] effectively 0 activation.
MAXIMUM_ACTIVATION = 1.0

# --- PID gains -----------------------------------------------------------
# Tune these to achieve good setpoint tracking.
Kp = 0.0  # Proportional gain
Kd = 0.0  # Derivative gain
Ki = 0.0  # Integral gain
# -------------------------------------------------------------------------

# TODO: initialise last_error to the actual first error so that the
# derivative term does not spike on the very first step.
last_error = 0.0
error_integral = 0.0

# Time between two consecutive control steps (seconds).
dt = env.unwrapped.sim.step_duration * env.unwrapped.frame_skip

action_history = []
setpoint_history = []
actual_angle_history = []

print("Target angle (degrees):", np.rad2deg(ELBOW_SETPOINT))

while not truncated:
    env.unwrapped.mj_render()
    setpoint_history.append(env.unwrapped.target_jnt_value[0])
    actual_angle_history.append(obs[0])

    # ------------------------------------------------------------------
    # TODO 1 - Compute the control error (radians).
    # The error is already in the observation vector but just to make it clear
    # be sure of the error convention as reference - actual, you can compute it 
    # explicitly yourself, you can access the actual angle from obs and the target
    # angle from env.unwrapped.target_jnt_value[0] as initialized above.
    error = 0.0

    # TODO 2 - Compute the time-derivative of the error.
    # Think about what last_error and dt are for.
    error_dot = 0.0

    # TODO 3 - Accumulate the integral of the error.
    # Recall how you can approximate integrals using sums of rectangles
    # and use the error and dt to do so.
    # error_integral += ...

    # TODO 4 - Update last_error for the next step.
    last_error = error

    # TODO: Not a direct TODO but at this point, you should carefully think about
    # the behaviour of each PID term and how they will contribute to the control signal u.

    # TODO 5 - Compute the PID control signal u.
    # u should represent the desired activation ratio offset.
    u = 0.0

    # TODO 6 - Convert u to an antagonistic muscle ratio a_r in [0, 1].
    #  a_r = 0.5 is neutral (equal co-activation).
    #  a_r = 1 drives the flexor;  a_r = 0 drives the extensor.
    # The point is, u does not directly represent a_r because a_r is
    # constrained to be between 0 and 1, but u can be any real number.
    a_r = 0.5

    # TODO 7 - Derive individual muscle activations from a_r.
    a_f = 0.0   # flexor  activation
    a_e = 0.0   # extensor activation
    # ------------------------------------------------------------------

    action = np.array([a_f, a_e])
    action_history.append(action)
    obs, _, _, truncated, _ = env.step(action)

action_history = np.asarray(action_history)
setpoint_history = np.asarray(setpoint_history)
actual_angle_history = np.asarray(actual_angle_history)

# --- Plotting -----------------------------------------------------------
# NOTE: plt.show() does NOT work with mjpython; save to a file instead.
time_steps = np.arange(len(setpoint_history)) * dt
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].plot(time_steps, np.rad2deg(setpoint_history), label="Setpoint (deg)",
           linestyle="--", color="black")
ax[0].plot(time_steps, np.rad2deg(actual_angle_history), label="Actual angle (deg)",
           color="tab:blue")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Angle (degrees)")
ax[0].set_title("Setpoint vs Actual Angle")
ax[0].legend()

ax[1].plot(time_steps, action_history[:, 0], label="Flexor activation")
ax[1].plot(time_steps, action_history[:, 1], label="Extensor activation")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Muscle activation")
ax[1].set_title("Muscle Activations")
ax[1].legend()

plt.tight_layout()
plt.savefig("ratio_pid_results.png", dpi=150)
print("Plot saved to ratio_pid_results.png")

env.close()
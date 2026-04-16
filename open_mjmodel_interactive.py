"""
Interactive MuJoCo model explorer.
Run with: uv run mjpython open_interactive.py

Drops into an IPython shell with the model loaded.
Useful variables:
  model  - MjModel  (static model data: joint limits, body names, etc.)
  data   - MjData   (live simulation state: qpos, qvel, ctrl, etc.)
  handle - passive viewer handle (call handle.close() to exit)

Quick reference:
  Joint names/limits : model.joint(i).name, model.jnt_range[i]  (radians)
  Joint positions    : data.qpos
  Joint velocities   : data.qvel
  Actuator ctrl      : data.ctrl
  Body names         : [model.body(i).name for i in range(model.nbody)]
  Step simulation    : mujoco.mj_step(model, data)
  Sync viewer        : handle.sync()
"""

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "mujoco_envs/xml/simple_arm/elbow.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

# Print a quick summary of the model
print(f"\nModel: {MODEL_PATH}")
print(f"  Bodies    : {model.nbody}")
print(f"  Joints    : {model.njnt}")
print(f"  Actuators : {model.nu}")
print(f"  Tendons   : {model.ntendon}")

print("\nJoints:")
for i in range(model.njnt):
    jnt = model.joint(i)
    lo, hi = np.degrees(model.jnt_range[i])
    print(f"  [{i}] {jnt.name:30s}  range: {lo:.1f}° – {hi:.1f}°")

print("\nActuators:")
for i in range(model.nu):
    act = model.actuator(i)
    lo, hi = model.actuator_ctrlrange[i]
    print(f"  [{i}] {act.name:30s}  ctrl range: {lo:.3f} – {hi:.3f}")

print("\nBodies:")
for i in range(model.nbody):
    print(f"  [{i}] {model.body(i).name}")

print()

handle = mujoco.viewer.launch_passive(model, data)

# Simulation runs in a background thread so the IPython shell stays responsive.
# Use sim.pause() / sim.resume() to freeze/unfreeze physics from the shell.
import threading
import time

class SimThread:
    def __init__(self):
        self.paused = False
        self._stop  = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop and handle.is_running():
            if not self.paused:
                mujoco.mj_step(model, data)
                handle.sync()
            time.sleep(model.opt.timestep)

    def pause(self):
        self.paused = True
        print("Simulation paused.")

    def resume(self):
        self.paused = False
        print("Simulation resumed.")

    def stop(self):
        self._stop = True

sim = SimThread()

try:
    import IPython
    IPython.embed(
        banner1="",
        banner2=(
            "Physics is running in the background.\n"
            "  sim.pause()   — freeze simulation\n"
            "  sim.resume()  — unfreeze\n"
            "  handle.close() — close viewer\n"
        ),
    )
except ImportError:
    print("IPython not found, dropping into plain Python REPL.")
    import code
    code.interact(local={"model": model, "data": data, "handle": handle, "sim": sim, "mujoco": mujoco, "np": np})
finally:
    sim.stop()
    handle.close()
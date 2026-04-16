# Biorobotics 2026 — MuJoCo Elbow Control Exercise

## Overview

This exercise asks you to implement a biologically-inspired joint-angle controller for a one-degree-of-freedom musculoskeletal elbow model simulated in [MuJoCo](https://mujoco.org/). The model has two antagonistic muscles - a **flexor** and an **extensor** - and your goal is to drive the elbow to a target angle by computing appropriate muscle activations using the **antagonistic muscle ratio** framework described in:

> Y. Honda, F. Miyazaki, and A. Nishikawa, "Angle control of pneumatically-driven musculoskeletal model using antagonistic muscle ratio and antagonistic muscle activity," in *Proceedings of the IEEE International Conference on Robotics and Biomimetics (ROBIO)*, 2010. (`ROBIO2010.pdf` is included in this repo.)

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10.x (exact) |
| [uv](https://docs.astral.sh/uv/) | latest |
| Git | any recent |

Install `uv` if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setup

### 1. Clone and fetch submodules

The muscle/skeleton assets live in a git submodule (`myo_sim`). After cloning/downloading, fetch it:

```bash
git init
git submodule update --init --recursive
```

### 2. Create the virtual environment and install dependencies

```bash
uv sync
```

This reads `pyproject.toml`, creates `.venv/`, and installs all dependencies (MuJoCo, Gymnasium, MyoSuite, Matplotlib, IPython).

---

## Running scripts

### Linux / Windows

```bash
uv run python <script>.py
# or with the venv activated:
python <script>.py
```

### macOS — important: you must use `mjpython`

Any script that opens a MuJoCo viewer window **must be launched with `mjpython`** on macOS, because the standard Python interpreter cannot host an OpenGL window on the main thread. `mjpython` is a thin wrapper installed alongside `mujoco` that handles this correctly.

```bash
uv run mjpython <script>.py
# or, with the venv activated:
mjpython <script>.py
```

#### macOS dylib issue

`uv`'s bundled Python does not ship with the shared library files (`.dylib`) that `mjpython` requires. You have two options — pick whichever is easier:

**Option A — install Python 3.10 via Homebrew to satisfy mjpython's dylib lookup**

Sometimes, simply installing the matching brew Python is sometimes enough for `mjpython` to find the dylib at runtime:

```bash
brew install python@3.10
```

**Option B — install Python 3.10 via Homebrew and point uv at it**

If the above doesn't work you can tell 'uv' to use the brew-installed Python 3.10, which includes the necessary dylibs:

```bash
brew install python@3.10
# Pin uv to use the brew-installed interpreter
uv python pin /opt/homebrew/bin/python3.10
uv sync   # rebuild the venv with the brew interpreter
```
This will still create a virtual environment, but it will use the brew Python as its base (via symbolic links in .venv), which should resolve the dylib issue for `mjpython`.

After either option, verify with:

```bash
uv run mjpython -c "import mujoco; print(mujoco.__version__)"
```

#### No interactive matplotlib on macOS with mjpython

`mjpython` occupies the main thread for the MuJoCo viewer, so **`plt.show()` will not work** — calling it will either hang or raise an error. Always save plots to a file instead:

```python
# Instead of plt.show():
plt.savefig("results.png", dpi=150)
```

The exercise files already do this. Use a non-interactive matplotlib backend at the top of your script:

```python
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
```

---

## Repository layout

```
.
├── mujoco_envs/
│   ├── __init__.py           # Gymnasium env registrations
│   ├── custom_env.py         # ElbowAngleEnv definition
│   └── xml/simple_arm/
│       └── elbow.xml         # MuJoCo model (muscles, joints, geometry)
├── myo_sim/                  # git submodule — MyoHub assets
├── open_mjmodel_interactive.py   # interactive model explorer (see below)
├── run_model.py              # sanity-check: runs the env with random actions
├── ratio_pid_challenge.py    # exercise — challenge version (barebones)
├── ratio_pid_guided.py       # exercise — guided version (with TODOs)
├── ROBIO2010.pdf             # reference paper
└── pyproject.toml
```

---

## The exercise

### The environment — `elbow_angle-v0`

The environment wraps a one-DOF musculoskeletal elbow. At each episode a **target elbow angle** is drawn at random from the range **5° – 130°**.

**Observation vector** (5 values):

| Index | Name | Description |
|---|---|---|
| 0 | `qpos` | Current elbow flexion angle (radians) |
| 1 | `qvel` | Angular velocity × dt |
| 2 | `pose_err` | `target_angle − qpos` (radians) |
| 3 | `act1` | Current flexor muscle activation state |
| 4 | `act2` | Current extensor muscle activation state |

**Action vector** (2 values):

| Index | Muscle | Effect |
|---|---|---|
| 0 | Flexor | Increases joint angle (flexion) |
| 1 | Extensor | Decreases joint angle (extension) |

`env.action_space.low` / `.high` report `[-1, 1]`, but the underlying muscle activations are clipped to `[0, 1]` internally — negative values behave the same as zero. In practice, keep your actions in `[0, 1]`. Equal activations → neutral; higher flexor → joint flexes; higher extensor → joint extends.

**Control timestep:**
```python
dt = env.unwrapped.sim.step_duration * env.unwrapped.frame_skip
```

### Exercise files

Two starter files are provided. Choose the one that matches your preferred level of challenge:

#### `ratio_pid_challenge.py` — challenge version

Contains only the environment setup and a loop that steps with random actions. Everything else is for you to figure out. Explore the codebase, read the paper, and implement the full controller from scratch.

```bash
uv run mjpython ratio_pid_challenge.py   # macOS
uv run python   ratio_pid_challenge.py   # Linux / Windows
```

#### `ratio_pid_guided.py` — guided version (released on demand)

Contains the full environment setup, plotting code, and structured `TODO` comments that walk you through each part of the implementation in order. Fill in the blanks.

```bash
uv run mjpython ratio_pid_guided.py   # macOS
uv run python   ratio_pid_guided.py   # Linux / Windows
```

If you face difficulties in implementing the controller due to unfamiliarity with the environment or the MuJoCo API, we recommend will share the guided version with you which includes most of the implementation details and step-by-step instructions, with only the controller logic and tuning left for you.
This can be a helpful resource to understand how to interact with the environment and implement the controller correctly.

### What to implement

Read **Section III** of `ROBIO2010.pdf` to understand the ratio PID controller for joint controlled via an antagonistic muscle pair. The core ideas are:

1. **Antagonistic muscle ratio** `Ar` — maps a desired angle to a ratio that splits activation between the two muscles.
2. **PID feedback** — compute a control signal `u` from the angle error, then derive `Ar` from `u`.
3. **Individual activations** — as given in the paper (eqs. 3 & 4): `Pe = Ar · Ac` and `Pf = (1 − Ar) · Ac`, where `Pe` is the extensor and `Pf` is the flexor. `Ac` is the total activity (joint stiffness knob; you can keep it fixed at 1.0 to start). Think carefully about whether this mapping applies directly to this simulation, or something needs toe be different here.

---

## Exploring the MuJoCo model interactively (optional)

`open_mjmodel_interactive.py` opens the MuJoCo viewer and drops you into an IPython shell with the model loaded. This is useful for inspecting joint limits, actuator names, and experimenting with `data.ctrl` values in real time.
You don't need to do this to complete the exercise, but it can be a helpful sandbox for understanding of MuJoCo's API.
You won't directly interact with MuJoCo in the main exercise since all MuJoCo calls are handled within the Gymnasium environment, but this is a good opportunity to get familiar with how MuJoCo works under the hood.

```bash
uv run mjpython open_mjmodel_interactive.py 
```

Useful snippets inside the shell:

```python
# Print all joint ranges
for i in range(model.njnt):
    lo, hi = np.degrees(model.jnt_range[i])
    print(model.joint(i).name, lo, hi)

# Manually set muscle activations and step
data.ctrl[:] = [0.8, 0.2]   # [flexor, extensor]
mujoco.mj_step(model, data)
handle.sync()

# Freeze / unfreeze physics
sim.pause()
sim.resume()
```

---

## Exploring the Gymnasium environment interactively (optional)

A great way to get comfortable with the Gymnasium API before writing your controller is to poke at the environment directly in an interactive Python session. You can do this in an **IPython shell** or a **JupyterLab notebook** — both are installed via `uv`.

```python
import gymnasium as gym
import mujoco_envs  # registers elbow_angle-v0

env = gym.make("elbow_angle-v0", render_mode="human")  # omit render_mode to run headless
obs, info = env.reset()
print("Observation space:", env.observation_space)
print("Action space:     ", env.action_space)
print("Initial obs:      ", obs)

# Step with a zero action and inspect the result
obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
print("After one step:   ", obs)

env.close()
```

Try things like `env.action_space.sample()`, inspect `env.unwrapped.sim`, or manually set actions and watch how the observation changes — it is a low-stakes way to build intuition before you implement the controller.

**macOS users — IPython must be launched via `mjpython`; JupyterLab does not.**

Because macOS requires the MuJoCo viewer to run on the main thread, IPython must be started through `mjpython`. JupyterLab is a web server and does not need `mjpython` — use the standard launcher for it on all platforms:

 ```bash
 # IPython shell (macOS — mjpython required for viewer access)
 uv run mjpython IPython

 # JupyterLab (all platforms, including macOS)
 uv run jupyter lab
 ```

> Note: inside a JupyterLab notebook, the MuJoCo interactive viewer may not open (the notebook kernel's main thread is managed by JupyterLab, although we have never tested this ourselves). We recommend using `render_mode=None` when creating the environment in a notebook.

---

## Relevant documentation

| Resource | Link |
|---|---|
| MuJoCo docs | https://mujoco.readthedocs.io |
| MuJoCo Python API | https://mujoco.readthedocs.io/en/stable/python.html |
| MuJoCo XML reference | https://mujoco.readthedocs.io/en/stable/XMLreference.html |
| Gymnasium Env API | https://gymnasium.farama.org/api/env/ |
| MyoSuite docs | https://myosuite.readthedocs.io |
| MyoSuite GitHub | https://github.com/MyoHub/myosuite |
| uv documentation | https://docs.astral.sh/uv/ |
| Reference paper | `ROBIO2010.pdf` (included) |

---

## Acknowledgements

- The initial minimal MuJoCo environment was created by [Jhon Charaja](https://github.com/JhonPool4/mujoco-simple-arm-model.git). This repository is a forked and modified version of that work.
- This README was drafted with the assistance of [Anthropic Claude](https://www.anthropic.com/claude).

---

## A note on AI-assisted work

Code generation tools increase productivity enormously and their use is highly encouraged in day-to-day engineering and research. That said, we ask that you complete this exercise without AI assistance.

Programming is a skill, and so is the ability to quickly understand and navigate a new API, both are essential for your future as an engineer or researcher. AI will generate a large fraction of the code written worldwide.
This is already happening as these lines are written, and academia is no exception. 
But AI is not a substitute for genuine understanding. 
The final research output: the model, the result, the paper, etc. will always need to be inspected, validated, and stood behind by the engineer or scientist putting it into the world. 
That requires brain work that no tool can do for you ... at least for the time being.

To that end, we recommend working through this exercise on your own. 
If you get stuck, please post a question on the course forum, the TAs will be happy to help you work through it.

Wishing you a great experience with Biorobotics 2026.
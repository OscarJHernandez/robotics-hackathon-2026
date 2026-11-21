# Run the simulated JetHexa in MuJoCo

We work on a simulated [Hiwonder JetHexa](https://www.hiwonder.com/products/jethexa) hexapod in
[MuJoCo](https://mujoco.readthedocs.io/). You need no robot and no GPU. It was tested on macOS (Apple Silicon)
with Python 3.12 and MuJoCo 4.0.1.

## Install the two tools

You need [uv](https://docs.astral.sh/uv/) (installs Python and the packages) and [just](https://just.systems/)
(runs the project's commands). You do not need to install Python yourself: uv downloads the right version.

macOS (with [Homebrew](https://brew.sh/)):

```bash
brew install uv just
```

Linux, or macOS without Homebrew:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv tool install rust-just
```

Windows (PowerShell):

```bash
winget install --id=astral-sh.uv -e
```

```bash
winget install --id=Casey.Just -e
```

Open a new terminal afterwards so the commands are found. Check with `uv --version` and `just --version`.

## Set up the project

From the repo root:

```bash
just setup
```

This installs Python 3.12 if needed, creates the virtual environment in `.venv/` with the exact package
versions in `uv.lock`, and downloads the robot's meshes (2.4 MB, from Hiwonder's GitHub repo). Run it again
any time; it only does what is missing.

The meshes make the robot look like the real JetHexa. They are not stored in this repo because Hiwonder
publishes them without a license. Without them everything still runs, with the robot drawn as capsules and
boxes ([sim/jethexa_simple.xml](../sim/jethexa_simple.xml)); the physics is the same.

## Commands

Run `just` on its own to list them.

| Command | What it does |
|---|---|
| `just setup` | First-time setup: Python, `.venv/`, packages and meshes |
| `just sim` | Walk for 5 s without a window and print where the robot ended up |
| `just sim --seconds 10` | The same, for 10 s |
| `just view` | Watch the robot walk in a live 3D window (20 s; `just view 60` for 60 s) |
| `just inspect` | Open the model in MuJoCo's viewer to move the joints by hand |
| `just snapshot` | Save a picture of the robot after walking to `walk.png` (`just snapshot other.png`) |
| `just test` | Run the tests |
| `just sync` | Update `.venv/` after someone changed the packages |
| `just meshes` | Download the meshes again |

## Check that it works

```bash
just sim
```

You should see a line like:

```text
MuJoCo 4.0.1: walked 5.0 s, body at x=0.544 m, y=0.024 m, height=0.080 m
```

Then watch it:

```bash
just view
```

In `just inspect` the robot starts with its legs straight out. To put it in the standing pose, open
"Simulation" on the left and press "Load key". The joint sliders are under "Control" on the right.

## Run your own code

Put your script anywhere in the repo and run it with `uv run`, which uses `.venv/` without activating it:

```bash
uv run python my_script.py
```

On macOS a script that opens a live window (`mujoco.viewer.launch_passive`) must run under `mjpython`:
`uv run mjpython my_script.py`.

Add a package with `uv add <name>` (for example `uv add pandas`). This updates `pyproject.toml` and `uv.lock`;
commit both so teammates get it with `just sync`. Do not use `pip install`.

To use the environment in an editor, point it at `.venv/bin/python` (`.venv\Scripts\python.exe` on Windows).

## Use it from Python

Add `sim/` to the import path, then:

```python
import sys
sys.path.insert(0, "sim")

from jethexa_sim import JetHexaSim, tripod_targets

sim = JetHexaSim()                       # starts in the standing pose
for _ in range(250):                     # 250 control steps of 20 ms = 5 s
    sim.set_targets(tripod_targets(sim.stand, sim.time))
    sim.step(0.02)

print(sim.body_position())               # (x, y, z) in metres, +x is forward
print(sim.joint_positions())             # 18 joint angles in radians
rgb = sim.render()                       # robot camera, uint8 array (360, 640, 3)
depth = sim.render(depth=True)           # metres, float array (360, 640)
```

To steer, pass `turn` to the gait: `tripod_targets(sim.stand, t, turn=0.1)` turns left, a negative value
turns right, and `swing=0.0` with a `turn` turns on the spot. `walk(sim, seconds, turn=0.1)` does the same.

More readings, all taken at the current simulation time:

```python
sim.body_yaw()          # heading in radians; 0 faces +x, grows when turning left
sim.imu()               # {"gyro": (x, y, z) rad/s, "accel": (x, y, z) m/s^2, "roll": rad, "pitch": rad}
sim.foot_contacts()     # {leg: force of the floor on the foot in newtons}, 0 in the air
sim.joint_velocities()  # {joint: rad/s}   (the real servos do not report this)
sim.actuator_forces()   # {joint: N m}     (the real servos do not report this)
sim.set_target(1.0, 0.5)  # move the red target pillar to x=1.0 m, y=0.5 m
sim.target()            # (x, y) of the pillar
```

`sim.model` and `sim.data` are the plain MuJoCo `MjModel` and `MjData`, for anything else.

## What is in the model

| Part | Detail |
|---|---|
| File | [sim/jethexa.xml](../sim/jethexa.xml) (with meshes), [sim/jethexa_simple.xml](../sim/jethexa_simple.xml) (fallback) |
| Joints | 18 hinge joints: `coxa_joint_<leg>`, `femur_joint_<leg>`, `tibia_joint_<leg>` |
| Legs | `LF`, `LM`, `LR`, `RF`, `RM`, `RR` (left/right, front/middle/rear) |
| Actuators | 18 position servos named `act_<joint>`; the control value is the target angle in radians |
| Base | Free: the robot can walk and fall over |
| Camera | `eye` on the body, looking forward, 640 x 360 |
| IMU | Sensors `imu_gyro`, `imu_accel`, `imu_quat` at the body centre (`body_site`) |
| Target marker | Body `target`: a red pillar, 0.3 m tall, that the camera sees. No mass, no collisions; the robot walks through it. Starts at x=1 m |
| Timestep | 2 ms (500 Hz); the demo sends new targets every 20 ms (50 Hz) |
| Start pose | Keyframe `stand` |
| Looks | Meshes in geom groups 1 and 2, no mass and no collisions |
| Collisions and mass | Capsules, a box and foot spheres in geom group 3, hidden by default |

## Limits to know about

- **The physics is not calibrated.** Masses, servo strength, damping and friction are placeholders. Motion looks
  plausible, but do not read forces or speeds as the real robot's.
- **The meshes are for looks only.** Contacts use the hidden capsules and foot spheres, so a leg can pass
  slightly through something that only its mesh touches. To see the collision shapes in the viewer, turn on
  geom group 3 under "Group enable".
- **The walking gait in `jethexa_sim.py` is a simple demo gait**, not the real robot's.
- **Not simulated yet:** lidar scan (the lidar on top is only a shape), microphones, servo temperature
  and voltage. The IMU has no noise or drift. The camera's 62° horizontal field of view is an assumption.
- **Both model files are generated** by the model generator in Oscar's private `sophy_jethexa` repo. Small edits
  here are fine (make them in both files), but a change to the robot's geometry should be made there and
  the files generated again.

## Problems

- **`just: command not found` or `uv: command not found`:** install them (see the top), then open a new
  terminal.
- **"JetHexa meshes not found, using simple shapes":** run `just meshes`.
- **`just inspect` fails with a missing mesh file:** same fix.
- **No MuJoCo package for your machine during `just setup`:** MuJoCo 4 ships for Apple Silicon Macs, Linux
  (x86-64 and ARM) and Windows (x86-64) only, with Python 3.10 to 3.13. Intel Macs are not supported.
- **The window does not open on macOS, or you get an error about `launch_passive`:** use `just view`, or run
  your script with `uv run mjpython`, not `python`.
- **`ARB_clip_control unavailable` warning when rendering depth on macOS:** harmless.
- **Rendering fails on a Linux machine without a screen:** run with `MUJOCO_GL=egl` (with a GPU) or
  `MUJOCO_GL=osmesa` (without) in front, for example `MUJOCO_GL=egl just sim`. This was not tested here.

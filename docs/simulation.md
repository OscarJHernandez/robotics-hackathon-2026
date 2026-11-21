# Run the simulated JetHexa in MuJoCo

We work on a simulated [Hiwonder JetHexa](https://www.hiwonder.com/products/jethexa) hexapod in
[MuJoCo](https://mujoco.readthedocs.io/). You need no robot and no GPU. It was tested on macOS (Apple Silicon)
with Python 3.12 and MuJoCo 4.0.1.

## Install

From the repo root, with Python 3.10 or newer:

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Then download the robot's meshes (2.4 MB, from Hiwonder's GitHub repo, once):

```bash
python sim/fetch_meshes.py
```

On Windows, activate with `.venv\Scripts\activate` instead. The other commands below assume the environment
is active.

The meshes make the robot look like the real JetHexa. They are not stored in this repo because Hiwonder
publishes them without a license. Without them everything still runs, with the robot drawn as capsules and
boxes ([sim/jethexa_simple.xml](../sim/jethexa_simple.xml)); the physics is the same.

## Check that it works

```bash
python sim/demo.py
```

The robot walks for 5 simulated seconds without opening a window. You should see a line like:

```text
MuJoCo 4.0.1: walked 5.0 s, body at x=0.544 m, y=0.024 m, height=0.080 m
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## Watch the robot

Live 3D window with the robot walking:

```bash
mjpython sim/demo.py --view --seconds 20
```

On macOS this must be `mjpython` (it is installed with MuJoCo); plain `python` fails there. On Linux and Windows
use `python sim/demo.py --view --seconds 20`.

To open the model and move the joints by hand (sliders are under "Control" on the right), use MuJoCo's own
viewer. It works with plain `python` on every system:

```bash
python -m mujoco.viewer --mjcf=sim/jethexa.xml
```

The robot starts with its legs straight out. To put it in the standing pose, open "Simulation" on the left and
press "Load key".

Save a picture without opening a window:

```bash
python sim/demo.py --snapshot walk.png
```

## Use it from Python

Run from the `sim/` folder, or add it to `PYTHONPATH`:

```python
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
- **Not simulated yet:** lidar scan (the lidar on top is only a shape), microphones, IMU, servo temperature
  and voltage. The camera's 62° horizontal field of view is an assumption.
- **Both model files are generated** by the model generator in Oscar's private `sophy_jethexa` repo. Small edits
  here are fine (make them in both files), but a change to the robot's geometry should be made there and
  the files generated again.

## Problems

- **"JetHexa meshes not found, using simple shapes":** run `python sim/fetch_meshes.py`.
- **`python -m mujoco.viewer --mjcf=sim/jethexa.xml` fails with a missing mesh file:** same fix, or open
  `sim/jethexa_simple.xml`.

- **The window does not open on macOS, or you get an error about `launch_passive`:** use `mjpython`, not
  `python`.
- **`ARB_clip_control unavailable` warning when rendering depth on macOS:** harmless.
- **Rendering fails on a Linux machine without a screen:** set `MUJOCO_GL=egl` (with a GPU) or
  `MUJOCO_GL=osmesa` (without) before the command. This was not tested here.

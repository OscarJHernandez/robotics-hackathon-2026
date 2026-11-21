"""Load and drive the simulated JetHexa.

The model is `jethexa.xml` next to this file. It needs the meshes from `fetch_meshes.py`; without them
`jethexa_simple.xml` (same robot drawn with capsules and boxes) is used. `JetHexaSim` loads it, starts in the standing pose and
gives joint targets by name. `tripod_targets` is a simple open-loop walking gait for demos and tests;
it is not the real robot's gait.
"""

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from fetch_meshes import meshes_available

SIM_DIR = Path(__file__).resolve().parent
MESH_MODEL_PATH = SIM_DIR / "jethexa.xml"
SIMPLE_MODEL_PATH = SIM_DIR / "jethexa_simple.xml"

LEGS = ("LF", "LM", "LR", "RF", "RM", "RR")
ROLES = ("coxa", "femur", "tibia")
# The two groups of three legs that alternate in a tripod gait.
TRIPOD_A = ("LF", "RM", "LR")
TRIPOD_B = ("RF", "LM", "RR")

CAMERA = "eye"
CAMERA_SIZE = (640, 360)  # width, height


def joint_name(leg, role):
    return "{}_joint_{}".format(role, leg)


class JetHexaSim:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MESH_MODEL_PATH
            if not meshes_available():
                print("JetHexa meshes not found, using simple shapes. For the full model run: "
                      "python sim/fetch_meshes.py", file=sys.stderr)
                model_path = SIMPLE_MODEL_PATH
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        self._ctrl = {}
        self._qpos = {}
        for leg in LEGS:
            for role in ROLES:
                name = joint_name(leg, role)
                self._ctrl[name] = self.model.actuator("act_" + name).id
                self._qpos[name] = int(self.model.joint(name).qposadr[0])
        self.reset()
        self.stand = dict((name, float(self.data.ctrl[i])) for name, i in self._ctrl.items())

    @property
    def time(self):
        return float(self.data.time)

    @property
    def joint_names(self):
        return list(self._ctrl)

    def reset(self):
        """Back to the standing pose at the origin."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("stand").id)
        mujoco.mj_forward(self.model, self.data)

    def set_targets(self, targets):
        """Set position targets in radians, as {joint name: angle}."""
        for name, angle in targets.items():
            self.data.ctrl[self._ctrl[name]] = angle

    def step(self, seconds):
        """Advance the physics by `seconds` (rounded to whole timesteps)."""
        for _ in range(max(1, round(seconds / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def joint_positions(self):
        return dict((name, float(self.data.qpos[adr])) for name, adr in self._qpos.items())

    def body_position(self):
        """Body centre (x, y, z) in metres. +x is forward."""
        return tuple(float(v) for v in self.data.body("body").xpos)

    def body_upright(self):
        """1.0 when the body is level, 0 on its side, -1 upside down."""
        return float(self.data.body("body").xmat.reshape(3, 3)[2, 2])

    def render(self, depth=False, camera=CAMERA):
        """Picture from a camera: RGB uint8 (360, 640, 3), or depth in metres (360, 640)."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=CAMERA_SIZE[1], width=CAMERA_SIZE[0])
        option = mujoco.MjvOption()
        if camera == CAMERA:
            option.geomgroup[2] = 0  # the camera's own housing would block its view
        self._renderer.update_scene(self.data, camera=camera, scene_option=option)
        if not depth:
            return self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        try:
            return self._renderer.render().copy()
        finally:
            self._renderer.disable_depth_rendering()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def tripod_targets(stand, t, period=1.0, swing=0.25, lift=0.35):
    """Joint targets at time `t` for walking forward with an open-loop tripod gait.

    Each tripod spends half the period on the floor pushing back and half in the air swinging
    forward. `swing` is the coxa amplitude and `lift` the femur lift, both in radians.
    """
    targets = dict(stand)
    for group, offset in ((TRIPOD_A, 0.0), (TRIPOD_B, 0.5)):
        phase = (t / period + offset) % 1.0
        if phase < 0.5:  # on the floor: the foot moves from front to back
            sweep = 1.0 - 4.0 * phase
            raise_leg = 0.0
        else:  # in the air: the foot moves from back to front
            sweep = 4.0 * (phase - 0.5) - 1.0
            raise_leg = math.sin(math.pi * (phase - 0.5) * 2.0)
        for leg in group:
            # A positive coxa angle turns a left foot backward and a right foot forward.
            side = -1.0 if leg.startswith("L") else 1.0
            targets[joint_name(leg, "coxa")] = stand[joint_name(leg, "coxa")] + side * swing * sweep
            targets[joint_name(leg, "femur")] = stand[joint_name(leg, "femur")] + lift * raise_leg
    return targets


def walk(sim, seconds, control_dt=0.02, on_step=None, **gait):
    """Walk forward for `seconds` of simulated time. `on_step(sim)` runs after every control step."""
    start = sim.time
    while sim.time - start < seconds:
        sim.set_targets(tripod_targets(sim.stand, sim.time - start, **gait))
        sim.step(control_dt)
        if on_step is not None:
            on_step(sim)
    return np.array(sim.body_position())

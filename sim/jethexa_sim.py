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
        self._qvel = {}
        for leg in LEGS:
            for role in ROLES:
                name = joint_name(leg, role)
                self._ctrl[name] = self.model.actuator("act_" + name).id
                self._qpos[name] = int(self.model.joint(name).qposadr[0])
                self._qvel[name] = int(self.model.joint(name).dofadr[0])
        self._floor = self.model.geom("floor").id
        self._feet = dict((self.model.geom("foot_" + leg).id, leg) for leg in LEGS)
        self._target = int(self.model.body("target").mocapid[0])
        self._lidar = self.model.geom("lidar_geom").id
        self._body = self.model.body("body").id
        m = self.model
        # Geoms a lidar ray passes through: those without collisions (the target pillar, the lidar itself)
        # and every geom of the robot's own body tree (legs included).
        self._lidar_skip = ((m.geom_contype == 0) & (m.geom_conaffinity == 0)) | (
            m.body_rootid[m.geom_bodyid] == m.body_rootid[self._body])
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

    def body_yaw(self):
        """Heading in radians, in (-pi, pi]: 0 faces +x and it grows when the robot turns left."""
        xmat = self.data.body("body").xmat.reshape(3, 3)
        return float(math.atan2(xmat[1, 0], xmat[0, 0]))

    def joint_velocities(self):
        """Joint speeds in rad/s, as {joint name: speed}. The real servos do not report this."""
        return dict((name, float(self.data.qvel[adr])) for name, adr in self._qvel.items())

    def actuator_forces(self):
        """Servo torques in N m, as {joint name: torque}. The real servos do not report this."""
        return dict((name, float(self.data.actuator_force[i])) for name, i in self._ctrl.items())

    def imu(self):
        """IMU at the body centre, in the body's frame.

        Returns {"gyro": (x, y, z) in rad/s, "accel": (x, y, z) in m/s^2 (about +9.81 in z at rest),
        "roll": radians, "pitch": radians}. The values are from the start of the last physics step.
        """
        w, x, y, z = (float(v) for v in self.data.sensor("imu_quat").data)
        return {
            "gyro": tuple(float(v) for v in self.data.sensor("imu_gyro").data),
            "accel": tuple(float(v) for v in self.data.sensor("imu_accel").data),
            "roll": math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            "pitch": math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))),
        }

    def foot_contacts(self):
        """Force of the floor on each foot in newtons, as {leg: force}. 0 when the foot is in the air."""
        forces = dict((leg, 0.0) for leg in LEGS)
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == self._floor:
                leg = self._feet.get(contact.geom2)
            elif contact.geom2 == self._floor:
                leg = self._feet.get(contact.geom1)
            else:
                leg = None
            if leg is not None:
                mujoco.mj_contactForce(self.model, self.data, i, wrench)
                forces[leg] += float(wrench[0])
        return forces

    def lidar(self, rays=36, min_range=0.15, max_range=8.0):
        """Ranges in metres of a 360 degree lidar at the lidar's position: float32 array (rays,).

        Ray 0 points straight ahead (the body's +x), ray i is turned i * 360/rays degrees to the LEFT
        (counter-clockwise seen from above). The scan plane is the body's own horizontal plane (it tilts with the
        body). No hit within max_range gives max_range; hits closer than min_range are reported as min_range.

        The origin is the geom `lidar_geom`. All rays are cast with `mujoco.mj_multiRay` from the body's
        `bodyexclude`; MuJoCo excludes only that one body, not the legs, and it does hit geoms without collisions.
        So a ray that hits a geom of the robot's own tree or a geom with contype and conaffinity 0 (the target
        pillar) is cast again from just behind that hit, until it hits something real or runs out of range.
        """
        m, d = self.model, self.data
        angles = np.arange(rays) * (2.0 * math.pi / rays)
        xmat = d.body(self._body).xmat.reshape(3, 3)
        vec = np.outer(np.cos(angles), xmat[:, 0]) + np.outer(np.sin(angles), xmat[:, 1])
        origin = d.geom_xpos[self._lidar].copy()
        geomid = np.empty(rays, dtype=np.int32)
        dist = np.empty(rays, dtype=np.float64)
        mujoco.mj_multiRay(m, d, origin, vec.ravel(), None, True, self._body, geomid, dist, None, rays, max_range)
        out = np.where(geomid >= 0, dist, max_range)
        for i in np.nonzero((geomid >= 0) & self._lidar_skip[geomid])[0]:
            travelled, gid = float(dist[i]), np.zeros(1, dtype=np.int32)
            while True:
                travelled += 1e-4
                if travelled >= max_range:
                    out[i] = max_range
                    break
                t = mujoco.mj_ray(m, d, origin + vec[i] * travelled, vec[i], None, 1, self._body, gid)
                if gid[0] < 0:
                    out[i] = max_range
                    break
                travelled += t
                if not self._lidar_skip[gid[0]]:
                    out[i] = travelled
                    break
        return np.clip(out, min_range, max_range).astype(np.float32)

    def target(self):
        """The target marker's (x, y) in metres."""
        return tuple(float(v) for v in self.data.mocap_pos[self._target][:2])

    def set_target(self, x, y):
        """Move the target marker (a red pillar without collisions) to (x, y) on the floor."""
        self.data.mocap_pos[self._target] = (x, y, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def render(self, depth=False, camera=CAMERA, segmentation=False):
        """Picture from a camera: RGB uint8 (360, 640, 3), or depth in metres (360, 640).

        With `segmentation=True` it returns the geom id at every pixel instead, int32 (360, 640); -1 where
        there is no geom (the sky). `model.geom(id).name` tells what it is.
        """
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=CAMERA_SIZE[1], width=CAMERA_SIZE[0])
        option = mujoco.MjvOption()
        if camera == CAMERA:
            option.geomgroup[2] = 0  # the camera's own housing would block its view
        self._renderer.update_scene(self.data, camera=camera, scene_option=option)
        if segmentation:
            self._renderer.enable_segmentation_rendering()
            try:
                ids = self._renderer.render()  # (h, w, 2): object id and object type
            finally:
                self._renderer.disable_segmentation_rendering()
            return np.where(ids[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), ids[..., 0], -1).astype(np.int32)
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


def tripod_targets(stand, t, period=1.0, swing=0.25, lift=0.35, turn=0.0):
    """Joint targets at time `t` for walking with an open-loop tripod gait.

    Each tripod spends half the period on the floor pushing back and half in the air swinging
    forward. `swing` is the coxa amplitude and `lift` the femur lift, both in radians. `turn`
    (radians) steers: the left legs swing by `swing - turn` and the right legs by `swing + turn`,
    so a positive `turn` turns the robot left (its yaw grows). With `swing=0` it turns on the spot.
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
            amplitude = swing + side * turn
            targets[joint_name(leg, "coxa")] = stand[joint_name(leg, "coxa")] + side * amplitude * sweep
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

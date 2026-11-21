"""The arena for camera data: a closed 50 m² square room whose furniture, colors and target change per episode.

`Arena` loads `sim/arena.xml` (or `arena_simple.xml` without the meshes), draws random layouts and robot poses,
and turns the simulator's segmentation picture into one class per pixel: floor, wall, obstacle, target.
"""

import colorsys
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))

from fetch_meshes import meshes_available  # noqa: E402
from jethexa_sim import SIM_DIR, JetHexaSim, tripod_targets  # noqa: E402

CLASSES = ("floor", "wall", "obstacle", "target")
FLOOR, WALL, OBSTACLE, TARGET = range(4)
IGNORE = 255  # the robot's own body and legs, and anything else that is none of the four classes

AREA = 50.0  # m²
HALF = math.sqrt(AREA) / 2  # 3.5355 m from the centre to the inside of each wall
ROBOT_RADIUS = 0.30  # metres: the standing robot with its legs, plus a little
PASSAGE = 0.50  # metres: the smallest gap left between two obstacles, so the robot can pass
TARGET_CLEARANCE = 0.15  # metres between the target pillar and any obstacle's footprint
AIMED_SHARE = 0.7  # share of poses that look roughly at the target
AIMED_DIST = (0.4, 3.0)  # metres from the target for those poses
HALF_VIEW = 0.45  # radians: the target stays inside the 62 degree wide picture with some margin
FALLEN = 0.9  # body_upright() below this after a walk: stand the robot up again


def _yaw_quat(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


class Arena:
    def __init__(self, simple=None):
        if simple is None:
            simple = not meshes_available()
        self.sim = JetHexaSim(SIM_DIR / ("arena_simple.xml" if simple else "arena.xml"))
        model = self.sim.model
        # The robot file's floor is 6 x 6 m; make it cover the room. (A second floor geom would break
        # foot_contacts(), which looks for the geom named "floor".)
        model.geom_size[model.geom("floor").id, :2] = HALF + 0.1
        model.light_active[0] = 0  # the robot file's lamp at the origin; the arena has its own even lights
        # A matte floor: the robot file's slightly shiny floor mirrors the red target, and those pixels would be
        # labelled floor in the segmentation picture.
        model.mat_reflectance[model.geom_matid[model.geom("floor").id]] = 0.0
        self._key = model.key("stand").id
        self._target_mocap = int(model.body("target").mocapid[0])
        self.obstacles = []  # one dict per obstacle body: mocap index, geom ids, footprint radius
        lookup = np.full(model.ngeom + 1, IGNORE, dtype=np.uint8)  # last entry: geom id -1 (no geom)
        for g in range(model.ngeom):
            name = model.geom(g).name
            if name == "floor":
                lookup[g] = FLOOR
            elif name.startswith("wall_"):
                lookup[g] = WALL
            elif name.startswith("obs_"):
                lookup[g] = OBSTACLE
            elif name == "target_geom":
                lookup[g] = TARGET
        self._lookup = lookup
        for b in range(model.nbody):
            name = model.body(b).name
            if not name.startswith("obs_"):
                continue
            geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
            radius = max(float(np.hypot(*model.geom_pos[g][:2])) + self._reach(g) for g in geoms)
            self.obstacles.append({"name": name, "mocap": int(model.body_mocapid[b]), "geoms": geoms,
                                   "radius": radius, "shade": [self._brightness(g) for g in geoms]})
        self.layout = None
        mujoco.mj_forward(model, self.sim.data)

    def _reach(self, geom):
        """How far a geom reaches sideways from its own centre (a circle that contains its footprint)."""
        size = self.sim.model.geom_size[geom]
        if self.sim.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
            return float(np.hypot(size[0], size[1]))
        return float(size[0])  # cylinder: its radius

    def _brightness(self, geom):
        return float(self.sim.model.geom_rgba[geom][:3].max())

    # ---- layouts -------------------------------------------------------------------------------------

    def randomize(self, rng, target_clearance=TARGET_CLEARANCE, target_wall=0.3):
        """New furniture positions and colors and a new target. Returns the layout (also in `self.layout`).

        The target is at least `target_clearance` m from every obstacle footprint and `target_wall` m from every wall.
        """
        placed = []
        for obstacle in sorted(self.obstacles, key=lambda o: -o["radius"]):  # big ones first: easier to fit
            limit = HALF - obstacle["radius"] - 0.05
            for _ in range(2000):
                x, y = rng.uniform(-limit, limit, size=2)
                if all(math.hypot(x - p["x"], y - p["y"]) > obstacle["radius"] + p["radius"] + PASSAGE
                       for p in placed):
                    break
            else:
                raise RuntimeError("could not place " + obstacle["name"])
            placed.append({"name": obstacle["name"], "x": float(x), "y": float(y),
                           "yaw": float(rng.uniform(-math.pi, math.pi)), "radius": obstacle["radius"],
                           "color": self._random_color(rng)})
        placed.sort(key=lambda p: p["name"])
        layout = {"obstacles": placed, "target": None}
        self.layout = layout
        limit = HALF - target_wall
        while True:
            tx, ty = rng.uniform(-limit, limit, size=2)
            if all(math.hypot(tx - o["x"], ty - o["y"]) > o["radius"] + target_clearance for o in placed):
                break
        layout["target"] = (float(tx), float(ty))
        self.apply(layout)
        return layout

    @staticmethod
    def _random_color(rng):
        """A random color that is not red (the target) and not near-white (the walls)."""
        hue = rng.uniform(0.08, 0.92)  # red sits at both ends of the hue circle
        return [float(c) for c in colorsys.hsv_to_rgb(hue, rng.uniform(0.3, 0.9), rng.uniform(0.45, 0.9))]

    def apply(self, layout):
        """Put the furniture and target where `layout` says. Survives sim.reset()."""
        model, data = self.sim.model, self.sim.data
        by_name = {o["name"]: o for o in self.obstacles}
        mpos = model.key_mpos[self._key].reshape(-1, 3)
        mquat = model.key_mquat[self._key].reshape(-1, 4)
        for item in layout["obstacles"]:
            obstacle = by_name[item["name"]]
            mpos[obstacle["mocap"]] = (item["x"], item["y"], 0.0)
            mquat[obstacle["mocap"]] = _yaw_quat(item["yaw"])
            top = max(obstacle["shade"])
            for geom, shade in zip(obstacle["geoms"], obstacle["shade"]):
                model.geom_rgba[geom][:3] = np.array(item["color"]) * (shade / top)  # keep the parts' shading
        mpos[self._target_mocap] = (layout["target"][0], layout["target"][1], 0.0)
        # reset() restores mocap bodies from the keyframe, so the layout is stored there as well.
        data.mocap_pos[:] = mpos
        data.mocap_quat[:] = mquat
        self.layout = layout
        mujoco.mj_forward(model, data)

    def is_free(self, x, y, clearance):
        """True when (x, y) is `clearance` metres clear of every wall and obstacle footprint."""
        if max(abs(x), abs(y)) > HALF - clearance:
            return False
        return all(math.hypot(x - o["x"], y - o["y"]) > o["radius"] + clearance
                   for o in self.layout["obstacles"])

    # ---- robot poses ---------------------------------------------------------------------------------

    def place(self, x, y, yaw):
        """Standing pose at (x, y) facing `yaw`."""
        self.sim.reset()
        self.sim.data.qpos[0], self.sim.data.qpos[1] = x, y
        self.sim.data.qpos[3:7] = _yaw_quat(yaw)
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def sample_pose(self, rng):
        """A robot pose (x, y, yaw) on free floor; most look roughly at the target. Places the robot there."""
        tx, ty = self.layout["target"]
        for _ in range(10000):
            if rng.random() < AIMED_SHARE:
                dist, angle = rng.uniform(*AIMED_DIST), rng.uniform(-math.pi, math.pi)
                x, y = tx + dist * math.cos(angle), ty + dist * math.sin(angle)
                yaw = math.atan2(ty - y, tx - x) + rng.uniform(-HALF_VIEW, HALF_VIEW)
            else:
                x, y = rng.uniform(-HALF, HALF, size=2)
                yaw = rng.uniform(-math.pi, math.pi)
            if self.is_free(x, y, ROBOT_RADIUS):
                pose = (float(x), float(y), float(math.atan2(math.sin(yaw), math.cos(yaw))))
                self.place(*pose)
                return pose
        raise RuntimeError("no free floor for the robot")

    def wander(self, rng, control_dt=0.02):
        """Walk at random for 1 to 3 s from the current pose and stop mid-step. False if the robot fell."""
        seconds = rng.uniform(1.0, 3.0)
        swing, turn = rng.uniform(0.0, 0.3), rng.uniform(-0.25, 0.25)
        start = self.sim.time
        while self.sim.time - start < seconds:
            self.sim.set_targets(tripod_targets(self.sim.stand, self.sim.time - start + 0.25, swing=swing, turn=turn))
            self.sim.step(control_dt)
        return self.sim.body_upright() > FALLEN

    def pose(self):
        x, y, _ = self.sim.body_position()
        return (x, y, self.sim.body_yaw())

    # ---- pictures ------------------------------------------------------------------------------------

    def class_map(self, seg=None):
        """One class per pixel, uint8 (360, 640): FLOOR, WALL, OBSTACLE, TARGET or IGNORE."""
        if seg is None:
            seg = self.sim.render(segmentation=True)
        return self._lookup[seg]  # geom id -1 indexes the last entry, IGNORE

    def close(self):
        self.sim.close()

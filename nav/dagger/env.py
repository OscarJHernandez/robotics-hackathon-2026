"""SimEnv: one JetHexa in the arena, steered with (stride, turn) once per 1 s gait cycle.

A scene id fixes the obstacle layout and the target pillar (the goal); the episode seed fixes the start pose.
`reset()` returns an observation, `step((stride, turn))` returns (observation, info):

    obs  = {"rgb": uint8 (360, 640, 3), "depth": float32 metres (360, 640), "seg": class map or None,
            "robot_pose": (x, y, yaw), "goal_point": (x, y)}
    info = {"collision": bool, "done": bool, "done_reason": "" | "goal" | "collision" | "timeout" | "fell"}

Collision = any contact between a robot geom and a wall or obstacle geom, checked every 20 ms.
"""

import math

import mujoco
import numpy as np

from nav.common import CONTROL_DT, CYCLE, FALLEN, PHASE0, clip_command

from . import REPO  # noqa: F401  (sets sys.path for sim/ and vision/)
from .timing import Timer

from jethexa_sim import tripod_targets  # noqa: E402
from scene import HALF, ROBOT_RADIUS, Arena  # noqa: E402

CLEARANCE_HEIGHT = 0.25  # metres: shapes whose bottom is above this (the table top) do not block the robot
LAYOUT_ATTEMPTS = 20


def scene_layout(arena, scene_id, target_clearance=None, target_wall=None):
    """Put the layout of `scene_id` into `arena` (same layout for the same id). Returns the layout.

    `target_clearance` / `target_wall` (metres from obstacles / walls to the target) default to the scene's own.
    """
    extra = {k: v for k, v in (("target_clearance", target_clearance), ("target_wall", target_wall)) if v is not None}
    for attempt in range(LAYOUT_ATTEMPTS):
        try:
            return arena.randomize(np.random.default_rng([int(scene_id), attempt]), **extra)
        except RuntimeError:  # furniture did not fit; try the next draw
            continue
    raise RuntimeError("no layout for scene {}".format(scene_id))


class SimEnv:
    def __init__(self, scene_id=0, seed=0, arena=None, simple=None, goal_radius=0.25, max_steps=300,
                 min_start_dist=1.0, segmentation=False, expert=None, timer=None, target_clearance=None,
                 target_wall=None):
        self.arena = arena or Arena(simple)
        self.sim = self.arena.sim
        model = self.sim.model
        robot = model.body("body").id
        self._robot_geom = np.array([model.body_rootid[model.geom_bodyid[g]] == robot for g in range(model.ngeom)])
        names = [model.geom(g).name for g in range(model.ngeom)]
        self._solid_geom = np.array([n.startswith(("wall_", "obs_")) for n in names])
        self._solid_ids = np.flatnonzero(self._solid_geom)
        self.goal_radius, self.max_steps, self.min_start_dist = goal_radius, max_steps, min_start_dist
        self.segmentation = segmentation
        self.expert = expert  # used by reset() to make sure the start can reach the goal
        self.timer = timer or Timer()
        self.scene_id, self.seed = scene_id, seed
        self.target_clearance, self.target_wall = target_clearance, target_wall
        self.layout = None
        self._footprints = None

    # ---- episode -------------------------------------------------------------------------------------

    def reset(self, scene_id=None, seed=None):
        if scene_id is not None:
            self.scene_id = scene_id
        if seed is not None:
            self.seed = seed
        with self.timer("reset"):
            self.layout = scene_layout(self.arena, self.scene_id, self.target_clearance, self.target_wall)
            self.goal = tuple(self.layout["target"])
            self._footprints = None
            rng = np.random.default_rng([int(self.scene_id), int(self.seed), 7])
            for _ in range(2000):
                x, y = rng.uniform(-HALF + 0.4, HALF - 0.4, size=2)
                if not self.arena.is_free(x, y, ROBOT_RADIUS):
                    continue
                if math.hypot(x - self.goal[0], y - self.goal[1]) < self.min_start_dist:
                    continue
                if self.expert is not None and not self.expert.reachable(self.state(pose=(x, y, 0.0)), x, y):
                    continue
                break
            else:
                raise RuntimeError("no reachable start in scene {}".format(self.scene_id))
            self.start = (float(x), float(y), float(rng.uniform(-math.pi, math.pi)))
            self.arena.place(*self.start)
            self.sim.set_target(*self.goal)
            self.gait = PHASE0
            self.steps = 0
            self.path_length = 0.0
            self.done = False
        return self.observe()

    def step(self, cmd, on_control=None):
        """One gait cycle with `cmd` = (stride, turn). `on_control(env)` runs after every 20 ms (videos)."""
        stride, turn = clip_command(*cmd)
        sim = self.sim
        collision = fell = reached = False
        last = self.pose()
        with self.timer("physics"):
            for _ in range(int(round(CYCLE / CONTROL_DT))):
                sim.set_targets(tripod_targets(sim.stand, self.gait, swing=stride, turn=turn))
                sim.step(CONTROL_DT)
                self.gait += CONTROL_DT
                if on_control is not None:
                    on_control(self)
                if self._touching():
                    collision = True
                    break
                if sim.body_upright() < FALLEN:
                    fell = True
                    break
                x, y, _ = self.pose()
                if math.hypot(x - self.goal[0], y - self.goal[1]) < self.goal_radius:
                    reached = True
                    break
        now = self.pose()
        self.path_length += math.hypot(now[0] - last[0], now[1] - last[1])
        self.steps += 1
        reason = ("collision" if collision else "fell" if fell else "goal" if reached
                  else "timeout" if self.steps >= self.max_steps else "")
        self.done = bool(reason)
        info = {"collision": collision, "done": self.done, "done_reason": reason, "stride": stride, "turn": turn}
        return (None if self.done else self.observe()), info  # no pictures after the last step

    # ---- readings ------------------------------------------------------------------------------------

    def pose(self):
        x, y, _ = self.sim.body_position()
        return (x, y, self.sim.body_yaw())

    def observe(self):
        with self.timer("render_rgb"):
            rgb = self.sim.render()
        with self.timer("render_depth"):
            depth = self.sim.render(depth=True).astype(np.float32)
        seg = None
        if self.segmentation:
            with self.timer("render_seg"):
                seg = self.arena.class_map()
        return {"rgb": rgb, "depth": depth, "seg": seg, "robot_pose": self.pose(), "goal_point": self.goal}

    def _touching(self):
        data = self.sim.data
        if data.ncon == 0:
            return False
        g1 = data.contact.geom1[:data.ncon]
        g2 = data.contact.geom2[:data.ncon]
        return bool(np.any((self._robot_geom[g1] & self._solid_geom[g2]) | (self._robot_geom[g2] & self._solid_geom[g1])))

    def footprints(self):
        """Floor footprints of every wall and obstacle shape that reaches below CLEARANCE_HEIGHT."""
        if self._footprints is None:
            model, data = self.sim.model, self.sim.data
            mujoco.mj_forward(model, data)
            out = []
            for g in self._solid_ids:
                size = model.geom_size[g]
                pos = data.geom_xpos[g]
                kind = model.geom_type[g]
                half_height = size[2] if kind == mujoco.mjtGeom.mjGEOM_BOX else size[1]
                if pos[2] - half_height > CLEARANCE_HEIGHT:
                    continue
                if kind == mujoco.mjtGeom.mjGEOM_BOX:
                    xmat = data.geom_xmat[g].reshape(3, 3)
                    out.append({"kind": "box", "x": float(pos[0]), "y": float(pos[1]),
                                "yaw": float(math.atan2(xmat[1, 0], xmat[0, 0])),
                                "hx": float(size[0]), "hy": float(size[1])})
                else:
                    out.append({"kind": "circle", "x": float(pos[0]), "y": float(pos[1]), "r": float(size[0])})
            self._footprints = out
        return self._footprints

    def state(self, pose=None):
        """What the expert needs: the true layout (as footprints), the goal and the robot pose."""
        return {"key": (self.scene_id, self.goal), "footprints": self.footprints(), "half": HALF,
                "goal": self.goal, "pose": pose or self.pose()}

    def close(self):
        self.arena.close()

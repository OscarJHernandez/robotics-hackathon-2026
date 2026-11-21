"""Shared pieces for walking the simulated JetHexa from a point a to a point b.

The robot is steered once per gait cycle (1 s) with two numbers: `stride` (how far the legs swing) and
`turn` (left/right difference). One gait cycle is one row of the data table. `Walker` runs one episode:
it places the robot, observes it, runs a cycle with a command and keeps the history the rows need.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))

from jethexa_sim import LEGS, JetHexaSim, tripod_targets  # noqa: E402

CYCLE = 1.0  # seconds per gait cycle = per table row
CONTROL_DT = 0.02  # new joint targets every 20 ms
# Gait clock offset. At this phase both tripods are in the middle of their sweep, so a new command
# starts without a jump in the joint targets.
PHASE0 = 0.25
STRIDE_RANGE = (0.0, 0.3)  # radians of coxa swing
TURN_RANGE = (-0.25, 0.25)  # radians of left/right swing difference; positive turns left
REACHED = 0.10  # metres: the episode succeeds when the body centre is this close to b
MAX_STEP = 0.13  # metres per cycle at full stride (measured)
MAX_HORIZON = 8  # cycles: the longest constant-command stretch used as a training example
FALLEN = 0.5  # body_upright() below this counts as a fall

__all__ = ["LEGS", "JetHexaSim", "Walker", "data_dir", "wrap", "to_body", "relative", "make_goal",
           "clip_command", "git_commit"]


def data_dir():
    """Where all recorded data lives: $NAV_DATA_DIR, else data/nav in the main checkout.

    The main checkout (not a temporary git worktree) is used so data survives across worktrees.
    """
    env = os.environ.get("NAV_DATA_DIR")
    if env:
        path = Path(env)
    else:
        root = REPO
        try:
            common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                    cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
            root = Path(common).parent
        except (OSError, subprocess.CalledProcessError):
            pass
        path = root / "data" / "nav"
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def wrap(angle):
    """Angle in (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def to_body(pose, x, y):
    """Point (x, y) in the frame of `pose` = (x, y, yaw): (forward, left) in metres."""
    px, py, yaw = pose
    dx, dy = x - px, y - py
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def relative(pose, x, y):
    """Distance and bearing (radians, positive = to the left) from `pose` to (x, y)."""
    fwd, left = to_body(pose, x, y)
    return math.hypot(fwd, left), math.atan2(left, fwd)


def make_goal(dist, bearing):
    """The policy's goal columns for a target at `dist` and `bearing`.

    `horizon` is the number of cycles to get there at full speed (at most MAX_HORIZON); a target that
    is farther away is pulled in along the same bearing to what MAX_HORIZON cycles can reach.
    `delta_x` (ahead) and `delta_y` (to the left) are the same pulled-in goal in the body frame, so
    they stay inside the range the hindsight examples cover.
    """
    horizon = int(min(MAX_HORIZON, max(1, math.ceil(dist / MAX_STEP))))
    reach = min(dist, MAX_HORIZON * MAX_STEP)
    return {"goal_dist": reach, "goal_sin": math.sin(bearing), "goal_cos": math.cos(bearing),
            "horizon": float(horizon), "delta_x": reach * math.cos(bearing), "delta_y": reach * math.sin(bearing)}


def clip_command(stride, turn):
    return (float(np.clip(stride, *STRIDE_RANGE)), float(np.clip(turn, *TURN_RANGE)))


def place(sim, x, y, yaw):
    """Standing pose at (x, y) facing `yaw`."""
    sim.reset()
    sim.data.qpos[0], sim.data.qpos[1] = x, y
    sim.data.qpos[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    mujoco.mj_forward(sim.model, sim.data)


class Walker:
    """One episode from `start` = (x, y, yaw) to `target` = (x, y)."""

    def __init__(self, sim, start, target, record_steps=True):
        self.sim = sim
        place(sim, *start)
        sim.set_target(*target)
        self.target = tuple(target)
        self.start = tuple(start)
        self.t0 = sim.time
        self.gait = PHASE0
        self.cycle = 0
        self.poses = [self.pose()]  # pose at the start of every cycle
        imu = sim.imu()
        self.tilts = [(imu["roll"], imu["pitch"])]
        self.prev = (0.0, 0.0)
        self.reached = False
        self.fell = False
        self.path_length = 0.0
        self.steps = []  # (t, x, y, yaw) every control step, for plots
        self.record_steps = record_steps
        if record_steps:
            self.steps.append((0.0,) + self.pose())

    @property
    def done(self):
        return self.reached or self.fell

    def pose(self):
        x, y, _ = self.sim.body_position()
        return (x, y, self.sim.body_yaw())

    def distance(self):
        x, y, _ = self.pose()
        return math.hypot(self.target[0] - x, self.target[1] - y)

    def observe(self):
        """The row for the cycle about to start: time, robot sensors, history, location, target."""
        sim = self.sim
        pose = self.poses[-1]
        row = {"t_rel": sim.time - self.t0, "cycle": self.cycle}
        for name, value in sim.joint_positions().items():
            row["j_pos_" + name.replace("_joint", "")] = value
        for name, value in sim.joint_velocities().items():
            row["j_vel_" + name.replace("_joint", "")] = value
        for name, value in sim.actuator_forces().items():
            row["j_tau_" + name.replace("_joint", "")] = value
        imu = sim.imu()
        for axis, value in zip("xyz", imu["gyro"]):
            row["gyro_" + axis] = value
        for axis, value in zip("xyz", imu["accel"]):
            row["accel_" + axis] = value
        row["roll"], row["pitch"] = imu["roll"], imu["pitch"]
        for leg, force in sim.foot_contacts().items():
            row["foot_" + leg] = force
        row["prev_stride"], row["prev_turn"] = self.prev
        for n in (1, 3):
            before = self.poses[max(0, len(self.poses) - 1 - n)]
            fwd, left = to_body(before, pose[0], pose[1])
            row["fwd_%d" % n], row["left_%d" % n] = fwd, left
            row["dyaw_%d" % n] = wrap(pose[2] - before[2])
        before = self.tilts[max(0, len(self.tilts) - 2)]
        row["droll_1"], row["dpitch_1"] = imu["roll"] - before[0], imu["pitch"] - before[1]
        row["x"], row["y"] = pose[0], pose[1]
        row["sin_yaw"], row["cos_yaw"] = math.sin(pose[2]), math.cos(pose[2])
        dist, bearing = relative(pose, *self.target)
        row["target_x"], row["target_y"] = self.target
        row["target_dist"], row["target_sin"], row["target_cos"] = dist, math.sin(bearing), math.cos(bearing)
        row.update(make_goal(dist, bearing))
        return row

    def run_cycle(self, stride, turn):
        """Walk one gait cycle with a fixed command; stops early when b is reached or the robot falls.

        Returns the outcome columns for the row: the command, the pose afterwards and flags.
        """
        sim = self.sim
        stride, turn = clip_command(stride, turn)
        last = self.pose()
        steps = int(round(CYCLE / CONTROL_DT))
        done_steps = 0
        for _ in range(steps):
            sim.set_targets(tripod_targets(sim.stand, self.gait, swing=stride, turn=turn))
            sim.step(CONTROL_DT)
            self.gait += CONTROL_DT
            done_steps += 1
            now = self.pose()
            self.path_length += math.hypot(now[0] - last[0], now[1] - last[1])
            last = now
            if self.record_steps:
                self.steps.append((sim.time - self.t0,) + now)
            if self.distance() < REACHED:
                self.reached = True
                break
            if sim.body_upright() < FALLEN:
                self.fell = True
                break
        pose = self.pose()
        imu = sim.imu()
        self.poses.append(pose)
        self.tilts.append((imu["roll"], imu["pitch"]))
        self.prev = (stride, turn)
        self.cycle += 1
        return {"stride": stride, "turn": turn, "x_next": pose[0], "y_next": pose[1], "yaw_next": pose[2],
                "dist_next": self.distance(), "partial": done_steps < steps, "reached": self.reached,
                "fell": self.fell}

    def summary(self):
        x, y, _ = self.pose()
        start_dist = math.hypot(self.target[0] - self.start[0], self.target[1] - self.start[1])
        return {"start_x": self.start[0], "start_y": self.start[1], "start_yaw": self.start[2],
                "target_x": self.target[0], "target_y": self.target[1], "start_dist": start_dist,
                "cycles": self.cycle, "seconds": self.sim.time - self.t0, "final_x": x, "final_y": y,
                "final_dist": self.distance(), "reached": self.reached, "fell": self.fell,
                "path_length": self.path_length,
                "directness": (max(0.0, start_dist - REACHED) / self.path_length
                               if self.reached and self.path_length > 0 else 0.0)}

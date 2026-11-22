"""The robot's own map and plan: lidar hits accumulated over the episode, a shortest path to the goal on that map,
and the bearing of the point `lookahead` metres along it. Only what the real robot has is used: the lidar scan, the
robot's pose (odometry) and the goal position. The expert plans the same way on the TRUE furniture; this plans on what
the lidar has seen so far, so it can be wrong about unseen places (treated as free).

The look-ahead ceiling (v08b, the expert's own look-ahead bearing as input) reached 100% goals with 0 collisions where
every non-privileged student plateaued at about 95%: the missing piece is where the path goes, not obstacle sensing.

  lidar_plan_columns(rows)   offline: (n, 2) sin/cos of the planned bearing for stored rows (any order)
  LidarPlanner               drive time: one per robot, reset per episode, step(pose, goal, ranges) -> (sin, cos)
Both use the same code, so the feature is identical at fit time and while driving.

  python -m nav.dagger.lidarmap add --name round0 [--procs 8]    writes data/<name>/rows.lp.parquet
"""

import argparse
import math
import time

import numpy as np
from scipy import ndimage

from .expert import cost_field, downhill_path, nearest_reachable

HALF = 3.5355  # the room's half width; only used to size the grid (the robot does not know the walls' places)
CELL = 0.1
INFLATE = 0.32  # the expert's safety margin
LOOKAHEAD = 0.4
GOAL_RADIUS = 0.25
COLUMNS = ["lp_sin", "lp_cos"]


def _disk(radius_cells):
    r = int(math.ceil(radius_cells))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= radius_cells * radius_cells


class _Grid:
    """Minimal grid with the interface `expert.cost_field` / `downhill_path` use."""

    def __init__(self, blocked, cell=CELL, half=HALF):
        self.blocked, self.cell, self.half = blocked, cell, half
        self.n = blocked.shape[0]
        self.xs = -half + cell * (np.arange(self.n) + 0.5)

    def index(self, x, y):
        i = int(np.clip((x + self.half) // self.cell, 0, self.n - 1))
        j = int(np.clip((y + self.half) // self.cell, 0, self.n - 1))
        return i, j

    def centre(self, i, j):
        return float(self.xs[i]), float(self.xs[j])


class LidarPlanner:
    def __init__(self, rays=36, max_range=8.0, cell=CELL, inflate=INFLATE, lookahead=LOOKAHEAD,
                 goal_radius=GOAL_RADIUS, half=HALF):
        self.rays, self.max_range, self.cell, self.half = rays, max_range, cell, half
        self.inflate, self.lookahead, self.goal_radius = inflate, lookahead, goal_radius
        self.n = int(math.ceil(2 * half / cell)) + 8  # a margin around the room
        self.half_grid = self.n * cell / 2
        self.angles = np.arange(rays) * 2 * math.pi / rays
        self.kernel = _disk(inflate / cell)
        centres = -self.half_grid + cell * (np.arange(self.n) + 0.5)
        self.gx, self.gy = np.meshgrid(centres, centres, indexing="ij")
        self.reset()

    def reset(self):
        self.hits = np.zeros((self.n, self.n), bool)

    def add_scan(self, pose, ranges):
        x, y, yaw = pose
        ranges = np.asarray(ranges, float)
        hit = ranges < self.max_range - 1e-6
        a = yaw + self.angles[hit]
        hx, hy = x + ranges[hit] * np.cos(a), y + ranges[hit] * np.sin(a)
        i = np.clip(((hx + self.half_grid) // self.cell).astype(int), 0, self.n - 1)
        j = np.clip(((hy + self.half_grid) // self.cell).astype(int), 0, self.n - 1)
        self.hits[i, j] = True

    def bearing(self, pose, goal):
        """Bearing (rad, + = left) of the look-ahead point on the plan over the map so far."""
        x, y, yaw = pose
        blocked = ndimage.binary_dilation(self.hits, structure=self.kernel)
        near_goal = np.hypot(self.gx - goal[0], self.gy - goal[1]) < self.goal_radius
        blocked &= ~(near_goal & ~self.hits)
        grid = _Grid(blocked, self.cell, self.half_grid)
        goal_dist = math.hypot(goal[0] - x, goal[1] - y)
        point = goal
        if goal_dist > self.lookahead:
            cost = cost_field(grid, goal)
            cell = grid.index(x, y)
            if not np.isfinite(cost[cell]):
                cell = nearest_reachable(cost, cell, int(round(0.6 / self.cell)))
            if cell is not None:
                path = downhill_path(grid, cost, cell, self.lookahead)
                if cost[path[-1]] > 0:
                    point = grid.centre(*path[-1])
        dx, dy = point[0] - x, point[1] - y
        return math.atan2(-math.sin(yaw) * dx + math.cos(yaw) * dy, math.cos(yaw) * dx + math.sin(yaw) * dy)

    def step(self, pose, goal, ranges):
        self.add_scan(pose, ranges)
        b = self.bearing(pose, goal)
        return math.sin(b), math.cos(b)


def _episode_columns(args):
    """One episode's rows (already in step order): (index array, (n, 2) sin/cos)."""
    index, poses, goals, ranges, rays, max_range = args
    planner = LidarPlanner(rays=rays, max_range=max_range)
    out = np.zeros((len(index), 2), np.float32)
    for k in range(len(index)):
        out[k] = planner.step(poses[k], goals[k], ranges[k])
    return index, out


def lidar_plan_columns(rows, procs=1, max_range=8.0):
    """(n, 2) float32 sin/cos of the planned bearing for stored rows (episode, step, x, y, yaw, goal_x, goal_y, lr_*)."""
    lr = sorted([c for c in rows.columns if c.startswith("lr_")], key=lambda c: int(c[3:]))
    order = np.lexsort((rows["step"].to_numpy(), rows["episode"].to_numpy()))
    ep = rows["episode"].to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    ends = np.r_[starts[1:], len(order)]
    poses = rows[["x", "y", "yaw"]].to_numpy(float)
    goals = rows[["goal_x", "goal_y"]].to_numpy(float)
    ranges = rows[lr].to_numpy(float)
    jobs = []
    for s, e in zip(starts, ends):
        idx = order[s:e]
        jobs.append((idx, poses[idx], goals[idx], ranges[idx], len(lr), max_range))
    out = np.zeros((len(rows), 2), np.float32)
    if procs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(procs) as pool:
            results = pool.map(_episode_columns, jobs, chunksize=4)
    else:
        results = [_episode_columns(j) for j in jobs]
    for idx, cols in results:
        out[idx] = cols
    return out


def main():
    import pandas as pd

    from .config import dataset_dir
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["add"])
    parser.add_argument("--name", required=True, nargs="+")
    parser.add_argument("--procs", type=int, default=8)
    args = parser.parse_args()
    for name in args.name:
        folder = dataset_dir(name)
        lr = ["lr_{}".format(i) for i in range(36)]
        rows = pd.read_parquet(folder / "rows.parquet", columns=["episode", "step", "x", "y", "yaw", "goal_x", "goal_y"] + lr)
        started = time.time()
        cols = lidar_plan_columns(rows, procs=args.procs)
        out = pd.DataFrame({"episode": rows["episode"].to_numpy(), "step": rows["step"].to_numpy(),
                            "lp_sin": cols[:, 0], "lp_cos": cols[:, 1]})
        out.to_parquet(folder / "rows.lp.parquet", index=False)
        print("{}: {} rows in {:.0f} s -> {}".format(name, len(out), time.time() - started, folder / "rows.lp.parquet"),
              flush=True)


if __name__ == "__main__":
    main()

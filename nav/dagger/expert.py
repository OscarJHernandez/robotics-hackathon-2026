"""The expert: a shortest path over the true obstacle layout plus pure pursuit, from any state.

The planner is A* rooted at the goal: one Dijkstra search from the goal cell gives the cost-to-go of every
free cell (A* with a zero heuristic, run once per layout instead of once per step). The shortest path
from any cell is then a walk downhill on that field, so the expert can label any state the student visits
at no extra cost. Pure pursuit steers at a point `lookahead` metres along that path.

Footprints come from the simulator (`SimEnv.footprints()`): every wall or obstacle shape that reaches
below the robot's height, as an oriented box or a circle on the floor.
"""

import heapq
import math

import numpy as np

from nav.common import STRIDE_RANGE, TURN_RANGE, relative

from . import STRIDE

SQRT2 = math.sqrt(2.0)
NEIGHBOURS = [(-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2), (0, -1, 1.0), (0, 1, 1.0), (1, -1, SQRT2),
              (1, 0, 1.0), (1, 1, SQRT2)]


def footprint_distance(footprint, px, py):
    """Distance in metres from points (px, py) to a footprint; 0 inside it.

    A footprint is {"kind": "box", "x", "y", "yaw", "hx", "hy"} or {"kind": "circle", "x", "y", "r"}.
    """
    dx, dy = px - footprint["x"], py - footprint["y"]
    if footprint["kind"] == "circle":
        return np.maximum(np.hypot(dx, dy) - footprint["r"], 0.0)
    c, s = math.cos(footprint["yaw"]), math.sin(footprint["yaw"])
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return np.hypot(np.maximum(np.abs(lx) - footprint["hx"], 0.0), np.maximum(np.abs(ly) - footprint["hy"], 0.0))


class Grid:
    """Occupancy grid over the square [-half, half]^2: True = the robot's centre must not go there."""

    def __init__(self, footprints, half, cell=0.05, inflate=0.32, goal=None, goal_radius=0.25):
        self.half, self.cell = half, cell
        self.n = int(math.ceil(2 * half / cell))
        centres = -half + cell * (np.arange(self.n) + 0.5)
        self.xs = centres
        gx, gy = np.meshgrid(centres, centres, indexing="ij")  # index [i, j] = (x_i, y_j)
        clearance = np.full(gx.shape, np.inf)
        for footprint in footprints:
            clearance = np.minimum(clearance, footprint_distance(footprint, gx, gy))
        self.clearance = clearance
        self.blocked = clearance < inflate
        # Outside the room counts as blocked even without wall footprints.
        edge = np.maximum(np.abs(gx), np.abs(gy)) > half - inflate
        self.blocked |= edge
        if goal is not None:
            # The target may stand closer to furniture than `inflate`: free a small disc around it
            # (but never cells inside a real shape).
            near = np.hypot(gx - goal[0], gy - goal[1]) < goal_radius
            self.blocked &= ~(near & (clearance > 0.02))

    def index(self, x, y):
        i = int(np.clip((x + self.half) // self.cell, 0, self.n - 1))
        j = int(np.clip((y + self.half) // self.cell, 0, self.n - 1))
        return i, j

    def centre(self, i, j):
        return float(self.xs[i]), float(self.xs[j])


def cost_field(grid, goal):
    """Cost-to-go in metres from every cell to the goal over free cells (inf where unreachable)."""
    n = grid.n
    cost = np.full((n, n), np.inf)
    gi, gj = grid.index(*goal)
    cost[gi, gj] = 0.0
    heap = [(0.0, gi, gj)]
    blocked = grid.blocked
    step = grid.cell
    while heap:
        c, i, j = heapq.heappop(heap)
        if c > cost[i, j]:
            continue
        for di, dj, w in NEIGHBOURS:
            a, b = i + di, j + dj
            if 0 <= a < n and 0 <= b < n and not blocked[a, b]:
                if di and dj and (blocked[i + di, j] or blocked[i, j + dj]):
                    continue  # no corner cutting past a blocked cell
                nc = c + w * step
                if nc < cost[a, b]:
                    cost[a, b] = nc
                    heapq.heappush(heap, (nc, a, b))
    return cost


def astar(grid, start, goal):
    """Plain A* from start to goal (x, y): list of (x, y) cell centres, or None. Used by tests and plots."""
    n = grid.n
    si, sj = grid.index(*start)
    gi, gj = grid.index(*goal)
    if grid.blocked[si, sj] or grid.blocked[gi, gj]:
        return None

    def h(i, j):
        di, dj = abs(i - gi), abs(j - gj)
        return grid.cell * (max(di, dj) + (SQRT2 - 1) * min(di, dj))

    best = {(si, sj): 0.0}
    parent = {}
    heap = [(h(si, sj), 0.0, si, sj)]
    while heap:
        _, c, i, j = heapq.heappop(heap)
        if (i, j) == (gi, gj):
            path = [(i, j)]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            return [grid.centre(a, b) for a, b in reversed(path)]
        if c > best.get((i, j), np.inf):
            continue
        for di, dj, w in NEIGHBOURS:
            a, b = i + di, j + dj
            if 0 <= a < n and 0 <= b < n and not grid.blocked[a, b]:
                if di and dj and (grid.blocked[i + di, j] or grid.blocked[i, j + dj]):
                    continue
                nc = c + w * grid.cell
                if nc < best.get((a, b), np.inf):
                    best[(a, b)] = nc
                    parent[(a, b)] = (i, j)
                    heapq.heappush(heap, (nc + h(a, b), nc, a, b))
    return None


def downhill_path(grid, cost, start_cell, length):
    """Follow the cost field from `start_cell` for about `length` metres: list of cells (incl. the start)."""
    path = [start_cell]
    travelled = 0.0
    i, j = start_cell
    n = grid.n
    while travelled < length and cost[i, j] > 0:
        best, nxt = cost[i, j], None
        for di, dj, w in NEIGHBOURS:
            a, b = i + di, j + dj
            if 0 <= a < n and 0 <= b < n and cost[a, b] < best:
                best, nxt = cost[a, b], (a, b, w)
        if nxt is None:
            break
        i, j = nxt[0], nxt[1]
        travelled += nxt[2] * grid.cell
        path.append((i, j))
    return path


def nearest_reachable(cost, cell, radius_cells):
    """The finite-cost cell closest to `cell` within a square of `radius_cells`, or None."""
    i, j = cell
    n = cost.shape[0]
    lo_i, hi_i = max(0, i - radius_cells), min(n, i + radius_cells + 1)
    lo_j, hi_j = max(0, j - radius_cells), min(n, j + radius_cells + 1)
    window = cost[lo_i:hi_i, lo_j:hi_j]
    finite = np.argwhere(np.isfinite(window))
    if len(finite) == 0:
        return None
    d = np.hypot(finite[:, 0] + lo_i - i, finite[:, 1] + lo_j - j)
    k = int(np.argmin(d))
    return int(finite[k, 0] + lo_i), int(finite[k, 1] + lo_j)


def pursuit_command(pose, point, goal_dist, gain=0.6, stop_turning=0.8, goal_radius=0.25):
    """(stride, turn) toward `point`: turn with the heading error, walk when roughly facing it."""
    if goal_dist < goal_radius:
        return 0.0, 0.0
    _, err = relative(pose, *point)
    turn = float(np.clip(gain * err, *TURN_RANGE))
    if abs(err) > stop_turning:
        return 0.0, turn
    stride = STRIDE_RANGE[1] * max(0.0, math.cos(err)) ** 2
    stride *= min(1.0, max(0.4, goal_dist / 0.5))  # slow down for the last half metre
    return float(np.clip(stride, *STRIDE_RANGE)), turn


def clearance(footprints, half, x, y):
    """Distance in metres from (x, y) to the nearest footprint or wall."""
    walls = min(half - abs(x), half - abs(y))
    if not footprints:
        return float(walls)
    return float(min(walls, min(float(footprint_distance(f, x, y)) for f in footprints)))


def recovery_command(state, recover_clearance=0.30, back_stride=-0.12, turn=0.2, cone=45.0, step=0.01,
                     side_clearance=0.31, side_back_stride=-0.06, side_turn=None, converge=0.02):
    """Back away from an edge the robot is too close to, or None.

    Two cases, by the bearing of the nearest furniture or wall (from the gradient of the clearance):
    - ahead (within `cone` degrees) and closer than `recover_clearance`: about to walk into it. Walk
      backwards (`back_stride`) while turning so the front swings away (dead ahead: toward the goal's side).
    - beside (cone to 135 degrees), closer than `side_clearance` and heading into it (a 10 cm step forward
      would cut the clearance by more than `converge`): the middle legs (0.26 m sideways) are about to clip
      it, the most common collision. The plain expert passes edges at 0.29-0.31 m running alongside them, so
      the heading condition keeps its own driving out. Step back slightly (`side_back_stride`) while turning
      away from that side.
    In both, when 10 cm further back is even tighter (something behind) it only turns. Behind (over 135
    degrees): no rule, path following moves away. The plain expert passes corners at about 0.29 to 0.31 m
    side-on (never under 0.25 m), so its own driving does not trigger these; states the student drove into do.
    """
    x, y, yaw = state["pose"]
    fps, half = state["footprints"], state["half"]
    c = clearance(fps, half, x, y)
    if c >= max(recover_clearance, side_clearance):
        return None
    gx = clearance(fps, half, x + step, y) - clearance(fps, half, x - step, y)
    gy = clearance(fps, half, x, y + step) - clearance(fps, half, x, y - step)
    if gx == 0 and gy == 0:
        return None
    toward = math.atan2(-gy, -gx) - yaw  # bearing of the obstacle in the body frame (left positive)
    toward = math.atan2(math.sin(toward), math.cos(toward))
    if abs(toward) <= math.radians(cone):
        if c >= recover_clearance:
            return None
        stride = back_stride
        if abs(toward) < math.radians(15):  # dead ahead: turn toward the goal's side
            _, goal_bearing = relative(state["pose"], *state["goal"])
            side = 1.0 if goal_bearing >= 0 else -1.0
        else:
            side = -1.0 if toward > 0 else 1.0  # obstacle front-left: turn right, and the other way round
    elif abs(toward) <= math.radians(135):
        ahead_c = clearance(fps, half, x + 0.1 * math.cos(yaw), y + 0.1 * math.sin(yaw))
        if c >= side_clearance or c - ahead_c < converge:
            return None  # far enough, or walking alongside / away from it
        stride = side_back_stride
        turn = turn if side_turn is None else side_turn
        side = -1.0 if toward > 0 else 1.0  # obstacle on the left: turn right, and the other way round
    else:
        return None
    if clearance(fps, half, x - 0.1 * math.cos(yaw), y - 0.1 * math.sin(yaw)) < c:
        return 0.0, side * turn  # backing up would get closer to something behind
    return float(np.clip(stride, *STRIDE)), side * turn


class Expert:
    """`act(state) -> (stride, turn)`. After each call `nopath` says whether no path was found and
    `recovering` whether it backed away from an edge (`recovery_command`).

    `state` = {"key": anything identifying the layout and goal, "footprints": [...], "half": room half-width,
    "goal": (x, y), "pose": (x, y, yaw)}. The grid and cost field are cached per key.
    """

    NO_PATH_COMMAND = (0.0, TURN_RANGE[1])  # turn left on the spot and look for a way

    def __init__(self, cell=0.05, inflate=0.32, lookahead=0.4, gain=0.6, stop_turning=0.8, goal_radius=0.25,
                 recover=None):
        self.cell, self.inflate, self.lookahead = cell, inflate, lookahead
        self.gain, self.stop_turning, self.goal_radius = gain, stop_turning, goal_radius
        self.recover = recover  # None = off; else {"clearance", "back_stride", "turn"}
        self._cache = {}
        self.nopath = False
        self.recovering = False

    @classmethod
    def from_config(cls, expert_cfg, goal_radius):
        return cls(cell=expert_cfg["cell"], inflate=expert_cfg["inflate"], lookahead=expert_cfg["lookahead"],
                   gain=expert_cfg["gain"], stop_turning=expert_cfg["stop_turning"], goal_radius=goal_radius,
                   recover=expert_cfg.get("recover"))

    def plan(self, state):
        key = state["key"]
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            grid = Grid(state["footprints"], state["half"], self.cell, self.inflate, state["goal"], self.goal_radius)
            self._cache[key] = (grid, cost_field(grid, state["goal"]))
        return self._cache[key]

    def reachable(self, state, x, y):
        grid, cost = self.plan(state)
        i, j = grid.index(x, y)
        return bool(np.isfinite(cost[i, j]))

    def act(self, state):
        self.recovering = False
        if self.recover:
            r = self.recover
            cmd = recovery_command(state, r.get("clearance", 0.30), r.get("back_stride", -0.12), r.get("turn", 0.2),
                                   r.get("cone", 45.0), side_clearance=r.get("side_clearance", 0.27),
                                   side_back_stride=r.get("side_back_stride", -0.06), side_turn=r.get("side_turn"),
                                   converge=r.get("converge", 0.02))
            if cmd is not None:
                self.recovering, self.nopath = True, False
                return cmd
        grid, cost = self.plan(state)
        pose, goal = state["pose"], state["goal"]
        goal_dist = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
        cell = grid.index(pose[0], pose[1])
        if not np.isfinite(cost[cell]):
            # Inside the safety margin (the student drove too close): head for the nearest free cell first.
            cell = nearest_reachable(cost, cell, int(round(0.6 / grid.cell)))
            if cell is None:
                self.nopath = True
                return self.NO_PATH_COMMAND
        self.nopath = False
        if goal_dist <= self.lookahead:
            point = goal
        else:
            path = downhill_path(grid, cost, cell, self.lookahead)
            point = goal if cost[path[-1]] == 0 else grid.centre(*path[-1])
        return pursuit_command(pose, point, goal_dist, self.gain, self.stop_turning, self.goal_radius)

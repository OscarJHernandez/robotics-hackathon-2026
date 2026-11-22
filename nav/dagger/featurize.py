"""Camera picture + depth + goal -> one feature row for the student.

Per patch (26 x 46 by default, the DINOv2 grid): its class (floor, wall, obstacle, target; 4 = robot's own
leg, only with simulator segmentation) and the mean depth of its pixels. Patch column = direction, depth =
distance, so the obstacles' position relative to the robot is in the row and changes as the robot moves.
A short summary from the same patches: nearest wall or obstacle (distance, bearing) and the free distance
to the left, ahead and right. The goal: distance and bearing from the robot. Nothing here uses the
simulator's true obstacle positions (only the expert does).

Perception is frozen: DINOv2 plus the trained patch probe (vision/probe.py), or with `perception: truth`
the simulator's segmentation picture (faster, for comparisons). `perception: none` uses no picture (depth
camera only); `perception: strip` runs DINOv2 on the horizon strip only and keeps 3 PCA values per cell
(vision/strip_pca.py). The lidar (env.py) adds 36 ranges all around and 360 degree summaries.
"""

import math

import numpy as np

from nav.common import relative

from . import REPO  # noqa: F401
from .timing import Timer

from features import COLS, PATCH, ROWS, extract, load_dino, patch_counts  # noqa: E402  (vision/features.py)
from scene import CLASSES, OBSTACLE, WALL  # noqa: E402

LEG = len(CLASSES)  # class 4: the robot's own body (simulator segmentation only)
FOVY = 37.348871  # degrees, the "eye" camera in both robot files
ASPECT = 640 / 360
HFOV = 2 * math.atan(math.tan(math.radians(FOVY) / 2) * ASPECT)  # about 62 degrees
GOAL = ["goal_dist", "goal_sin", "goal_cos"]
SUMMARY = ["near_dist", "near_bearing", "free_left", "free_ahead", "free_right"]
# From the depth camera alone (no probe): obstacle points 3 to 25 cm above the floor, in metres from the body
# centre. dc_ahead = nearest point in a corridor as wide as the robot straight ahead; dc_left / dc_right =
# nearest in the left / right third of the view; dc_near = nearest anywhere in view.
DEPTH = ["dc_ahead", "dc_left", "dc_right", "dc_near"]
PREV = ["prev_stride", "prev_turn"]
# depth_scan = n: the nearest obstacle point (same height rule) in each of n equal directions across the view,
# ds_0 = leftmost ... ds_<n-1> = rightmost, like a small lidar made from the depth camera.
# Horizontal lidar (env.py SimEnv.lidar): lr_k = range of beam k (k * 10 degrees counter-clockwise from ahead);
# lc_ahead = nearest hit in the robot-wide corridor ahead, lc_left / lc_rear / lc_right = nearest in the
# 90 degree quadrant around 90 / 180 / 270 degrees, lc_near = nearest anywhere.
LIDAR_SUMMARY = ["lc_ahead", "lc_left", "lc_right", "lc_rear", "lc_near"]
OBSTACLE_Z = (0.03, 0.25)  # metres above the floor; above 0.25 (table tops) nothing blocks the robot (env.py)
CORRIDOR = 0.25  # metres: half the robot's width with its middle feet (about 0.2) plus a margin
GOAL_CLEAR = 0.30  # metres around the target pillar that are not obstacles (it has no collisions)


def grid_shape(pool):
    return int(math.ceil(ROWS / pool)), int(math.ceil(COLS / pool))


def scan_columns(n):
    return ["ds_{}".format(k) for k in range(n)]


def lidar_columns(n):
    return ["lr_{}".format(k) for k in range(n)]


def strip_columns(cells, components=3):
    return ["dp_{}_{}".format(c, k) for c in range(cells) for k in range(components)]


def has_classes(perception):
    """Patch classes and the probe summary exist for these perception modes."""
    return perception in ("probe", "truth")


def history_keys(depth_summary=False, perception="probe", lidar=False):
    """Columns repeated for past steps: the probe's obstacle summary (with patch classes), the depth
    distances and the lidar summary."""
    return ((SUMMARY if has_classes(perception) else []) + (DEPTH if depth_summary else [])
            + (LIDAR_SUMMARY if lidar else []))


def columns(pool=1, use_prev_command=False, depth_summary=False, history=0, perception="probe",
            patch_depths=True, depth_scan=0, lidar_beams=0, strip_cells=0):
    """(FEATURE_COLUMNS, CATEGORICAL_COLUMNS) in their fixed order.

    history = k adds the obstacle summary (and the depth distances and lidar summary, if on) of the k
    previous steps as h1_<name> (one step ago) ... hk_<name>. Perception "none" and "strip" have no patch
    classes and no probe summary; "strip" adds 3 PCA values per horizon cell (dp_<cell>_<k>).
    """
    rows, cols = grid_shape(pool)
    vision = has_classes(perception)
    classes = ["pc_{}_{}".format(r, c) for r in range(rows) for c in range(cols)] if vision else []
    depths = ["pd_{}_{}".format(r, c) for r in range(rows) for c in range(cols)] if patch_depths else []
    keys = history_keys(depth_summary, perception, bool(lidar_beams))
    past = ["h{}_{}".format(k, name) for k in range(1, history + 1) for name in keys]
    feature = (classes + depths + GOAL + (SUMMARY if vision else []) + (DEPTH if depth_summary else [])
               + scan_columns(depth_scan) + lidar_columns(lidar_beams) + (LIDAR_SUMMARY if lidar_beams else [])
               + (strip_columns(strip_cells) if perception == "strip" else []) + past
               + (PREV if use_prev_command else []))
    return feature, classes


COLUMN_GROUPS = {  # for student.column_groups: fit on a subset of the recorded columns
    "patch_class": lambda c: c.startswith("pc_"), "patch_depth": lambda c: c.startswith("pd_"),
    "goal": lambda c: c in GOAL, "summary": lambda c: c in SUMMARY, "depth": lambda c: c in DEPTH,
    "scan": lambda c: c.startswith("ds_"), "history": lambda c: c[:1] == "h" and c[1:2].isdigit() and "_" in c,
    "prev": lambda c: c in PREV, "lidar": lambda c: c.startswith("lr_"), "lidar_summary": lambda c: c in LIDAR_SUMMARY,
    "dino_strip": lambda c: c.startswith("dp_"),
}


def select_columns(all_columns, groups=None):
    """The columns of `all_columns` in the named groups (all of them when groups is None), order kept."""
    if not groups:
        return list(all_columns)
    unknown = [g for g in groups if g not in COLUMN_GROUPS]
    if unknown:
        raise ValueError("unknown column groups {}; known: {}".format(unknown, sorted(COLUMN_GROUPS)))
    return [c for c in all_columns if any(COLUMN_GROUPS[g](c) for g in groups)]


FEATURE_COLUMNS, CATEGORICAL_COLUMNS = columns()


def _patch_index(height, width):
    rows = np.arange(ROWS * PATCH) * height // (ROWS * PATCH)
    cols = np.arange(COLS * PATCH) * width // (COLS * PATCH)
    return rows, cols


def patch_depth(depth, cap=10.0):
    """Mean depth per patch (26, 46) of a depth picture (h, w) in metres, capped at `cap`."""
    rows, cols = _patch_index(*depth.shape)
    big = np.minimum(depth, cap)[rows][:, cols].reshape(ROWS, PATCH, COLS, PATCH)
    return big.mean(axis=(1, 3)).astype(np.float32)


def truth_classes(class_map):
    """Majority class per patch from a per-pixel class map; IGNORE (robot) pixels count as class LEG."""
    counts = patch_counts(class_map)  # (26, 46, 5): floor, wall, obstacle, target, ignore
    return counts.argmax(-1).astype(np.int8)  # index 4 = ignore = LEG


def pool_grid(classes, depths, pool):
    """Blocks of pool x pool patches: majority class and mean depth."""
    if pool == 1:
        return classes, depths
    rows, cols = grid_shape(pool)
    out_c = np.zeros((rows, cols), np.int8)
    out_d = np.zeros((rows, cols), np.float32)
    for r in range(rows):
        for c in range(cols):
            block_c = classes[r * pool:(r + 1) * pool, c * pool:(c + 1) * pool]
            out_c[r, c] = np.bincount(block_c.reshape(-1).astype(np.int64), minlength=LEG + 1).argmax()
            out_d[r, c] = depths[r * pool:(r + 1) * pool, c * pool:(c + 1) * pool].mean()
    return out_c, out_d


def column_bearings(cols):
    """Bearing (radians, positive = left) of the centre of every patch column."""
    u = (np.arange(cols) + 0.5) / cols  # 0 at the left edge of the picture
    return -np.arctan((2 * u - 1) * math.tan(HFOV / 2))


def obstacle_summary(classes, depths, cap=10.0):
    """Nearest wall/obstacle patch (distance, bearing) and the free distance left, ahead and right."""
    solid = (classes == WALL) | (classes == OBSTACLE)
    d = np.where(solid, depths, cap)
    bearings = column_bearings(classes.shape[1])
    flat = int(np.argmin(d))
    r, c = divmod(flat, classes.shape[1])
    near = float(d[r, c])
    third = classes.shape[1] / 3
    left = d[:, :int(round(third))].min()
    ahead = d[:, int(round(third)):int(round(2 * third))].min()
    right = d[:, int(round(2 * third)):].min()
    return {"near_dist": near, "near_bearing": float(bearings[c]) if near < cap else 0.0,
            "free_left": float(left), "free_ahead": float(ahead), "free_right": float(right)}


def depth_obstacles(depth, camera, robot_pose, goal=None, cap=10.0, stride=4, scan=0):
    """Obstacle distances from the depth picture alone (see DEPTH): every `stride`-th pixel back-projected
    with the camera's world pose, kept if 3 to 25 cm above the floor and not the target pillar."""
    height, width = depth.shape
    v, u = np.mgrid[stride // 2:height:stride, stride // 2:width:stride]
    d = depth[v, u].astype(np.float64)
    tan_y = math.tan(math.radians(FOVY) / 2)
    tan_x = tan_y * width / height
    local = np.stack([d * (2 * (u + 0.5) / width - 1) * tan_x,  # camera frame: x right, y up, looking along -z
                      d * (1 - 2 * (v + 0.5) / height) * tan_y,
                      -d], axis=-1).reshape(-1, 3)
    world = np.asarray(camera["pos"]) + local @ np.asarray(camera["mat"]).T
    keep = (d.reshape(-1) > 0.01) & (world[:, 2] > OBSTACLE_Z[0]) & (world[:, 2] < OBSTACLE_Z[1])
    if goal is not None:
        keep &= np.hypot(world[:, 0] - goal[0], world[:, 1] - goal[1]) > GOAL_CLEAR
    x, y, yaw = robot_pose
    dx, dy = world[keep, 0] - x, world[keep, 1] - y
    fwd = math.cos(yaw) * dx + math.sin(yaw) * dy
    side = -math.sin(yaw) * dx + math.cos(yaw) * dy  # positive = left
    rng = np.hypot(fwd, side)
    bearing = np.arctan2(side, fwd)

    def nearest(values, mask):
        return float(min(cap, values[mask].min())) if mask.any() else cap

    third = HFOV / 6
    out = {"dc_ahead": nearest(fwd, (np.abs(side) < CORRIDOR) & (fwd > 0)),
           "dc_left": nearest(rng, bearing > third), "dc_right": nearest(rng, bearing < -third),
           "dc_near": nearest(rng, np.ones_like(rng, bool))}
    if scan:
        edges = np.linspace(HFOV / 2, -HFOV / 2, scan + 1)  # left to right
        for k, name in enumerate(scan_columns(scan)):
            out[name] = nearest(rng, (bearing <= edges[k]) & (bearing > edges[k + 1]))
    return out


def lidar_summary(ranges, cap=10.0):
    """LIDAR_SUMMARY from n ranges (beam k at k * 360/n degrees counter-clockwise from ahead)."""
    ranges = np.minimum(np.asarray(ranges, np.float64), cap)
    angles = np.arange(len(ranges)) * (2 * math.pi / len(ranges))
    fwd, side = ranges * np.cos(angles), ranges * np.sin(angles)
    bearing = np.arctan2(side, fwd)
    hit = ranges < cap

    def nearest(values, mask):
        mask = mask & hit
        return float(min(cap, values[mask].min())) if mask.any() else cap

    quarter = math.pi / 4
    return {"lc_ahead": nearest(fwd, (np.abs(side) < CORRIDOR) & (fwd > 0)),
            "lc_left": nearest(ranges, (bearing > quarter) & (bearing <= 3 * quarter)),
            "lc_right": nearest(ranges, (bearing < -quarter) & (bearing >= -3 * quarter)),
            "lc_rear": nearest(ranges, np.abs(bearing) > 3 * quarter),
            "lc_near": nearest(ranges, np.ones_like(hit))}


class Featurizer:
    """Builds feature rows for a batch of observations (one DINOv2 call for the whole batch).

    Per robot it keeps a small state between steps: {"cmd": last executed (stride, turn), "hist": the obstacle
    summaries of the last steps, most recent first, "depth": this step's depth distances}.
    """

    def __init__(self, perception="probe", probe_path=None, device=None, patch_pool=1, depth_cap=10.0,
                 use_prev_command=False, timer=None, depth_summary=False, history=0, patch_depths=True,
                 depth_scan=0, lidar_beams=0, strip_pca=None):
        self.perception = perception
        self.pool, self.cap, self.use_prev = patch_pool, depth_cap, use_prev_command
        self.depth_summary, self.history = depth_summary, history
        self.patch_depths, self.depth_scan, self.lidar_beams = patch_depths, depth_scan, lidar_beams
        self.timer = timer or Timer()
        self.probe = self.dino = self.pca = None
        strip_cells = 0
        if perception == "probe":
            from probe import load_probe  # vision/probe.py
            self.probe = load_probe(probe_path)
            self.dino = load_dino(device)
        elif perception == "strip":
            from strip_pca import load_pca  # vision/strip_pca.py
            self.pca = strip_pca if isinstance(strip_pca, dict) else load_pca(strip_pca)
            strip_cells = int(self.pca["cells"])
            self.dino = load_dino(device) if not self.pca.get("fake") else None
        elif perception not in ("truth", "none"):
            raise ValueError("perception must be 'probe', 'truth', 'strip' (DINOv2 horizon strip + PCA) "
                             "or 'none' (no picture)")
        self.columns, self.categorical = columns(patch_pool, use_prev_command, depth_summary, history, perception,
                                                 patch_depths, depth_scan, lidar_beams, strip_cells)

    @classmethod
    def from_config(cls, features_cfg, vision_dir, device=None, timer=None):
        probe = pca = None
        if features_cfg["perception"] == "probe":
            probe = vision_dir / features_cfg["probe"]
            if not probe.is_file():
                raise SystemExit("no patch probe at {}: run just vision-data, vision-features, vision-probe "
                                 "(or copy data/vision/ from the laptop)".format(probe))
        if features_cfg["perception"] == "strip":
            pca = vision_dir / features_cfg["strip_pca"]
            if not pca.is_file():
                raise SystemExit("no strip PCA at {}: run python vision/strip_pca.py full60".format(pca))
        return cls(features_cfg["perception"], probe, device, features_cfg["patch_pool"], features_cfg["depth_cap"],
                   features_cfg["use_prev_command"], timer, features_cfg.get("depth_summary", False),
                   features_cfg.get("history", 0), features_cfg.get("patch_depths", True),
                   features_cfg.get("depth_scan", 0), features_cfg.get("lidar_beams", 0), pca)

    def patch_classes(self, observations):
        """(n, 26, 46) int8 classes for a batch of observations (all floor without patch classes)."""
        if not has_classes(self.perception):
            return np.zeros((len(observations), ROWS, COLS), np.int8)
        if self.perception == "truth":
            with self.timer("perception"):
                return np.stack([truth_classes(o["seg"]) for o in observations])
        from probe import predict_probs
        with self.timer("perception"):
            feats = extract(self.dino, np.stack([o["rgb"] for o in observations]))
            return predict_probs(self.probe, feats).argmax(-1).astype(np.int8)

    def strip_values(self, observations):
        """(n, cells * 3) DINOv2 PCA values of the horizon strip, or None without perception strip."""
        if self.perception != "strip":
            return None
        from strip_pca import project, strip_cells
        with self.timer("perception"):
            if self.dino is None:  # tests: a fake PCA with precomputed band features in obs["band"]
                tokens = np.stack([o["band"] for o in observations])
            else:
                tokens = extract(self.dino, np.stack([o["rgb"] for o in observations]), band=int(self.pca["band"]))
            values = project(strip_cells(tokens), self.pca)
        return values.reshape(len(observations), -1).astype(np.float32)

    def build_batch(self, observations, prev_states):
        """Feature matrix (n, len(columns)) float32 and the new per-robot states.

        A prev state is None at an episode's first step (the history then repeats the current values), a
        state dict from the last call, or a bare (stride, turn) tuple. Set state["cmd"] after the step.
        """
        classes = self.patch_classes(observations)
        strip = self.strip_values(observations)
        keys = history_keys(self.depth_summary, self.perception, bool(self.lidar_beams))
        vision = has_classes(self.perception)
        with self.timer("featurize"):
            rows, states = [], []
            for i, (obs, cls, prev) in enumerate(zip(observations, classes, prev_states)):
                depths = patch_depth(obs["depth"], self.cap)
                cls, depths = pool_grid(cls, depths, self.pool)
                dist, bearing = relative(obs["robot_pose"], *obs["goal_point"])
                summary = obstacle_summary(cls, depths, self.cap) if vision else {}
                state = prev if isinstance(prev, dict) else {"cmd": prev, "hist": []}
                if self.depth_summary or self.depth_scan or "camera" in obs:
                    summary.update(depth_obstacles(obs["depth"], obs["camera"], obs["robot_pose"],
                                                   obs["goal_point"], self.cap, scan=self.depth_scan))
                tail = [dist, math.sin(bearing), math.cos(bearing)]
                if vision:
                    tail += [summary[k] for k in SUMMARY]
                if self.depth_summary:
                    tail += [summary[k] for k in DEPTH]
                tail += [summary[k] for k in scan_columns(self.depth_scan)]
                if self.lidar_beams:
                    ranges = np.minimum(np.asarray(obs["lidar"], np.float32), self.cap)
                    if len(ranges) != self.lidar_beams:
                        raise ValueError("lidar has {} beams, config says {}".format(len(ranges), self.lidar_beams))
                    summary.update(lidar_summary(ranges, self.cap))
                    tail += ranges.tolist() + [summary[k] for k in LIDAR_SUMMARY]
                if strip is not None:
                    tail += strip[i].tolist()
                current = [summary[k] for k in keys]
                past = state["hist"] or [current] * self.history
                for k in range(self.history):
                    tail += past[min(k, len(past) - 1)]
                if self.use_prev:
                    tail += list(state["cmd"] or (0.0, 0.0))
                parts = ([cls.reshape(-1).astype(np.float32)] if vision else []) + (
                    [depths.reshape(-1)] if self.patch_depths else [])
                rows.append(np.concatenate(parts + [np.asarray(tail, np.float32)]))
                states.append({"cmd": state["cmd"], "hist": ([current] + past)[:self.history],
                               "depth": {k: summary[k] for k in DEPTH if k in summary},
                               "lidar": None if obs.get("lidar") is None else np.asarray(obs["lidar"], np.float32)})
            return np.stack(rows), states


def build_features(obs, prev_state, perception):
    """The spec's single-observation interface: (feature dict, new prev state)."""
    matrix, states = perception.build_batch([obs], [prev_state])
    return dict(zip(perception.columns, matrix[0].tolist())), states[0]

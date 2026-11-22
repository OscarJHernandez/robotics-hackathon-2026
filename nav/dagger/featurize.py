"""Camera picture + depth + goal -> one feature row for the student.

Per patch (26 x 46 by default, the DINOv2 grid): its class (floor, wall, obstacle, target; 4 = robot's own
leg, only with simulator segmentation) and the mean depth of its pixels. Patch column = direction, depth =
distance, so the obstacles' position relative to the robot is in the row and changes as the robot moves.
A short summary from the same patches: nearest wall or obstacle (distance, bearing) and the free distance
to the left, ahead and right. The goal: distance and bearing from the robot. Nothing here uses the
simulator's true obstacle positions (only the expert does).

Perception is frozen: DINOv2 plus the trained patch probe (vision/probe.py), or with `perception: truth`
the simulator's segmentation picture (faster, for comparisons).
"""

import math

import numpy as np
import pandas as pd

from nav.common import relative

from . import REPO  # noqa: F401
from .timing import Timer

from features import COLS, PATCH, ROWS, extract, load_dino, patch_counts  # noqa: E402  (vision/features.py)
from scene import CLASSES, OBSTACLE, TARGET, WALL  # noqa: E402

LEG = len(CLASSES)  # class 4: the robot's own body (simulator segmentation only)
FOVY = 37.348871  # degrees, the "eye" camera in both robot files
ASPECT = 640 / 360
HFOV = 2 * math.atan(math.tan(math.radians(FOVY) / 2) * ASPECT)  # about 62 degrees
GOAL = ["goal_dist", "goal_sin", "goal_cos"]
SUMMARY = ["near_dist", "near_bearing", "free_left", "free_ahead", "free_right"]
PREV = ["prev_stride", "prev_turn"]


def grid_shape(pool):
    return int(math.ceil(ROWS / pool)), int(math.ceil(COLS / pool))


def columns(pool=1, use_prev_command=False):
    """(FEATURE_COLUMNS, CATEGORICAL_COLUMNS) in their fixed order."""
    rows, cols = grid_shape(pool)
    classes = ["pc_{}_{}".format(r, c) for r in range(rows) for c in range(cols)]
    depths = ["pd_{}_{}".format(r, c) for r in range(rows) for c in range(cols)]
    feature = classes + depths + GOAL + SUMMARY + (PREV if use_prev_command else [])
    return feature, classes


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


# ---- derived feature variants: computed from the stored BASE rows, at fit time and at drive time ----------------

BAND_ROWS = (11, 15)  # patch rows of the horizon band (the camera looks level from 15 cm above the floor)
META_BASE = ["x", "y", "yaw", "episode", "step"]  # base columns that are bookkeeping, never features


def lidar_columns(rays=36):
    return ["lr_{}".format(i) for i in range(rays)]


def stored_columns(rays=36):
    """Feature-bearing columns stored with every v2 row: full-resolution grids, goal, summary-3, lidar."""
    pc, pd_ = base_columns()
    return pc + pd_ + GOAL + SUMMARY + lidar_columns(rays)


DEFAULT_VARIANT = {"pool": 1, "pool_rule": "majority", "summary": "3", "history": 0, "patches": True,
                   "categorical": True, "cap": 10.0}
SECTORS = 7


def base_columns():
    """Names of the stored base columns holding the full-resolution grids: (classes, depths)."""
    return (["pc_{}_{}".format(r, c) for r in range(ROWS) for c in range(COLS)],
            ["pd_{}_{}".format(r, c) for r in range(ROWS) for c in range(COLS)])


def _pool_batch(classes, depths, pool, rule):
    """Vectorised pooling of (n, R, C) grids into blocks of pool x pool patches (edge blocks may be smaller)."""
    if pool == 1:
        return classes, depths
    n, rows, cols = classes.shape
    out_r, out_c = -(-rows // pool), -(-cols // pool)
    pr, pc = out_r * pool, out_c * pool
    cls = np.full((n, pr, pc), -1, np.int16)
    cls[:, :rows, :cols] = classes
    dep = np.full((n, pr, pc), np.nan, np.float64)
    dep[:, :rows, :cols] = depths

    def blocks(a):
        return a.reshape(n, out_r, pool, out_c, pool)

    counts = np.stack([(blocks(cls) == k).sum(axis=(2, 4)) for k in range(LEG + 1)], axis=-1)  # (n, r, c, 5)
    if rule == "majority":
        out_cls = counts.argmax(-1)
    elif rule == "any_solid":
        out_cls = np.zeros((n, out_r, out_c), np.int64)  # FLOOR
        out_cls[counts[..., WALL] > 0] = WALL
        out_cls[counts[..., OBSTACLE] > 0] = OBSTACLE
        out_cls[counts[..., TARGET] > 0] = TARGET
    else:
        raise ValueError("pool_rule must be 'majority' or 'any_solid'")
    out_dep = np.nanmean(blocks(dep), axis=(2, 4))
    return out_cls.astype(np.int8), out_dep.astype(np.float32)


def _solid_depth(classes, depths, cap):
    """(n, R, C): depth of wall/obstacle patches, cap everywhere else."""
    solid = (classes == WALL) | (classes == OBSTACLE)
    return np.where(solid, np.minimum(depths, cap), cap)


def _nearest(d, cap):
    """Nearest solid patch of (n, R, C) distances: (near_dist, near_bearing) as (n,) arrays."""
    n, _, cols = d.shape
    flat = d.reshape(n, -1)
    idx = flat.argmin(1)
    near = flat[np.arange(n), idx]
    bearing = column_bearings(cols)[idx % cols]
    return near, np.where(near < cap, bearing, 0.0)


def _summary3(classes, depths, cap):
    d = _solid_depth(classes, depths, cap)
    third = d.shape[2] / 3
    a, b = int(round(third)), int(round(2 * third))
    col_min = d.min(axis=1)
    near, bearing = _nearest(d, cap)
    return {"near_dist": near, "near_bearing": bearing, "free_left": col_min[:, :a].min(1),
            "free_ahead": col_min[:, a:b].min(1), "free_right": col_min[:, b:].min(1)}


def sector_edges(cols, sectors=SECTORS):
    """Column boundaries of `sectors` equal sectors, left to right."""
    return np.round(np.linspace(0, cols, sectors + 1)).astype(int)


def _summary7(classes, depths, cap, goal_sin, goal_cos):
    d = _solid_depth(classes, depths, cap)
    n, _, cols = d.shape
    col_min = d.min(axis=1)
    edges = sector_edges(cols)
    free = np.stack([col_min[:, edges[s]:edges[s + 1]].min(1) for s in range(SECTORS)], axis=1)  # (n, 7)
    bearings = column_bearings(cols)
    centres = np.array([bearings[edges[s]:edges[s + 1]].mean() for s in range(SECTORS)])
    near, near_bearing = _nearest(d, cap)
    goal_bearing = np.arctan2(goal_sin, goal_cos)
    half = HFOV / 2
    inside = np.abs(goal_bearing) < half
    u = (1 - np.tan(np.clip(goal_bearing, -half, half)) / math.tan(half)) / 2
    col = np.clip((u * cols).astype(int), 0, cols - 1)
    sector = np.clip(np.searchsorted(edges, col, side="right") - 1, 0, SECTORS - 1)
    out = {"near_dist": near, "near_bearing": near_bearing}
    for s in range(SECTORS):
        out["free_s{}".format(s)] = free[:, s]
    out["widest_bearing"] = centres[free.argmax(1)]
    out["goal_free"] = np.where(inside, free[np.arange(n), sector], cap)
    return out, free


def derive(base, variant=None):
    """Feature variant from stored base rows -> (float32 matrix (n, k), column names, categorical column names).

    `base`: DataFrame of base columns (pc_*, pd_* at 26 x 46, goal_dist, goal_sin, goal_cos, and for history
    also episode, step, x, y, yaw), rows of many episodes. The same function runs at drive time (see
    `derive_live`), so training and driving use identical features. Variant keys (all optional):
      pool        1 or 2: blocks of pool x pool patches                              (default 1)
      pool_rule   "majority" (most common class) or "any_solid" (target > obstacle > wall > floor
                  if ANY patch of the block has it; LEG counts as nothing)           (default "majority")
      summary     "3": near_dist, near_bearing, free_left/ahead/right, computed on the POOLED grid as today;
                  "7": near_dist, near_bearing, free_s0..free_s6 (left to right), widest_bearing, goal_free,
                  computed on the full-resolution grid                               (default "3")
      history     k >= 0: for j = 1..k the row j steps back: its free distances (h{j}_free_*), goal sin/cos
                  (h{j}_goal_sin/cos) and the odometry since then (odo_fwd_j, odo_left_j, odo_dyaw_j) in the
                  robot frame of the earlier pose; same episode only, early steps repeat the first row.
                  Never the previous command.                                        (default 0)
      patches     include the pc_ / pd_ columns (False: summary + goal only)         (default True)
      categorical list the pc_ columns as categorical                                (default True)
      cap         depth cap in metres                                                (default 10)
    """
    v = dict(DEFAULT_VARIANT)
    v.update(variant or {})
    pool, cap, hist = int(v["pool"]), float(v["cap"]), int(v["history"])
    if pool not in (1, 2):
        raise ValueError("pool must be 1 or 2")
    if str(v["summary"]) not in ("3", "7"):
        raise ValueError("summary must be '3' or '7'")
    n = len(base)
    pc_names, pd_names = base_columns()
    classes = base[pc_names].to_numpy().reshape(n, ROWS, COLS)
    depths = base[pd_names].to_numpy().astype(np.float32).reshape(n, ROWS, COLS)
    goal = base[GOAL].to_numpy(np.float32)
    pooled_c, pooled_d = _pool_batch(classes, depths, pool, v["pool_rule"])

    names, parts, categorical = [], [], []
    if v["patches"]:
        rows, cols = pooled_c.shape[1:]
        pcn = ["pc_{}_{}".format(r, c) for r in range(rows) for c in range(cols)]
        pdn = ["pd_{}_{}".format(r, c) for r in range(rows) for c in range(cols)]
        names += pcn + pdn
        parts += [pooled_c.reshape(n, -1).astype(np.float32), pooled_d.reshape(n, -1)]
        if v["categorical"]:
            categorical = pcn
    names += GOAL
    parts.append(goal)

    if str(v["summary"]) == "3":
        summ = _summary3(pooled_c, pooled_d, cap)
        dist_names = ["free_left", "free_ahead", "free_right"]
        dist = np.stack([summ[k] for k in dist_names], axis=1)
    else:
        summ, dist = _summary7(classes, depths, cap, goal[:, 1], goal[:, 2])
        dist_names = ["free_s{}".format(s) for s in range(SECTORS)]
    summ_names = list(summ)
    names += summ_names
    parts.append(np.stack([summ[k] for k in summ_names], axis=1).astype(np.float32))

    if hist > 0:
        order = np.lexsort((base["step"].to_numpy(), base["episode"].to_numpy()))
        ep = base["episode"].to_numpy()[order]
        pos = np.arange(n)
        first = np.maximum.accumulate(np.where(np.r_[True, ep[1:] != ep[:-1]], pos, 0))  # sorted index of ep start
        inv = np.empty(n, np.int64)
        inv[order] = pos
        x, y, yaw = (base[k].to_numpy(np.float64)[order] for k in ("x", "y", "yaw"))
        dist_s, goal_s = dist[order], goal[order]
        for j in range(1, hist + 1):
            prev = np.maximum(pos - j, first)
            c, s = np.cos(yaw[prev]), np.sin(yaw[prev])
            dx, dy = x - x[prev], y - y[prev]
            odo = np.stack([c * dx + s * dy, -s * dx + c * dy,
                            np.arctan2(np.sin(yaw - yaw[prev]), np.cos(yaw - yaw[prev]))], axis=1)
            block = np.concatenate([dist_s[prev], goal_s[prev][:, 1:3], odo], axis=1)
            names += (["h{}_{}".format(j, m) for m in dist_names]
                      + ["h{}_goal_sin".format(j), "h{}_goal_cos".format(j), "odo_fwd_{}".format(j),
                         "odo_left_{}".format(j), "odo_dyaw_{}".format(j)])
            parts.append(block[inv].astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32), names, categorical


def drop_constant(matrix, columns, threshold):
    """Boolean keep-mask: False for columns whose most common value covers more than `threshold` of the rows.

    Use at fit time; store the kept column names and apply the same selection at predict time.
    """
    matrix = np.asarray(matrix)
    if matrix.shape[1] != len(columns):
        raise ValueError("columns do not match the matrix")
    keep = np.ones(matrix.shape[1], bool)
    if matrix.shape[0] == 0:
        return keep
    for j in range(matrix.shape[1]):
        _, counts = np.unique(matrix[:, j], return_counts=True)
        keep[j] = counts.max() / matrix.shape[0] <= threshold
    return keep


def derive_live(base, prev_states, variant=None):
    """Drive-time `derive` for one row per robot.

    `base`: DataFrame from `Featurizer.base_rows` (one row per robot, in order). `prev_states[i]`: None at the
    start of an episode, else the DataFrame returned for robot i by the previous call (its last `history` base
    rows). Returns (matrix (n, k), columns, categorical, new_prev_states). History is read only from the
    robot's own earlier base rows, so a robot never sees another robot's or episode's rows.
    """
    hist = int((variant or {}).get("history", 0))
    stacked, new_prev = [], []
    for i in range(len(base)):
        cur = base.iloc[[i]]
        old = prev_states[i] if prev_states[i] is not None else cur.iloc[:0]
        rows = pd.concat([old, cur], ignore_index=True)
        rows["episode"] = i
        rows["step"] = np.arange(len(rows))
        stacked.append(rows)
        new_prev.append(rows.tail(hist).reset_index(drop=True) if hist else None)
    full = pd.concat(stacked, ignore_index=True)
    matrix, names, categorical = derive(full, variant)
    last = full.groupby("episode").tail(1).index.to_numpy()  # last row of every robot, in robot order
    return matrix[last], names, categorical, new_prev


class Featurizer:
    """Builds feature rows for a batch of observations (one DINOv2 call for the whole batch)."""

    def __init__(self, perception="probe", probe_path=None, device=None, patch_pool=1, depth_cap=10.0,
                 use_prev_command=False, timer=None):
        self.perception = perception
        self.pool, self.cap, self.use_prev = patch_pool, depth_cap, use_prev_command
        self.columns, self.categorical = columns(patch_pool, use_prev_command)
        self.timer = timer or Timer()
        self.probe = self.dino = None
        self.last_band = None  # set by patch_classes() with probe perception
        if perception == "probe":
            from probe import load_probe  # vision/probe.py
            self.probe = load_probe(probe_path)
            self.dino = load_dino(device)
        elif perception != "truth":
            raise ValueError("perception must be 'probe' or 'truth'")

    @classmethod
    def from_config(cls, features_cfg, vision_dir, device=None, timer=None):
        probe = None
        if features_cfg["perception"] == "probe":
            probe = vision_dir / features_cfg["probe"]
            if not probe.is_file():
                raise SystemExit("no patch probe at {}: run just vision-data, vision-features, vision-probe "
                                 "(or copy data/vision/ from the laptop)".format(probe))
            expected = features_cfg.get("probe_sha256")
            if expected:
                import hashlib
                digest = hashlib.sha256()
                with open(probe, "rb") as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise SystemExit("patch probe {} has SHA-256 {} but dagger.yaml pins {}: wrong or damaged "
                                     "copy; copy it again (nav.dagger.transfer) or update features.probe_sha256"
                                     .format(probe, digest.hexdigest(), expected))
        return cls(features_cfg["perception"], probe, device, features_cfg["patch_pool"], features_cfg["depth_cap"],
                   features_cfg["use_prev_command"], timer)

    def patch_classes(self, observations):
        """(n, 26, 46) int8 classes for a batch of observations."""
        if self.perception == "truth":
            with self.timer("perception"):
                return np.stack([truth_classes(o["seg"]) for o in observations])
        from probe import predict_probs
        with self.timer("perception"):
            feats = extract(self.dino, np.stack([o["rgb"] for o in observations]))
            # Horizon band: mean over patch rows BAND_ROWS for each of the 46 patch columns, (n, 46, 384) float16
            self.last_band = feats[:, BAND_ROWS[0]:BAND_ROWS[1]].mean(axis=1).astype(np.float16)
            return predict_probs(self.probe, feats).argmax(-1).astype(np.int8)

    def build_batch(self, observations, prev_states):
        """Feature matrix (n, len(columns)) float32 and the new prev states (the commands are set later)."""
        classes = self.patch_classes(observations)
        with self.timer("featurize"):
            rows = []
            for obs, cls, prev in zip(observations, classes, prev_states):
                depths = patch_depth(obs["depth"], self.cap)
                cls, depths = pool_grid(cls, depths, self.pool)
                dist, bearing = relative(obs["robot_pose"], *obs["goal_point"])
                summary = obstacle_summary(cls, depths, self.cap)
                tail = [dist, math.sin(bearing), math.cos(bearing)] + [summary[k] for k in SUMMARY]
                if self.use_prev:
                    tail += list(prev or (0.0, 0.0))
                rows.append(np.concatenate([cls.reshape(-1).astype(np.float32), depths.reshape(-1),
                                            np.asarray(tail, np.float32)]))
            return np.stack(rows), list(prev_states)

    def base_rows(self, observations, prev_states):
        """Base rows for live observations: DataFrame with the full-resolution pc_/pd_ grids, goal_dist/sin/cos
        and x, y, yaw (one row per observation), plus episode (= robot index) and step (= rows seen so far).

        `prev_states[i]` is None at an episode start, else the last `history` base rows of robot i (a DataFrame,
        as returned by `derive_live`). Drive-time history: call `derive_live(base, prev_states, variant)`; it
        stacks each robot's stored rows with the current one, runs `derive` (the same code as at fit time) and
        returns the features of the current rows plus the new prev_states. `build_batch` is unchanged.
        """
        classes = self.patch_classes(observations)
        with self.timer("featurize"):
            n = len(observations)
            depths = np.stack([patch_depth(o["depth"], self.cap) for o in observations])
            pc_names, pd_names = base_columns()
            frame = pd.concat([pd.DataFrame(classes.reshape(n, -1).astype(np.int8), columns=pc_names),
                               pd.DataFrame(depths.reshape(n, -1), columns=pd_names)], axis=1)
            extra = []
            for k, obs in enumerate(observations):
                dist, bearing = relative(obs["robot_pose"], *obs["goal_point"])
                x, y, yaw = obs["robot_pose"]
                step = 0 if prev_states[k] is None else int(prev_states[k]["step"].iloc[-1]) + 1
                extra.append({"goal_dist": dist, "goal_sin": math.sin(bearing), "goal_cos": math.cos(bearing),
                              "x": x, "y": y, "yaw": yaw, "episode": k, "step": step})
            summ = _summary3(classes, depths, self.cap)
            parts = [frame, pd.DataFrame(extra), pd.DataFrame({k: np.asarray(v, np.float32) for k, v in summ.items()})]
            if observations and observations[0].get("lidar") is not None:
                lidar = np.stack([o["lidar"] for o in observations]).astype(np.float32)
                parts.append(pd.DataFrame(lidar, columns=lidar_columns(lidar.shape[1])))
            return pd.concat(parts, axis=1)


def build_features(obs, prev_state, perception):
    """The spec's single-observation interface: (feature dict, new prev state)."""
    matrix, states = perception.build_batch([obs], [prev_state])
    return dict(zip(perception.columns, matrix[0].tolist())), states[0]


# ---- v2 feature pipeline: one object turns stored base rows into a student's inputs, at fit and at drive time ----

DEFAULT_FEATURES = {
    "patches": True, "pool": 1, "pool_rule": "majority", "summary": "3", "history": 0, "categorical": True,
    "cap": 10.0, "drop_constant": None,
    "scan": 0,            # k > 0: nearest wall/obstacle depth (probe classes) in k sectors of the horizon band
    "band_depth": 0,      # k > 0: nearest depth of anything in k sectors of the horizon band (no classes)
    "band_pca": 0,        # k > 0: first k PCA components of the DINOv2 features of every band sector
    "band_sectors": 15,   # sectors for band_pca
    "lidar": False,       # the 360 degree lidar ranges lr_*
    "lookahead": False,   # PRIVILEGED: the expert's look-ahead bearing (sin, cos); ceiling experiments only
    "lidar_plan": False,  # the robot's own plan on its lidar map: look-ahead bearing (sin, cos), see lidarmap.py
}


def _sector_min(values, sectors):
    """(n, rows, cols) -> (n, sectors): minimum over rows and over the columns of each sector."""
    edges = sector_edges(values.shape[2], sectors)
    col_min = values.min(axis=1)
    return np.stack([col_min[:, edges[s]:edges[s + 1]].min(axis=1) for s in range(sectors)], axis=1)


def band_sector_means(band, sectors):
    """(n, 46, d) horizon-band features -> (n, sectors, d) means per sector."""
    edges = sector_edges(band.shape[1], sectors)
    return np.stack([band[:, edges[s]:edges[s + 1]].astype(np.float32).mean(axis=1) for s in range(sectors)], axis=1)


class Features:
    """Stored base rows (+ the band side array) -> the student's input matrix.

    `fit_transform` (context rows only) fixes everything that is learnt from data: the PCA of the band features and
    the constant-column mask. `transform` (tables of many episodes) and `live` (one row per robot while driving)
    then apply exactly that, so the columns at fit time and at drive time are the same by construction; `check`
    raises if they are not. No TabPFN weights are involved: these are transforms of the inputs.
    """

    def __init__(self, cfg=None):
        unknown = set(cfg or {}) - set(DEFAULT_FEATURES)
        if unknown:
            raise ValueError("unknown feature keys: {}".format(sorted(unknown)))
        self.cfg = dict(DEFAULT_FEATURES)
        self.cfg.update(cfg or {})
        if str(self.cfg["summary"]) not in ("3", "7", "none"):
            raise ValueError("summary must be '3', '7' or 'none'")
        self.columns = self.categorical = None
        self.keep = None  # column mask from drop_constant
        self.pca = None  # (mean, scale, components) of the band features
        self.needs_band = bool(self.cfg["band_pca"])

    # -- pieces --
    def _variant(self):
        c = self.cfg
        return {"pool": c["pool"], "pool_rule": c["pool_rule"], "summary": "3" if str(c["summary"]) == "none" else c["summary"],
                "history": c["history"], "patches": c["patches"], "categorical": c["categorical"], "cap": c["cap"]}

    def _extras(self, base, band):
        """Blocks that depend only on the current row: (matrix, names)."""
        c, n = self.cfg, len(base)
        names, parts = [], []
        if c["scan"] or c["band_depth"]:
            pc_names, pd_names = base_columns()
            classes = base[pc_names].to_numpy().reshape(n, ROWS, COLS)[:, BAND_ROWS[0]:BAND_ROWS[1]]
            depths = np.minimum(base[pd_names].to_numpy(np.float32).reshape(n, ROWS, COLS)[:, BAND_ROWS[0]:BAND_ROWS[1]],
                                c["cap"])
            if c["scan"]:
                parts.append(_sector_min(_solid_depth(classes, depths, c["cap"]), int(c["scan"])))
                names += ["scan_{}".format(i) for i in range(int(c["scan"]))]
            if c["band_depth"]:
                parts.append(_sector_min(depths, int(c["band_depth"])))
                names += ["bd_{}".format(i) for i in range(int(c["band_depth"]))]
        if c["lidar"]:
            cols = [k for k in base.columns if k.startswith("lr_")]
            if not cols:
                raise ValueError("lidar features asked for, but the rows have no lr_ columns")
            parts.append(np.minimum(base[cols].to_numpy(np.float32), c["cap"]))
            names += cols
        if c["band_pca"]:
            if band is None:
                raise ValueError("band_pca features need the horizon-band side array")
            k, sectors = int(c["band_pca"]), int(c["band_sectors"])
            means = band_sector_means(np.asarray(band), sectors)  # (n, S, d)
            flat = means.reshape(-1, means.shape[-1])
            if self.pca is None:
                mean = flat.mean(axis=0)
                scale = flat.std(axis=0)
                scale = np.where(scale > 1e-6, scale, 1.0)
                z = (flat - mean) / scale
                # principal directions from the covariance (d x d): cheap and exact for d = 384
                cov = (z.T @ z) / max(len(z) - 1, 1)
                values, vectors = np.linalg.eigh(cov.astype(np.float64))
                order = np.argsort(values)[::-1][:k]
                self.pca = (mean.astype(np.float32), scale.astype(np.float32), vectors[:, order].astype(np.float32),
                            (values[order] / values.sum()).tolist())
            mean, scale, comps = self.pca[:3]
            parts.append((((flat - mean) / scale) @ comps).reshape(n, sectors * k))
            names += ["bp_{}_{}".format(s, j) for s in range(sectors) for j in range(k)]
        if c["lidar_plan"]:
            if "lp_sin" not in base.columns:
                raise ValueError("lidar_plan features need lp_sin/lp_cos: run `python -m nav.dagger.lidarmap add`")
            parts.append(base[["lp_sin", "lp_cos"]].to_numpy(np.float32))
            names += ["lp_sin", "lp_cos"]
        if c["lookahead"]:
            look = base["look_bearing"].to_numpy(np.float32)
            parts.append(np.stack([np.sin(look), np.cos(look)], axis=1))
            names += ["look_sin", "look_cos"]
        if not parts:
            return np.zeros((n, 0), np.float32), []
        return np.concatenate(parts, axis=1).astype(np.float32), names

    def _assemble(self, derived, names, categorical, extras, extra_names, fit):
        if str(self.cfg["summary"]) == "none":
            drop = set(SUMMARY) | {"near_dist", "near_bearing"}
            keep = [i for i, name in enumerate(names) if name not in drop]
            derived, names = derived[:, keep], [names[i] for i in keep]
        matrix = np.concatenate([derived, extras], axis=1).astype(np.float32)
        names = list(names) + list(extra_names)
        if fit:
            threshold = self.cfg["drop_constant"]
            self.keep = (drop_constant(matrix, names, threshold) if threshold else np.ones(len(names), bool))
            for i, name in enumerate(names):  # the goal columns are never dropped
                if name in GOAL:
                    self.keep[i] = True
            self.columns = [name for name, k in zip(names, self.keep) if k]
            self.categorical = [name for name in categorical if name in set(self.columns)]
            self._all = names
        elif names != self._all:
            raise ValueError("feature columns differ from fit time: {} vs {}".format(len(names), len(self._all)))
        return matrix[:, self.keep]

    # -- public --
    def fit_transform(self, base, band=None):
        derived, names, categorical = derive(base, self._variant())
        self.pca = None
        extras, extra_names = self._extras(base, band)
        return self._assemble(derived, names, categorical, extras, extra_names, fit=True)

    def transform(self, base, band=None):
        self.check_fitted()
        derived, names, categorical = derive(base, self._variant())
        extras, extra_names = self._extras(base, band)
        return self._assemble(derived, names, categorical, extras, extra_names, fit=False)

    def live(self, base, band, prev_states):
        """Drive time: `base` from Featurizer.base_rows (one row per robot). Returns (matrix, new prev states)."""
        self.check_fitted()
        derived, names, categorical, new_prev = derive_live(base, prev_states, self._variant())
        extras, extra_names = self._extras(base, band)
        return self._assemble(derived, names, categorical, extras, extra_names, fit=False), new_prev

    def check_fitted(self):
        if self.columns is None:
            raise RuntimeError("Features.fit_transform has not been called")

    def describe(self):
        return {"columns": len(self.columns or []), "categorical": len(self.categorical or []),
                "dropped": int((~self.keep).sum()) if self.keep is not None else 0,
                "pca_variance": [round(v, 4) for v in self.pca[3]] if self.pca else None, "config": self.cfg}

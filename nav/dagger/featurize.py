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


class Featurizer:
    """Builds feature rows for a batch of observations (one DINOv2 call for the whole batch)."""

    def __init__(self, perception="probe", probe_path=None, device=None, patch_pool=1, depth_cap=10.0,
                 use_prev_command=False, timer=None):
        self.perception = perception
        self.pool, self.cap, self.use_prev = patch_pool, depth_cap, use_prev_command
        self.columns, self.categorical = columns(patch_pool, use_prev_command)
        self.timer = timer or Timer()
        self.probe = self.dino = None
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


def build_features(obs, prev_state, perception):
    """The spec's single-observation interface: (feature dict, new prev state)."""
    matrix, states = perception.build_batch([obs], [prev_state])
    return dict(zip(perception.columns, matrix[0].tolist())), states[0]

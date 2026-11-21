"""TabPFN as a model of the robot's motion, used to plan each step toward b.

Instead of predicting the command directly (policy.py), TabPFN predicts what a command does:

    state (previous command, last motion) + command (stride, turn)  ->  motion over the next cycle
                                                                        (forward m, left m, yaw rad)

Each cycle the planner asks TabPFN about a grid of 45 candidate commands for every robot in one
batch and picks the command whose predicted end pose is closest to b (with a small cost for not
facing b). The examples are plain recorded cycles: no relabelling is needed.

TabPFN is used only through `fit()` and `predict()` (rule in AGENTS.md). Build one with
`just controller --kind planner`.
"""

import math

import numpy as np

from common import STRIDE_RANGE, TURN_RANGE
from policy import fit_regressor, predict

STATE = ["prev_stride", "prev_turn", "fwd_1", "left_1", "dyaw_1"]
COMMAND = ["stride", "turn"]
FEATURES = STATE + COMMAND
MOTION = ["m_fwd", "m_left", "m_dyaw"]
CANDIDATES = [(s, t) for s in np.linspace(*STRIDE_RANGE, 5) for t in np.linspace(*TURN_RANGE, 9)]
FACING_COST = 0.08  # metres of distance one radian of bearing error is worth


def motion_rows(cycles):
    """Examples: every full cycle (not cut short, no fall) with the body-frame motion it produced."""
    full = cycles[~cycles["partial"].astype(bool) & ~cycles["fell"].astype(bool)].copy()
    yaw = np.arctan2(full["sin_yaw"], full["cos_yaw"])
    dx, dy = full["x_next"] - full["x"], full["y_next"] - full["y"]
    full["m_fwd"] = np.cos(yaw) * dx + np.sin(yaw) * dy
    full["m_left"] = -np.sin(yaw) * dx + np.cos(yaw) * dy
    full["m_dyaw"] = np.arctan2(np.sin(full["yaw_next"] - yaw), np.cos(full["yaw_next"] - yaw))
    return full.reset_index(drop=True)


def sample_rows(table, rows, seed=0):
    if len(table) <= rows:
        return table
    return table.sample(n=rows, random_state=seed).sort_index()


class DynamicsPlanner:
    """Picks the candidate command whose TabPFN-predicted end pose is best."""

    kind = "planner"
    name = "tabpfn-planner"
    labels = MOTION

    def __init__(self, context, device="cpu", n_estimators=4, seed=0):
        X = context[FEATURES].to_numpy(np.float32)
        self.models = {label: fit_regressor(X, context[label], device, n_estimators, seed) for label in MOTION}

    def predict_labels(self, table):
        X = table[FEATURES].to_numpy(np.float32)
        return {label: predict(self.models[label], X) for label in MOTION}

    def act(self, rows):
        cands = np.array(CANDIDATES, dtype=np.float32)
        X = np.concatenate([np.hstack([np.tile([[r[c] for c in STATE]], (len(cands), 1)), cands]) for r in rows])
        motion = np.stack([predict(self.models[label], X) for label in MOTION], axis=1)
        motion = motion.reshape(len(rows), len(cands), 3)
        out = []
        for row, m in zip(rows, motion):
            bearing = math.atan2(row["target_sin"], row["target_cos"])
            # b in the current body frame, then relative to each predicted end pose
            tf, tl = row["target_dist"] * math.cos(bearing), row["target_dist"] * math.sin(bearing)
            rf, rl = tf - m[:, 0], tl - m[:, 1]
            c, s = np.cos(m[:, 2]), np.sin(m[:, 2])
            ef, el = c * rf + s * rl, -s * rf + c * rl
            cost = np.hypot(ef, el) + FACING_COST * np.abs(np.arctan2(el, ef))
            best = cands[int(np.argmin(cost))]
            out.append((float(best[0]), float(best[1])))
        return np.array(out)

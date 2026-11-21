"""Turn recorded cycles into a→b training examples for TabPFN.

Every example says: "from this state, to reach a goal at (goal_dist, goal bearing) in `horizon`
cycles, use this (stride, turn)". The examples are derived from the `cycles` table each time and
never stored, so the database always holds only what really happened.

Hindsight relabelling: a stretch of K cycles with one command, starting at cycle t, reached some
point P. Relabel: P is the goal, K the horizon, the command the label.
"""

import math

import pandas as pd

from common import MAX_HORIZON, relative

GOAL = ["goal_dist", "goal_sin", "goal_cos", "horizon"]
HISTORY = ["prev_stride", "prev_turn", "fwd_1", "left_1", "dyaw_1", "fwd_3", "left_3", "dyaw_3",
           "droll_1", "dpitch_1"]
LOCATION = ["x", "y", "sin_yaw", "cos_yaw"]
TIME = ["t_rel", "cycle"]
LABELS = ["stride", "turn"]
# Only the goal by default. "history" (previous command, recent motion) leaks the label in
# random-walk data, where a command is held for several cycles: the model learns "repeat the
# previous command". It stays available as an opt-in group.
DEFAULT_GROUPS = ("goal",)


def robot_columns(frame):
    return [c for c in frame.columns if c.startswith(("j_pos_", "j_vel_", "j_tau_", "gyro_", "accel_", "foot_"))
            ] + ["roll", "pitch"]


def feature_columns(groups, frame):
    columns = []
    for group in groups:
        columns += {"goal": GOAL, "history": HISTORY, "location": LOCATION, "time": TIME}.get(group) or (
            robot_columns(frame) if group == "robot" else [])
        if group not in ("goal", "history", "location", "time", "robot"):
            raise ValueError("unknown feature group: " + group)
    return columns


def hindsight_rows(cycles):
    """Relabelled examples from stretches of constant command. `cycles` holds one or more runs."""
    out = []
    for (_, _), ep in cycles.sort_values(["run_id", "episode", "cycle"]).groupby(["run_id", "episode"], sort=False):
        ep = ep.reset_index(drop=True)
        stride, turn = ep["stride"].to_numpy(), ep["turn"].to_numpy()
        bad = (ep["partial"].astype(bool) | ep["fell"].astype(bool)).to_numpy()
        xs, ys = ep["x_next"].to_numpy(), ep["y_next"].to_numpy()
        for t in range(len(ep)):
            start = ep.iloc[t]
            pose = (start["x"], start["y"], math.atan2(start["sin_yaw"], start["cos_yaw"]))
            for k in range(1, MAX_HORIZON + 1):
                end = t + k - 1
                if end >= len(ep) or bad[end] or stride[end] != stride[t] or turn[end] != turn[t]:
                    break
                dist, bearing = relative(pose, xs[end], ys[end])
                row = start.to_dict()
                row.update({"goal_dist": dist, "goal_sin": math.sin(bearing), "goal_cos": math.cos(bearing),
                            "horizon": float(k), "kind": "hindsight"})
                out.append(row)
    return pd.DataFrame(out)

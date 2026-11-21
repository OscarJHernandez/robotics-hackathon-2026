"""Turn recorded cycles into a→b training examples for TabPFN.

Every example says: "from this state, to reach a goal at (goal_dist, goal bearing) in `horizon`
cycles, use this (stride, turn)". The examples are derived from the `cycles` table each time and
never stored, so the database always holds only what really happened.

- Hindsight rows (any run): a stretch of K cycles with one command, starting at cycle t, reached
  some point P. Relabel: P is the goal, K the horizon, the command the label.
- Success rows (runs where a policy walked to b): each cycle of an episode that reached b, with
  its real goal and the command used, repeated by how direct the path was (fine-tuning has no
  sample weights, so a weight means repeating a row).
"""

import math

import numpy as np
import pandas as pd

from common import MAX_HORIZON, make_goal, relative

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


def success_rows(cycles, episodes, max_repeat=3):
    """Cycles of the better half of the episodes that reached b (by path directness, per run).

    An episode at the median directness is used once, the most direct one `max_repeat` times.
    Keeping only the better half is what pushes each round toward straighter paths: when every
    attempt reaches b, taking all of them would also teach the detours.
    """
    good = episodes[episodes["reached"]].copy()
    if good.empty:
        return pd.DataFrame()
    good["rank"] = good.groupby("run_id")["directness"].rank(pct=True)
    good = good[good["rank"] >= 0.5]
    merged = cycles.merge(good[["run_id", "episode", "rank"]], on=["run_id", "episode"])
    merged = merged[~merged["partial"].astype(bool) | merged["reached"].astype(bool)].assign(kind="success")
    repeat = np.round(1 + (max_repeat - 1) * (merged["rank"].to_numpy() - 0.5) / 0.5)
    return merged.loc[merged.index.repeat(np.clip(repeat, 1, max_repeat).astype(int))].reset_index(drop=True)


def training_table(cycles, episodes=None):
    """All examples for fine-tuning: hindsight rows from every run plus success rows from policy runs."""
    parts = [hindsight_rows(cycles)]
    if episodes is not None:
        policy_cycles = cycles[cycles["source"] != "random"]
        if not policy_cycles.empty:
            parts.append(success_rows(policy_cycles, episodes))
    table = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    return table


def goal_for(row):
    """Goal columns for an observed row (the real target), as the policy sees it."""
    bearing = math.atan2(row["target_sin"], row["target_cos"])
    return make_goal(row["target_dist"], bearing)

"""Lock-step rollouts: many robots step together; the student predicts for all of them in one call.

Each step: features for every active robot (one perception batch), one `policy.act_batch`, the expert's
command for every robot, then `executed = expert` with probability beta, else the student's. Both commands
are stored; the label is always the expert's. Finished robots start the next planned episode.
"""

import math
import time
from collections import Counter

import numpy as np
import pandas as pd

from .timing import Timer

META_FLOAT = ["x", "y", "yaw", "goal_x", "goal_y", "student_stride", "student_turn", "expert_stride",
              "expert_turn", "executed_stride", "executed_turn", "y_stride", "y_turn", "beta"]


class RoundResult:
    def __init__(self):
        self.features = []  # float32 arrays (n_features,)
        self.meta = []  # dicts
        self.episodes = []  # one summary dict per finished episode

    def take(self):
        """Rows and episodes gathered since the last take(), then forget them."""
        out = (self.features, self.meta, self.episodes)
        self.features, self.meta, self.episodes = [], [], []
        return out


def to_frame(features, meta, columns, categorical):
    """One DataFrame: feature columns (classes as int8) plus the bookkeeping columns."""
    if not meta:
        return pd.DataFrame()
    feats = pd.DataFrame(np.stack(features), columns=columns)
    for name in categorical:
        feats[name] = feats[name].astype(np.int8)
    return pd.concat([feats, pd.DataFrame(meta)], axis=1)


def run_episodes(envs, featurizer, expert, policy, beta, plan, round_, rng, log=None, timer=None,
                 on_chunk=None, chunk_episodes=25, heartbeat_every=30.0, deadline=None):
    """Run every episode in `plan` ([{"episode", "scene_id", "seed"}]) on the robots in `envs`.

    `on_chunk(result)` is called every `chunk_episodes` finished episodes and at the end (write to disk there).
    `deadline` (time.time() value, for benchmarks) stops early. Returns (outcome counts, steps).
    """
    timer = timer or Timer()
    queue = list(plan)
    total = len(queue)
    slots = [None] * len(envs)
    obs = [None] * len(envs)
    prev = [None] * len(envs)
    result = RoundResult()
    counts = Counter()
    finished = steps = 0
    pending_chunk = 0

    def start(k):
        if not queue:
            slots[k] = None
            return
        episode = dict(queue.pop(0))
        obs[k] = envs[k].reset(episode["scene_id"], episode["seed"])
        prev[k] = None
        episode["t0"] = time.time()
        episode["nopath_steps"] = 0
        slots[k] = episode

    for k in range(len(envs)):
        start(k)
    if log:
        log.event("round", "round {} beta {}: {} episodes on {} robots".format(round_, beta, total, len(envs)),
                  round=round_, beta=beta, episodes=total, robots=len(envs))
        log.start_rollout()
    while any(slot is not None for slot in slots):
        if deadline is not None and time.time() > deadline:
            break
        active = [k for k, slot in enumerate(slots) if slot is not None]
        matrix, _ = featurizer.build_batch([obs[k] for k in active], [prev[k] for k in active])
        student = None
        if policy is not None and beta < 1.0:
            with timer("student_predict"):
                student = policy.act_batch(matrix)
        for row, k in enumerate(active):
            env, episode = envs[k], slots[k]
            pose = obs[k]["robot_pose"]
            with timer("expert"):
                expert_cmd = expert.act(env.state())
            nopath = bool(expert.nopath)
            episode["nopath_steps"] += nopath
            student_cmd = (float(student[row, 0]), float(student[row, 1])) if student is not None else (math.nan, math.nan)
            use_expert = student is None or rng.random() < beta
            executed = expert_cmd if use_expert else student_cmd
            obs[k], info = env.step(executed)
            result.features.append(matrix[row])
            result.meta.append({
                "round": round_, "episode": episode["episode"], "step": env.steps - 1, "scene_id": episode["scene_id"],
                "seed": episode["seed"], "x": pose[0], "y": pose[1], "yaw": pose[2],
                "goal_x": env.goal[0], "goal_y": env.goal[1],
                "student_stride": student_cmd[0], "student_turn": student_cmd[1],
                "expert_stride": expert_cmd[0], "expert_turn": expert_cmd[1],
                "executed_stride": info["stride"], "executed_turn": info["turn"], "used_expert": use_expert,
                "y_stride": expert_cmd[0], "y_turn": expert_cmd[1], "beta": beta,
                "collision": info["collision"], "expert_nopath": nopath, "done_reason": info["done_reason"]})
            prev[k] = (info["stride"], info["turn"])
            if info["done"]:
                x, y, _ = env.pose()
                summary = {"round": round_, "episode": episode["episode"], "scene_id": episode["scene_id"],
                           "seed": episode["seed"], "steps": env.steps, "done_reason": info["done_reason"],
                           "wall_s": round(time.time() - episode["t0"], 2), "path_length": round(env.path_length, 3),
                           "start_x": env.start[0], "start_y": env.start[1], "start_yaw": env.start[2],
                           "goal_x": env.goal[0], "goal_y": env.goal[1],
                           "final_dist": round(math.hypot(env.goal[0] - x, env.goal[1] - y), 3),
                           "nopath_steps": episode["nopath_steps"]}
                result.episodes.append(summary)
                counts[info["done_reason"]] += 1
                finished += 1
                pending_chunk += 1
                if log:
                    log.event("episode", "episode {} scene {} {} after {} steps in {:.1f} s ({} of {})".format(
                        summary["episode"], summary["scene_id"], summary["done_reason"], summary["steps"],
                        summary["wall_s"], finished, total), **summary)
                start(k)
        steps += len(active)
        if log:
            log.heartbeat(finished, total, steps, counts, timer, every=heartbeat_every)
        if on_chunk is not None and pending_chunk >= chunk_episodes:
            with timer("write"):
                on_chunk(result)
            pending_chunk = 0
    if on_chunk is not None:
        with timer("write"):
            on_chunk(result)
    if log:
        log.heartbeat(finished, total, steps, counts, timer, force=True)
    return counts, steps

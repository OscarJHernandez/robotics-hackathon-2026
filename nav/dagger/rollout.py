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


# ---- v2: stored base rows (full grids, goal, summary, lidar), features built by a pipeline at drive time ----------

class Collected:
    def __init__(self):
        self.base, self.meta, self.band, self.episodes = [], [], [], []

    def take(self):
        out = (self.base, self.meta, self.band, self.episodes)
        self.base, self.meta, self.band, self.episodes = [], [], [], []
        return out


def frame_v2(base, meta, columns):
    """Stored rows as one DataFrame: the base columns (patch classes as int8) plus the bookkeeping columns."""
    if not meta:
        return pd.DataFrame()
    feats = pd.DataFrame(np.stack(base), columns=columns)
    for name in columns:
        if name.startswith("pc_"):
            feats[name] = feats[name].astype(np.int8)
    return pd.concat([feats, pd.DataFrame(meta)], axis=1)


def run_v2(envs, featurizer, expert, plan, rng, stored_columns, round_=0, policy=None, features=None, beta=1.0,
           perturb=None, store_band=False, log=None, timer=None, on_chunk=None, chunk_episodes=25,
           heartbeat_every=30.0):
    """Run every planned episode. Each step stores the base row; the student (if any) gets `features.live(...)`.

    `perturb` (expert collections only): {"command": p, "kick": p}. With probability `command` per step a random
    command is executed for 1 to 3 steps; with probability `kick` the robot is turned by a random angle after a
    step. The label is always the expert's command for the state the robot is in (the DART idea).
    Returns (outcome counts, steps).
    """
    from nav.common import STRIDE_RANGE, TURN_RANGE
    timer = timer or Timer()
    queue = list(plan)
    total = len(queue)
    n = len(envs)
    slots, obs, hist = [None] * n, [None] * n, [None] * n
    out = Collected()
    counts = Counter()
    finished = steps = pending = 0
    perturb = perturb or {}

    def start(k):
        if not queue:
            slots[k] = None
            return
        episode = dict(queue.pop(0))
        obs[k] = envs[k].reset(episode["scene_id"], episode["seed"])
        hist[k] = None
        episode.update(t0=time.time(), nopath_steps=0, random_left=0, random_cmd=None, kicked=False, perturbed_steps=0)
        slots[k] = episode

    for k in range(n):
        start(k)
    if log:
        log.event("round", "{} episodes on {} robots, beta {}, perturb {}".format(total, n, beta, perturb or "off"),
                  episodes=total, robots=n, beta=beta)
        log.start_rollout()
    while any(slot is not None for slot in slots):
        active = [k for k, slot in enumerate(slots) if slot is not None]
        base = featurizer.base_rows([obs[k] for k in active], [None] * len(active))
        band = featurizer.last_band
        stored = base[stored_columns].to_numpy(np.float32)
        experts = []
        with timer("expert"):
            for k in active:
                cmd = expert.act(envs[k].state())
                experts.append((cmd, bool(expert.nopath), float(expert.look_bearing)))
        base["look_bearing"] = [e[2] for e in experts]  # used only by the privileged ceiling variant
        student = None
        if policy is not None and beta < 1.0:
            with timer("features_live"):
                X, new_hist = features.live(base, band, [hist[k] for k in active])
            for row, k in enumerate(active):
                hist[k] = new_hist[row]
            with timer("student_predict"):
                student = policy.act_batch(X)
        for row, k in enumerate(active):
            env, episode = envs[k], slots[k]
            pose = obs[k]["robot_pose"]
            expert_cmd, nopath, look = experts[row]
            episode["nopath_steps"] += nopath
            student_cmd = ((float(student[row, 0]), float(student[row, 1])) if student is not None
                           else (math.nan, math.nan))
            use_expert = student is None or rng.random() < beta
            executed, perturbed = (expert_cmd if use_expert else student_cmd), False
            if student is None and perturb.get("command"):
                if episode["random_left"] == 0 and rng.random() < perturb["command"]:
                    episode["random_left"] = int(rng.integers(1, 4))
                    episode["random_cmd"] = (float(rng.uniform(*STRIDE_RANGE)), float(rng.uniform(*TURN_RANGE)))
                if episode["random_left"] > 0:
                    executed, perturbed = episode["random_cmd"], True
                    episode["random_left"] -= 1
                    episode["perturbed_steps"] += 1
            obs[k], info = env.step(executed)
            out.base.append(stored[row])
            if store_band and band is not None:
                out.band.append(band[row])
            out.meta.append({
                "round": round_, "episode": episode["episode"], "step": env.steps - 1, "scene_id": episode["scene_id"],
                "seed": episode["seed"], "x": pose[0], "y": pose[1], "yaw": pose[2],
                "goal_x": env.goal[0], "goal_y": env.goal[1],
                "student_stride": student_cmd[0], "student_turn": student_cmd[1],
                "expert_stride": expert_cmd[0], "expert_turn": expert_cmd[1],
                "executed_stride": info["stride"], "executed_turn": info["turn"], "used_expert": use_expert,
                "y_stride": expert_cmd[0], "y_turn": expert_cmd[1], "beta": beta, "look_bearing": look,
                "perturbed": perturbed, "kicked": episode["kicked"],
                "collision": info["collision"], "expert_nopath": nopath, "done_reason": info["done_reason"]})
            episode["kicked"] = False
            if not info["done"] and student is None and perturb.get("kick") and rng.random() < perturb["kick"]:
                obs[k] = env.kick(float(rng.uniform(-math.pi, math.pi)))
                episode["kicked"] = True
                episode["perturbed_steps"] += 1
            if info["done"]:
                x, y, _ = env.pose()
                summary = {"round": round_, "episode": episode["episode"], "scene_id": episode["scene_id"],
                           "seed": episode["seed"], "steps": env.steps, "done_reason": info["done_reason"],
                           "wall_s": round(time.time() - episode["t0"], 2), "path_length": round(env.path_length, 3),
                           "start_x": env.start[0], "start_y": env.start[1], "start_yaw": env.start[2],
                           "goal_x": env.goal[0], "goal_y": env.goal[1], "start_dist": round(env.start_dist, 3),
                           "final_dist": round(math.hypot(env.goal[0] - x, env.goal[1] - y), 3),
                           "nopath_steps": episode["nopath_steps"], "perturbed_steps": episode["perturbed_steps"]}
                out.episodes.append(summary)
                counts[info["done_reason"]] += 1
                finished += 1
                pending += 1
                if log:
                    log.event("episode", "episode {} scene {} {} after {} steps in {:.1f} s ({} of {})".format(
                        summary["episode"], summary["scene_id"], summary["done_reason"], summary["steps"],
                        summary["wall_s"], finished, total), **summary)
                start(k)
        steps += len(active)
        if log:
            log.heartbeat(finished, total, steps, counts, timer, every=heartbeat_every)
        if on_chunk is not None and pending >= chunk_episodes:
            with timer("write"):
                on_chunk(out)
            pending = 0
    if on_chunk is not None:
        with timer("write"):
            on_chunk(out)
    if log:
        log.heartbeat(finished, total, steps, counts, timer, force=True)
    return counts, steps

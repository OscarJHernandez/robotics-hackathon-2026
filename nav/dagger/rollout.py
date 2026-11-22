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


def safety_command(cmd, depth, safety):
    """Depth-camera stop: when the corridor ahead is closer than safety["stop_distance"], no stride, and a turn
    on the spot toward the freer side (left if dc_left >= dc_right). Returns (command, stopped)."""
    if not safety or not safety.get("enabled", True) or not depth or "dc_ahead" not in depth:
        return cmd, False
    if depth["dc_ahead"] >= safety.get("stop_distance", 0.45):
        return cmd, False
    side = 1.0 if depth["dc_left"] >= depth["dc_right"] else -1.0
    return (0.0, side * abs(safety.get("turn", 0.25))), True


LIDAR_OFFSET = -0.023  # metres: the lidar sits this far behind the body centre (lidar_geom in jethexa.xml)


def lidar_points(ranges, cap=10.0):
    """Lidar hits (beam k at k * 360/n degrees) as body-frame points (x ahead, y left) from the body centre."""
    ranges = np.asarray(ranges, np.float64)
    angles = np.arange(len(ranges)) * (2 * math.pi / len(ranges))
    hit = ranges < cap - 1e-6
    x = LIDAR_OFFSET + ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return x[hit], y[hit]


def lidar_safety(cmd, ranges, safety):
    """Leg-aware lidar safety for the student's command. The legs reach about 0.27 m ahead and behind,
    0.26 m sideways and 0.29 m at the corners (measured in the sim while walking and turning).

    1. stop: an obstacle in the corridor |y| < corridor ahead, closer than stop_distance -> no stride;
       rear: backing up (negative stride) with an obstacle closer than rear_distance behind -> no stride
    2. side: walking with something closer than side_clearance beside the legs (|x| < 0.27) -> turn away
       at least `nudge` (straight on when both sides are tight)
    3. sweep: turning on the spot swings the front legs toward the turn side and the rear legs the other
       way; a left turn is unsafe with an obstacle within sweep_radius front-left or rear-right (a right
       turn: front-right or rear-left). An unsafe turn is flipped when the other way is safe, else dropped,
       except after a stop, where turning is the only way out: then the robot turns toward the freer side.
    Returns (command, rule): rule is "" when nothing changed, else "+"-joined rule names.
    """
    stride, turn = float(cmd[0]), float(cmd[1])
    x, y = lidar_points(ranges)
    rules = []
    corridor = safety.get("corridor", 0.32)
    ahead = (np.abs(y) < corridor) & (x > 0)
    if stride > 0 and ahead.any() and x[ahead].min() < safety.get("stop_distance", 0.48):
        stride = 0.0
        rules.append("stop")
    behind = (np.abs(y) < corridor) & (x < 0)
    if stride < 0 and behind.any() and -x[behind].max() < safety.get("rear_distance", 0.40):
        stride = 0.0  # backing up into something behind the rear legs
        rules.append("rear")
    if stride > 0:
        beside = np.abs(x) < 0.27
        side_clear = safety.get("side_clearance", 0.32)
        left_tight = (beside & (y > 0) & (y < side_clear)).any()
        right_tight = (beside & (y < 0) & (y > -side_clear)).any()
        nudge = safety.get("nudge", 0.1)
        if left_tight and right_tight:
            turn, rules = 0.0, rules + ["side"]
        elif left_tight and turn > -nudge:
            turn, rules = -nudge, rules + ["side"]
        elif right_tight and turn < nudge:
            turn, rules = nudge, rules + ["side"]
    near = np.hypot(x, y) < safety.get("sweep_radius", 0.35)
    front, rear = x > 0.05, x < -0.05  # straight beside (|x| <= 5 cm) the middle legs move along it, not into it
    unsafe_left = (near & ((front & (y > 0)) | (rear & (y < 0)))).any()
    unsafe_right = (near & ((front & (y < 0)) | (rear & (y > 0)))).any()
    spin = abs(safety.get("turn", 0.25))
    if "stop" in rules and abs(turn) < 0.05:  # stopped without a clear turn: pick the safe / freer side
        free_left = np.min(np.hypot(x, y)[y > 0], initial=10.0)
        free_right = np.min(np.hypot(x, y)[y < 0], initial=10.0)
        turn = spin if free_left >= free_right else -spin
    stuck = "stop" in rules  # stopped: a turn is the only way out, so keep the freer one even if both are tight
    if turn > 0 and unsafe_left and not (unsafe_right and stuck):
        turn = -max(abs(turn), 0.1) if not unsafe_right else 0.0
        rules.append("sweep")
    elif turn < 0 and unsafe_right and not (unsafe_left and stuck):
        turn = max(abs(turn), 0.1) if not unsafe_left else 0.0
        rules.append("sweep")
    return (stride, turn), "+".join(rules)


def run_episodes(envs, featurizer, expert, policy, beta, plan, round_, rng, log=None, timer=None,
                 on_chunk=None, chunk_episodes=25, heartbeat_every=30.0, deadline=None, safety=None):
    """Run every episode in `plan` ([{"episode", "scene_id", "seed"}]) on the robots in `envs`.

    `on_chunk(result)` is called every `chunk_episodes` finished episodes and at the end (write to disk there).
    `deadline` (time.time() value, for benchmarks) stops early. `safety` ({"stop_distance", "turn"}) overrides
    the student's command when the depth camera sees an obstacle close ahead (never the expert's).
    Returns (outcome counts, steps).
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
        matrix, states = featurizer.build_batch([obs[k] for k in active], [prev[k] for k in active])
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
            recovering = bool(getattr(expert, "recovering", False))
            episode["nopath_steps"] += nopath
            student_cmd = (float(student[row, 0]), float(student[row, 1])) if student is not None else (math.nan, math.nan)
            use_expert = student is None or rng.random() < beta
            state = states[row] if isinstance(states[row], dict) else {"cmd": None, "hist": []}
            executed, stopped = expert_cmd, False
            rule = ""
            if not use_expert:
                if safety and safety.get("enabled", True) and safety.get("source") == "lidar" \
                        and state.get("lidar") is not None:
                    executed, rule = lidar_safety(student_cmd, state["lidar"], safety)
                    stopped = bool(rule)
                else:
                    executed, stopped = safety_command(student_cmd, state.get("depth"), safety)
                    rule = "depth_stop" if stopped else ""
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
                "collision": info["collision"], "expert_nopath": nopath, "expert_recover": recovering, "done_reason": info["done_reason"],
                "safety_stop": stopped, "safety_rule": rule})
            state["cmd"] = (info["stride"], info["turn"])
            prev[k] = state
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

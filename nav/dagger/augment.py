"""Add depth-camera columns to recorded rounds by replaying them: no DINOv2, no student, no API.

    python -m nav.dagger.augment --run lidar4 --rounds 0 1 --out lidar5 --config nav/dagger/lidar3.yaml

Every episode of data/nav/dagger/<run>/round_<R>.parquet is replayed exactly (same scene, start seed and
executed commands; the physics is deterministic, as in video.py), and at every recorded step the depth
picture gives the columns the config's featurizer would have written: dc_ahead, dc_left, dc_right, dc_near
(depth_summary), ds_0..ds_<n-1> (depth_scan) and their history h1_dc_*, h2_dc_* (history; the first step
repeats the present, as Featurizer.build_batch does). All other columns are kept. The result goes to
data/nav/dagger/<out>/round_<R>.parquet with round_<R>.json and the plans. An episode whose replay deviates
more than --max-deviation metres from the recorded poses, or ends differently, stops the run.
"""

import argparse
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .config import load_config, run_dir, runs_dir
from .env import SimEnv
from .expert import Expert
from .featurize import DEPTH, depth_obstacles, scan_columns


def new_columns(scan, history):
    return DEPTH + scan_columns(scan) + ["h{}_{}".format(k, name) for k in range(1, history + 1) for name in DEPTH]


def replay_episode(env, rows, scan, history, cap):
    """Depth columns for one episode's rows (sorted by step): (DataFrame indexed like rows, max deviation, outcome)."""
    obs = env.reset(int(rows["scene_id"].iloc[0]), int(rows["seed"].iloc[0]))
    out, past, deviation, info = [], [], 0.0, {"done_reason": ""}
    for _, row in rows.iterrows():
        x, y, _ = env.pose()
        deviation = max(deviation, math.hypot(x - row["x"], y - row["y"]))
        d = depth_obstacles(obs["depth"], obs["camera"], obs["robot_pose"], obs["goal_point"], cap, scan=scan)
        current = [d[k] for k in DEPTH]
        hist = past or [current] * history
        values = current + [d[k] for k in scan_columns(scan)]
        for k in range(history):
            values += hist[min(k, len(hist) - 1)]
        out.append(values)
        past = ([current] + hist)[:history]
        obs, info = env.step((row["executed_stride"], row["executed_turn"]))
        if info["done"]:
            break
    frame = pd.DataFrame(out, columns=new_columns(scan, history), index=rows.index[:len(out)], dtype=np.float32)
    return frame, deviation, info["done_reason"]


def _job(args):
    config_path, episodes = args
    config = load_config(config_path)
    env_cfg, features = config["env"], config["features"]
    expert = Expert.from_config(config["expert"], env_cfg["goal_radius"])
    env = SimEnv(simple=env_cfg["simple"], goal_radius=env_cfg["goal_radius"], max_steps=env_cfg["max_steps"],
                 min_start_dist=env_cfg["min_start_dist"], expert=expert)
    results = []
    for rows in episodes:
        frame, deviation, outcome = replay_episode(env, rows, features.get("depth_scan", 0), features.get("history", 0),
                                                   features["depth_cap"])
        results.append((frame, deviation, outcome, rows["done_reason"].iloc[-1], int(rows["episode"].iloc[0]),
                        len(frame) == len(rows)))
    env.close()
    return results


def augment_round(table, config_path, procs=8):
    """table + the depth columns, and a report. Raises when a replay does not match the recording."""
    episodes = [rows.sort_values("step") for _, rows in table.groupby("episode", sort=False)]
    jobs = [(config_path, episodes[k::procs]) for k in range(procs) if episodes[k::procs]]
    frames, worst, mismatched = [], 0.0, []
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        for results in pool.map(_job, jobs):
            for frame, deviation, outcome, recorded, episode, complete in results:
                frames.append(frame)
                worst = max(worst, deviation)
                if outcome != recorded or not complete:
                    mismatched.append((episode, recorded, outcome))
    added = pd.concat(frames).reindex(table.index)
    out = pd.concat([table.drop(columns=[c for c in added.columns if c in table.columns]), added], axis=1)
    return out, {"rows": len(table), "episodes": len(episodes), "max_deviation_m": round(worst, 4),
                 "mismatched": mismatched[:10], "mismatched_count": len(mismatched)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="source run in data/nav/dagger/")
    parser.add_argument("--rounds", nargs="+", required=True, help="round numbers, or 'heldout'")
    parser.add_argument("--out", required=True, help="new run name")
    parser.add_argument("--config", required=True, help="the config whose features the new columns follow")
    parser.add_argument("--procs", type=int, default=8)
    parser.add_argument("--max-deviation", type=float, default=0.01, help="metres")
    args = parser.parse_args()
    src = runs_dir() / args.run
    dst = run_dir(args.out)
    for r in args.rounds:
        name = "round_{}".format(99 if r == "heldout" else int(r))
        table = pd.read_parquet(src / (name + ".parquet"))
        out, report = augment_round(table, args.config, args.procs)
        print(name, json.dumps(report), flush=True)
        if report["max_deviation_m"] > args.max_deviation or report["mismatched_count"]:
            raise SystemExit("replay does not match the recording in {}: {}".format(name, report))
        out.to_parquet(dst / (name + ".parquet"), index=False)
        if (src / (name + ".json")).is_file():
            shutil.copy2(src / (name + ".json"), dst / (name + ".json"))
    if (src / "plans").is_dir() and not (dst / "plans").exists():
        shutil.copytree(src / "plans", dst / "plans")
    print("written to {}".format(dst))


if __name__ == "__main__":
    main()

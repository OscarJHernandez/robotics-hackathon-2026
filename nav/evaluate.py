"""Run a policy on a→b tasks and record every episode in the database.

    uv run --group ml python nav/evaluate.py --model demo --round 2     # trained vs pretrained vs baseline

Episodes run in lockstep (one simulator per task) so TabPFN predicts all robots in one call.
The 20 held-out tasks in eval_tasks.json are never used for training.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import JetHexaSim, Walker, data_dir
from db import Recorder, query
from images import save_camera

NAV_DIR = Path(__file__).resolve().parent
EVAL_TASKS = NAV_DIR / "eval_tasks.json"
MAX_CYCLES = 40


def make_tasks(n, seed, first_id=0):
    """Random tasks: start at the origin facing a random way, b at 0.5 to 1.5 m in any direction."""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n):
        yaw = float(rng.uniform(-math.pi, math.pi))
        dist, bearing = float(rng.uniform(0.5, 1.5)), float(rng.uniform(-math.pi, math.pi))
        tasks.append({"task": first_id + i, "start": [0.0, 0.0, yaw],
                      "target": [round(dist * math.cos(yaw + bearing), 4), round(dist * math.sin(yaw + bearing), 4)]})
    return tasks


def eval_tasks():
    return json.loads(EVAL_TASKS.read_text())


def run_episodes(policy, tasks, recorder, max_cycles=MAX_CYCLES, images=False, sims=None):
    """Run every task to the end (reached b, fell, or `max_cycles`) and record it. Returns summaries."""
    sims = sims or [JetHexaSim() for _ in tasks]
    walkers = [Walker(sims[i], t["start"], t["target"]) for i, t in enumerate(tasks)]
    rows = [[] for _ in tasks]
    active = list(range(len(tasks)))
    summaries = [None] * len(tasks)
    while active:
        observed = []
        for i in active:
            row = walkers[i].observe()
            row["image"] = save_camera(walkers[i].sim, recorder, i, walkers[i].cycle) if images else ""
            row["source"] = policy.name
            row["hold"] = -1
            observed.append(row)
        commands = policy.act(observed)
        still = []
        for i, row, (stride, turn) in zip(active, observed, commands):
            row.update(walkers[i].run_cycle(stride, turn))
            rows[i].append(row)
            if walkers[i].done or walkers[i].cycle >= max_cycles:
                summaries[i] = walkers[i].summary()
                recorder.add_episode(i, rows[i], walkers[i].steps, summaries[i], task=tasks[i]["task"])
            else:
                still.append(i)
        active = still
    return summaries


def report(summaries, name):
    frame = pd.DataFrame(summaries)
    line = "{:<22} success {:>5.0%}  mean final distance {:.3f} m  directness {:.2f}  fell {}".format(
        name, frame["reached"].mean(), frame["final_dist"].mean(), frame["directness"].mean(), int(frame["fell"].sum()))
    print(line, flush=True)
    return line


def evaluate(policy, model="", round_=None, tasks=None, images=False):
    tasks = tasks or eval_tasks()
    recorder = Recorder("eval", policy=policy.name, model=model, round_=round_,
                        settings={"tasks": len(tasks), "max_cycles": MAX_CYCLES}, images=images)
    summaries = run_episodes(policy, tasks, recorder, images=images)
    recorder.close()
    return report(summaries, policy.name if round_ is None else "{} r{}".format(policy.name, round_))


def main():
    from policy import BaselinePolicy, TabPFNNavPolicy, checkpoint_paths
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="trained model name (folder in data/nav/models)")
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()
    folder = data_dir() / "models" / args.model
    context = pd.read_parquet(folder / "round_{}_context.parquet".format(args.round))
    evaluate(BaselinePolicy())
    evaluate(TabPFNNavPolicy(context), args.model, args.round)
    evaluate(TabPFNNavPolicy(context, checkpoints=checkpoint_paths(folder, args.round)), args.model, args.round)


if __name__ == "__main__":
    main()

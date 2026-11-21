"""Build a TabPFN controller for walking a→b, check it and evaluate it.

    uv run --group ml python nav/controller.py                                  # default: planner, 300 examples
    uv run --group ml python nav/controller.py --name goal --kind goal          # goal -> command

"Building" means choosing the examples that go into TabPFN's built-in `fit()`. No weights are
trained (rule in AGENTS.md); it takes seconds. The steps:

1. Examples from the random-walk runs in the database (`just collect` first):
   kind `goal`: hindsight a→b examples (hindsight.py); kind `planner`: motion examples (dynamics.py).
2. Hold out 10 % of the episodes. Pick `--context-rows` examples from the rest: the controller's context.
3. `fit()` TabPFN on the context. Predict the held-out examples (table `predictions`) and the
   command over a grid of goals (table `policy_map`).
4. Walk the 20 held-out a→b tasks with the controller and with the hand-written baseline.

The context is saved to data/nav/controllers/<name>/context.parquet; loading a controller reads it
and calls `fit()` again.
"""

import argparse
import json
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd

from common import data_dir, git_commit, make_goal
from db import connect, query
from dynamics import DEFAULT_SCORE, FEATURES, SCORES, DynamicsPlanner, motion_rows, sample_rows
from evaluate import evaluate
from hindsight import DEFAULT_GROUPS, hindsight_rows
from policy import BaselinePolicy, TabPFNNavPolicy, sample_context

KINDS = ("planner", "goal")
MAX_VAL_ROWS = 1000
# Examples given to fit() by default. More did not help on the 20 held-out tasks (2026-11-21):
# planner 300 and 2000 both reached directness 0.89 with the polar score (2000 is 9x slower); 300 with the
# hybrid score: 0.96. goal 150: 0.77 (0.84 with the delta inputs), 3000: 0.72.
DEFAULT_CONTEXT = {"planner": 300, "goal": 150}


def folder(name):
    return data_dir() / "controllers" / name


def make_policy(kind, context, groups=DEFAULT_GROUPS, device="cpu", seed=0, score=DEFAULT_SCORE):
    if kind == "goal":
        return TabPFNNavPolicy(context, groups=groups, device=device, seed=seed)
    if kind == "planner":
        return DynamicsPlanner(context, device=device, seed=seed, score=score)
    raise ValueError("unknown controller kind: " + kind)


def load(name, device="cpu"):
    """The saved controller `name`: reads its context and calls TabPFN's fit() on it."""
    path = folder(name)
    if not (path / "controller.json").exists():
        raise SystemExit("no controller named '{}': build it with `just controller --name {}`".format(name, name))
    info = json.loads((path / "controller.json").read_text())
    return make_policy(info["kind"], pd.read_parquet(path / "context.parquet"), tuple(info["groups"]), device,
                       info["seed"], info.get("score") or "polar")  # built before scores existed: polar


def split_by_episode(table, fraction, seed):
    keys = table[["run_id", "episode"]].drop_duplicates().reset_index(drop=True)
    pick = keys.iloc[np.random.default_rng(seed).permutation(len(keys))[: max(1, int(len(keys) * fraction))]]
    is_val = table.set_index(["run_id", "episode"]).index.isin(pd.MultiIndex.from_frame(pick))
    return table[~is_val], table[is_val]


def goal_grid():
    """Observed rows for the steering map: b at bearings every 10 degrees and three distances, robot at rest."""
    rows = []
    for dist in (0.2, 0.5, 1.0):
        for degrees in range(-180, 181, 10):
            bearing = math.radians(degrees)
            row = {"target_dist": dist, "target_sin": math.sin(bearing), "target_cos": math.cos(bearing),
                   "bearing": float(degrees), "prev_stride": 0.0, "prev_turn": 0.0, "fwd_1": 0.0, "left_1": 0.0,
                   "dyaw_1": 0.0}
            row.update(make_goal(dist, bearing))
            rows.append(row)
    return rows


def insert(table, frame):
    with connect() as con:
        con.register("frame", frame)
        con.execute("INSERT INTO {} BY NAME SELECT * FROM frame".format(table))
        con.unregister("frame")


def build(name, kind="planner", context_rows=None, groups=DEFAULT_GROUPS, device="cpu", seed=0, run_eval=True,
          score=DEFAULT_SCORE):
    context_rows = context_rows or DEFAULT_CONTEXT[kind]
    cycles = query("SELECT c.* FROM cycles c JOIN runs r USING (run_id) WHERE r.kind = 'collect'")
    if cycles.empty:
        raise SystemExit("no random-walk data: run `just collect` first")
    table = hindsight_rows(cycles) if kind == "goal" else motion_rows(cycles)
    train, val = split_by_episode(table, 0.1, seed)
    context = sample_context(train, context_rows, seed) if kind == "goal" else sample_rows(train, context_rows, seed)
    val = val.sample(n=min(len(val), MAX_VAL_ROWS), random_state=seed)
    path = folder(name)
    path.mkdir(parents=True, exist_ok=True)
    context.to_parquet(path / "context.parquet")

    tic = time.time()
    policy = make_policy(kind, context, groups, device, seed, score)  # TabPFN's fit()
    fit_seconds = time.time() - tic
    features = policy.columns if kind == "goal" else FEATURES
    info = {"name": name, "kind": kind, "context_rows": len(context), "total_examples": len(table),
            "val_rows": len(val), "groups": list(groups), "features": features, "outputs": list(policy.labels),
            "seed": seed, "score": score if kind == "planner" else None, "source_runs": sorted(cycles["run_id"].unique().tolist()), "git_commit": git_commit(),
            "created_at": datetime.now().isoformat(timespec="seconds")}
    (path / "controller.json").write_text(json.dumps(info, indent=2))

    with connect() as con:  # a rebuilt controller replaces its old rows
        for old in ("controllers WHERE name", "predictions WHERE model", "policy_map WHERE model"):
            con.execute("DELETE FROM {} = ?".format(old), [name])
        con.execute("INSERT INTO controllers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [name, kind, datetime.now(), len(context), len(table), len(val), ",".join(features),
                     ",".join(policy.labels), fit_seconds, str(path / "context.parquet"), git_commit(),
                     json.dumps(info)])
    print("controller '{}' ({}): fit() on {} of {} examples in {:.1f} s".format(
        name, kind, len(context), len(table), fit_seconds), flush=True)

    predicted = policy.predict_labels(val)
    for label, values in predicted.items():
        frame = pd.DataFrame({"model": name, "round": None, "output": label, "weights": "pretrained",
                              "run_id": val["run_id"].to_numpy(), "episode": val["episode"].to_numpy(),
                              "cycle": val["cycle"].to_numpy(), "goal_dist": val["goal_dist"].to_numpy(),
                              "goal_sin": val["goal_sin"].to_numpy(), "goal_cos": val["goal_cos"].to_numpy(),
                              "horizon": val["horizon"].to_numpy(), "actual": val[label].to_numpy(),
                              "predicted": values})
        frame["round"] = frame["round"].astype("Int64")
        insert("predictions", frame)
        actual = val[label].to_numpy()
        print("  held-out {}: RMSE {:.4f}, R2 {:.3f}".format(
            label, float(np.sqrt(np.mean((values - actual) ** 2))),
            1 - float(np.sum((values - actual) ** 2) / np.sum((actual - actual.mean()) ** 2))), flush=True)

    grid = goal_grid()
    commands = policy.act(grid)
    for i, label in enumerate(("stride", "turn")):
        frame = pd.DataFrame({"model": name, "round": None, "output": label, "weights": "pretrained",
                              "goal_dist": [g["target_dist"] for g in grid], "bearing": [g["bearing"] for g in grid],
                              "horizon": [g["horizon"] for g in grid], "predicted": commands[:, i]})
        frame["round"] = frame["round"].astype("Int64")
        insert("policy_map", frame)

    if run_eval:
        evaluate(BaselinePolicy(), name)
        evaluate(policy, name)
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="default", help="controller name; files go to data/nav/controllers/<name>")
    parser.add_argument("--kind", choices=KINDS, default="planner")
    parser.add_argument("--context-rows", type=int, help="examples given to TabPFN's fit() (default: planner 300, goal 150)")
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS), help="feature groups (kind goal only)")
    parser.add_argument("--score", choices=SCORES, default=DEFAULT_SCORE,
                        help="how the planner scores candidates (kind planner only): polar = distance + bearing, "
                             "delta = b's offset ahead and sideways, hybrid = delta + facing b")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(args.name, args.kind, args.context_rows, tuple(args.groups.split(",")), args.device, args.seed,
          score=args.score)


if __name__ == "__main__":
    main()

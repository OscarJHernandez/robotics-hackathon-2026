"""TabPFN as a learned model of the robot's motion, used to plan each step toward b.

Why: copying hindsight commands (policy.py) teaches TabPFN the random curved arcs of the training
data, and TabPFN already reproduces those labels almost exactly, so fine-tuning has nothing to
improve. Here TabPFN instead predicts what a command does:

    state (previous command, last motion) + command (stride, turn)  ->  motion over the next cycle
                                                                        (forward m, left m, yaw rad)

Each cycle the planner asks TabPFN about a grid of candidate commands for every robot in one batch,
and picks the command whose predicted end pose is closest to b (with a small cost for not facing b).
Better motion predictions mean straighter paths, so the validation error of the motion model and
the a→b results measure the same thing, and fine-tuning can improve both.

    uv run --group ml python nav/dynamics.py --name dyn --rounds 2
"""

import argparse
import json
import math
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from common import STRIDE_RANGE, TURN_RANGE, data_dir, git_commit, wrap
from db import Recorder, query
from evaluate import evaluate, make_tasks, run_episodes
from policy import make_regressor, sample_context

STATE = ["prev_stride", "prev_turn", "fwd_1", "left_1", "dyaw_1"]
COMMAND = ["stride", "turn"]
FEATURES = STATE + COMMAND
MOTION = ["m_fwd", "m_left", "m_dyaw"]
CANDIDATES = [(s, t) for s in np.linspace(*STRIDE_RANGE, 5) for t in np.linspace(*TURN_RANGE, 9)]
FACING_COST = 0.08  # metres of distance one radian of bearing error is worth


def motion_rows(cycles):
    """Training rows: every full cycle (not cut short, no fall) with the motion it produced."""
    full = cycles[~cycles["partial"].astype(bool) & ~cycles["fell"].astype(bool)].copy()
    yaw = np.arctan2(full["sin_yaw"], full["cos_yaw"])
    dx, dy = full["x_next"] - full["x"], full["y_next"] - full["y"]
    full["m_fwd"] = np.cos(yaw) * dx + np.sin(yaw) * dy
    full["m_left"] = -np.sin(yaw) * dx + np.cos(yaw) * dy
    full["m_dyaw"] = np.arctan2(np.sin(full["yaw_next"] - yaw), np.cos(full["yaw_next"] - yaw))
    # sample_context bins by goal columns; give every row the same so it samples uniformly
    return full.reset_index(drop=True)


def sample_rows(table, rows, seed):
    if len(table) <= rows:
        return table
    return table.sample(n=rows, random_state=seed).sort_index()


class DynamicsPlanner:
    """Picks the candidate command whose TabPFN-predicted end pose is best."""

    def __init__(self, context, checkpoints=None, context_rows=150, device="cpu", n_estimators=4, seed=0,
                 noise=None, name=None):
        self.context = sample_rows(context, context_rows, seed)
        self.name = name or ("dynamics-finetuned" if checkpoints else "dynamics-pretrained")
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        X = self.context[FEATURES].to_numpy(np.float32)
        self.models = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for label in MOTION:
                model = make_regressor((checkpoints or {}).get(label), device, n_estimators, seed)
                model.fit(X, self.context[label].to_numpy(np.float32))
                self.models[label] = model

    def predict(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.stack([self.models[label].predict(X) for label in MOTION], axis=1)

    def act(self, rows):
        cands = np.array(CANDIDATES, dtype=np.float32)
        X = np.concatenate([np.hstack([np.tile([[r[c] for c in STATE]], (len(cands), 1)), cands]) for r in rows])
        motion = self.predict(X.astype(np.float32)).reshape(len(rows), len(cands), 3)
        out = []
        for row, m in zip(rows, motion):
            bearing = math.atan2(row["target_sin"], row["target_cos"])
            # target in the current body frame
            tf, tl = row["target_dist"] * math.cos(bearing), row["target_dist"] * math.sin(bearing)
            # target relative to each predicted end pose
            rf, rl = tf - m[:, 0], tl - m[:, 1]
            c, s = np.cos(m[:, 2]), np.sin(m[:, 2])
            ef, el = c * rf + s * rl, -s * rf + c * rl
            cost = np.hypot(ef, el) + FACING_COST * np.abs(np.arctan2(el, ef))
            best = cands[int(np.argmin(cost))].astype(float)
            if self.noise is not None:
                best = best + self.rng.normal(0.0, self.noise)
            out.append((float(np.clip(best[0], *STRIDE_RANGE)), float(np.clip(best[1], *TURN_RANGE))))
        return np.array(out)


def checkpoint_paths(folder, round_):
    return {label: folder / "dyn_round_{}_{}.ckpt".format(round_, label) for label in MOTION}


def split_by_episode(table, fraction, seed):
    keys = table[["run_id", "episode"]].drop_duplicates().reset_index(drop=True)
    pick = keys.iloc[np.random.default_rng(seed).permutation(len(keys))[: max(1, int(len(keys) * fraction))]]
    is_val = table.set_index(["run_id", "episode"]).index.isin(pd.MultiIndex.from_frame(pick))
    return table[~is_val], table[is_val]


def main():
    from train import finetune, record_training, pretrained_checkpoint
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--rounds", type=int, default=2, help="rounds after round 0")
    parser.add_argument("--attempts", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-rows", type=int, default=250)
    parser.add_argument("--train-rows", type=int, default=6000)
    parser.add_argument("--context-rows", type=int, default=150)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    folder = data_dir() / "models" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    settings = dict(vars(args), kind="dynamics", features=FEATURES, outputs=MOTION, git_commit=git_commit(),
                    started=datetime.now().isoformat())
    (folder / "train.json").write_text(json.dumps(settings, indent=2))

    collect = query("SELECT c.* FROM cycles c JOIN runs r USING (run_id) WHERE r.kind = 'collect'")
    if collect.empty:
        raise SystemExit("no random-walk data: run `just collect` first")
    from policy import BaselinePolicy
    evaluate(BaselinePolicy(), args.name, None)
    extra_runs, context = [], None
    for round_ in range(args.rounds + 1):
        print("\n=== round {} ===".format(round_), flush=True)
        if round_ > 0:
            planner = DynamicsPlanner(context, checkpoint_paths(folder, round_ - 1), args.context_rows,
                                      seed=round_, noise=(0.03, 0.04), name="dynamics-finetuned")
            recorder = Recorder("attempts", policy="dynamics-finetuned+noise", model=args.name, round_=round_ - 1,
                                settings={"attempts": args.attempts})
            done = run_episodes(planner, make_tasks(args.attempts, seed=2000 + round_, first_id=1000 * round_), recorder)
            recorder.close()
            extra_runs.append(recorder.run_id)
            print("attempts: {}/{} reached b".format(sum(s["reached"] for s in done), len(done)), flush=True)
        cycles = collect
        if extra_runs:
            marks = ",".join("'{}'".format(r) for r in extra_runs)
            cycles = pd.concat([collect, query("SELECT * FROM cycles WHERE run_id IN ({})".format(marks))],
                               ignore_index=True)
        table = motion_rows(cycles)
        train, val = split_by_episode(table, 0.1, args.seed)
        train = sample_rows(train, args.train_rows, args.seed + round_)
        context = train
        train.to_parquet(folder / "dyn_round_{}_context.parquet".format(round_))
        print("training rows {} (of {}), validation rows {}".format(len(train), len(table), len(val)), flush=True)
        for label, out in checkpoint_paths(folder, round_).items():
            start = pretrained_checkpoint() if round_ == 0 else checkpoint_paths(folder, round_ - 1)[label]
            tic = time.time()
            before, after, change = finetune(train, val, FEATURES, label, start, out, args, round_)
            record_training(args.name, round_, label, len(train), len(val), time.time() - tic, before, after,
                            change, out, start, settings)
            print("{}: validation MSE {:.6f} -> {:.6f} ({:+.1%}), weights moved {:.2e}".format(
                label, before, after, after / before - 1 if before else 0, change), flush=True)
        evaluate(DynamicsPlanner(context, None, args.context_rows), args.name, round_)
        evaluate(DynamicsPlanner(context, checkpoint_paths(folder, round_), args.context_rows), args.name, round_)


if __name__ == "__main__":
    main()

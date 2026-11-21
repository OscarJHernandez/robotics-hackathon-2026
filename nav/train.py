"""Fine-tune TabPFN's weights for walking from a to b, in rounds.

    uv run --group ml python nav/train.py --name demo --rounds 3

Round 0: fine-tune the pretrained TabPFN v2 on hindsight examples from the random-walk runs
         (`just collect` first).
Round r: the round r-1 model, with a little exploration noise, attempts random a→b tasks. Those
         episodes are recorded; their hindsight rows and the cycles of episodes that reached b are
         added to the training table, and the model is fine-tuned again from its round r-1 weights.

After every round the model is evaluated on the 20 held-out tasks against the pretrained TabPFN
with the same context rows, and against the hand-written baseline (once). Everything goes into
the database (tables runs, episodes, cycles, steps, training) and models/<name>/ holds the
checkpoints, each round's context table and train.json.
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

from common import data_dir, git_commit, make_goal
from db import Recorder, connect, query
from evaluate import evaluate, make_tasks, run_episodes
from hindsight import DEFAULT_GROUPS, LABELS, feature_columns, training_table
from policy import BaselinePolicy, TabPFNNavPolicy, checkpoint_paths, make_regressor, sample_context


def pretrained_checkpoint():
    from tabpfn.model_loading import ModelSource, prepend_cache_path
    return prepend_cache_path(ModelSource.get_regressor_v2().default_filename)


def split_by_episode(table, fraction, seed):
    keys = table[["run_id", "episode"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    val_keys = keys.iloc[rng.permutation(len(keys))[: max(1, int(len(keys) * fraction))]]
    is_val = table.set_index(["run_id", "episode"]).index.isin(pd.MultiIndex.from_frame(val_keys))
    return table[~is_val], table[is_val]


class DatabaseLogger:
    """Receives the fine-tuner's step and epoch metrics and stores every value in `training_log`."""

    def __init__(self, model, round_, output):
        self.key = (model, round_, output)
        self.rows = []

    def setup(self, config):
        pass

    def _add(self, kind, metrics, step):
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                self.rows.append(self.key + (kind, int(step), name, float(value), datetime.now()))

    def log_step(self, metrics, step):
        self._add("step", metrics, step)

    def log_epoch(self, metrics, step):
        self._add("epoch", metrics, step)
        shown = ", ".join("{} {:.5f}".format(k, v) for k, v in metrics.items() if isinstance(v, float))
        print("  epoch {}: {}".format(step, shown), flush=True)

    def finish(self):
        if self.rows:
            with connect() as con:
                con.executemany("INSERT INTO training_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)", self.rows)


def weight_change(start_ckpt, out_ckpt):
    """Relative size of the change in weights: ||w_after - w_before|| / ||w_before||, over shared tensors."""
    def tensors(path):
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        return {k: v.float() for k, v in state.items() if torch.is_tensor(v) and not k.startswith("criterion.")}
    a, b = tensors(start_ckpt), tensors(out_ckpt)
    shared = [k for k in b if k in a and a[k].shape == b[k].shape]
    if not shared:  # the v2 file uses older parameter names than the loaded model: compare via a load
        a = {k: v.detach().cpu().float() for k, v in load_weights(start_ckpt).items()}
        shared = [k for k in b if k in a and a[k].shape == b[k].shape]
    diff = sum(float(((b[k] - a[k]) ** 2).sum()) for k in shared)
    base = sum(float((a[k] ** 2).sum()) for k in shared)
    return (diff / base) ** 0.5 if base > 0 else float("nan")


def load_weights(ckpt):
    model = make_regressor(ckpt, "cpu", n_estimators=1)
    model.fit(np.zeros((4, 1), np.float32), np.arange(4, dtype=np.float32))
    return model.models_[0].state_dict()


def store_predictions(model_name, round_, label, weights, val, predicted):
    frame = val[["run_id", "episode", "cycle", "goal_dist", "goal_sin", "goal_cos", "horizon"]].copy()
    frame.insert(0, "weights", weights)
    frame.insert(0, "output", label)
    frame.insert(0, "round", round_)
    frame.insert(0, "model", model_name)
    frame["actual"] = val[label].to_numpy()
    frame["predicted"] = predicted
    with connect() as con:
        con.register("frame", frame)
        con.execute("INSERT INTO predictions BY NAME SELECT * FROM frame")
        con.unregister("frame")


def goal_grid():
    """Goals for the policy map: bearings every 10 degrees at three distances."""
    rows = []
    for dist in (0.2, 0.5, 1.0):
        for degrees in range(-180, 181, 10):
            goal = make_goal(dist, math.radians(degrees))
            goal["bearing"] = float(degrees)
            rows.append(goal)
    return pd.DataFrame(rows)


def store_policy_map(model_name, round_, label, weights, grid, predicted):
    frame = grid[["goal_dist", "bearing", "horizon"]].copy()
    frame.insert(0, "weights", weights)
    frame.insert(0, "output", label)
    frame.insert(0, "round", round_)
    frame.insert(0, "model", model_name)
    frame["predicted"] = predicted
    with connect() as con:
        con.register("frame", frame)
        con.execute("INSERT INTO policy_map BY NAME SELECT * FROM frame")
        con.unregister("frame")


def finetune(train, val, columns, label, start_ckpt, out_ckpt, args, round_):
    """Fine-tune one output from `start_ckpt`; save the weights to `out_ckpt`.

    Validation MSE before and after uses the same context rows, so the only difference is the weights.
    Returns (mse_before, mse_after, weight_change).
    """
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning import FinetunedTabPFNRegressor
    from tabpfn.model_loading import save_tabpfn_model

    X, y = train[columns].to_numpy(np.float32), train[label].to_numpy(np.float32)
    Xv, yv = val[columns].to_numpy(np.float32), val[label].to_numpy(np.float32)
    ctx = sample_context(train, args.context_rows, args.seed)
    Xc, yc = ctx[columns].to_numpy(np.float32), ctx[label].to_numpy(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = make_regressor(start_ckpt, args.device, n_estimators=2, seed=args.seed)
        before.fit(Xc, yc)
        pred_before = before.predict(Xv)
        grid = goal_grid()
        map_before = before.predict(grid[columns].to_numpy(np.float32)) if set(columns) <= set(grid) else None
        model = FinetunedTabPFNRegressor(
            device=args.device, epochs=args.epochs, learning_rate=args.lr, model_version=ModelVersion.V2,
            n_finetune_ctx_plus_query_samples=args.batch_rows, early_stopping=args.early_stopping, early_stopping_patience=3,
            n_estimators_finetune=2, n_estimators_validation=2, n_estimators_final_inference=2,
            random_state=args.seed, save_checkpoint_interval=None,
            experiment_logger=DatabaseLogger(args.name, round_, label),
            extra_regressor_kwargs={"model_path": str(start_ckpt), "ignore_pretraining_limits": True})
        model.fit(X, y, X_val=Xv, y_val=yv)
        # TabPFN's own writer: weights, config and the regression bin borders, loadable as model_path.
        save_tabpfn_model(model.finetuned_inference_regressor_, out_ckpt)
        after = make_regressor(out_ckpt, args.device, n_estimators=2, seed=args.seed)
        after.fit(Xc, yc)
        pred_after = after.predict(Xv)
        map_after = after.predict(grid[columns].to_numpy(np.float32)) if map_before is not None else None
    store_predictions(args.name, round_, label, "start", val, pred_before)
    store_predictions(args.name, round_, label, "finetuned", val, pred_after)
    if map_before is not None:  # only when the policy sees nothing but the goal
        store_policy_map(args.name, round_, label, "start", grid, map_before)
        store_policy_map(args.name, round_, label, "finetuned", grid, map_after)
    return (float(np.mean((pred_before - yv) ** 2)), float(np.mean((pred_after - yv) ** 2)),
            weight_change(start_ckpt, out_ckpt))


def evaluated(model, policy, round_):
    """True when a finished evaluation run of this policy and round exists."""
    frame = query("SELECT count(*) AS n FROM runs WHERE kind = 'eval' AND model = ? AND policy = ? AND "
                  "round IS NOT DISTINCT FROM ? AND finished_at IS NOT NULL", [model, policy, round_])
    return int(frame["n"][0]) > 0


def record_training(model, round_, label, rows, val_rows, seconds, before, after, change, ckpt, start, settings):
    with connect() as con:
        con.execute("INSERT INTO training VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [model, round_, label, datetime.now(), rows, val_rows, settings["epochs"], seconds, before,
                     after, change, str(ckpt), str(start), json.dumps(settings)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="model name; files go to data/nav/models/<name>")
    parser.add_argument("--rounds", type=int, default=3, help="rounds after round 0")
    parser.add_argument("--attempts", type=int, default=60, help="a→b attempts per round")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-rows", type=int, default=2000, help="context + query rows per training step")
    parser.add_argument("--train-rows", type=int, default=8000, help="cap on training examples per round")
    parser.add_argument("--context-rows", type=int, default=3000, help="examples TabPFN sees when walking")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="reuse rounds of this model that were already trained (after a crash)")
    parser.add_argument("--early-stopping", action="store_true",
                        help="keep the best weights by validation MSE (the start weights if nothing improves). "
                             "Off by default: the hindsight labels are already predicted almost exactly, so "
                             "validation MSE cannot tell better steering apart; the a→b evaluation judges instead")
    args = parser.parse_args()
    groups = tuple(args.groups.split(","))
    folder = data_dir() / "models" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    settings = dict(vars(args), groups=list(groups), git_commit=git_commit(), started=datetime.now().isoformat())
    name = "train.json" if not (args.resume and (folder / "train.json").exists()) else \
        "train-resumed-{:%Y%m%d-%H%M%S}.json".format(datetime.now())
    (folder / name).write_text(json.dumps(settings, indent=2))

    collect = query("SELECT c.* FROM cycles c JOIN runs r USING (run_id) WHERE r.kind = 'collect'")
    if collect.empty:
        raise SystemExit("no random-walk data: run `just collect` first")
    attempts_runs = []
    if not (args.resume and evaluated(args.name, "baseline", None)):
        evaluate(BaselinePolicy(), args.name, None)

    for round_ in range(args.rounds + 1):
        print("\n=== round {} ===".format(round_), flush=True)
        if args.resume:
            earlier = query("SELECT run_id FROM runs WHERE kind = 'attempts' AND model = ? AND round = ? "
                            "AND finished_at IS NOT NULL ORDER BY started_at DESC LIMIT 1", [args.name, round_ - 1])
            done = query("SELECT count(DISTINCT output) AS n FROM training WHERE model = ? AND round = ?",
                         [args.name, round_])["n"][0] == len(LABELS)
            saved = all(p.exists() for p in checkpoint_paths(folder, round_).values())
            context_file = folder / "round_{}_context.parquet".format(round_)
            if done and saved and context_file.exists() and (round_ == 0 or not earlier.empty):
                if round_ > 0:
                    attempts_runs.append(earlier["run_id"][0])
                context = pd.read_parquet(context_file)
                print("round {} already trained: reusing its checkpoints".format(round_), flush=True)
                for policy in (TabPFNNavPolicy(context, groups=groups, context_rows=args.context_rows),
                               TabPFNNavPolicy(context, checkpoints=checkpoint_paths(folder, round_),
                                               groups=groups, context_rows=args.context_rows)):
                    if not evaluated(args.name, policy.name, round_):
                        evaluate(policy, args.name, round_)
                continue
        if round_ > 0:  # the previous model attempts a→b tasks; their data joins the table
            previous = TabPFNNavPolicy(context, checkpoints=checkpoint_paths(folder, round_ - 1), groups=groups,
                                       context_rows=args.context_rows, seed=round_, noise=(0.03, 0.04),
                                       name="tabpfn-finetuned")
            recorder = Recorder("attempts", policy="tabpfn-finetuned+noise", model=args.name, round_=round_ - 1,
                                settings={"attempts": args.attempts, "noise": [0.03, 0.04]})
            summaries = run_episodes(previous, make_tasks(args.attempts, seed=1000 + round_, first_id=1000 * round_),
                                     recorder)
            recorder.close()
            attempts_runs.append(recorder.run_id)
            print("attempts: {}/{} reached b".format(sum(s["reached"] for s in summaries), len(summaries)), flush=True)
        cycles, episodes = collect, None
        if attempts_runs:
            marks = ",".join("'{}'".format(r) for r in attempts_runs)
            cycles = pd.concat([collect, query("SELECT * FROM cycles WHERE run_id IN ({})".format(marks))],
                               ignore_index=True)
            episodes = query("SELECT * FROM episodes WHERE run_id IN ({})".format(marks))
        table = training_table(cycles, episodes)
        columns = feature_columns(groups, table)
        train, val = split_by_episode(table, 0.1, args.seed)
        # Keep every success row (the signal that improves steering); fill the rest with hindsight rows.
        success = train[train["kind"] == "success"].iloc[: args.train_rows // 2]
        hindsight = sample_context(train[train["kind"] == "hindsight"], args.train_rows - len(success),
                                   args.seed + round_)
        train = pd.concat([hindsight, success], ignore_index=True)
        context = train
        context.to_parquet(folder / "round_{}_context.parquet".format(round_))
        print("training rows {} (of {}), validation rows {}".format(len(train), len(table), len(val)), flush=True)
        for label, out in checkpoint_paths(folder, round_).items():
            start = pretrained_checkpoint() if round_ == 0 else checkpoint_paths(folder, round_ - 1)[label]
            tic = time.time()
            before, after, change = finetune(train, val, columns, label, start, out, args, round_)
            seconds = time.time() - tic
            record_training(args.name, round_, label, len(train), len(val), seconds, before, after, change, out,
                            start, settings)
            print("{}: validation MSE {:.5f} -> {:.5f}, weights moved {:.2e} ({:.0f} s)".format(
                label, before, after, change, seconds), flush=True)
        evaluate(TabPFNNavPolicy(context, groups=groups, context_rows=args.context_rows), args.name, round_)
        evaluate(TabPFNNavPolicy(context, checkpoints=checkpoint_paths(folder, round_), groups=groups,
                                 context_rows=args.context_rows), args.name, round_)


if __name__ == "__main__":
    main()

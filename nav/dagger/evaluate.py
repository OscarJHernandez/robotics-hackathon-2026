"""Evaluation harness for the DAgger research loop: held-out plans, closed-loop summaries, offline metrics.

python -m nav.dagger.evaluate plan --run R --set screen-50 --seed 99 --host vendetta   # writes plans/eval-*.json
python -m nav.dagger.evaluate summarize --run R --round N [--name NAME]                # round_N/episodes -> eval_NAME.json
python -m nav.dagger.evaluate compare --run R --a ROUND_A --b ROUND_B                  # paired, same scene and start
python -m nav.dagger.evaluate offline --rows rows.parquet --pred pred.parquet [--run R --name NAME]

Driving is done by nav.dagger.worker (give it the plan file); this module only makes plans and reads results.
Sets: screen-50 = held-out scenes 100000..100049, final-200 = 100000..100199. beta is 0 and max_steps comes from
dagger.yaml. Use --seed 98 for the confirmation run (episode ids never collide with seed 99).
"""

import argparse
import json
import math
import statistics

from .stats import paired, sign_test, spearman, wilcoxon_p, wilson

SETS = {"screen-50": 50, "final-200": 200}
OUTCOMES = ["goal", "collision", "timeout", "fell"]
EPISODE_BASE = 90_000_000
SEED_STRIDE = 1_000_000
DEFAULT_SEED = 99
STOP_EXPERT = 0.02  # expert stop-and-turn: y_stride below this
STOP_PRED = 0.05  # predicted stop-and-turn: pred_stride below this
FREE_BINS = [("<0.5", None, 0.5), ("0.5-1", 0.5, 1.0), (">1", 1.0, None)]


def eval_plan(set_name, seed=DEFAULT_SEED):
    """Episodes of an evaluation set: [{"episode", "scene_id", "seed"}], one start per held-out scene."""
    from .config import HELDOUT_START
    n = SETS[set_name]
    base = EPISODE_BASE + SEED_STRIDE * (seed - DEFAULT_SEED)
    return [{"episode": base + i, "scene_id": HELDOUT_START + i, "seed": seed} for i in range(n)]


def write_plan(run, set_name, seed, host):
    from .config import run_dir
    folder = run_dir(run) / "plans"
    folder.mkdir(exist_ok=True)
    path = folder / "eval-{}-s{}-{}.json".format(set_name, seed, host)
    path.write_text(json.dumps(eval_plan(set_name, seed), indent=1))
    return path


def _episodes(table_or_list):
    if hasattr(table_or_list, "to_dict"):
        table_or_list = table_or_list.to_dict("records")
    episodes = list(table_or_list)
    if episodes and all("episode" in e for e in episodes):
        episodes = list({e["episode"]: e for e in episodes}.values())  # one per episode id
    return episodes


def closed_loop_summary(table_or_episode_list):
    """Counts, rates (with Wilson 95% intervals) per outcome, and steps to goal. Input: episode dicts or a DataFrame
    with `done_reason` and `steps` (one row per episode)."""
    episodes = _episodes(table_or_episode_list)
    n = len(episodes)
    out = {"episodes": n, "outcomes": {}}
    for name in OUTCOMES + sorted({e["done_reason"] for e in episodes} - set(OUTCOMES)):
        k = sum(e["done_reason"] == name for e in episodes)
        low, high = wilson(k, n)
        out["outcomes"][name] = {"count": k, "rate": round(k / n, 4) if n else None,
                                 "ci_low": round(low, 4), "ci_high": round(high, 4)}
    steps = [e["steps"] for e in episodes if e["done_reason"] == "goal"]
    out["mean_steps_to_goal"] = round(statistics.fmean(steps), 2) if steps else None
    out["median_steps_to_goal"] = float(statistics.median(steps)) if steps else None
    return out


def _bin_mask(df, low, high):
    mask = df["free_ahead"].notna()
    if low is not None:
        mask &= df["free_ahead"] >= low
    if high is not None:
        mask &= df["free_ahead"] < high
    return mask


def _mae(df):
    if len(df) == 0:
        return {"n": 0, "mae_stride": None, "mae_turn": None}
    return {"n": int(len(df)), "mae_stride": round(float((df["pred_stride"] - df["y_stride"]).abs().mean()), 4),
            "mae_turn": round(float((df["pred_turn"] - df["y_turn"]).abs().mean()), 4)}


def offline_metrics(df):
    """df columns: y_stride, y_turn, pred_stride, pred_turn, free_ahead (metres)."""
    out = {"all": _mae(df), "by_free_ahead": {}, "by_mode": {}}
    for label, low, high in FREE_BINS:
        out["by_free_ahead"][label] = _mae(df[_bin_mask(df, low, high)])
    expert_stop = df["y_stride"] < STOP_EXPERT
    out["by_mode"]["walk"] = _mae(df[~expert_stop])
    out["by_mode"]["stop_and_turn"] = _mae(df[expert_stop])
    pred_stop = df["pred_stride"] < STOP_PRED
    hit = int((expert_stop & pred_stop).sum())
    out["stop_and_turn"] = {
        "expert_stops": int(expert_stop.sum()), "predicted_stops": int(pred_stop.sum()),
        "recall": round(hit / int(expert_stop.sum()), 4) if expert_stop.any() else None,
        "precision": round(hit / int(pred_stop.sum()), 4) if pred_stop.any() else None}
    return out


# ---------------------------------------------------------------------------------------------------------------
# Diagnostics: one row per episode, failure types, paired comparison, offline report, rank agreement.
# Pure functions over pandas objects; no simulator needed.

WALL = 3.5355  # inner wall of the square room: |x|, |y| = this
STOP = 0.02  # stride below this is "stop and turn on the spot" (expert labels and student commands alike)
NEAR_WALL = 0.6
HARD_TURN = 0.2
LIDAR_RAYS = 36
START_SIDES = ["ahead", "side", "behind"]
DIST_BINS = ["<2", "2-3", "3-4", ">4"]


def _start_side(deg):
    return "ahead" if deg < 45 else ("side" if deg <= 120 else "behind")


def _dist_bin(d):
    return "<2" if d < 2 else ("2-3" if d < 3 else ("3-4" if d <= 4 else ">4"))


def _nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _episode_records(episodes):
    if hasattr(episodes, "drop_duplicates"):
        return episodes.drop_duplicates("episode").to_dict("records")
    return list(episodes)


def episode_table(rows, episodes, expert_episodes=None):
    """One row per episode: outcome, progress, path ratio against the expert, start geometry, student-vs-expert
    divergence counts, last-step lidar, and a `failure` label (rules applied in this order: goal, collision,
    never_turned, walked_past, spinning, else done_reason)."""
    import numpy as np
    import pandas as pd

    expert_path = {}
    for e in _episode_records(expert_episodes) if expert_episodes is not None else []:
        if e["done_reason"] == "goal" and e["path_length"] > 0:
            expert_path[(e["scene_id"], e["seed"])] = float(e["path_length"])
    lidar_cols = ["lr_{}".format(i) for i in range(LIDAR_RAYS)]
    has_lidar = all(c in rows.columns for c in lidar_cols)
    has_student = "student_stride" in rows.columns
    groups = {k: g.sort_values("step") for k, g in rows.groupby("episode")}
    out = []
    for e in _episode_records(episodes):
        g = groups.get(e["episode"])
        if g is None or len(g) == 0:
            continue
        last = g.iloc[-1]
        start_dist = e.get("start_dist")
        if _nan(start_dist):
            start_dist = math.hypot(e["goal_x"] - e["start_x"], e["goal_y"] - e["start_y"])
        start_dist, final_dist = float(start_dist), float(e["final_dist"])
        path = float(e["path_length"])
        outcome = e["done_reason"]
        bearings = np.degrees(np.abs(np.arctan2(g["goal_sin"].to_numpy(float), g["goal_cos"].to_numpy(float))))
        start_bearing = float(bearings[0])
        min_goal = float(g["goal_dist"].min())
        first_div, stop_ignored, wrong_stop = float("nan"), 0, 0
        if has_student:
            has = g["student_stride"].notna().to_numpy()
            s_stop = g["student_stride"].to_numpy(float) < STOP
            x_stop = g["expert_stride"].to_numpy(float) < STOP
            differ = has & (s_stop != x_stop)
            if differ.any():
                first_div = float(g["step"].to_numpy()[int(np.argmax(differ))])
            stop_ignored = int((has & x_stop & ~s_stop).sum())
            wrong_stop = int((has & s_stop & ~x_stop).sum())
        lidar_min = lidar_bearing = float("nan")
        if has_lidar:
            r = last[lidar_cols].to_numpy(float)
            if not np.isnan(r).all():
                i = int(np.nanargmin(r))
                lidar_min = float(r[i])
                lidar_bearing = float(i * 10 if i * 10 <= 180 else i * 10 - 360)
        near_wall = False
        if outcome == "collision":
            near_wall = bool(WALL - max(abs(float(last["x"])), abs(float(last["y"]))) <= NEAR_WALL)
        goal = outcome == "goal"
        if goal:
            failure = "goal"
        elif outcome == "collision":
            if _nan(lidar_bearing):
                failure = "collision"
            else:
                failure = "collision_front" if abs(lidar_bearing) <= 60 else "collision_side"
        elif float(bearings.min()) >= 45:
            failure = "never_turned"
        elif min_goal < 0.6 and final_dist > min_goal + 0.5:
            failure = "walked_past"
        elif path < 1.0 and len(g) >= 40:
            failure = "spinning"
        else:
            failure = outcome
        ref = expert_path.get((e["scene_id"], e["seed"]))
        out.append({
            "episode": e["episode"], "scene_id": e["scene_id"], "seed": e["seed"], "outcome": outcome,
            "goal": goal, "steps": int(e.get("steps", len(g))), "start_dist": start_dist, "final_dist": final_dist,
            "progress": (start_dist - final_dist) / start_dist if start_dist > 0 else float("nan"),
            "path_length": path, "path_ratio": path / ref if ref else float("nan"),
            "start_bearing_deg": start_bearing, "start_side": _start_side(start_bearing),
            "dist_bin": _dist_bin(start_dist), "first_divergence": first_div, "stop_ignored": stop_ignored,
            "wrong_stop": wrong_stop, "min_goal_dist": min_goal, "last_lidar_min": lidar_min,
            "last_lidar_bearing_deg": lidar_bearing,
            "last_free_ahead": float(last["free_ahead"]) if "free_ahead" in g.columns else float("nan"),
            "failure": failure, "near_wall": near_wall})
    return pd.DataFrame(out)


def _rate(k, n):
    low, high = wilson(k, n)
    return {"count": int(k), "rate": round(k / n, 4) if n else None, "ci_low": round(low, 4),
            "ci_high": round(high, 4)}


def summarize(table):
    """Rates with Wilson 95% intervals, progress, failure counts, goal rate by start side and distance bin,
    stop_ignored / wrong_stop totals, and the share of collisions that are collision_side."""
    n = len(table)
    out = {"n": n}
    for name in ["goal", "collision", "timeout", "stuck", "fell"]:
        out[name] = _rate(int((table["outcome"] == name).sum()), n)
    out["mean_progress"] = round(float(table["progress"].mean()), 4) if n else None
    out["median_progress"] = round(float(table["progress"].median()), 4) if n else None
    goal_steps = table.loc[table["goal"], "steps"]
    out["mean_steps_goal"] = round(float(goal_steps.mean()), 2) if len(goal_steps) else None
    out["failures"] = {str(k): int(v) for k, v in table["failure"].value_counts().items()}
    for key, col, order in [("by_start_side", "start_side", START_SIDES), ("by_dist_bin", "dist_bin", DIST_BINS)]:
        out[key] = {}
        for label in order:
            sub = table[table[col] == label]
            out[key][label] = {"n": int(len(sub)),
                               "goal_rate": round(float(sub["goal"].mean()), 4) if len(sub) else None}
    out["stop_ignored"] = int(table["stop_ignored"].sum())
    out["wrong_stop"] = int(table["wrong_stop"].sum())
    coll = table[table["outcome"] == "collision"]
    out["collision_side_share"] = (round(float((coll["failure"] == "collision_side").mean()), 4)
                                   if len(coll) else None)
    return out


def compare(table_a, table_b):
    """Paired comparison on the (scene_id, seed) pairs present in both tables."""
    import numpy as np
    a = table_a.drop_duplicates(["scene_id", "seed"]).set_index(["scene_id", "seed"])
    b = table_b.drop_duplicates(["scene_id", "seed"]).set_index(["scene_id", "seed"])
    keys = a.index.intersection(b.index)
    a, b = a.loc[keys], b.loc[keys]
    wins = int((a["goal"] & ~b["goal"]).sum())
    losses = int((~a["goal"] & b["goal"]).sum())
    diff = (a["progress"] - b["progress"]).to_numpy(float)
    diff = diff[~np.isnan(diff)]
    return {"pairs": int(len(keys)), "wins": wins, "losses": losses, "ties": int(len(keys)) - wins - losses,
            "sign_p": sign_test(wins, losses),
            "mean_progress_diff": float(diff.mean()) if len(diff) else None,
            "wilcoxon_p": wilcoxon_p(diff),
            "collisions_a": int((a["outcome"] == "collision").sum()),
            "collisions_b": int((b["outcome"] == "collision").sum())}


def _mae_pair(df, pred_stride, pred_turn):
    if len(df) == 0:
        return {"n": 0, "mae_stride": None, "mae_turn": None}
    return {"n": int(len(df)), "mae_stride": float((df[pred_stride] - df["y_stride"]).abs().mean()),
            "mae_turn": float((df[pred_turn] - df["y_turn"]).abs().mean())}


def offline_report(df, pred_stride="pred_stride", pred_turn="pred_turn"):
    """Offline errors against the expert's labels: MAE overall, by free_ahead bin and expert mode; stop recall and
    precision (predicted stop = stride < 0.02); and the hard turns (abs expert turn >= 0.2): predicted vs expert
    turn size and sign agreement. Plain floats and ints."""
    out = {"n": int(len(df)), "all": _mae_pair(df, pred_stride, pred_turn), "by_free_ahead": {}, "by_mode": {}}
    for label, low, high in FREE_BINS:
        out["by_free_ahead"][label] = _mae_pair(df[_bin_mask(df, low, high)], pred_stride, pred_turn)
    expert_stop = df["y_stride"] < STOP
    out["by_mode"]["walk"] = _mae_pair(df[~expert_stop], pred_stride, pred_turn)
    out["by_mode"]["stop"] = _mae_pair(df[expert_stop], pred_stride, pred_turn)
    pred_stop = df[pred_stride] < STOP
    hit = int((expert_stop & pred_stop).sum())
    n_exp, n_pred = int(expert_stop.sum()), int(pred_stop.sum())
    out["stop"] = {"expert_stops": n_exp, "predicted_stops": n_pred, "hits": hit,
                   "recall": hit / n_exp if n_exp else None, "precision": hit / n_pred if n_pred else None}
    hard = df[df["y_turn"].abs() >= HARD_TURN]
    if len(hard):
        pred_abs, exp_abs = float(hard[pred_turn].abs().mean()), float(hard["y_turn"].abs().mean())
        same = float(((hard[pred_turn] > 0) == (hard["y_turn"] > 0)).mean())
        out["hard_turns"] = {"n": int(len(hard)), "mean_abs_pred_turn": pred_abs, "mean_abs_expert_turn": exp_abs,
                             "ratio": pred_abs / exp_abs, "same_sign_share": same}
    else:
        out["hard_turns"] = {"n": 0, "mean_abs_pred_turn": None, "mean_abs_expert_turn": None, "ratio": None,
                             "same_sign_share": None}
    return out


def rank_agreement(offline_scores, closed_scores):
    """Spearman rho between offline and closed-loop scores of the same variants (higher = better in both; negate
    an error before passing it), over the shared keys, and each score's two top-ranked keys."""
    keys = sorted(offline_scores.keys() & closed_scores.keys())
    rho = spearman([offline_scores[k] for k in keys], [closed_scores[k] for k in keys])

    def top(scores):
        return sorted(keys, key=lambda k: -scores[k])[:2]
    return {"n": len(keys), "spearman": rho, "top2_offline": top(offline_scores), "top2_closed": top(closed_scores)}


def outcomes_by_key(episodes):
    """{(scene_id, seed): done_reason}"""
    return {(e["scene_id"], e["seed"]): e["done_reason"] for e in episodes}


def compare_rounds(a_episodes, b_episodes):
    a, b = outcomes_by_key(a_episodes), outcomes_by_key(b_episodes)
    out = paired(a, b)
    out["matched"] = len(a.keys() & b.keys())
    out["a_goal_rate"] = round(sum(v == "goal" for v in a.values()) / max(1, len(a)), 4)
    out["b_goal_rate"] = round(sum(v == "goal" for v in b.values()) / max(1, len(b)), 4)
    return out


def _print_summary(summary, log=print):
    log("{} episodes".format(summary["episodes"]))
    log("{:10s} {:>6s} {:>7s}  {}".format("outcome", "count", "rate", "95% Wilson"))
    for name, v in summary["outcomes"].items():
        rate = "-" if v["rate"] is None else "{:.1%}".format(v["rate"])
        log("{:10s} {:>6d} {:>7s}  {:.1%} to {:.1%}".format(name, v["count"], rate, v["ci_low"], v["ci_high"]))
    log("steps to goal: mean {}, median {}".format(summary["mean_steps_to_goal"], summary["median_steps_to_goal"]))


def _fmt(x):
    return "-" if x is None else "{:.4f}".format(x)


def _print_offline(metrics, log=print):
    log("{:22s} {:>8s} {:>11s} {:>9s}".format("group", "n", "MAE stride", "MAE turn"))

    def row(name, m):
        log("{:22s} {:>8d} {:>11s} {:>9s}".format(name, m["n"], _fmt(m["mae_stride"]), _fmt(m["mae_turn"])))
    row("all", metrics["all"])
    for k, m in metrics["by_free_ahead"].items():
        row("free_ahead " + k + " m", m)
    for k, m in metrics["by_mode"].items():
        row("mode " + k, m)
    s = metrics["stop_and_turn"]
    log("stop-and-turn: expert {} predicted {} recall {} precision {}".format(
        s["expert_stops"], s["predicted_stops"], s["recall"], s["precision"]))


def main():
    from .config import run_dir, save_json
    from .merge import read_episodes

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="write an evaluation plan for worker.py")
    p.add_argument("--run", required=True)
    p.add_argument("--set", dest="set_name", choices=sorted(SETS), required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--host", required=True)
    s = sub.add_parser("summarize", help="closed-loop summary of a merged evaluation round")
    s.add_argument("--run", required=True)
    s.add_argument("--round", required=True)
    s.add_argument("--name", default=None)
    c = sub.add_parser("compare", help="paired comparison of two rounds on the same scenes and starts")
    c.add_argument("--run", required=True)
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    o = sub.add_parser("offline", help="offline errors of predictions against the expert's labels")
    o.add_argument("--rows", required=True)
    o.add_argument("--pred", required=True)
    o.add_argument("--run", default=None)
    o.add_argument("--name", default="offline")
    args = parser.parse_args()

    if args.cmd == "plan":
        path = write_plan(args.run, args.set_name, args.seed, args.host)
        print("wrote {} ({} episodes)".format(path, len(json.loads(path.read_text()))))
    elif args.cmd == "summarize":
        summary = closed_loop_summary(read_episodes(args.run, args.round))
        summary["round"] = args.round
        _print_summary(summary)
        save_json(run_dir(args.run) / "eval_{}.json".format(args.name or "round{}".format(args.round)), summary)
    elif args.cmd == "compare":
        result = compare_rounds(read_episodes(args.run, args.a), read_episodes(args.run, args.b))
        result.update({"a": args.a, "b": args.b})
        print("A (round {}) goal {:.1%}, B (round {}) goal {:.1%}; matched {}; A wins {}, B wins {}, ties {}; "
              "sign test p = {:.4f}".format(args.a, result["a_goal_rate"], args.b, result["b_goal_rate"],
                                            result["matched"], result["wins"], result["losses"], result["ties"],
                                            result["p"]))
        save_json(run_dir(args.run) / "eval_compare_{}_vs_{}.json".format(args.a, args.b), result)
    else:
        import pandas as pd
        rows, pred = pd.read_parquet(args.rows), pd.read_parquet(args.pred)
        if len(rows) != len(pred):
            raise SystemExit("rows ({}) and pred ({}) differ in length".format(len(rows), len(pred)))
        df = rows[["y_stride", "y_turn", "free_ahead"]].reset_index(drop=True).join(
            pred[["pred_stride", "pred_turn"]].reset_index(drop=True))
        metrics = offline_metrics(df)
        _print_offline(metrics)
        if args.run:
            save_json(run_dir(args.run) / "eval_{}.json".format(args.name), metrics)


if __name__ == "__main__":
    main()

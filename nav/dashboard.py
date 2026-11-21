"""Write a self-contained HTML dashboard of one model's fine-tuning rounds.

    uv run --group ml python nav/dashboard.py [--model NAME] [--out PATH]

Reads the navigation database read-only (see nav/db.py) and writes one HTML file with all data embedded
as JSON: KPIs of the latest round, metrics across rounds, the fine-tuning audit (training rows, loss
curves), how the regression changed (actual vs predicted, the steering map), every evaluation episode
as a path plot with its per-cycle table, and an inventory of the runs. No external scripts: the file
works offline. Default model: the one with the most recent `training` row. Default output:
<data folder>/dashboard.html.
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from db import db_path, query  # noqa: E402

TEMPLATE = HERE / "dashboard_template.html"
STEP_BUDGET = 400_000  # most path points embedded in total; steps are thinned to stay under it
MIN_STEP_STRIDE = 5  # keep at most every 5th 20 ms pose (one per 0.1 s)


def tables():
    return set(query("SELECT table_name FROM information_schema.tables")["table_name"])


def clean(value):
    """A JSON-safe Python value (no NaN, no numpy, timestamps as text)."""
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (int, bool, str)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ", timespec="seconds") if isinstance(value, datetime) else str(value)
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def records(frame, digits=None):
    out = []
    for row in frame.itertuples(index=False):
        rec = {}
        for name, value in zip(frame.columns, row):
            value = clean(value)
            if digits is not None and isinstance(value, float):
                value = round(value, digits)
            rec[name] = value
        out.append(rec)
    return out


def rnd(value, digits=4):
    value = clean(value)
    return round(value, digits) if isinstance(value, float) else value


def default_model():
    frame = query("SELECT model FROM training ORDER BY created_at DESC LIMIT 1")
    if len(frame):
        return frame["model"][0]
    frame = query("SELECT model FROM runs WHERE kind = 'eval' AND model <> '' ORDER BY started_at DESC LIMIT 1")
    if len(frame):
        return frame["model"][0]
    raise SystemExit("no trained model in {}: run nav/train.py first".format(db_path()))


def eval_runs(model):
    """The latest eval run per (policy, round) of this model, plus the latest baseline run."""
    runs = query("""
        SELECT run_id, policy, round, model, started_at FROM runs
        WHERE kind = 'eval' AND finished_at IS NOT NULL AND (model = ? OR policy = 'baseline')
        ORDER BY started_at""", [model])
    chosen = {}
    for row in runs.itertuples(index=False):
        if row.policy == "baseline":
            key = ("baseline", None)
            # prefer this model's baseline; else the most recent baseline of any model
            if key in chosen and chosen[key]["model"] == model and row.model != model:
                continue
        else:
            key = (row.policy, None if clean(row.round) is None else int(row.round))
        chosen[key] = {"run_id": row.run_id, "policy": row.policy, "round": key[1], "model": row.model}
    return list(chosen.values())


def paired(run_a, run_b, max_seconds=40.0, seed=0):
    """Compare two eval runs task by task. Positive gains mean run_a is better.

    Seconds to b counts a trip that never reached b as the time limit (40 s). Returns per-task gains,
    wins/losses and a bootstrap 95% interval of the mean gain for seconds and for directness.
    """
    import numpy as np
    frame = query("""
        SELECT a.task, a.reached AS ra, b.reached AS rb,
               CASE WHEN a.reached THEN a.seconds ELSE ? END AS sa, CASE WHEN b.reached THEN b.seconds ELSE ? END AS sb,
               a.directness AS da, b.directness AS db
        FROM episodes a JOIN episodes b USING (task)
        WHERE a.run_id = ? AND b.run_id = ? ORDER BY task""", [max_seconds, max_seconds, run_a, run_b])
    rng = np.random.default_rng(seed)
    out = {"tasks": len(frame)}
    for name, gain in (("seconds", frame["sb"] - frame["sa"]), ("directness", frame["da"] - frame["db"])):
        g = gain.to_numpy(float)
        boots = [rng.choice(g, len(g)).mean() for _ in range(2000)] if len(g) else [0.0]
        out[name] = {"mean": float(g.mean()) if len(g) else 0.0, "low": float(np.percentile(boots, 2.5)),
                     "high": float(np.percentile(boots, 97.5)), "wins": int((g > 1e-9).sum()),
                     "losses": int((g < -1e-9).sum())}
    out["reached"] = (int(frame["ra"].sum()), int(frame["rb"].sum()))
    return out


def verdict(summary, training):
    """A plain, honest answer to "is training making the robot better?".

    Trained and untrained TabPFN (same examples) do the same 20 held-out trips; compare them trip by
    trip on seconds to reach b. "Yes" only when the trained model is faster on most trips AND the
    95% interval of the mean gain is above zero. A gain inside the noise is reported as "no clear
    difference", never as progress.
    """
    def pick(policy, round_):
        return next((s for s in summary if s["policy"] == policy and s["round"] == round_), None)

    base = next((s for s in summary if s["policy"] == "baseline"), None)
    rounds = sorted({s["round"] for s in summary if s["policy"] == "tabpfn-finetuned" and s["round"] is not None})
    rounds = [r for r in rounds if pick("tabpfn-pretrained", r)]
    out = {"status": "waiting", "baseline": base, "round": None, "rounds": len(rounds), "checks": []}
    if not rounds:
        out["headline"] = "Not known yet."
        out["detail"] = "No training round has been tested yet. The answer appears when round 0 is tested."
        return out
    r = rounds[-1]
    ft, pre = pick("tabpfn-finetuned", r), pick("tabpfn-pretrained", r)
    p = paired(ft["run_id"], pre["run_id"])
    sec, dirn = p["seconds"], p["directness"]
    moved = [t for t in training if t["round"] == r and (t["weight_change"] or 0) > 1e-6]
    clear = sec["low"] > 0 and sec["wins"] > p["tasks"] / 2
    worse = sec["high"] < 0
    out.update(round=r, paired=p)
    out["checks"] = [
        {"ok": bool(moved), "text": "Training changed the model's weights" if moved else
         "Training did not change the model's weights"},
        {"ok": clear, "text": "Trip by trip (20 test trips): the trained model was faster on {} and slower on {}. "
         "Average gain {:+.1f} s per trip; likely range {:+.1f} to {:+.1f} s.".format(
             sec["wins"], sec["losses"], sec["mean"], sec["low"], sec["high"])},
        {"ok": dirn["low"] > 0, "text": "Path straightness: better on {}, worse on {} trips; average {:+.2f} "
         "(likely range {:+.2f} to {:+.2f}).".format(dirn["wins"], dirn["losses"], dirn["mean"], dirn["low"], dirn["high"])},
    ]
    if clear and moved:
        out["status"], out["headline"] = "working", "Yes: training is making the robot better."
        out["detail"] = "After round {}, the trained model reaches b faster on most trips, by more than the noise.".format(r)
    elif worse:
        out["status"], out["headline"] = "not_working", "No: training is making the robot worse."
        out["detail"] = "After round {}, the trained model is clearly slower than the untrained one.".format(r)
    else:
        out["status"], out["headline"] = "unclear", "Not yet: no clear difference."
        out["detail"] = ("After round {}, the trained and untrained models perform about the same; any difference "
                         "is within the noise. This is not progress yet.".format(r))
    return out

def fmt_cycles(s):
    return "-" if s.get("cycles_reached") is None else "{:.1f}".format(s["cycles_reached"])


def build(model, out_path):
    have = tables()
    meta = {
        "model": model, "db": str(db_path()), "generated": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "out": str(out_path),
    }
    commits = query("SELECT DISTINCT git_commit FROM runs WHERE model = ? AND git_commit <> ''", [model])
    meta["git_commits"] = sorted(commits["git_commit"].tolist())

    # ---- evaluation episodes
    chosen = eval_runs(model)
    run_ids = [r["run_id"] for r in chosen]
    episodes = []
    summary = []
    steps_by_ep = {}
    cycles_by_ep = {}
    if run_ids:
        marks = ",".join("?" * len(run_ids))
        ep = query("SELECT * FROM episodes WHERE run_id IN ({}) ORDER BY run_id, episode".format(marks), run_ids)
        info = {r["run_id"]: r for r in chosen}
        ep["policy"] = ep["run_id"].map(lambda r: info[r]["policy"])
        ep["round"] = ep["run_id"].map(lambda r: info[r]["round"])
        n_steps = query("SELECT count(*) AS n FROM steps WHERE run_id IN ({})".format(marks), run_ids)["n"][0]
        stride = max(MIN_STEP_STRIDE, int(math.ceil(n_steps / STEP_BUDGET)))
        st = query("""
            SELECT run_id, episode, x, y FROM (
                SELECT *, row_number() OVER (PARTITION BY run_id, episode ORDER BY t) - 1 AS i,
                       count(*) OVER (PARTITION BY run_id, episode) AS n
                FROM steps WHERE run_id IN ({})) WHERE i % {} = 0 OR i = n - 1
            ORDER BY run_id, episode, t""".format(marks, stride), run_ids)
        for (run_id, episode), part in st.groupby(["run_id", "episode"], sort=False):
            steps_by_ep[(run_id, int(episode))] = [[round(float(x), 3), round(float(y), 3)]
                                                   for x, y in zip(part["x"], part["y"])]
        cyc_cols = ["cycle", "t_rel", "target_dist", "goal_dist", "goal_sin", "goal_cos", "horizon", "stride",
                    "turn", "dist_next", "reached"]
        cyc = query("SELECT run_id, episode, {} FROM cycles WHERE run_id IN ({}) ORDER BY run_id, episode, cycle"
                    .format(", ".join(cyc_cols), marks), run_ids)
        for (run_id, episode), part in cyc.groupby(["run_id", "episode"], sort=False):
            rows = []
            for r in part.itertuples(index=False):
                bearing = math.degrees(math.atan2(r.goal_sin, r.goal_cos))
                rows.append([int(r.cycle), rnd(r.t_rel, 2), rnd(r.target_dist), rnd(r.goal_dist),
                             round(bearing, 1), rnd(r.horizon, 1), rnd(r.stride), rnd(r.turn), rnd(r.dist_next),
                             bool(r.reached)])
            cycles_by_ep[(run_id, int(episode))] = rows
        for rec in records(ep, digits=4):
            key = (rec["run_id"], rec["episode"])
            rec["path"] = steps_by_ep.get(key, [])
            rec["has_cycles"] = key in cycles_by_ep
            episodes.append(rec)
        summary = records(query("""
            SELECT run_id, count(*) AS n, avg(reached::INT) AS success, avg(final_dist) AS final_dist,
                   avg(directness) AS directness, avg(CASE WHEN reached THEN cycles END) AS cycles_reached,
                   avg(fell::INT) AS fell
            FROM episodes WHERE run_id IN ({}) GROUP BY run_id""".format(marks), run_ids), digits=5)
        for rec in summary:
            rec["policy"] = info[rec["run_id"]]["policy"]
            rec["round"] = info[rec["run_id"]]["round"]
        meta["step_stride"] = stride
    cycles_json = {"{}|{}".format(k[0], k[1]): v for k, v in cycles_by_ep.items()}

    # ---- training audit
    training = records(query("""
        SELECT round, output, created_at, train_rows, val_rows, epochs, seconds, val_mse_before, val_mse_after,
               weight_change, checkpoint, start_checkpoint
        FROM training WHERE model = ? ORDER BY round, output, created_at""", [model]))
    log = {}
    frame = query("""
        SELECT round, output, kind, step, metric, value FROM training_log
        WHERE model = ? AND metric IN ('train/loss', 'val/MSE', 'train/mean_loss', 'train/lr')
        ORDER BY round, output, metric, step""", [model])
    for (round_, output, metric), part in frame.groupby(["round", "output", "metric"], sort=False):
        log.setdefault("{}|{}".format(int(round_), output), {})[metric] = [
            [int(s), float(v)] for s, v in zip(part["step"], part["value"]) if clean(v) is not None]

    # ---- regression: validation predictions and the steering map
    preds = {}
    frame = query("""
        SELECT round, output, weights, actual, predicted FROM predictions WHERE model = ?
        ORDER BY round, output, weights""", [model])
    for (round_, output, weights), part in frame.groupby(["round", "output", "weights"], sort=False):
        preds["{}|{}|{}".format(int(round_), output, weights)] = [
            [round(float(a), 5), round(float(p), 5)] for a, p in zip(part["actual"], part["predicted"])]
    policy_map = None
    if "policy_map" in have:
        policy_map = {}
        frame = query("""
            SELECT round, output, weights, goal_dist, bearing, horizon, predicted FROM policy_map WHERE model = ?
            ORDER BY round, output, weights, goal_dist, bearing""", [model])
        for (round_, output, weights), part in frame.groupby(["round", "output", "weights"], sort=False):
            policy_map["{}|{}|{}".format(int(round_), output, weights)] = [
                [round(float(d), 3), round(float(b), 2), round(float(h), 1), round(float(p), 5)]
                for d, b, h, p in zip(part["goal_dist"], part["bearing"], part["horizon"], part["predicted"])]

    # ---- inventory
    runs = records(query("""
        SELECT r.run_id, r.kind, r.policy, r.model, r.round, r.started_at, r.finished_at, r.git_commit,
               (SELECT count(*) FROM episodes e WHERE e.run_id = r.run_id) AS n_episodes,
               (SELECT count(*) FROM cycles c WHERE c.run_id = r.run_id) AS n_cycles
        FROM runs r ORDER BY r.started_at""" if "cycles" in have else """
        SELECT r.run_id, r.kind, r.policy, r.model, r.round, r.started_at, r.finished_at, r.git_commit,
               (SELECT count(*) FROM episodes e WHERE e.run_id = r.run_id) AS n_episodes, 0 AS n_cycles
        FROM runs r ORDER BY r.started_at"""))
    sql_examples = [
        ("Success rate by policy and round (evaluation runs)",
         "SELECT e.policy, e.round, count(*) AS episodes,\n       avg(e.reached::INT) AS success_rate,\n"
         "       avg(e.final_dist) AS mean_final_dist_m, avg(e.directness) AS mean_directness\n"
         "FROM episodes e JOIN runs r USING (run_id)\nWHERE r.kind = 'eval' AND (e.model = '{m}' OR e.policy = 'baseline')\n"
         "GROUP BY ALL ORDER BY e.round NULLS FIRST, e.policy;".format(m=model)),
        ("Validation error before and after fine-tuning",
         "SELECT round, output, train_rows, val_mse_before, val_mse_after,\n"
         "       100 * (val_mse_after - val_mse_before) / val_mse_before AS pct_change\n"
         "FROM training WHERE model = '{m}' ORDER BY round, output;".format(m=model)),
        ("Every command one episode received (first eval run of the latest round)",
         "SELECT cycle, t_rel, target_dist, degrees(atan2(goal_sin, goal_cos)) AS bearing_deg,\n"
         "       horizon, stride, turn, dist_next, reached\nFROM cycles\n"
         "WHERE run_id = (SELECT run_id FROM runs WHERE kind = 'eval' AND model = '{m}'\n"
         "                ORDER BY round DESC, started_at DESC LIMIT 1)\n  AND episode = 0\nORDER BY cycle;"
         .format(m=model)),
    ]
    return {
        "verdict": verdict(summary, training),
        "meta": meta, "summary": summary, "episodes": episodes, "cycles": cycles_json, "training": training,
        "log": log, "preds": preds, "policy_map": policy_map, "runs": runs,
        "sql": [{"title": t, "sql": s} for t, s in sql_examples],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="model name (default: the most recently trained)")
    parser.add_argument("--out", help="output HTML file (default: <data folder>/dashboard.html)")
    args = parser.parse_args()
    if not db_path().exists():
        raise SystemExit("no navigation data yet: {} does not exist".format(db_path()))
    model = args.model or default_model()
    out = Path(args.out) if args.out else db_path().parent / "dashboard.html"
    data = build(model, out.resolve())
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("__DASHBOARD_DATA__", payload).replace(
        "__DASHBOARD_TITLE__", "Navigation training: " + model.replace("<", "&lt;"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(out.resolve())
    print("{:.1f} MB, {} evaluation episodes, {} training rows".format(
        len(html.encode()) / 1e6, len(data["episodes"]), len(data["training"])), file=sys.stderr)


if __name__ == "__main__":
    main()

"""Write a self-contained HTML dashboard of the TabPFN controllers.

    uv run --group ml python nav/dashboard.py [--controller NAME] [--out PATH]

A controller is a set of recorded examples given to TabPFN's built-in `fit()` (see nav/controller.py);
no weights are trained. The dashboard reads the navigation database read-only (see nav/db.py) and
writes one HTML file with all data embedded as JSON: a plain answer to "does TabPFN walk the robot
from a to b?", one row per controller (examples, held-out prediction quality, evaluation results)
next to the hand-written baseline, a comparison chart, how each controller's regression behaves
(actual vs predicted, the steering map), every evaluation episode as a path plot with its per-cycle
table, and an inventory of the runs. No external scripts: the file works offline.
Default output: <data folder>/dashboard.html.
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
MAX_SCATTER = 800  # held-out points embedded per controller and output
MAX_SECONDS = 40.0  # time limit of an evaluation trip; a trip that never reached b counts as this


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


def controllers(only=None):
    """One row per controller (the newest if a name appears twice)."""
    if "controllers" not in tables():
        return []
    rows = records(query("""
        SELECT name, kind, created_at, context_rows, total_examples, val_rows, features, outputs, fit_seconds,
               context_file, git_commit, settings
        FROM controllers QUALIFY row_number() OVER (PARTITION BY name ORDER BY created_at DESC) = 1
        ORDER BY created_at"""))
    if only:
        rows = [r for r in rows if r["name"] == only]
    return rows


def latest_eval_runs(names):
    """The latest eval run per controller name, and the baseline run to compare with.

    Old runs from a removed experiment (a `round` value, kind 'attempts') are ignored. A finished
    run is preferred; an unfinished one is used only when the controller has nothing else.
    """
    if not names:
        return {}, {}
    marks = ",".join("?" * len(names))
    runs = records(query("""
        SELECT run_id, policy, model, started_at, finished_at FROM runs
        WHERE kind = 'eval' AND round IS NULL AND model IN ({}) ORDER BY started_at""".format(marks), names))
    chosen, baselines = {}, {}
    for r in runs:
        target = baselines if r["policy"] == "baseline" else chosen
        prev = target.get(r["model"])
        if prev is None or r["finished_at"] is not None or prev["finished_at"] is None:
            target[r["model"]] = r
    return chosen, baselines


def episode_summary(run_ids):
    if not run_ids:
        return {}
    marks = ",".join("?" * len(run_ids))
    frame = query("""
        SELECT run_id, count(*) AS n, sum(reached::INT) AS reached, avg(reached::INT) AS success,
               avg(final_dist) AS final_dist, avg(directness) AS directness,
               avg(CASE WHEN reached THEN seconds END) AS seconds_reached, avg(fell::INT) AS fell
        FROM episodes WHERE run_id IN ({}) GROUP BY run_id""".format(marks), run_ids)
    return {r["run_id"]: r for r in records(frame, digits=5)}


def paired(run_a, run_b, seed=0):
    """Compare two eval runs trip by trip. Positive gains mean run_a is better.

    Seconds to b counts a trip that never reached b as the time limit. Returns per-metric wins and
    losses and a bootstrap 95% interval of the mean gain, for seconds and for directness.
    """
    import numpy as np
    frame = query("""
        SELECT a.task, CASE WHEN a.reached THEN a.seconds ELSE ? END AS sa,
               CASE WHEN b.reached THEN b.seconds ELSE ? END AS sb, a.directness AS da, b.directness AS db
        FROM episodes a JOIN episodes b USING (task)
        WHERE a.run_id = ? AND b.run_id = ? ORDER BY task""", [MAX_SECONDS, MAX_SECONDS, run_a, run_b])
    rng = np.random.default_rng(seed)
    out = {"tasks": len(frame)}
    for name, gain in (("seconds", frame["sb"] - frame["sa"]), ("directness", frame["da"] - frame["db"])):
        g = gain.to_numpy(float)
        boots = [rng.choice(g, len(g)).mean() for _ in range(2000)] if len(g) else [0.0]
        out[name] = {"mean": float(g.mean()) if len(g) else 0.0, "low": float(np.percentile(boots, 2.5)),
                     "high": float(np.percentile(boots, 97.5)), "wins": int((g > 1e-9).sum()),
                     "losses": int((g < -1e-9).sum())}
    return out


def answer(rows, baseline):
    """A plain, honest answer to "does TabPFN walk the robot from a to b?".

    Judged on the best controller (highest mean directness on its latest evaluation): "Yes" when it
    reaches b on at least 90% of the held-out trips, "Partly" from 50%, else "No". The comparison
    with the hand-written baseline is trip by trip; a difference inside the noise is called that.
    """
    tested = [r for r in rows if r.get("eval")]
    out = {"status": "waiting", "best": None, "baseline": baseline, "checks": []}
    if not tested:
        out["headline"] = "Not known yet."
        out["detail"] = ("No controller has been evaluated yet. Build one with `just controller`; the answer "
                         "appears when its 20 test trips are recorded.")
        return out
    best = tested[0]
    e = best["eval"]
    out["best"] = best["name"]
    n, reached = int(e["n"] or 0), int(e["reached"] or 0)
    rate = reached / n if n else 0.0
    if rate >= 0.9:
        out["status"], word = "working", "Yes"
    elif rate >= 0.5:
        out["status"], word = "partly", "Partly"
    else:
        out["status"], word = "not_working", "No"
    out["headline"] = "{}: controller {} reaches b on {} of {} test trips.".format(word, best["name"], reached, n)
    sec = "{:.1f} s".format(e["seconds_reached"]) if e["seconds_reached"] is not None else "no trip reached b"
    out["detail"] = ("Its paths have a mean directness of {:.2f} (1 = a straight line from a to b) and the trips "
                     "that reached b took {} on average. It is the best of {} evaluated controller{}.".format(
                         e["directness"] or 0, sec, len(tested), "" if len(tested) == 1 else "s"))
    if baseline and best.get("baseline_run"):
        p = paired(best["run_id"], best["baseline_run"])
        out["paired"] = p
        s, d = p["seconds"], p["directness"]

        def verdict(m, unit_better, unit_worse):
            if m["low"] > 0:
                return unit_better
            if m["high"] < 0:
                return unit_worse
            return "about the same (the difference is within the trip-to-trip noise)"

        out["checks"] = [
            {"ok": rate >= 0.9, "text": "Reaches b: {} of {} trips (baseline: {} of {}).".format(
                reached, n, int(baseline["reached"] or 0), int(baseline["n"] or 0))},
            {"ok": s["low"] > 0, "text": "Time to b, trip by trip against the baseline: faster on {}, slower on {}; "
             "average {:+.1f} s per trip (likely range {:+.1f} to {:+.1f} s). Verdict: {}.".format(
                 s["wins"], s["losses"], s["mean"], s["low"], s["high"],
                 verdict(s, "faster than the baseline", "slower than the baseline"))},
            {"ok": d["low"] > 0, "text": "Path straightness against the baseline: better on {}, worse on {} trips; "
             "average {:+.2f} (likely range {:+.2f} to {:+.2f}). Verdict: {}.".format(
                 d["wins"], d["losses"], d["mean"], d["low"], d["high"],
                 verdict(d, "straighter than the baseline", "less straight than the baseline"))},
        ]
        faster = verdict(s, "faster", "slower")
        straighter = verdict(d, "straighter", "less straight")
        if faster.startswith("about") and straighter.startswith("about"):
            out["detail"] += " Against the hand-written baseline it is about the same: the differences are within the noise."
        else:
            parts = ["time to b: " + (faster if not faster.startswith("about") else "about the same"),
                     "paths: " + (straighter if not straighter.startswith("about") else "about the same")]
            out["detail"] += " Against the hand-written baseline on the same trips: {}.".format("; ".join(parts))
    return out


def regression(rows, have):
    """Held-out actual vs predicted per controller and output, its RMSE and R², and the steering map."""
    names = [r["name"] for r in rows]
    preds, quality, policy_map = {}, {}, {}
    if not names:
        return preds, quality, policy_map
    marks = ",".join("?" * len(names))
    if "predictions" in have:
        q = query("""
            SELECT model, output, count(*) AS n, sqrt(avg((predicted - actual) ^ 2)) AS rmse,
                   1 - sum((predicted - actual) ^ 2) / nullif(sum((actual - a.mu) ^ 2), 0) AS r2
            FROM predictions p JOIN (SELECT model, output, avg(actual) AS mu FROM predictions
                                     WHERE round IS NULL GROUP BY ALL) a USING (model, output)
            WHERE p.round IS NULL AND p.model IN ({}) GROUP BY ALL""".format(marks), names)
        for r in records(q, digits=6):
            quality.setdefault(r["model"], {})[r["output"]] = {"n": r["n"], "rmse": r["rmse"], "r2": r["r2"]}
        frame = query("""
            SELECT model, output, actual, predicted FROM (
                SELECT *, row_number() OVER (PARTITION BY model, output ORDER BY run_id, episode, cycle) AS i
                FROM predictions WHERE round IS NULL AND model IN ({})) WHERE i <= {}
            ORDER BY model, output""".format(marks, MAX_SCATTER), names)
        for (model, output), part in frame.groupby(["model", "output"], sort=False):
            preds["{}|{}".format(model, output)] = [
                [round(float(a), 5), round(float(p), 5)] for a, p in zip(part["actual"], part["predicted"])]
    if "policy_map" in have:
        frame = query("""
            SELECT model, output, goal_dist, bearing, predicted FROM policy_map
            WHERE round IS NULL AND model IN ({}) ORDER BY model, output, goal_dist, bearing""".format(marks), names)
        for (model, output), part in frame.groupby(["model", "output"], sort=False):
            policy_map["{}|{}".format(model, output)] = [
                [round(float(d), 3), round(float(b), 2), round(float(p), 5)]
                for d, b, p in zip(part["goal_dist"], part["bearing"], part["predicted"])]
    return preds, quality, policy_map


def episodes_of(run_ids, have):
    """Every episode of these runs with its thinned path and its per-cycle commands."""
    if not run_ids:
        return [], {}, MIN_STEP_STRIDE
    marks = ",".join("?" * len(run_ids))
    ep = query("SELECT * FROM episodes WHERE run_id IN ({}) ORDER BY run_id, episode".format(marks), run_ids)
    n_steps = query("SELECT count(*) AS n FROM steps WHERE run_id IN ({})".format(marks), run_ids)["n"][0]
    stride = max(MIN_STEP_STRIDE, int(math.ceil(n_steps / STEP_BUDGET)))
    st = query("""
        SELECT run_id, episode, x, y FROM (
            SELECT *, row_number() OVER (PARTITION BY run_id, episode ORDER BY t) - 1 AS i,
                   count(*) OVER (PARTITION BY run_id, episode) AS n
            FROM steps WHERE run_id IN ({})) WHERE i % {} = 0 OR i = n - 1
        ORDER BY run_id, episode, t""".format(marks, stride), run_ids)
    steps_by_ep = {}
    for (run_id, episode), part in st.groupby(["run_id", "episode"], sort=False):
        steps_by_ep[(run_id, int(episode))] = [[round(float(x), 3), round(float(y), 3)]
                                               for x, y in zip(part["x"], part["y"])]
    cycles_by_ep = {}
    if "cycles" in have:
        cyc_cols = ["cycle", "t_rel", "target_dist", "goal_dist", "goal_sin", "goal_cos", "horizon", "stride",
                    "turn", "dist_next", "reached"]
        cyc = query("SELECT run_id, episode, {} FROM cycles WHERE run_id IN ({}) ORDER BY run_id, episode, cycle"
                    .format(", ".join(cyc_cols), marks), run_ids)
        for (run_id, episode), part in cyc.groupby(["run_id", "episode"], sort=False):
            rows = []
            for r in part.itertuples(index=False):
                bearing = math.degrees(math.atan2(r.goal_sin, r.goal_cos)) if clean(r.goal_sin) is not None else None
                rows.append([int(r.cycle), rnd(r.t_rel, 2), rnd(r.target_dist), rnd(r.goal_dist),
                             None if bearing is None else round(bearing, 1), rnd(r.horizon, 1), rnd(r.stride),
                             rnd(r.turn), rnd(r.dist_next), bool(r.reached)])
            cycles_by_ep["{}|{}".format(run_id, int(episode))] = rows
    out = []
    for rec in records(ep, digits=4):
        key = (rec["run_id"], rec["episode"])
        rec["path"] = steps_by_ep.get(key, [])
        rec.pop("round", None)
        out.append(rec)
    return out, cycles_by_ep, stride


def build(out_path, only=None):
    have = tables()
    meta = {"db": str(db_path()), "generated": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "out": str(out_path), "filter": only}
    rows = controllers(only)
    names = [r["name"] for r in rows]
    chosen, baselines = latest_eval_runs(names)
    summary = episode_summary([r["run_id"] for r in chosen.values()] + [r["run_id"] for r in baselines.values()])
    preds, quality, policy_map = regression(rows, have)

    for r in rows:
        r.pop("settings", None)
        r["features"] = (r["features"] or "").split(",") if r["features"] else []
        r["outputs"] = (r["outputs"] or "").split(",") if r["outputs"] else []
        run = chosen.get(r["name"])
        r["run_id"] = run["run_id"] if run else None
        r["run_finished"] = bool(run and run["finished_at"])
        r["eval"] = summary.get(run["run_id"]) if run else None
        base = baselines.get(r["name"])
        r["baseline_run"] = base["run_id"] if base else None
        r["quality"] = quality.get(r["name"], {})
    rows.sort(key=lambda r: (r["eval"] is None, -((r["eval"] or {}).get("directness") or 0),
                             -((r["eval"] or {}).get("success") or 0)))

    # the baseline shown everywhere: the one evaluated with the best controller, else the latest one
    base_run = next((r["baseline_run"] for r in rows if r["eval"] and r["baseline_run"]), None)
    if base_run is None and baselines:
        base_run = max(baselines.values(), key=lambda b: str(b["started_at"]))["run_id"]
    baseline = dict(summary.get(base_run, {}), run_id=base_run) if base_run else None
    if baseline is not None:
        baseline["all_runs"] = len(baselines)
        others = [summary[b["run_id"]] for b in baselines.values() if b["run_id"] in summary]
        baseline["identical"] = len({(o["reached"], o["directness"], o["seconds_reached"]) for o in others}) <= 1

    ep_runs = [r["run_id"] for r in rows if r["run_id"]] + ([base_run] if base_run else [])
    episodes, cycles, stride = episodes_of(ep_runs, have)
    meta["step_stride"] = stride
    run_label = {r["run_id"]: r["name"] for r in rows if r["run_id"]}
    if base_run:
        run_label[base_run] = "baseline"
    for e in episodes:
        e["controller"] = run_label.get(e["run_id"], e["run_id"])

    # ---- inventory: runs of this project's controllers and the random-walk data they were built from
    n_cycles = "(SELECT count(*) FROM cycles c WHERE c.run_id = r.run_id)" if "cycles" in have else "0"
    all_names = [c["name"] for c in controllers()] if only else names
    marks_all = ",".join("?" * len(all_names)) or "''"
    runs = records(query("""
        SELECT r.run_id, r.kind, r.policy, r.model, r.started_at, r.finished_at, r.git_commit,
               (SELECT count(*) FROM episodes e WHERE e.run_id = r.run_id) AS n_episodes, {} AS n_cycles
        FROM runs r WHERE r.round IS NULL AND r.kind <> 'attempts' AND (r.kind = 'collect' OR r.model IN ({}))
        ORDER BY r.started_at""".format(n_cycles, marks_all), all_names))
    hidden = query("SELECT count(*) AS n FROM runs").iloc[0]["n"] - len(runs)
    meta["hidden_runs"] = int(hidden)

    sql_examples = [
        ("Success, directness and time to b per controller (latest evaluation of each)",
         "SELECT r.model AS controller, e.policy, count(*) AS trips,\n"
         "       avg(e.reached::INT) AS success_rate, avg(e.directness) AS mean_directness,\n"
         "       avg(CASE WHEN e.reached THEN e.seconds END) AS mean_seconds_to_b,\n"
         "       avg(e.final_dist) AS mean_final_dist_m\n"
         "FROM episodes e JOIN runs r USING (run_id)\n"
         "WHERE r.kind = 'eval' AND r.model IN (SELECT name FROM controllers)\n"
         "  AND r.started_at = (SELECT max(started_at) FROM runs x\n"
         "                      WHERE x.kind = 'eval' AND x.model = r.model AND x.policy = r.policy)\n"
         "GROUP BY ALL ORDER BY mean_directness DESC;"),
        ("Held-out prediction error per controller and output",
         "SELECT model AS controller, output, count(*) AS examples,\n"
         "       sqrt(avg((predicted - actual) ^ 2)) AS rmse,\n"
         "       1 - sum((predicted - actual) ^ 2)\n"
         "         / sum((actual - (SELECT avg(actual) FROM predictions q\n"
         "                          WHERE q.model = p.model AND q.output = p.output)) ^ 2) AS r2\n"
         "FROM predictions p WHERE model IN (SELECT name FROM controllers)\nGROUP BY ALL ORDER BY controller, output;"),
        ("The controllers: examples given to fit() and how long fit() took",
         "SELECT name, kind, context_rows, total_examples, val_rows, features, outputs, fit_seconds\n"
         "FROM controllers ORDER BY created_at;"),
        ("Every command one trip received (latest evaluation of the best controller{})".format(
            ", " + rows[0]["name"] if rows and rows[0]["eval"] else ""),
         "SELECT cycle, t_rel, target_dist, degrees(atan2(goal_sin, goal_cos)) AS bearing_deg,\n"
         "       horizon, stride, turn, dist_next, reached\nFROM cycles\n"
         "WHERE run_id = '{}'\n  AND episode = 0\nORDER BY cycle;".format(
             rows[0]["run_id"] if rows and rows[0]["run_id"] else "<run_id>")),
    ]
    commits = sorted({r["git_commit"] for r in rows if r.get("git_commit")})
    meta["git_commits"] = commits
    return {
        "answer": answer(rows, baseline), "meta": meta, "controllers": rows, "baseline": baseline,
        "episodes": episodes, "cycles": cycles, "preds": preds, "policy_map": policy_map, "runs": runs,
        "sql": [{"title": t, "sql": s} for t, s in sql_examples],
    }


def render_html(data):
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return TEMPLATE.read_text().replace("__DASHBOARD_DATA__", payload).replace(
        "__DASHBOARD_TITLE__", "TabPFN controllers")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller", help="show only this controller (and the baseline)")
    parser.add_argument("--out", help="output HTML file (default: <data folder>/dashboard.html)")
    args = parser.parse_args()
    if not db_path().exists():
        raise SystemExit("no navigation data yet: {} does not exist".format(db_path()))
    out = Path(args.out) if args.out else db_path().parent / "dashboard.html"
    data = build(out.resolve(), args.controller)
    if not data["controllers"]:
        print("note: no controller {}; build one with `just controller`".format(
            "named " + args.controller if args.controller else "in the database yet"), file=sys.stderr)
    html = render_html(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(out.resolve())
    print("{:.1f} MB, {} controllers, {} evaluation episodes".format(
        len(html.encode()) / 1e6, len(data["controllers"]), len(data["episodes"])), file=sys.stderr)


if __name__ == "__main__":
    main()

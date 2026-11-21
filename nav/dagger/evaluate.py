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
import statistics

from .stats import paired, wilson

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

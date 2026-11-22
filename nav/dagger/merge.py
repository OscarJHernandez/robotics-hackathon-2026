"""Put a round's parts together and check them; summarise rounds and timings from the logs.

python -m nav.dagger.merge --run NAME --round 0     # round_0.parquet + checks + summary
python -m nav.dagger.merge --run NAME --timings     # timings.json and a table per machine and round
"""

import argparse
import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .config import run_dir, save_json
from .worker import round_number


def planned_episodes(run, round_):
    plans = sorted((run_dir(run) / "plans").glob("round_{}-*.json".format(round_number(round_))))
    return [e for p in plans for e in json.loads(p.read_text())]


def read_episodes(run, round_):
    folder = run_dir(run) / "round_{}".format(round_number(round_)) / "episodes"
    return [json.loads(line) for path in sorted(folder.glob("*.jsonl")) for line in path.read_text().splitlines() if line]


def merge_round(run, round_, log=print):
    """Concatenate the parts into round_<R>.parquet. Returns the round summary (also round_<R>.json)."""
    r = round_number(round_)
    folder = run_dir(run) / "round_{}".format(r)
    parts = sorted((folder / "parts").glob("*.parquet"))
    if not parts:
        raise SystemExit("no parts in {}".format(folder / "parts"))
    table = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    table = table.drop_duplicates(subset=["episode", "step"]).sort_values(["episode", "step"]).reset_index(drop=True)
    path = run_dir(run) / "round_{}.parquet".format(r)
    table.to_parquet(path, index=False)

    episodes = read_episodes(run, round_)
    seen = Counter(e["episode"] for e in episodes)
    planned = planned_episodes(run, round_)
    planned_ids = {e["episode"] for e in planned}
    missing = sorted(planned_ids - set(seen)) if planned else []
    duplicate = sorted(e for e, n in seen.items() if n > 1)
    outcomes = Counter(e["done_reason"] for e in {e["episode"]: e for e in episodes}.values())
    n = max(1, sum(outcomes.values()))
    student = table.dropna(subset=["student_stride"])
    summary = {
        "round": r, "rows": len(table), "episodes": sum(outcomes.values()), "planned": len(planned_ids),
        "missing": missing[:50], "missing_count": len(missing), "duplicates": duplicate[:50],
        "outcomes": dict(outcomes), "rates": {k: round(v / n, 4) for k, v in outcomes.items()},
        "mean_steps": round(float(np.mean([e["steps"] for e in episodes])), 2) if episodes else 0,
        "expert_share_executed": round(float(table["used_expert"].mean()), 4),
        "expert_nopath_rows": int(table["expert_nopath"].sum()),
        "student_vs_expert": None if student.empty else {
            "mean_abs_stride": round(float((student["student_stride"] - student["expert_stride"]).abs().mean()), 4),
            "mean_abs_turn": round(float((student["student_turn"] - student["expert_turn"]).abs().mean()), 4)},
        "parquet": str(path), "parquet_mb": round(path.stat().st_size / 1e6, 1), "parts": len(parts)}
    save_json(run_dir(run) / "round_{}.json".format(r), summary)
    log("round {}: {} episodes ({} planned, {} missing, {} duplicates), {} rows, outcomes {}".format(
        r, summary["episodes"], summary["planned"], summary["missing_count"], len(duplicate), summary["rows"],
        summary["rates"]))
    return summary


def timings(run, log=print):
    """Per machine and round: rollout seconds, steps/s, episodes, fit seconds, phases. Writes timings.json."""
    rows = defaultdict(lambda: {"procs": 0, "rollout_s": 0.0, "steps": 0, "rows": 0, "episodes": 0,
                                "fit_s": 0.0, "setup_s": 0.0, "phases": defaultdict(float), "start": None, "end": None})
    for path in sorted((run_dir(run) / "logs").glob("*.jsonl")):
        match = re.search(r"-r(\d+)-p\d+$", path.stem)
        if not match:  # worker and loop logs: no per-process timings
            continue
        round_ = match.group(1)
        for line in path.read_text().splitlines():
            event = json.loads(line)
            key = (event["host"], round_)
            row = rows[key]
            row["start"] = row["start"] or event["time"]
            row["end"] = event["time"]
            if event["event"] == "setup":
                row["setup_s"] += event.get("seconds", 0.0)
            elif event["event"] == "fit" and "seconds" in event:
                row["fit_s"] = max(row["fit_s"], event["seconds"])
            elif event["event"] == "end":
                row["procs"] += 1
                row["rollout_s"] = max(row["rollout_s"], event["rollout_s"])
                row["steps"] += event["steps"]
                row["rows"] += event["rows"]
                row["episodes"] += sum(event["outcomes"].values())
                for phase, value in event["phases"].items():
                    row["phases"][phase] += value["seconds"]
    out = []
    for (host, round_), row in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        total = sum(row["phases"].values()) or 1.0
        out.append({"host": host, "round": round_, "procs": row["procs"], "episodes": row["episodes"],
                    "steps": row["steps"], "rollout_s": round(row["rollout_s"], 1),
                    "steps_per_s": round(row["steps"] / row["rollout_s"], 2) if row["rollout_s"] else None,
                    "fit_s": round(row["fit_s"], 1), "setup_s": round(row["setup_s"], 1),
                    "start": row["start"], "end": row["end"],
                    "phase_share": {k: round(v / total, 3) for k, v in sorted(row["phases"].items(), key=lambda kv: -kv[1])}})
    save_json(run_dir(run) / "timings.json", out)
    log("{:10s} {:>5s} {:>5s} {:>8s} {:>8s} {:>10s} {:>8s} {:>7s}  top phases".format(
        "host", "round", "procs", "episodes", "steps", "rollout_s", "steps/s", "fit_s"))
    for r in out:
        top = ", ".join("{} {:.0%}".format(k, v) for k, v in list(r["phase_share"].items())[:4])
        log("{:10s} {:>5s} {:>5d} {:>8d} {:>8d} {:>10.1f} {:>8s} {:>7.1f}  {}".format(
            r["host"], r["round"], r["procs"], r["episodes"], r["steps"], r["rollout_s"],
            "-" if r["steps_per_s"] is None else str(r["steps_per_s"]), r["fit_s"], top))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--round", default=None)
    parser.add_argument("--timings", action="store_true")
    args = parser.parse_args()
    if args.round is not None:
        merge_round(args.run, args.round)
    if args.timings:
        timings(args.run)


if __name__ == "__main__":
    main()

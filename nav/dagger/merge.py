"""Put a round's parts together and check them; summarise rounds and timings from the logs.

python -m nav.dagger.merge --run NAME --round 0     # round_0.parquet + checks + summary
python -m nav.dagger.merge --run NAME --timings     # timings.json and a table per machine and round
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import run_dir, save_json
from .transfer import sha256
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


class MergeError(RuntimeError):
    pass


SIDE_SUFFIX = ".hz.npy"
CHUNK_ROWS = 2000


def _side_path(part):
    return part.with_name(part.stem + SIDE_SUFFIX)


def _load_episodes(folder):
    episodes, seen = [], Counter()
    for path in sorted((folder / "episodes").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                episode = json.loads(line)
                episodes.append(episode)
                seen[episode["episode"]] += 1
    return episodes, seen


def missing_episodes(folder, plan):
    """Plan entries that have no episode summary yet (to resume a half-finished collection)."""
    _, seen = _load_episodes(Path(folder))
    return [e for e in plan if e["episode"] not in seen]


def merge_dir(folder, plan=None, log=print):
    """Strictly merge folder/parts/*.parquet (+ optional <stem>.hz.npy side arrays) and folder/episodes/*.jsonl.

    Writes rows.parquet, rows.hz.npy (when there are side files), episodes.json and merge.json; returns merge.json.
    Raises MergeError on any inconsistency, before anything is written.
    """
    folder = Path(folder)
    parts = sorted((folder / "parts").glob("*.parquet"))
    if not parts:
        raise MergeError("no parts in {}".format(folder / "parts"))
    sides = [_side_path(p) for p in parts]
    have = [s.exists() for s in sides]
    if any(have) and not all(have):
        raise MergeError("some parts have side files and some do not: without {}".format(
            [p.name for p, h in zip(parts, have) if not h][:10]))
    has_side = all(have)

    tables, side_arrays = [], []
    for i, part in enumerate(parts):
        table = pd.read_parquet(part).copy()
        if has_side:
            side = np.load(sides[i], mmap_mode="r")
            if side.shape[0] != len(table):
                raise MergeError("{}: {} rows but side file {} has {}".format(
                    part.name, len(table), sides[i].name, side.shape[0]))
            if side.dtype != np.float16:
                raise MergeError("{}: side file must be float16, got {}".format(sides[i].name, side.dtype))
            if side_arrays and side.shape[1:] != side_arrays[0].shape[1:]:
                raise MergeError("{}: side shape {} differs from {}".format(
                    sides[i].name, side.shape[1:], side_arrays[0].shape[1:]))
            side_arrays.append(side)
        table["_part"] = i
        table["_row"] = np.arange(len(table))
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)

    dup = table.duplicated(subset=["episode", "step"], keep=False)
    if dup.any():
        pairs = sorted({(int(e), int(s)) for e, s in table.loc[dup, ["episode", "step"]].itertuples(index=False)})
        raise MergeError("{} duplicated (episode, step) pairs, first {}".format(len(pairs), pairs[:10]))
    table = table.sort_values(["episode", "step"], kind="mergesort").reset_index(drop=True)

    gaps = table.groupby("episode")["step"].apply(lambda s: bool((s.to_numpy() != np.arange(len(s))).any()))
    bad = [int(e) for e in gaps[gaps].index]
    if bad:
        raise MergeError("{} episodes whose steps are not 0..k without gaps, first {}".format(len(bad), bad[:10]))

    episodes, seen = _load_episodes(folder)
    repeated = sorted(e for e, n in seen.items() if n > 1)
    if repeated:
        raise MergeError("duplicated episode summaries: {}".format(repeated[:10]))
    row_ids = {int(e) for e in table["episode"].unique()}
    summary_ids = set(seen)
    if row_ids != summary_ids:
        raise MergeError("episodes in rows but without summary: {}; with summary but no rows: {}".format(
            sorted(row_ids - summary_ids)[:10], sorted(summary_ids - row_ids)[:10]))
    if plan is not None:
        planned = {e["episode"] for e in plan}
        missing, unexpected = sorted(planned - summary_ids), sorted(summary_ids - planned)
        if missing or unexpected:
            raise MergeError("episodes differ from the plan: missing {} ({} in all), unexpected {} ({} in all)".format(
                missing[:20], len(missing), unexpected[:20], len(unexpected)))

    src_part, src_row = table.pop("_part").to_numpy(), table.pop("_row").to_numpy()
    rows_path = folder / "rows.parquet"
    table.to_parquet(rows_path, index=False)

    side_shape = None
    side_out = folder / "rows.hz.npy"
    if has_side:
        side_shape = [len(table), *side_arrays[0].shape[1:]]
        out = np.lib.format.open_memmap(side_out, mode="w+", dtype=np.float16, shape=tuple(side_shape))
        for start in range(0, len(table), CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, len(table))
            block = np.empty((stop - start, *side_shape[1:]), dtype=np.float16)
            chunk_part, chunk_row = src_part[start:stop], src_row[start:stop]
            for p in np.unique(chunk_part):
                where = np.nonzero(chunk_part == p)[0]
                rows = chunk_row[where]
                order = np.argsort(rows)  # ascending reads from the memory-mapped part
                block[where[order]] = side_arrays[int(p)][rows[order]]
            out[start:stop] = block
        out.flush()
        del out
    else:
        side_out.unlink(missing_ok=True)

    (folder / "episodes.json").write_text(json.dumps(sorted(episodes, key=lambda e: e["episode"])))
    result = {"rows": len(table), "episodes": len(episodes), "parts": len(parts), "has_side": has_side,
              "side_shape": side_shape, "outcomes": dict(Counter(e["done_reason"] for e in episodes)),
              "rows_sha256": sha256(rows_path)}
    (folder / "merge.json").write_text(json.dumps(result, indent=1))
    log("merged {}: {} rows, {} episodes, {} parts, side {}".format(
        folder, result["rows"], result["episodes"], result["parts"], side_shape))
    return result


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

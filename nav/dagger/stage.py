"""Run a queue of variants on this machine, then analyse them: python -m nav.dagger.stage run v03-reference v11-lidar ...

  run      one variant after another (fit, offline predictions, closed loop on the screening layouts); a variant
           whose result.json is already "done" is skipped, so the queue can be restarted after a crash.
  analyse  merge every experiment, build its episode table and summary, compare each with the reference on the same
           layouts, score its offline predictions, write result.json, failures.json and leaderboard.json, print a table.
Anchors checked here: the expert reference on the screening layouts (goal >= 98%, no stuck), every experiment's
episode count equals its plan, columns at fit = columns at predict (the run would have failed otherwise).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .config import dataset_dir, experiment_dir, set_plan, v2_dir

VARIANTS = Path(__file__).resolve().parent / "variants"


def run_queue(ids, closed_count, n_envs):
    from .journal import read_result
    for vid in ids:
        folder = experiment_dir(vid)
        try:
            if read_result(folder).get("status") == "done":
                print("skip {} (done)".format(vid), flush=True)
                continue
        except (FileNotFoundError, OSError, ValueError):
            pass
        cmd = [sys.executable, "-m", "nav.dagger.experiment", "run", "--variant", str(VARIANTS / (vid + ".yaml")),
               "--closed-count", str(closed_count), "--n-envs", str(n_envs), "--procs", "1"]
        started = time.time()
        print("RUN {} ...".format(vid), flush=True)
        with open(folder / "run.out", "a") as out:
            code = subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT)
        print("RUN {} exit {} after {:.0f} s".format(vid, code, time.time() - started), flush=True)
        _write(folder, vid, status="done" if code == 0 else "failed",
               notes="" if code == 0 else "experiment.py exited with {}; see run.out".format(code))


def _write(folder, vid, **fields):
    import yaml

    from .journal import read_result, write_result
    path = VARIANTS / (vid + ".yaml")
    if not path.exists():  # a tagged run such as v15-lidar-allrows-confirm uses its variant's file
        path = VARIANTS / (vid.rsplit("-", 1)[0] + ".yaml")
    variant = yaml.safe_load(path.read_text())
    try:
        current = read_result(folder)
    except (FileNotFoundError, OSError, ValueError):
        current = {}
    base = {"id": vid, "hypothesis": variant.get("hypothesis", ""), "parent": variant.get("parent") or "",
            "change": variant.get("change", ""), "kill_criterion": variant.get("kill_criterion", "")}
    base.update({k: v for k, v in current.items() if k in ("metrics", "cost", "verdict", "started", "git_commit")})
    base.update(fields)
    write_result(folder, **base)


def _closed(folder, plan):
    """Merge every episode the folder has (strict checks), then keep the planned ones; all of them must be there."""
    import pandas as pd

    from .merge import MergeError, merge_dir
    info = merge_dir(folder, plan=None, log=lambda *a: None)
    rows = pd.read_parquet(folder / "rows.parquet")
    episodes = json.loads((folder / "episodes.json").read_text())
    wanted = {e["episode"] for e in plan}
    have = {e["episode"] for e in episodes}
    missing = sorted(wanted - have)
    if missing:
        raise MergeError("{} planned episodes missing, e.g. {}".format(len(missing), missing[:5]))
    rows = rows[rows["episode"].isin(wanted)].reset_index(drop=True)
    episodes = [e for e in episodes if e["episode"] in wanted]
    info = dict(info, used_episodes=len(episodes), extra_episodes=len(have - wanted))
    return rows, episodes, info


def analyse(ids, reference, closed_count, closed_set="screen"):
    import pandas as pd

    from .evaluate import compare, episode_table, offline_report, summarize
    from .journal import leaderboard, write_failures
    from .merge import MergeError
    plan = set_plan(closed_set, 0, closed_count)
    keep = {e["episode"] for e in plan}
    expert_set = closed_set if (dataset_dir(closed_set) / "episodes.json").exists() else "screen"
    screen_eps = json.loads((dataset_dir(expert_set) / "episodes.json").read_text())
    expert = [e for e in screen_eps if e["episode"] in keep]
    expert_rows = pd.read_parquet(dataset_dir(expert_set) / "rows.parquet")
    expert_table = episode_table(expert_rows[expert_rows["episode"].isin(keep)], expert)
    anchor = summarize(expert_table)
    anchors = {"expert_goal_rate": anchor["goal"]["rate"], "expert_stuck": anchor["stuck"]["count"],
               "expert_ok": anchor["goal"]["rate"] >= 0.98 and anchor["stuck"]["count"] == 0}
    tables, out = {}, {}
    for vid in ids:
        folder = experiment_dir(vid)
        try:
            rows, episodes, info = _closed(folder, plan)
        except (MergeError, FileNotFoundError) as error:
            print("{}: not analysed ({})".format(vid, error), flush=True)
            continue
        table = episode_table(rows, episodes, expert_episodes=expert)
        tables[vid] = table
        summary = summarize(table)
        offline = {}
        for path in sorted(folder.glob("offline_*.parquet")):
            offline[path.stem[len("offline_"):]] = offline_report(pd.read_parquet(path))
        jobs = [json.loads(p.read_text()) for p in sorted(folder.glob("job-*.json"))]
        fit = [r.get("info", {}) for j in jobs for r in j.get("results", [])]
        cost = {"fit_s": max([f.get("fit_seconds", 0) for f in fit] or [0]),
                "columns": max([f.get("columns_fit", 0) for f in fit] or [0]),
                "context_rows": max([f.get("context_rows", 0) for f in fit] or [0]),
                "closed_s": round(sum(j.get("seconds", 0) for j in jobs), 1)}
        out[vid] = {"closed": summary, "offline": offline, "merge": info}
        write_failures(folder, table)
        _write(folder, vid, status="done", metrics={"closed": dict(summary, goal_rate=summary["goal"]["rate"]),
                                                     "offline": offline, "anchors": anchors}, cost=cost)
    if reference in tables:
        for vid, table in tables.items():
            if vid != reference:
                out[vid]["vs_reference"] = compare(table, tables[reference])
    board = leaderboard(v2_dir())
    (v2_dir() / "stage_summary.json").write_text(json.dumps({"anchors": anchors, "experiments": out}, indent=1,
                                                            default=str))
    print("expert on screen {}: goal {:.1%}, stuck {} -> anchor {}".format(
        closed_count, anchors["expert_goal_rate"], anchors["expert_stuck"], "OK" if anchors["expert_ok"] else "FAILED"))
    print("{:16s} {:>4s} {:>16s} {:>6s} {:>6s} {:>6s} {:>6s} {:>8s} {:>7s} {:>7s} {:>9s}".format(
        "variant", "n", "goal (95% CI)", "coll", "side", "stuck", "tmout", "progress", "stopR", "turn%", "vs ref p"))
    for vid in ids:
        if vid not in out:
            continue
        s, o = out[vid]["closed"], out[vid]["offline"].get("offline_expert", {})
        g = s["goal"]
        vs = out[vid].get("vs_reference")
        print("{:16s} {:>4d} {:>5.0%} ({:.0%}-{:.0%}) {:>6.0%} {:>6.0%} {:>6.0%} {:>6.0%} {:>8.2f} {:>7} {:>7} {:>9}".format(
            vid, s["n"], g["rate"], g["ci_low"], g["ci_high"], s["collision"]["rate"], s.get("collision_side_share") or 0,
            s["stuck"]["rate"], s["timeout"]["rate"], s["mean_progress"],
            "{:.0%}".format(o["stop"]["recall"]) if o.get("stop") else "-",
            "{:.0%}".format(o["hard_turns"]["ratio"]) if o.get("hard_turns") and o["hard_turns"].get("ratio") is not None else "-",
            "{:+d}/{} p={:.2f}".format(vs["wins"] - vs["losses"], vs["pairs"], vs["sign_p"]) if vs else "-"))
    return out, board


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["run", "analyse"])
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--closed-count", type=int, default=100)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--reference", default="v03-reference")
    parser.add_argument("--closed-set", default="screen", choices=["screen", "confirm", "final"])
    args = parser.parse_args()
    if args.command == "run":
        run_queue(args.ids, args.closed_count, args.n_envs)
    else:
        analyse(args.ids, args.reference, args.closed_count, args.closed_set)


if __name__ == "__main__":
    main()

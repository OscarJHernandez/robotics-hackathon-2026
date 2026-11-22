"""The DAgger loop: just dagger local --run NAME [--smoke], or the pieces for running on several machines.

  local    every round on this machine: plan, rollout, merge, context, fit, ..., then the held-out test
  plan     write the episode plan of a round, split between machines: --weights laptop=1,vendetta=2
  context  aggregate rows of rounds 0..R-1 into context_<R>.parquet (the student's fit() input for round R)
  merge    see nav.dagger.merge (round_<R>.parquet, checks, timings)
  report   rounds.json and a table of every merged round
"""

import argparse
import json
import time

import pandas as pd

from .config import episode_plan, load_config, run_dir, save_json, split_plan
from .merge import merge_round, timings
from .policy import choose_context
from .timing import RunLog, host_name, machine_info
from .worker import HELDOUT_ROUND, round_number, run_plan


def write_plan(run, config, round_, weights):
    plan = episode_plan(config, round_ if round_ == "heldout" else int(round_))
    folder = run_dir(run) / "plans"
    folder.mkdir(exist_ok=True)
    shares = split_plan(plan, weights)
    paths = {}
    for host, episodes in shares.items():
        path = folder / "round_{}-{}.json".format(round_number(round_), host)
        path.write_text(json.dumps(episodes))
        paths[host] = path
        print("{}: {} episodes -> {}".format(host, len(episodes), path))
    return paths


def build_context(run, config, round_):
    """All rows of rounds before `round_` (the DAgger aggregate), capped if max_context_rows is set."""
    paths = [run_dir(run) / "round_{}.parquet".format(r) for r in range(int(round_))]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit("merge these rounds first: {}".format(", ".join(p.name for p in missing)))
    table = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    context = choose_context(table, config["student"]["max_context_rows"], config["seed"])
    path = run_dir(run) / "context_{}.parquet".format(int(round_))
    context.to_parquet(path, index=False)
    print("context for round {}: {} of {} aggregated rows -> {} ({:.1f} MB)".format(
        round_, len(context), len(table), path, path.stat().st_size / 1e6))
    return path


def report(run):
    rows = []
    for path in sorted(run_dir(run).glob("round_*.json")):
        rows.append(json.loads(path.read_text()))
    rows.sort(key=lambda r: r["round"])
    save_json(run_dir(run) / "rounds.json", rows)
    print("{:>7s} {:>8s} {:>8s} {:>6s} {:>9s} {:>7s} {:>6s} {:>9s} {:>9s}".format(
        "round", "episodes", "rows", "goal", "collision", "timeout", "fell", "|d stride|", "|d turn|"))
    for r in rows:
        rates, diff = r["rates"], r["student_vs_expert"] or {}
        print("{:>7s} {:>8d} {:>8d} {:>6.1%} {:>9.1%} {:>7.1%} {:>6.1%} {:>9} {:>9}".format(
            "heldout" if r["round"] == HELDOUT_ROUND else str(r["round"]), r["episodes"], r["rows"],
            rates.get("goal", 0), rates.get("collision", 0), rates.get("timeout", 0), rates.get("fell", 0),
            diff.get("mean_abs_stride", "-"), diff.get("mean_abs_turn", "-")))
    return rows


def local(args):
    config = load_config(args.config, args.smoke)
    host = host_name()
    log = RunLog(run_dir(args.run) / "logs", "{}-loop".format(host))
    log.event("start", "DAgger on {}: {} rounds, then the held-out test".format(host, len(config["rounds"])),
              machine=machine_info(), config=config)
    started = time.time()
    rounds = list(range(len(config["rounds"]))) + ["heldout"]
    for round_ in rounds:
        t0 = time.time()
        beta = 0.0 if round_ == "heldout" else config["rounds"][round_]["beta"]
        plan = json.loads(write_plan(args.run, config, round_, {host: 1})[host].read_text())
        context = None
        if beta < 1.0:
            last = len(config["rounds"]) if round_ == "heldout" else round_
            context = build_context(args.run, config, last)
        log.event("round", "round {} (beta {}): {} episodes".format(round_, beta, len(plan)), round=str(round_))
        run_plan(args.run, round_, plan, beta, args.procs, args.n_envs or config["env"]["n_envs"], args.threads,
                 context, args.config, args.smoke, args.dino_device, args.heartbeat)
        summary = merge_round(args.run, round_, log=lambda m: log.event("merge", m))
        log.event("round_done", "round {} done in {:.1f} s: {}".format(round_, time.time() - t0, summary["rates"]),
                  round=str(round_), seconds=time.time() - t0, summary=summary)
    report(args.run)
    timings(args.run, log=lambda m: log.event("timings", m))
    log.event("end", "all rounds done in {:.1f} s".format(time.time() - started), seconds=time.time() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_local = sub.add_parser("local")
    p_plan = sub.add_parser("plan")
    p_context = sub.add_parser("context")
    p_report = sub.add_parser("report")
    for p in (p_local, p_plan, p_context, p_report):
        p.add_argument("--run", required=True)
        p.add_argument("--config", default=None)
        p.add_argument("--smoke", action="store_true")
    p_local.add_argument("--procs", type=int, default=1)
    p_local.add_argument("--n-envs", type=int, default=None)
    p_local.add_argument("--threads", type=int, default=None)
    p_local.add_argument("--dino-device", default=None)
    p_local.add_argument("--heartbeat", type=float, default=30.0)
    p_plan.add_argument("--round", required=True)
    p_plan.add_argument("--weights", required=True, help="host=weight,... e.g. laptop=1,vendetta=2")
    p_context.add_argument("--round", required=True, type=int)
    args = parser.parse_args()
    if args.command == "local":
        local(args)
    elif args.command == "plan":
        weights = {k: float(v) for k, v in (item.split("=") for item in args.weights.split(","))}
        write_plan(args.run, load_config(args.config, args.smoke), args.round, weights)
    elif args.command == "context":
        build_context(args.run, load_config(args.config, args.smoke), args.round)
    elif args.command == "report":
        report(args.run)


if __name__ == "__main__":
    main()

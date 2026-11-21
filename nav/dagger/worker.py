"""Run one machine's share of a round: python -m nav.dagger.worker --run NAME --round 0 --plan plan.json --procs 8

The plan file lists the episodes ([{"episode", "scene_id", "seed"}]). They are dealt out to `--procs`
processes; each runs `--n-envs` robots in lock-step and writes Parquet parts every 25 episodes to
data/nav/dagger/<run>/round_<R>/parts/, episode summaries to .../episodes/ and its logs to <run>/logs/.
Student rounds (beta < 1) need `--context rows.parquet`: every process calls TabPFN's fit() on it once.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

HELDOUT_ROUND = 99


def round_number(value):
    return HELDOUT_ROUND if value == "heldout" else int(value)


def round_dir(run, round_):
    from .config import run_dir
    path = run_dir(run) / "round_{}".format(round_number(round_))
    (path / "parts").mkdir(parents=True, exist_ok=True)
    (path / "episodes").mkdir(parents=True, exist_ok=True)
    return path


def run_shard(job):
    """One process: build robots, perception and (for student rounds) the student, then run its episodes."""
    threads = job["threads"]
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(threads)

    from .config import load_config, run_dir, vision_dir
    from .env import SimEnv
    from .expert import Expert
    from .featurize import Featurizer
    from .policy import TabPFNPolicy, choose_context
    from .rollout import run_episodes, to_frame
    from .timing import RunLog, Timer, host_name, machine_info, peak_memory_mb

    config = load_config(job["config"], job["smoke"])
    round_ = round_number(job["round"])
    host = host_name()
    name = "{}-r{}-p{}".format(host, round_, job["proc"])
    log = RunLog(run_dir(job["run"]) / "logs", name)
    timer = Timer()
    log.event("start", "process {} of {}: {} episodes, {} robots, {} threads, beta {}".format(
        job["proc"] + 1, job["procs"], len(job["plan"]), job["n_envs"], threads, job["beta"]),
        machine=machine_info(), job={k: v for k, v in job.items() if k != "plan"}, config=config,
        scenes=[e["scene_id"] for e in job["plan"]])
    out = round_dir(job["run"], job["round"])

    env_cfg = config["env"]
    expert = Expert.from_config(config["expert"], env_cfg["goal_radius"])
    with timer("setup_envs"):
        envs = [SimEnv(simple=env_cfg["simple"], goal_radius=env_cfg["goal_radius"], max_steps=env_cfg["max_steps"],
                       min_start_dist=env_cfg["min_start_dist"],
                       target_clearance=env_cfg.get("target_clearance"), target_wall=env_cfg.get("target_wall"),
                       segmentation=config["features"]["perception"] == "truth", expert=expert, timer=timer)
                for _ in range(min(job["n_envs"], max(1, len(job["plan"]))))]
    from fetch_meshes import meshes_available
    model = "simple" if env_cfg["simple"] or (env_cfg["simple"] is None and not meshes_available()) else "meshes"
    log.event("setup", "{} robots ready in {:.1f} s (robot model: {})".format(
        len(envs), timer.total["setup_envs"], model), seconds=timer.total["setup_envs"], robot_model=model)
    with timer("setup_perception"):
        featurizer = Featurizer.from_config(config["features"], vision_dir(), job.get("dino_device"), timer)
    log.event("setup", "perception '{}' ready in {:.1f} s ({} feature columns)".format(
        config["features"]["perception"], timer.total["setup_perception"], len(featurizer.columns)),
        seconds=timer.total["setup_perception"], features=len(featurizer.columns))

    policy = None
    if job["beta"] < 1.0:
        context = pd.read_parquet(job["context"])
        context = choose_context(context, config["student"]["max_context_rows"], config["seed"])
        policy = TabPFNPolicy.from_config(config["student"], featurizer.columns, featurizer.categorical, config["seed"])
        log.event("fit", "TabPFN {} fit() on {} rows x {} columns, device {} ...".format(
            policy.model_version, len(context), len(featurizer.columns), policy.device),
            rows=len(context), columns=len(featurizer.columns))
        seconds = policy.fit(context)
        timer.add("student_fit", seconds)
        log.event("fit", "fit() done in {:.1f} s".format(seconds), seconds=seconds, rows=len(context))

    chunk = {"n": 0, "rows": 0, "bytes": 0}

    def write(result):
        features, meta, episodes = result.take()
        if meta:
            path = out / "parts" / "{}-p{}-{:04d}.parquet".format(host, job["proc"], chunk["n"])
            to_frame(features, meta, featurizer.columns, featurizer.categorical).to_parquet(path, index=False)
            chunk["n"] += 1
            chunk["rows"] += len(meta)
            chunk["bytes"] += path.stat().st_size
        with open(out / "episodes" / "{}-p{}.jsonl".format(host, job["proc"]), "a") as f:
            for e in episodes:
                f.write(json.dumps(e) + "\n")

    rng = np.random.default_rng([config["seed"], round_, job["proc"], 12345])
    started = time.time()
    counts, steps = run_episodes(envs, featurizer, expert, policy, job["beta"], job["plan"], round_, rng, log=log,
                                 timer=timer, on_chunk=write, heartbeat_every=job["heartbeat"])
    seconds = time.time() - started
    log.event("end", "{} episodes, {} steps, {} rows in {} parts ({:.1f} MB) in {:.1f} s; {:.2f} steps/s; "
              "peak memory {} MB".format(sum(counts.values()), steps, chunk["rows"], chunk["n"], chunk["bytes"] / 1e6,
                                         seconds, steps / max(seconds, 1e-9), peak_memory_mb()),
              outcomes=dict(counts), steps=steps, rows=chunk["rows"], parts=chunk["n"], bytes=chunk["bytes"],
              rollout_s=seconds, steps_per_s=steps / max(seconds, 1e-9), phases=timer.summary(per=max(steps, 1)),
              peak_mem_mb=peak_memory_mb())
    for env in envs:
        env.close()
    log.close()
    return {"proc": job["proc"], "outcomes": dict(counts), "steps": steps, "rows": chunk["rows"], "seconds": seconds}


def run_plan(run, round_, plan, beta, procs=1, n_envs=16, threads=None, context=None, config=None, smoke=False,
             dino_device=None, heartbeat=30.0):
    """Deal `plan` out to `procs` processes (round robin) and wait for all. Returns their results.

    The process count is lowered to what fits in available memory next to the reserve (guard.py).
    """
    from .guard import available_gb, fit_procs, reserve_gb
    wanted = max(1, min(procs, len(plan)))
    procs = fit_procs(wanted)
    if procs < wanted:
        print("memory: {:.1f} GB available, reserve {:.1f} GB -> {} processes instead of {}".format(
            available_gb(), reserve_gb(), procs, wanted), flush=True)
    threads = threads or max(1, (os.cpu_count() or 1) // procs)
    jobs = [{"run": run, "round": round_, "beta": beta, "plan": plan[k::procs], "proc": k, "procs": procs,
             "n_envs": n_envs, "threads": threads, "context": str(context) if context else None,
             "config": str(config) if config else None, "smoke": smoke, "dino_device": dino_device,
             "heartbeat": heartbeat} for k in range(procs)]
    # Always in child processes, also for one: then the memory guard covers TabPFN's fit() as well.
    return run_jobs(run_shard, jobs)


class MemoryLow(RuntimeError):
    pass


def run_jobs(fn, jobs, log=print, reserve=None, every=5.0):
    """Run fn(job) for every job in its own process and return the results in order.

    - A process that dies raises BrokenProcessPool here instead of hanging forever like Pool.map.
    - Children exit by themselves if this parent dies (`guard.die_with_parent`): no orphans.
    - A memory guard stops every child when available memory drops under the reserve, and this raises
      MemoryLow. Parts already written stay on disk; merge.py lists the missing episodes.
    """
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    from .guard import MemoryGuard, die_with_parent

    executor = ProcessPoolExecutor(len(jobs), mp_context=mp.get_context("spawn"), initializer=die_with_parent)

    def stop_children(available):
        for process in list(getattr(executor, "_processes", {}).values()):
            try:
                process.kill()
            except Exception:
                pass

    with MemoryGuard(stop_children, reserve=reserve, every=every, log=log) as guard:
        try:
            return list(executor.map(fn, jobs))
        except BrokenProcessPool:
            if guard.tripped:
                raise MemoryLow("stopped: available memory fell under {:.1f} GB".format(guard.reserve)) from None
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--round", required=True, help="round number, or 'heldout'")
    parser.add_argument("--plan", required=True, help="JSON file with this machine's episodes")
    parser.add_argument("--beta", type=float, default=None, help="default: from the config")
    parser.add_argument("--procs", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None, help="torch threads per process")
    parser.add_argument("--context", default=None, help="Parquet rows for the student's fit()")
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dino-device", default=None, help="cpu, mps or cuda (default: best available)")
    parser.add_argument("--heartbeat", type=float, default=30.0, help="seconds between progress lines")
    parser.add_argument("--no-lock", action="store_true", help="do not take this machine's job lock")
    args = parser.parse_args()
    if args.no_lock:
        return run_main(args)
    from .queue import Busy, JobLock
    job_lock = JobLock()
    try:
        job_lock.acquire(args.run)
    except Busy as error:
        raise SystemExit(str(error))
    try:
        return run_main(args)
    finally:
        job_lock.release()


def run_main(args):
    from .config import load_config, run_dir
    from .timing import RunLog, host_name, machine_info

    config = load_config(args.config, args.smoke)
    plan = json.loads(Path(args.plan).read_text())
    round_ = args.round
    beta = args.beta if args.beta is not None else (0.0 if round_ == "heldout" else config["rounds"][int(round_)]["beta"])
    log = RunLog(run_dir(args.run) / "logs", "{}-r{}-worker".format(host_name(), round_number(round_)))
    log.event("start", "worker: {} episodes, {} processes, beta {}".format(len(plan), args.procs, beta),
              machine=machine_info(), episodes=len(plan), procs=args.procs, beta=beta)
    started = time.time()
    results = run_plan(args.run, round_, plan, beta, args.procs, args.n_envs or config["env"]["n_envs"], args.threads,
                       args.context, args.config, args.smoke, args.dino_device, args.heartbeat)
    seconds = time.time() - started
    steps = sum(r["steps"] for r in results)
    outcomes = {}
    for r in results:
        for k, v in r["outcomes"].items():
            outcomes[k] = outcomes.get(k, 0) + v
    log.event("end", "worker done: {} episodes, {} steps in {:.1f} s ({:.2f} steps/s) {}".format(
        sum(outcomes.values()), steps, seconds, steps / max(seconds, 1e-9), outcomes),
        seconds=seconds, steps=steps, outcomes=outcomes, results=results)
    log.close()


if __name__ == "__main__":
    main()

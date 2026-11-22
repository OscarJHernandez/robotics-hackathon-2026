"""Measure this machine's DAgger speed and project how long its share of a round takes.

python -m nav.dagger.bench --procs 1,4,8 --seconds 60 --episodes 1000 [--tabpfn --rows 1000,5000,20000]

1. Preflight: build the arena and render one RGB and one depth picture (checks headless OpenGL).
2. Expert rollouts with perception for `--seconds` at every process count in `--procs`: steps per second,
   episodes per minute, time per phase.
3. With --tabpfn: TabPFN fit() and predict() time on random rows with the real number of feature columns.
4. Projection: the time for `--episodes` expert episodes at the best process count.
Writes data/nav/dagger/bench/<host>.json and logs to data/nav/dagger/bench/logs/.
"""

import argparse
import json
import os
import time

from .config import load_config, runs_dir, save_json
from .timing import RunLog, host_name, machine_info, peak_memory_mb


def _rollout_job(job):
    threads = job["threads"]
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    import numpy as np
    import torch
    torch.set_num_threads(threads)

    from .config import vision_dir
    from .env import SimEnv
    from .expert import Expert
    from .featurize import Featurizer
    from .rollout import run_episodes
    from .timing import Timer

    config = load_config(job["config"])
    env_cfg = config["env"]
    timer = Timer()
    start = time.time()
    expert = Expert.from_config(config["expert"], env_cfg["goal_radius"])
    envs = [SimEnv(simple=env_cfg["simple"], goal_radius=env_cfg["goal_radius"], max_steps=env_cfg["max_steps"],
                   min_start_dist=env_cfg["min_start_dist"], segmentation=job["perception"] == "truth",
                   expert=expert, timer=timer) for _ in range(job["n_envs"])]
    features = dict(config["features"], perception=job["perception"])
    featurizer = Featurizer.from_config(features, vision_dir(), job.get("dino_device"), timer)
    setup = time.time() - start
    plan = [{"episode": job["proc"] * 100000 + i, "scene_id": 500000 + job["proc"] * 1000 + i, "seed": 0}
            for i in range(10000)]
    rng = np.random.default_rng(job["proc"])
    started = time.time()
    episodes = []

    def keep(result):
        episodes.extend(result.take()[2])

    counts, steps = run_episodes(envs, featurizer, expert, None, 1.0, plan, 0, rng, timer=timer, on_chunk=keep,
                                 chunk_episodes=1, deadline=started + job["seconds"])
    seconds = time.time() - started
    for env in envs:
        env.close()
    return {"proc": job["proc"], "setup_s": setup, "seconds": seconds, "steps": steps,
            "episodes": len(episodes), "episode_steps": [e["steps"] for e in episodes], "outcomes": dict(counts),
            "phases": timer.summary(per=max(steps, 1)), "peak_mem_mb": peak_memory_mb()}


def bench_rollouts(procs, seconds, n_envs, perception, config, dino_device, log):
    from .worker import run_jobs
    threads = max(1, (os.cpu_count() or 1) // procs)
    jobs = [{"proc": k, "threads": threads, "seconds": seconds, "n_envs": n_envs, "perception": perception,
             "config": config, "dino_device": dino_device} for k in range(procs)]
    log.event("bench", "rollouts: {} processes x {} robots, {} threads each, {} s, perception {}".format(
        procs, n_envs, threads, seconds, perception))
    started = time.time()
    if procs == 1:
        results = [_rollout_job(jobs[0])]
    else:
        results = run_jobs(_rollout_job, jobs)
    wall = time.time() - started
    steps = sum(r["steps"] for r in results)
    rollout = max(r["seconds"] for r in results)
    episode_steps = [s for r in results for s in r["episode_steps"]]
    out = {"procs": procs, "threads": threads, "n_envs": n_envs, "perception": perception, "wall_s": round(wall, 1),
           "rollout_s": round(rollout, 1), "setup_s": round(max(r["setup_s"] for r in results), 1), "steps": steps,
           "steps_per_s": round(steps / rollout, 2), "episodes_done": len(episode_steps),
           "mean_episode_steps": round(sum(episode_steps) / len(episode_steps), 1) if episode_steps else None,
           "phases_ms_per_step": {k: v["ms_per_step"] for k, v in results[0]["phases"].items()},
           "peak_mem_mb_per_proc": max(r["peak_mem_mb"] for r in results)}
    log.event("bench", "{} processes: {:.2f} steps/s ({} steps, {} episodes finished), setup {:.1f} s; "
              "ms per step in process 0: {}".format(procs, out["steps_per_s"], steps, len(episode_steps),
                                                     out["setup_s"], out["phases_ms_per_step"]), **out)
    return out


def bench_tabpfn(rows_list, n_features, n_categorical, student_cfg, log, batch=16):
    import numpy as np

    from .policy import TabPFNPolicy
    columns = ["f{}".format(i) for i in range(n_features)]
    out = []
    rng = np.random.default_rng(0)
    for rows in rows_list:
        import pandas as pd
        X = rng.random((rows, n_features), dtype=np.float32)
        X[:, :n_categorical] = rng.integers(0, 4, size=(rows, n_categorical))
        table = pd.DataFrame(X, columns=columns)
        table["y_stride"] = rng.uniform(0, 0.3, rows)
        table["y_turn"] = rng.uniform(-0.25, 0.25, rows)
        policy = TabPFNPolicy(columns, columns[:n_categorical], student_cfg["device"], student_cfg["n_estimators"],
                              student_cfg["model_version"])
        log.event("bench", "TabPFN {} fit() on {} rows x {} columns ...".format(policy.model_version, rows, n_features))
        try:
            fit = policy.fit(table)
            start = time.perf_counter()
            policy.act_batch(table.iloc[:batch][columns])  # first predict builds caches
            first = time.perf_counter() - start
            start = time.perf_counter()
            for _ in range(3):
                policy.act_batch(table.iloc[:batch][columns])
            predict = (time.perf_counter() - start) / 3
            result = {"rows": rows, "features": n_features, "fit_s": round(fit, 2), "first_predict_s": round(first, 2),
                      "predict_s_per_batch": round(predict, 3), "batch": batch, "ok": True}
        except Exception as error:  # report and keep going: the table should show what fails where
            result = {"rows": rows, "features": n_features, "ok": False, "error": "{}: {}".format(
                type(error).__name__, str(error)[:300])}
        log.event("bench", "TabPFN {} rows: {}".format(rows, result), **result)
        out.append(result)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--procs", default="1", help="process counts to try, e.g. 1,4,8")
    parser.add_argument("--seconds", type=float, default=60.0, help="rollout seconds per process count")
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--perception", default=None, help="probe or truth (default: from the config)")
    parser.add_argument("--episodes", type=int, default=1000, help="episodes to project the time for")
    parser.add_argument("--tabpfn", action="store_true", help="also time TabPFN fit() and predict()")
    parser.add_argument("--rows", default="1000,5000,20000")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dino-device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    folder = runs_dir() / "bench"
    host = host_name()
    log = RunLog(folder / "logs", "{}-bench".format(host))
    log.event("start", "benchmark on {}".format(host), machine=machine_info())

    started = time.time()
    from .env import SimEnv
    env = SimEnv(simple=config["env"]["simple"])
    env.reset(0, 0)
    env.close()
    log.event("preflight", "arena built and rendered RGB + depth in {:.1f} s (MUJOCO_GL={})".format(
        time.time() - started, os.environ.get("MUJOCO_GL", "default")), seconds=time.time() - started)

    perception = args.perception or config["features"]["perception"]
    n_envs = args.n_envs or config["env"]["n_envs"]
    rollouts = [bench_rollouts(int(p), args.seconds, n_envs, perception, args.config, args.dino_device, log)
                for p in args.procs.split(",")]
    best = max(rollouts, key=lambda r: r["steps_per_s"])
    episode_steps = best["mean_episode_steps"] or 36.0
    projected = args.episodes * episode_steps / best["steps_per_s"]
    result = {"host": host, "machine": machine_info(), "perception": perception, "rollouts": rollouts,
              "best_procs": best["procs"], "best_steps_per_s": best["steps_per_s"],
              "mean_episode_steps": episode_steps, "episodes_projected": args.episodes,
              "projected_round_s": round(projected, 1), "episodes_per_min": round(60 * best["steps_per_s"] / episode_steps, 2)}
    if args.tabpfn:
        from .featurize import columns
        feature, categorical = columns(config["features"]["patch_pool"], config["features"]["use_prev_command"],
                                      config["features"].get("depth_summary", False),
                                      config["features"].get("history", 0))
        result["tabpfn"] = bench_tabpfn([int(r) for r in args.rows.split(",")], len(feature), len(categorical),
                                        config["student"], log)
    save_json(folder / "{}.json".format(host), result)
    log.event("end", "best: {} processes, {:.2f} steps/s = {:.1f} episodes/min; {} episodes would take {:.1f} min "
              "on this machine alone. Benchmark took {:.1f} min.".format(
                  best["procs"], best["steps_per_s"], result["episodes_per_min"], args.episodes, projected / 60,
                  (time.time() - started) / 60), **{k: v for k, v in result.items() if k != "rollouts"})
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()

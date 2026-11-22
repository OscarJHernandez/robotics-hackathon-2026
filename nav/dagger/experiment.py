"""Run one v2 job on this machine: collect a data set, or fit and test one student variant.

  python -m nav.dagger.experiment collect --name round0 --first 0 --count 600 --procs 4 [--perturb 0.04,0.03]
  python -m nav.dagger.experiment run --variant variants/v11.yaml [--closed-count 100] [--procs 1] [--n-envs 32]

`collect`: the expert drives the planned episodes of a set (config.SETS); every step is a stored base row (full
patch grids, goal, summary, lidar) and, with probe perception, the horizon band's DINOv2 features in a side file.
`run`: reads the variant YAML (id, student, features, context, optional offline / closed), fits the student with
TabPFN's fit() on the chosen context rows, scores it offline on held-out rows and lets it drive alone (beta 0)
on the screening layouts. Everything lands in data/nav/dagger/v2/{data/<name>,experiments/<id>}/.
Finished episodes are skipped, so a job can be started again after a crash. One job per machine (job lock).
"""

import argparse
import json
import os
import time
from pathlib import Path

VARIANT_KEYS = {"id", "hypothesis", "parent", "change", "kill_criterion", "student", "features", "context",
                "offline", "closed", "perception"}
CONTEXT_KEYS = {"sources", "max_rows", "quotas", "seed"}
CLOSED_DEFAULT = {"set": "screen", "first": 0, "count": 100, "max_steps": 150, "stuck_steps": 40}


def load_variant(path):
    import yaml
    variant = yaml.safe_load(Path(path).read_text())
    unknown = set(variant) - VARIANT_KEYS
    if unknown:
        raise SystemExit("unknown keys in {}: {}".format(path, sorted(unknown)))
    unknown = set(variant.get("context", {})) - CONTEXT_KEYS
    if unknown:
        raise SystemExit("unknown context keys in {}: {}".format(path, sorted(unknown)))
    from .policy import STUDENT_KINDS
    if variant["student"].get("kind", "tabpfn") not in STUDENT_KINDS:
        raise SystemExit("unknown student kind {!r}".format(variant["student"].get("kind")))
    if variant["student"].get("shield") is not None:
        raise SystemExit("the safety filter is switched off in v2 (see docs/navigation.md)")
    return variant


def done_episodes(folder):
    done = set()
    for path in sorted((Path(folder) / "episodes").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["episode"])
    return done


def load_rows(name, need_band, columns=None):
    """Merged rows of a data set and, if asked, its band side array (memory-mapped)."""
    import numpy as np
    import pandas as pd

    from .config import dataset_dir
    folder = dataset_dir(name)
    path = folder / "rows.parquet"
    if not path.exists():
        raise SystemExit("data set {} is not merged yet: {} missing".format(name, path))
    rows = pd.read_parquet(path, columns=columns)
    band = None
    if need_band:
        side = folder / "rows.hz.npy"
        if not side.exists():
            raise SystemExit("data set {} has no band side file {}".format(name, side))
        band = np.load(side, mmap_mode="r")
        if len(band) != len(rows):
            raise SystemExit("band side file of {} has {} rows, the table {}".format(name, len(band), len(rows)))
    return rows, band


def build_student(variant, config, log, timer):
    """Context rows -> fitted feature pipeline and student. Returns (features, student, info)."""
    import numpy as np
    import pandas as pd

    from .featurize import Features
    from .policy import choose_context, context_summary, make_student
    ctx_cfg = variant["context"]
    features = Features(variant.get("features"))
    tables, bands = [], []
    for name in ctx_cfg["sources"]:
        rows, band = load_rows(name, features.needs_band)
        rows["_source"], rows["_row"] = name, np.arange(len(rows))
        tables.append(rows)
        bands.append(band)
    table = pd.concat(tables, ignore_index=True)
    table["_all"] = np.arange(len(table))
    context = choose_context(table, ctx_cfg.get("max_rows"), ctx_cfg.get("seed", 0), quotas=ctx_cfg.get("quotas", "uniform"))
    band = None
    if features.needs_band:
        offsets = np.cumsum([0] + [len(t) for t in tables])
        src = np.searchsorted(offsets, context["_all"].to_numpy(), side="right") - 1
        band = np.stack([bands[s][r] for s, r in zip(src, context["_row"].to_numpy())])
    with timer("features_fit"):
        X = features.fit_transform(context, band)
    student_cfg = dict(config["student"])
    student_cfg.update(variant["student"])
    student = make_student(student_cfg, features.columns, features.categorical, ctx_cfg.get("seed", 0))
    frame = pd.DataFrame(X, columns=features.columns)
    frame["y_stride"], frame["y_turn"] = context["y_stride"].to_numpy(), context["y_turn"].to_numpy()
    info = {"student": type(student).__name__, "student_kind": student_cfg.get("kind", "tabpfn"), "context_rows": len(context),
            "context_groups": context_summary(context), "sources": {t["_source"].iloc[0]: len(t) for t in tables},
            "features": features.describe(), "columns_fit": len(features.columns)}
    log.event("fit", "{} fit() on {} rows x {} columns ({})".format(info["student"], len(frame), len(features.columns),
                                                                     info["context_groups"]), **info)
    seconds = student.fit(frame)
    timer.add("student_fit", seconds)
    info["fit_seconds"] = round(seconds, 2)
    log.event("fit", "fit() done in {:.1f} s".format(seconds), seconds=seconds)
    return features, student, info


def offline_predictions(variant, features, student, folder, log, timer, chunk=1024):
    """Predict on every held-out offline set; one small Parquet per set with predictions next to the labels."""
    import numpy as np
    keep = ["episode", "step", "scene_id", "seed", "y_stride", "y_turn", "free_ahead", "goal_dist", "goal_sin",
            "goal_cos", "perturbed", "kicked", "used_expert"]
    out = {}
    for name in variant.get("offline", ["offline_expert"]):
        rows, band = load_rows(name, features.needs_band)
        start = time.perf_counter()
        preds = []
        for i in range(0, len(rows), chunk):
            part = rows.iloc[i:i + chunk]
            X = features.transform(part, None if band is None else np.asarray(band[i:i + chunk]))
            if X.shape[1] != len(features.columns):
                raise RuntimeError("columns at predict ({}) differ from fit ({})".format(X.shape[1], len(features.columns)))
            preds.append(student.act_batch(X))
        seconds = time.perf_counter() - start
        timer.add("offline_predict", seconds)
        pred = np.concatenate(preds)
        table = rows[[c for c in keep if c in rows.columns]].copy()
        table["pred_stride"], table["pred_turn"] = pred[:, 0], pred[:, 1]
        path = Path(folder) / "offline_{}.parquet".format(name)
        table.to_parquet(path, index=False)
        out[name] = {"rows": len(table), "seconds": round(seconds, 1), "file": path.name}
        log.event("offline", "{}: {} rows predicted in {:.1f} s".format(name, len(table), seconds), **out[name])
    return out


def run_part(job):
    """One process: robots, perception, optionally a fitted student; then its share of the episodes."""
    threads = job["threads"]
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    import numpy as np
    import torch
    torch.set_num_threads(threads)

    from .config import load_config, vision_dir
    from .env import SimEnv
    from .expert import Expert
    from .featurize import Featurizer, stored_columns
    from .rollout import frame_v2, run_v2
    from .timing import RunLog, Timer, host_name, machine_info, peak_memory_mb

    config = load_config(job["config"])
    folder = Path(job["folder"])
    host = host_name()
    tag = "{}-p{}-{}".format(host, job["proc"], job["stamp"])
    log = RunLog(folder / "logs", tag)
    timer = Timer()
    variant = job.get("variant")
    log.event("start", "{} {}: {} episodes, {} robots, {} threads".format(job["kind"], job["label"], len(job["plan"]),
                                                                        job["n_envs"], threads),
              machine=machine_info(), job={k: v for k, v in job.items() if k not in ("plan", "variant")},
              variant=variant, config=config)
    env_cfg, lidar = config["env"], config.get("lidar", {"rays": 36, "min_range": 0.15, "max_range": 8.0})
    perception = (variant or {}).get("perception", config["features"]["perception"])
    expert = Expert.from_config(config["expert"], env_cfg["goal_radius"])
    info = {}
    features = student = None
    if variant is not None:
        features, student, info = build_student(variant, config, log, timer)
        if job["proc"] == 0 and job.get("offline", True):
            info["offline"] = offline_predictions(variant, features, student, folder, log, timer)
    result = {"proc": job["proc"], "info": info, "outcomes": {}, "steps": 0, "rows": 0, "seconds": 0.0}
    if job["plan"]:
        with timer("setup_envs"):
            envs = [SimEnv(simple=env_cfg["simple"], goal_radius=env_cfg["goal_radius"], max_steps=job["max_steps"],
                           min_start_dist=env_cfg["min_start_dist"], segmentation=perception == "truth",
                           expert=expert, timer=timer, target_clearance=env_cfg.get("target_clearance"),
                           target_wall=env_cfg.get("target_wall"), lidar_rays=lidar["rays"],
                           lidar_range=(lidar["min_range"], lidar["max_range"]), stuck_steps=job["stuck_steps"])
                    for _ in range(min(job["n_envs"], len(job["plan"])))]
        with timer("setup_perception"):
            featurizer = Featurizer.from_config(dict(config["features"], perception=perception), vision_dir(),
                                                job.get("dino_device"), timer)
        columns = stored_columns(lidar["rays"])
        state = {"n": 0, "rows": 0, "bytes": 0}

        def write(collected):
            base, meta, band, episodes = collected.take()
            if meta:
                stem = "{}-{:04d}".format(tag, state["n"])
                path = folder / "parts" / (stem + ".parquet")
                frame_v2(base, meta, columns).to_parquet(path, index=False)
                if job["store_band"]:
                    if len(band) != len(meta):
                        raise RuntimeError("band rows {} != table rows {}".format(len(band), len(meta)))
                    np.save(folder / "parts" / (stem + ".hz.npy"), np.stack(band).astype(np.float16))
                state["n"] += 1
                state["rows"] += len(meta)
                state["bytes"] += path.stat().st_size
            with open(folder / "episodes" / "{}.jsonl".format(tag), "a") as f:
                for e in episodes:
                    f.write(json.dumps(e) + "\n")

        rng = np.random.default_rng([config["seed"], job["proc"], job["rng_salt"]])
        started = time.time()
        counts, steps = run_v2(envs, featurizer, expert, job["plan"], rng, columns, round_=job["round"],
                               policy=student, features=features, beta=job["beta"], perturb=job.get("perturb"),
                               store_band=job["store_band"], log=log, timer=timer, on_chunk=write,
                               heartbeat_every=job["heartbeat"])
        seconds = time.time() - started
        for env in envs:
            env.close()
        result.update(outcomes=dict(counts), steps=steps, rows=state["rows"], seconds=seconds, parts=state["n"])
        log.event("end", "{} episodes, {} steps, {} rows in {} parts ({:.1f} MB) in {:.1f} s; {:.2f} steps/s; peak "
                  "memory {} MB".format(sum(counts.values()), steps, state["rows"], state["n"], state["bytes"] / 1e6,
                                        seconds, steps / max(seconds, 1e-9), peak_memory_mb()),
                  outcomes=dict(counts), steps=steps, rows=state["rows"], rollout_s=seconds,
                  steps_per_s=steps / max(seconds, 1e-9), phases=timer.summary(per=max(steps, 1)),
                  peak_mem_mb=peak_memory_mb())
    result["phases"] = timer.summary(per=max(result["steps"], 1))
    result["peak_mem_mb"] = peak_memory_mb()
    log.close()
    return result


def launch(kind, label, folder, plan, config_path, procs, n_envs, max_steps, stuck_steps, beta, variant=None,
           perturb=None, store_band=False, round_=0, heartbeat=30.0, dino_device=None, threads=None):
    """Split the still-missing episodes over `procs` processes (memory permitting) and run them."""
    from .guard import fit_procs
    from .worker import run_jobs
    done = done_episodes(folder)
    todo = [e for e in plan if e["episode"] not in done]
    wanted = max(1, min(procs, max(len(todo), 1)))
    procs = fit_procs(wanted)
    if procs < wanted:
        print("memory: {} processes instead of {}".format(procs, wanted), flush=True)
    threads = threads or max(1, (os.cpu_count() or 1) // procs)
    stamp = time.strftime("%H%M%S")
    jobs = [{"kind": kind, "label": label, "folder": str(folder), "plan": todo[k::procs], "proc": k, "procs": procs,
             "n_envs": n_envs, "threads": threads, "config": str(config_path) if config_path else None,
             "max_steps": max_steps, "stuck_steps": stuck_steps, "beta": beta, "variant": variant, "perturb": perturb,
             "store_band": store_band, "round": round_, "heartbeat": heartbeat, "dino_device": dino_device,
             "stamp": stamp, "rng_salt": int(time.time()) % 100000, "offline": True} for k in range(procs)]
    print("{} {}: {} of {} episodes still to run, {} processes".format(kind, label, len(todo), len(plan), procs),
          flush=True)
    started = time.time()
    results = run_jobs(run_part, jobs)
    return {"host": __import__("nav.dagger.timing", fromlist=["host_name"]).host_name(), "kind": kind, "label": label,
            "planned": len(plan), "ran": len(todo), "skipped_done": len(plan) - len(todo), "procs": procs,
            "seconds": round(time.time() - started, 1), "results": results}


def main():
    from .config import SETS, dataset_dir, experiment_dir, load_config, save_json, set_plan
    from .joblock import Busy, JobLock
    from nav.common import git_commit

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--name", required=True, choices=sorted(SETS))
    c.add_argument("--first", type=int, default=0)
    c.add_argument("--count", type=int, default=None)
    c.add_argument("--perturb", default=None, help="P_COMMAND,P_KICK per step, e.g. 0.04,0.03")
    c.add_argument("--max-steps", type=int, default=300)
    c.add_argument("--stuck-steps", type=int, default=None)
    r = sub.add_parser("run")
    r.add_argument("--variant", required=True)
    r.add_argument("--closed-first", type=int, default=None)
    r.add_argument("--closed-count", type=int, default=None)
    r.add_argument("--no-closed", action="store_true")
    for p in (c, r):
        p.add_argument("--procs", type=int, default=1)
        p.add_argument("--n-envs", type=int, default=None)
        p.add_argument("--config", default=None)
        p.add_argument("--dino-device", default=None)
        p.add_argument("--heartbeat", type=float, default=30.0)
        p.add_argument("--no-lock", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    n_envs = args.n_envs or config["env"]["n_envs"]

    if args.command == "collect":
        label, folder = args.name, dataset_dir(args.name)
        plan = set_plan(args.name, args.first, args.count)
        perturb = None
        if args.perturb:
            p_cmd, p_kick = (float(v) for v in args.perturb.split(","))
            perturb = {"command": p_cmd, "kick": p_kick}
        kwargs = dict(kind="collect", label=label, folder=folder, plan=plan, config_path=args.config, procs=args.procs,
                      n_envs=n_envs, max_steps=args.max_steps, stuck_steps=args.stuck_steps, beta=1.0,
                      perturb=perturb, store_band=config["features"]["perception"] == "probe")
    else:
        variant = load_variant(args.variant)
        label, folder = variant["id"], experiment_dir(variant["id"])
        (folder / "variant.yaml").write_text(Path(args.variant).read_text())
        closed = dict(CLOSED_DEFAULT)
        closed.update(variant.get("closed") or {})
        if args.closed_first is not None:
            closed["first"] = args.closed_first
        if args.closed_count is not None:
            closed["count"] = args.closed_count
        plan = [] if args.no_closed else set_plan(closed["set"], closed["first"], closed["count"])
        kwargs = dict(kind="run", label=label, folder=folder, plan=plan, config_path=args.config, procs=args.procs,
                      n_envs=n_envs, max_steps=closed["max_steps"], stuck_steps=closed["stuck_steps"], beta=0.0,
                      variant=variant)
    kwargs.update(heartbeat=args.heartbeat, dino_device=args.dino_device)
    lock = None
    if not args.no_lock:
        lock = JobLock()
        try:
            lock.acquire(label)
        except Busy as error:
            raise SystemExit(str(error))
    try:
        summary = launch(**kwargs)
    finally:
        if lock is not None:
            lock.release()
    summary["git_commit"] = git_commit()
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    name = "job-{}-{}.json".format(summary["host"], time.strftime("%H%M%S"))
    save_json(folder / name, summary)
    outcomes = {}
    for res in summary["results"]:
        for k, v in res["outcomes"].items():
            outcomes[k] = outcomes.get(k, 0) + v
    print("DONE {} {}: ran {} episodes in {} s, outcomes {} -> {}".format(summary["kind"], label, summary["ran"],
                                                                         summary["seconds"], outcomes, folder / name),
          flush=True)


if __name__ == "__main__":
    main()

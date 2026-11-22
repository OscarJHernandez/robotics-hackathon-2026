"""The student in the flats: legs as the app's search walks them, the student and A* + a-to-b on the same legs.

  just student-bench --flats 40 --budget-calls 80                    the real student, through the TabPFN API
  just student-bench --flats 1 --max-legs 2 --policy fake --no-baseline    a dry run: no TabPFN, no credits

Legs: in each of N random flats (homes/flats.py), "home" (as built) and "scattered" (the loose pieces moved,
`flats.scatter`), the robot walks dock -> viewing spot -> next viewing spot, zones in a random order, as the search
does. Each leg is its own episode in a plain FlatEnv (homes/sim_bridge.py: the training arena with the flat's
furniture, which is what the app's plain twin shows the student; tests/test_student_twin.py), starting at the leg's
start point facing the way the robot would face there.

All legs step in lock-step: one perception batch and ONE predict() call per step for every robot still walking, at
most `--max-steps` steps (80), with the trainer's stuck rule (no new best of 0.2 m in 40 steps) and the trainer's
collision rule (a touch ends the leg). So the real student costs at most max_steps API calls (10,000 credits each),
whatever the number of legs. `--budget-calls` caps the calls (the run stops before the call that would pass it and
reports how far it got); the credits left are checked before anything starts.

The baseline (local, no credits): the app's A* route (app/robot.py plan_route) walked by the TabPFN a-to-b controller
(`--controller default`, nav/controller.py), on the same legs.

Writes <out>/bench.json (every leg, the summary) and <out>/bench.md (a table with 95% Wilson intervals).
fit()/predict() only (AGENTS.md).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from . import REPO
from .stats import wilson

for _p in (REPO / "homes", REPO / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LIDAR = (36, 0.15, 8.0)
CONDITIONS = ("home", "scattered")


def make_legs(flats, conditions=CONDITIONS, seed=0, max_legs=None):
    """[{"leg", "flat", "condition", "from", "to", "start": (x, y, yaw), "goal": (x, y)}] for every flat x condition."""
    import flats as F
    legs = []
    for k in range(flats):
        home = F.random_flat(np.random.default_rng([seed, k, 0]), k)
        order = [int(i) for i in np.random.default_rng([seed, k, 1]).permutation(len(home["zones"]))]
        for condition in conditions:
            flat = home if condition == "home" else F.scatter(home, np.random.default_rng([seed, k, 2]))
            zones = [flat["zones"][i] for i in order]
            at, name = flat["dock"], "dock"
            yaw = math.atan2(-at["y"], -at["x"])
            point = (at["x"], at["y"])
            for z in zones:
                legs.append({"leg": len(legs), "flat_id": k, "flat": flat, "condition": condition, "from": name,
                             "to": z["name"], "start": (point[0], point[1], yaw), "goal": (z["view_x"], z["view_y"])})
                point, name, yaw = (z["view_x"], z["view_y"]), z["name"], math.radians(z["heading_deg"])
    if max_legs is not None:
        legs = legs[:max_legs]
    return legs


def _env(leg, simple, render, max_steps, stuck_steps):
    from sim_bridge import FlatEnv
    env = FlatEnv(leg["flat"], render=render, simple=simple, max_steps=max_steps, stuck_steps=stuck_steps,
                  lidar_rays=LIDAR[0] if render else 0, lidar_range=LIDAR[1:])
    env.arena.place(*leg["start"])
    env.sim.set_target(*leg["goal"])  # the red pillar at the goal, as in training
    env.set_goal(*leg["goal"])
    return env


def _outcome(leg, env, info, steps, driver):
    x, y, _ = env.pose()
    return {"leg": leg["leg"], "flat": leg["flat_id"], "condition": leg["condition"], "from": leg["from"],
            "to": leg["to"], "driver": driver, "reason": info.get("done_reason") or "not_run", "cycles": steps,
            "start_dist": round(math.hypot(leg["start"][0] - leg["goal"][0], leg["start"][1] - leg["goal"][1]), 3),
            "final_dist": round(math.hypot(x - leg["goal"][0], y - leg["goal"][1]), 3),
            "metres": round(env.path_length, 3)}


def run_student(legs, policy, features, featurizer, simple=None, max_steps=80, stuck_steps=40, log=print):
    """Every leg at once, lock-step, one policy.act_batch per step. Returns (outcomes, calls, stopped_reason)."""
    from .lidarmap import LidarPlanner
    from .policy import OutOfCredits
    envs = [_env(leg, simple, True, max_steps, stuck_steps) for leg in legs]
    planners = [LidarPlanner(rays=LIDAR[0], max_range=LIDAR[2]) for _ in legs]
    obs = [env.observe() for env in envs]
    done = [None] * len(legs)
    calls, stopped = 0, None
    started = time.time()
    for step in range(max_steps):
        active = [i for i, d in enumerate(done) if d is None]
        if not active:
            break
        base = featurizer.base_rows([obs[i] for i in active], [None] * len(active))
        band = featurizer.last_band
        lp = [planners[i].step(obs[i]["robot_pose"], envs[i].goal, obs[i]["lidar"]) for i in active]
        base["lp_sin"], base["lp_cos"] = [v[0] for v in lp], [v[1] for v in lp]
        X, _ = features.live(base, band, [None] * len(active))
        try:
            commands = policy.act_batch(X)
        except OutOfCredits as error:
            stopped = str(error)
            break
        calls += 1
        for row, i in enumerate(active):
            obs[i], info = envs[i].step((float(commands[row, 0]), float(commands[row, 1])))
            if info["done"]:
                done[i] = _outcome(legs[i], envs[i], info, envs[i].steps, "student")
        log("step {}: {} legs walking, {} done, {:.0f} s".format(step + 1, len(active), sum(d is not None for d in done),
                                                               time.time() - started))
    for i, d in enumerate(done):
        if d is None:
            reason = {"done_reason": "timeout" if stopped is None else "not_run"}
            done[i] = _outcome(legs[i], envs[i], reason, envs[i].steps, "student")
        envs[i].close()
    return done, calls, stopped


def run_baseline(legs, controller="default", simple=None, max_steps=80, stuck_steps=40, log=print):
    """A* route (app/robot.py plan_route) + the TabPFN a-to-b controller, lock-step (one local predict per step)."""
    from robot import WAYPOINT, load_tabpfn, nav_row, plan_route
    policy = load_tabpfn(controller)
    envs = [_env(leg, simple, False, max_steps, stuck_steps) for leg in legs]
    routes = []
    for leg, env in zip(legs, envs):
        routes.append(plan_route(env.footprints(), math.sqrt(50.0) / 2, leg["start"][:2], leg["goal"]))
    poses = [[] for _ in legs]
    prev = [(0.0, 0.0)] * len(legs)
    done = [None] * len(legs)
    started = time.time()
    for step in range(max_steps):
        active = [i for i, d in enumerate(done) if d is None]
        if not active:
            break
        rows = []
        for i in active:
            pose = envs[i].pose()
            if len(routes[i]) > 1 and math.hypot(routes[i][0][0] - pose[0], routes[i][0][1] - pose[1]) < WAYPOINT:
                routes[i] = routes[i][1:]
            rows.append(nav_row(pose, poses[i], prev[i], routes[i][0]))
        commands = policy.act(rows)
        for row, i in enumerate(active):
            poses[i].append(envs[i].pose())
            del poses[i][:-4]
            prev[i] = (float(commands[row, 0]), float(commands[row, 1]))
            _, info = envs[i].step(prev[i])
            if info["done"]:
                done[i] = _outcome(legs[i], envs[i], info, envs[i].steps, "astar+ab")
        if step % 10 == 9:
            log("baseline step {}: {} walking, {:.0f} s".format(step + 1, len(active), time.time() - started))
    for i, d in enumerate(done):
        if d is None:
            done[i] = _outcome(legs[i], envs[i], {"done_reason": "timeout"}, envs[i].steps, "astar+ab")
        envs[i].close()
    return done


def summarize(outcomes):
    """{(driver, condition): {"legs", "goal", "goal_ci", "collision", "stuck", "timeout", "not_run", "cycles"}}."""
    out = {}
    for d in sorted({o["driver"] for o in outcomes}):
        for c in CONDITIONS + ("all",):
            rows = [o for o in outcomes if o["driver"] == d and (c == "all" or o["condition"] == c) and o["reason"] != "not_run"]
            if not rows:
                continue
            n = len(rows)
            k = {r: sum(o["reason"] == r for o in rows) for r in ("goal", "collision", "stuck", "timeout", "fell")}
            goals = [o["cycles"] for o in rows if o["reason"] == "goal"]
            out["{}/{}".format(d, c)] = {
                "driver": d, "condition": c, "legs": n, "not_run": sum(o["driver"] == d and o["reason"] == "not_run"
                                                                       and (c == "all" or o["condition"] == c)
                                                                       for o in outcomes),
                **{r: v for r, v in k.items()}, "goal_rate": k["goal"] / n,
                "goal_ci": [round(v, 3) for v in wilson(k["goal"], n)],
                "collision_ci": [round(v, 3) for v in wilson(k["collision"], n)],
                "mean_cycles_goal": round(float(np.mean(goals)), 1) if goals else None}
    return out


def markdown(summary, meta):
    lines = ["# Student in the flats ({})".format(meta["started"]), "",
             "{} flats, {} legs per driver, max {} steps, stuck rule {} steps; student: {}, {} calls ({:,} API "
             "credits).".format(meta["flats"], meta["legs"], meta["max_steps"], meta["stuck_steps"], meta["policy"],
                                meta.get("calls", 0), meta.get("credits_used", 0)), "",
             "| driver | condition | legs | goal (no contact) | 95% interval | collision | 95% interval | stuck | timeout | cycles to goal |",
             "|---|---|---:|---:|---|---:|---|---:|---:|---:|"]
    for s in summary.values():
        lines.append("| {driver} | {condition} | {legs} | {g:.1%} | {g0:.1%} to {g1:.1%} | {c:.1%} | {c0:.1%} to {c1:.1%} "
                     "| {stuck} | {timeout} | {cyc} |".format(
                         g=s["goal_rate"], g0=s["goal_ci"][0], g1=s["goal_ci"][1], c=s["collision"] / s["legs"],
                         c0=s["collision_ci"][0], c1=s["collision_ci"][1], cyc=s["mean_cycles_goal"], **s))
    if meta.get("stopped"):
        lines += ["", "Stopped early: {}. Legs not run are left out of the rates.".format(meta["stopped"])]
    return "\n".join(lines) + "\n"


def main():
    from nav.common import git_commit

    from .config import load_config, v2_dir, vision_dir
    from .export import load_export
    from .featurize import Featurizer
    from .policy import CREDITS_PER_PREDICT, CreditBudget, RulePolicy, client_login, credits_left, make_student
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--flats", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-legs", type=int, default=None, help="only the first N legs (dry runs)")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--stuck-steps", type=int, default=40)
    parser.add_argument("--policy", choices=("client", "fake"), default="client",
                        help="client: the student through the TabPFN API; fake: a heading rule (no TabPFN, no credits)")
    parser.add_argument("--budget-calls", type=int, default=None, help="at most this many API calls (required for client)")
    parser.add_argument("--min-credits", type=int, default=150_000, help="never go below this many credits left")
    parser.add_argument("--export", default=None, help="the v29 export folder (default data/nav/dagger/v2/student/v29)")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--controller", default="default")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    legs = make_legs(args.flats, seed=args.seed, max_legs=args.max_legs)
    print("{} legs in {} flats ({})".format(len(legs), args.flats, ", ".join(CONDITIONS)), flush=True)
    folder = Path(args.export) if args.export else v2_dir() / "student" / "v29"
    features, table, export_meta = load_export(folder)
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "flats": args.flats, "legs": len(legs),
            "max_steps": args.max_steps, "stuck_steps": args.stuck_steps, "policy": args.policy,
            "export": export_meta.get("variant"), "git_commit": git_commit(), "credits_per_call": CREDITS_PER_PREDICT}
    if args.policy == "client":
        if args.budget_calls is None:
            raise SystemExit("--budget-calls is required with the client policy (each call costs {:,} credits)".format(
                CREDITS_PER_PREDICT))
        if not client_login():
            raise SystemExit("no TABPFN_TOKEN (environment or the repo's .env)")
        left = credits_left()
        meta["credits_before"] = left
        need = min(args.budget_calls, args.max_steps) * CREDITS_PER_PREDICT
        if left is not None and left - need < args.min_credits:
            raise SystemExit("{:,} credits left; this run may use {:,} and must leave {:,}: lower --budget-calls".format(
                left, need, args.min_credits))
        print("credits left: {}; this run uses at most {:,}".format(left, need), flush=True)
        config = load_config()
        policy = make_student(dict(config["student"], kind="tabpfn", backend="client"), features.columns,
                              features.categorical, budget=CreditBudget(args.budget_calls))
        seconds = policy.fit(table)
        print("fit() on {:,} rows in {:.0f} s (no credits)".format(len(table), seconds), flush=True)
    else:
        policy = RulePolicy(features.columns)
    config = load_config()
    featurizer = Featurizer.from_config(config["features"], vision_dir())
    started = time.time()
    outcomes, calls, stopped = run_student(legs, policy, features, featurizer, max_steps=args.max_steps,
                                           stuck_steps=args.stuck_steps)
    meta.update(calls=calls, stopped=stopped, student_seconds=round(time.time() - started, 1),
                credits_used=calls * CREDITS_PER_PREDICT if args.policy == "client" else 0)
    if args.policy == "client":
        meta["credits_after"] = credits_left()
    if not args.no_baseline:
        started = time.time()
        outcomes += run_baseline(legs, args.controller, max_steps=args.max_steps, stuck_steps=args.stuck_steps)
        meta["baseline_seconds"] = round(time.time() - started, 1)
    summary = summarize(outcomes)
    out = Path(args.out) if args.out else v2_dir() / "student" / "bench-{}".format(time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps({"meta": meta, "summary": summary, "legs": outcomes}, indent=1,
                                               default=str))
    (out / "bench.md").write_text(markdown(summary, meta))
    print(markdown(summary, meta), flush=True)
    print("-> {}".format(out), flush=True)


if __name__ == "__main__":
    main()

"""The whole search with the robot's own eyes, many times, without the web page: just search-bench [--runs 40]

Each run: a random flat at a random mess level (as built, clean, medium or messy), Margarete in a random zone and spot, lying or
standing (or nobody home), the robot on its dock. The app's search (app/apartment.py) runs exactly as in the
browser: the most likely zone, walk there (the detector watches), look around, a close-up check of anything it
saw, update the belief, next zone. The simulator runs as fast as it can (realtime off).

Driver: `--driver planner` (A* on the known furniture + pure pursuit; no TabPFN, no credits) by default;
`--driver student` walks with the v29 student through the TabPFN API (10,000 credits per gait cycle).

Prints and saves (data/nav/searchbench/<stamp>.json): found rate (and in the right zone), nobody-home answers,
false alarms, looks and metres per search, how far from her the robot confirmed her, and the look-around miss
rate (looks in her zone that did not see her), which is the miss rate the Bayesian update should assume.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent
REPO = APP.parent
for p in (APP, REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ZONES = ("bedroom", "living_room", "dining_area", "kitchen", "hallway")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def main():
    from apartment import Apartment
    from robot import LiveRobot
    from spotter import Spotter
    from nav.common import data_dir

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--first-seed", type=int, default=7000)
    parser.add_argument("--driver", default="planner", choices=("planner", "student", "rule"))
    parser.add_argument("--student-calls", type=int, default=None)
    parser.add_argument("--out-share", type=float, default=0.1, help="share of runs with nobody home")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds per search")
    args = parser.parse_args()
    rng = np.random.default_rng(args.first_seed)
    spotter = Spotter().load()
    if not spotter.ready:
        raise SystemExit("the person detector did not load: {}".format(spotter.reason))
    robot = LiveRobot("apartment", args.driver, ab="planner" if args.driver != "rule" else "rule", realtime=False,
                      student_calls=args.student_calls, spotter=spotter)
    robot.wait_ready(300)
    apartment = Apartment(seed=args.first_seed, robot=robot)
    results = []
    started = time.time()
    for k in range(args.runs):
        seed = args.first_seed + k
        apartment.reset(seed)
        level = ("home", "clean", "medium", "messy")[int(rng.integers(4))]
        if level != "home":
            apartment.shuffle(level)
        time.sleep(0.5)
        out = rng.random() < args.out_share
        where = "out" if out else ZONES[int(rng.integers(len(ZONES)))]
        pose = "lying" if rng.random() < 0.5 else "standing"
        if not apartment.start(where, pose=pose):
            raise SystemExit("could not start the search")
        t0 = time.time()
        while apartment.state()["search"]["running"] and time.time() - t0 < args.timeout:
            time.sleep(1.0)
        if apartment.state()["search"]["running"]:
            apartment.stop()
        st = apartment.state()
        run = st["search"]
        steps = run["steps"]
        found_zone = steps[-1]["zone"] if run["status"] == "found" and steps else None
        # looks at her true zone that came back empty (the detector's misses)
        missed = sum(1 for s in steps if s["zone"] == where and not s["seen"])
        looked_hers = sum(1 for s in steps if s["zone"] == where)
        sightings = run.get("sightings", [])
        result = {"seed": seed, "level": level, "truth": where, "pose": run.get("pose"), "status": run["status"],
                  "found_zone": found_zone, "right_zone": found_zone == where, "looks": len(steps),
                  "metres": round(run.get("metres", 0.0), 2), "false_alarms": run.get("false_alarms", 0),
                  "found_error": run.get("found_error"), "missed_looks": missed, "looks_in_her_zone": looked_hers,
                  "robot_to_person": run.get("robot_to_person"),
                  "approach_updates": (run.get("approach") or {}).get("updates"),
                  "approach_reason": (run.get("approach") or {}).get("reason"),
                  "confirmed_from_m": sightings[-1]["dist"] if sightings and run["status"] == "found" else None,
                  "seen_on_the_way": bool(sightings and sightings[-1]["while"] == "walking"),
                  "fallbacks": run.get("fallbacks", 0), "contacts": run.get("contacts", 0),
                  "look_by": run.get("look_by"), "seconds": round(time.time() - t0, 1)}
        results.append(result)
        print("{:3d} seed {} {:11s} {:8s} -> {:11s} {:11s} looks {} false {} err {} ({:.0f} s, total {:.0f} s)".format(
            k, seed, where, str(result["pose"]), run["status"], str(found_zone), len(steps), result["false_alarms"],
            result["found_error"], result["seconds"], time.time() - started), flush=True)
    home = [r for r in results if r["truth"] != "out"]
    away = [r for r in results if r["truth"] == "out"]
    found = [r for r in home if r["status"] == "found" and r["right_zone"]]
    looks_hers = sum(r["looks_in_her_zone"] for r in home)
    missed = sum(r["missed_looks"] for r in home)
    summary = {
        "runs": len(results), "driver": args.driver, "look_by": sorted({r["look_by"] for r in results}),
        "found": len(found), "home": len(home), "found_rate": round(len(found) / max(len(home), 1), 3),
        "found_ci": wilson(len(found), len(home)),
        "found_wrong_zone": sum(1 for r in home if r["status"] == "found" and not r["right_zone"]),
        "by_pose": {p: "{}/{}".format(sum(1 for r in found if r["pose"] == p), sum(1 for r in home if r["pose"] == p))
                    for p in ("lying", "standing")},
        "nobody_home_right": "{}/{}".format(sum(1 for r in away if r["status"] == "nobody_home"), len(away)),
        "found_when_out": sum(1 for r in away if r["status"] == "found"),
        "false_alarms": sum(r["false_alarms"] for r in results),
        "looks_mean": round(float(np.mean([r["looks"] for r in home])), 2) if home else None,
        "metres_mean": round(float(np.mean([r["metres"] for r in home])), 2) if home else None,
        "seen_on_the_way": sum(r["seen_on_the_way"] for r in found),
        "found_error_median": round(float(np.median([r["found_error"] for r in found])), 2) if found else None,
        "within_1m_by_pose": {p: "{}/{}".format(
            sum(1 for r in found if r["pose"] == p and r["found_error"] is not None and r["found_error"] <= 1.0),
            sum(1 for r in home if r["pose"] == p)) for p in ("lying", "standing")},
        "robot_to_person_median": round(float(np.median([r["robot_to_person"] for r in found
                                                         if r["robot_to_person"] is not None])), 2) if found else None,
        "robot_near_her": "{}/{}".format(sum(1 for r in found if r["robot_to_person"] is not None
                                             and 0.3 <= r["robot_to_person"] <= 1.0), len(found)),
        "approach_updates_mean": round(float(np.mean([r["approach_updates"] or 0 for r in found])), 1) if found else None,
        "confirmed_from_median": round(float(np.median([r["confirmed_from_m"] for r in found
                                                        if r["confirmed_from_m"] is not None])), 2) if found else None,
        "look_miss_rate": round(missed / max(looks_hers, 1), 3), "looks_in_her_zone": looks_hers,
        "contacts": sum(r["contacts"] for r in results), "fallbacks": sum(r["fallbacks"] for r in results),
        "statuses": {s: sum(1 for r in results if r["status"] == s) for s in sorted({r["status"] for r in results})},
    }
    folder = data_dir() / "searchbench"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "{}-{}.json".format(time.strftime("%Y%m%d-%H%M%S"), args.driver)
    path.write_text(json.dumps({"summary": summary, "runs": results}, indent=1))
    print(json.dumps(summary, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()

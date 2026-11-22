"""How well the target detector works in the simulator, without TabPFN:

  python -m nav.dagger.target_check --set final --count 40 [--variant nav/dagger/variants/v32-target-search.yaml]

Two checks per held-out episode (same layouts and starts as the student's test):
  walk   the expert drives from the start to the true target (as in the recorded expert episodes); every step
         the camera's horizon band is matched (the same DINOv2 band the student uses) and the detection is
         compared with the truth: hit rate by distance while the target is in the camera's view, position
         error, and false detections (a match more than 0.5 m from the target).
  look   the robot is put at the true target's viewing point (target.candidates) facing a random way and spins
         on the spot for one turn, as the search does: is the target found (`confirm` agreeing detections)?

Writes data/nav/dagger/v2/target_check/<set>.json and prints the tables.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .featurize import HFOV

BINS = (0.0, 1.0, 2.0, 3.0, 4.0, 9.0)


def main():
    from .config import load_config, set_plan, v2_dir, vision_dir
    from .env import SimEnv
    from .expert import Expert
    from .experiment import load_variant
    from .featurize import Featurizer
    from .target import Detector, TargetSearch, candidates, load_signature, target_config
    from nav.common import relative

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", default="final")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--variant", default="nav/dagger/variants/v32-target-search.yaml")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--dino-device", default=None)
    parser.add_argument("--no-walk", action="store_true", help="only the look-arounds")
    parser.add_argument("--target", action="append", default=[], metavar="KEY=VALUE",
                        help="override a target setting, e.g. --target spin_turn=0.15")
    args = parser.parse_args()
    config = load_config(args.config)
    import yaml
    overrides = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in args.target)}
    cfg = target_config(dict(load_variant(args.variant)["target"], **overrides))
    detector = Detector(load_signature(Path(vision_dir()) / cfg["signature"]), cfg)
    env_cfg, lidar = config["env"], config["lidar"]
    expert = Expert.from_config(config["expert"], env_cfg["goal_radius"])
    env = SimEnv(simple=env_cfg["simple"], goal_radius=env_cfg["goal_radius"], max_steps=args.max_steps,
                 min_start_dist=env_cfg["min_start_dist"], expert=expert, target_clearance=env_cfg["target_clearance"],
                 target_wall=env_cfg["target_wall"], lidar_rays=lidar["rays"],
                 lidar_range=(lidar["min_range"], lidar["max_range"]))
    featurizer = Featurizer.from_config(config["features"], vision_dir(), args.dino_device)

    def look(o):
        featurizer.patch_classes([o])
        return detector.detect(featurizer.last_band[0], o["depth"], o["lidar"], o["robot_pose"])

    walk, looks = [], []
    started = time.time()
    for ep in set_plan(args.set, args.first, args.count):
        o = env.reset(ep["scene_id"], ep["seed"])
        goal = env.goal
        while o is not None and not args.no_walk:
            dist, bearing = relative(o["robot_pose"], *goal)
            sim, d = look(o)
            walk.append({"scene_id": ep["scene_id"], "dist": dist, "in_view": abs(bearing) < HFOV / 2 - 0.02,
                         "sim": sim, "hit": d is not None, "source": d["source"] if d else "",
                         "error": math.hypot(d["x"] - goal[0], d["y"] - goal[1]) if d else math.nan})
            o, info = env.step(expert.act(env.state()))
        # the look-around at the true target's viewing point
        state = env.state()
        rng = np.random.default_rng([ep["scene_id"], ep["seed"], 11])
        locs = candidates(rng, goal, state["footprints"], state["half"], 1, cfg["min_gap"], tuple(cfg["view_dist"]),
                          cfg["clearance"], cfg["wall"])
        vx, vy = locs[0]["x"], locs[0]["y"]
        yaw = float(rng.uniform(-math.pi, math.pi))
        env.arena.place(vx, vy, yaw)
        env.done, env.steps = False, 0
        o = env.observe()
        search = TargetSearch(dict(cfg, candidates=1), detector, locs, _First(), truth=goal)
        found, steps = False, 0
        for steps in range(1, 41):
            search.update(*_inputs(featurizer, o))
            if search.confirmed:
                found = True
                break
            if search.command is None or search.phase == "not_found":
                break
            o, info = env.step(search.command)
            if o is None:
                break
        looks.append({"scene_id": ep["scene_id"], "view_dist": math.hypot(vx - goal[0], vy - goal[1]), "found": found,
                      "steps": steps, "error": math.hypot(search.est[0] - goal[0], search.est[1] - goal[1])
                      if found and search.est else math.nan})
        print("scene {}: walk {} steps, look-around {} after {} steps ({:.0f} s)".format(
            ep["scene_id"], sum(1 for w in walk if w["scene_id"] == ep["scene_id"]),
            "found" if found else "NOT found", steps, time.time() - started), flush=True)
    report = summarize(walk, looks)
    report["walk"], report["looks"] = walk, looks
    report.update(set=args.set, first=args.first, count=args.count, target=cfg)
    out = v2_dir() / "target_check"
    out.mkdir(exist_ok=True)
    tag = "".join("-{}{}".format(k, v) for k, v in overrides.items())
    path = out / "{}-{}-{}{}.json".format(args.set, args.first, args.count, tag)
    path.write_text(json.dumps(report, indent=1))
    print_report(report)
    print("saved", path)


class _First:
    def next(self, locations, checked, pose):
        return next((i for i, c in enumerate(checked) if not c), None)


def _inputs(featurizer, o):
    featurizer.patch_classes([o])
    return featurizer.last_band[0], o["depth"], o["lidar"], o["robot_pose"]


def summarize(walk, looks):
    by = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        rows = [w for w in walk if w["in_view"] and lo <= w["dist"] < hi]
        hits = [w for w in rows if w["hit"]]
        good = [w for w in hits if w["error"] <= 0.5]
        by.append({"from_m": lo, "to_m": hi, "steps_in_view": len(rows), "hit_rate": len(hits) / max(len(rows), 1),
                   "good_rate": len(good) / max(len(rows), 1),
                   "median_error_m": float(np.median([w["error"] for w in good])) if good else None})
    hits = [w for w in walk if w["hit"]]
    false = [w for w in hits if w["error"] > 0.5]
    out_of_view = [w for w in walk if not w["in_view"]]
    return {"false_examples": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in w.items()}
                               for w in false[:20]],
            "walk_steps": len(walk), "by_distance": by, "detections": len(hits), "false_detections": len(false),
            "false_rate_per_step": len(false) / max(len(walk), 1),
            "hits_out_of_view": sum(w["hit"] for w in out_of_view), "steps_out_of_view": len(out_of_view),
            "sources": {s: sum(w["source"] == s for w in hits) for s in ("depth", "lidar")},
            "look_arounds": len(looks), "look_found": sum(lk["found"] for lk in looks),
            "look_found_rate": sum(lk["found"] for lk in looks) / max(len(looks), 1),
            "look_steps_mean": float(np.mean([lk["steps"] for lk in looks if lk["found"]] or [math.nan])),
            "look_error_median_m": float(np.nanmedian([lk["error"] for lk in looks] or [math.nan]))
            if any(lk["found"] for lk in looks) else None}


def print_report(r):
    print("walk ({} steps): detections {}, false (> 0.5 m off) {} = {:.2%} of steps, sources {}".format(
        r["walk_steps"], r["detections"], r["false_detections"], r["false_rate_per_step"], r["sources"]))
    print("  target in view, by distance: hit rate / within 0.5 m / median error")
    for b in r["by_distance"]:
        if b["steps_in_view"]:
            print("  {:.0f}-{:.0f} m: {:5.1%} / {:5.1%} / {} ({} steps)".format(
                b["from_m"], b["to_m"], b["hit_rate"], b["good_rate"],
                "-" if b["median_error_m"] is None else "{:.2f} m".format(b["median_error_m"]), b["steps_in_view"]))
    print("look-around at the viewing point: found {} of {} ({:.1%}), mean {:.1f} steps, median error {}".format(
        r["look_found"], r["look_arounds"], r["look_found_rate"], r["look_steps_mean"],
        "-" if r["look_error_median_m"] is None else "{:.2f} m".format(r["look_error_median_m"])))


if __name__ == "__main__":
    main()

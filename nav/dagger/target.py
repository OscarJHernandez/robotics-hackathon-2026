"""Find the target by its DINOv2 look, range it, and give the student a goal it found itself (test only).

Without this the student is handed the true target. Here it gets candidate locations and its own sensors:

  Detector        every step, the camera's horizon band (46 columns of DINOv2 features, already computed for the
                  student) is matched to the target's signature (vision/target_signature.py) by cosine
                  similarity. The best column gives the bearing; the depth camera (and the lidar where it can
                  see the target) gives the range; pose + range and bearing give a world position.
  candidates()    the locations to visit, each with a probability: in the simulator test the viewing point of the
                  true target (`view_dist` from it, in open space) plus decoys at random free places.
  Schedule        which location next: `RandomSchedule` (the first one; a seeded random order) or `ProbabilitySchedule`.
                  Another model plugs in as another Schedule (config target.schedule), with no other change.
  TargetSearch    one per robot and episode: walk to a location, look around (spin on the spot, the detector runs
                  every step), go to the next one; a confirmed detection (on the way or while looking) becomes the
                  goal. Its goal replaces obs["goal_point"] for the student's goal columns and the lidar-map plan;
                  success, the expert's labels and the logged goal_x/goal_y stay the true target.

Which sensor does what: appearance only from the camera; the lidar has none. The target pillar is not seen by the
lidar at all (it passes through by design), so `range_source: depth` is the default; `lidar` and `auto` are for
targets the lidar hits. Everything here is numpy; no simulator, no TabPFN.
"""

import math
from collections import deque

import numpy as np

from .expert import footprint_distance
from .featurize import BAND_ROWS, HFOV

W, H = 640, 360           # camera picture
COLS = 46                 # band columns (patch columns)
CAM_X = 0.0625            # camera, metres ahead of the body centre (sim/jethexa.xml "eye")
LIDAR_X = -0.023          # lidar, metres ahead of the body centre (lidar_geom)
FOCAL = (W / 2) / math.tan(HFOV / 2)
HORIZON = H / 2

DEFAULT_TARGET = {
    "signature": "full60/signature-pillar.npz",  # in data/vision (config.vision_dir)
    "threshold": None,        # None: the signature's own
    "margin": 0.05,           # best cell this much above the best cell more than 2 cells away
    "window": 3, "min_hits": 1,  # debounce: raw matches in at least min_hits of the last window steps
    # (1: a look-around turns 0.53 rad per step, so the target is in the 62 degree view for only ~2 steps;
    # `confirm` agreeing detections are the real debounce)
    "range_source": "depth",  # depth | lidar | auto
    "range_tol": 0.25,        # metres: lidar range accepted only this close to the depth camera's
    "lidar_window_deg": 5.0,
    "offset": 0.03,           # metres from the nearest visible point to the target's centre (pillar radius)
    # false: the lidar passes through the target (the pillar), so a match the lidar ALSO sees at the depth camera's
    # range is something solid, a look-alike: rejected. On v32's first 100 final layouts this removed 29 of 53
    # false matches and 0 of 1,452 true ones. true (a person): no such check.
    "lidar_sees_target": False,
    "depth_cap": 8.0, "max_range": 4.0,
    "confirm": 2, "fuse_radius": 0.35, "max_jump": 0.75, "fuse_decay": 0.7, "stale_steps": 6,
    "candidates": 4, "min_gap": 1.0, "view_dist": [0.8, 1.5], "clearance": 0.35, "wall": 0.45,
    "schedule": "random",
    "arrive_radius": 0.5,     # at a location: start looking around
    # at a detected estimate that did not end the episode (the episode ends 0.25 m from the true target, so only
    # closer than that, or after `near_steps` steps within `near_radius` without reaching it): look around again
    "estimate_radius": 0.15, "near_radius": 0.5, "near_steps": 8,
    # 0.15: about 0.32 rad per step, 20 steps a turn; found the pillar in 92.5% of 40 look-arounds at its viewing
    # point (target_check), against 75% at the full turn rate 0.25 (only ~2 steps with the target in view)
    "spin_turn": 0.15, "spin_angle": 2 * math.pi,
    "spin_clearance": 0.33,   # metres (lidar): closer than this, move into open space before spinning
    "reposition_steps": 3, "reposition_tries": 3, "reposition_dist": 0.6,
    "detect_range": 2.5,      # metres: a look-around checks every location this close
    "checked_p": 0.02, "passes": 1,
}


def target_config(cfg):
    unknown = set(cfg or {}) - set(DEFAULT_TARGET)
    if unknown:
        raise SystemExit("unknown target keys: {}".format(sorted(unknown)))
    out = dict(DEFAULT_TARGET)
    out.update(cfg or {})
    if out["range_source"] not in ("depth", "lidar", "auto"):
        raise SystemExit("target.range_source must be depth, lidar or auto")
    return out


# ---- signature matching ----------------------------------------------------------------------------------------

def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def load_signature(path):
    data = np.load(path)
    return {"vectors": data["vectors"].astype(np.float32), "center": data["center"].astype(np.float32),
            "threshold": float(data["threshold"]), "name": str(data["name"])}


def column_similarity(band_row, signature):
    """(46, 384) band -> (46,) cosine similarity to the nearest signature vector (after centring)."""
    x = np.asarray(band_row, dtype=np.float32) - signature["center"]
    return (_unit(x) @ _unit(signature["vectors"]).T).max(axis=-1)


def column_bearings(cols=COLS):
    """Same as featurize.column_bearings: bearing of every column centre, positive = left."""
    u = (np.arange(cols) + 0.5) / cols
    return -np.arctan((2 * u - 1) * math.tan(HFOV / 2))


def pixel_bearing(px):
    return -math.atan((px + 0.5 - W / 2) / FOCAL)


def band_pixel_rows(height=H):
    """Pixel rows of the band that are at or above the horizon (no floor there)."""
    top = BAND_ROWS[0] * 14 * height // (26 * 14)
    return top, int(HORIZON * height / H)


def column_pixels(c0, c1, width=W):
    """Pixel columns of band columns c0..c1 (inclusive)."""
    return c0 * width // COLS, -(-(c1 + 1) * width // COLS)


class Detector:
    def __init__(self, signature, cfg):
        self.sig, self.cfg = signature, cfg
        self.threshold = cfg["threshold"] if cfg.get("threshold") is not None else signature["threshold"]
        self.bearings = column_bearings()

    def match(self, band_row):
        """(similarities (46,), (c0, c1, best column) of the matched run or None)."""
        sim = column_similarity(band_row, self.sig)
        best = int(np.argmax(sim))
        if sim[best] < self.threshold:
            return sim, None
        c0 = c1 = best
        while c0 > 0 and sim[c0 - 1] >= self.threshold:
            c0 -= 1
        while c1 < COLS - 1 and sim[c1 + 1] >= self.threshold:
            c1 += 1
        # margin: the matched stretch must beat every place outside it (and 2 cells either side of it). Only
        # outside: a person close by fills most of the view, and then the whole view is one match.
        cells = sim[:COLS // 2 * 2].reshape(-1, 2).max(axis=1)
        idx = np.arange(len(cells))
        far = (idx < c0 // 2 - 2) | (idx > c1 // 2 + 2)
        if far.any() and sim[best] - cells[far].max() < self.cfg["margin"]:
            return sim, None
        return sim, (c0, c1, best)

    def bearing(self, sim, c0, c1):
        w = np.maximum(sim[c0:c1 + 1] - self.threshold, 0) + 1e-3
        return float((self.bearings[c0:c1 + 1] * w).sum() / w.sum())

    def depth_range(self, depth, c0, c1, bearing):
        """Horizontal range from the CAMERA along `bearing` to the nearest valid pixel of the matched columns."""
        h, w = depth.shape
        r0, r1 = band_pixel_rows(h)
        p0, p1 = column_pixels(c0, c1, w)
        patch = depth[r0:r1, p0:p1]
        valid = (patch > 0.05) & (patch < self.cfg["depth_cap"])
        if not valid.any():
            return None
        z = float(patch[valid].min())  # z-depth (along the camera axis)
        return z / max(math.cos(bearing), 0.2)

    def lidar_range(self, lidar, body_bearing):
        """Range from the LIDAR along `body_bearing` (interpolated between neighbouring beams within the window)."""
        if lidar is None:
            return None
        n = len(lidar)
        step = 2 * math.pi / n
        a = body_bearing % (2 * math.pi)
        i0 = int(a // step) % n
        i1 = (i0 + 1) % n
        f = (a - i0 * step) / step
        near = [(i, abs(f if i == i0 else 1 - f) * step) for i in (i0, i1)]
        near = [(i, d) for i, d in near if d <= math.radians(self.cfg["lidar_window_deg"]) + 1e-9]
        if not near:
            return None
        r = [float(lidar[i]) for i, _ in near]
        if len(r) == 2 and abs(r[0] - r[1]) < self.cfg["range_tol"]:
            return (1 - f) * r[0] + f * r[1]
        return r[int(np.argmin([d for _, d in near]))]

    def detect(self, band_row, depth, lidar, pose):
        """(best similarity, detection dict or None)."""
        if band_row is None:
            return float("nan"), None
        sim, run = self.match(band_row)
        best_sim = float(sim.max())
        if run is None:
            return best_sim, None
        c0, c1, best = run
        bearing = self.bearing(sim, c0, c1)
        cam = self.depth_range(depth, c0, c1, bearing) if depth is not None else None
        offset = self.cfg["offset"]
        point_depth = None
        if cam is not None:
            r = cam + offset
            point_depth = (CAM_X + r * math.cos(bearing), r * math.sin(bearing))
        point, source = point_depth, "depth"
        if not self.cfg["lidar_sees_target"] and point_depth is not None and lidar is not None:
            lb = math.atan2(point_depth[1], point_depth[0] - LIDAR_X)
            lr = self.lidar_range(lidar, lb)
            if lr is not None and abs(lr + offset - math.hypot(point_depth[0] - LIDAR_X, point_depth[1])) \
                    < self.cfg["range_tol"]:
                return best_sim, None  # the lidar sees something solid there: not the target
        if self.cfg["range_source"] in ("lidar", "auto"):
            ref = point_depth or (CAM_X + math.cos(bearing), math.sin(bearing))
            lb = math.atan2(ref[1], ref[0] - LIDAR_X)
            lr = self.lidar_range(lidar, lb)
            point_lidar = None
            if lr is not None:
                r = lr + offset
                point_lidar = (LIDAR_X + r * math.cos(lb), r * math.sin(lb))
            agree = (point_lidar is not None and point_depth is not None
                     and abs(math.hypot(*point_lidar) - math.hypot(*point_depth)) < self.cfg["range_tol"])
            if agree:
                point, source = point_lidar, "lidar"
            elif self.cfg["range_source"] == "lidar":
                point = None
        if point is None:
            return best_sim, None
        rng = math.hypot(*point)
        if rng > self.cfg["max_range"]:
            return best_sim, None
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        return best_sim, {"sim": float(sim[best]), "column": best, "bearing": bearing, "range": rng,
                          "body_bearing": math.atan2(point[1], point[0]), "source": source,
                          "edge": best < 2 or best > COLS - 3,
                          "x": x + c * point[0] - s * point[1], "y": y + s * point[0] + c * point[1]}


# ---- candidate locations and the visit order -------------------------------------------------------------------

def _free(x, y, footprints, half, clearance, wall):
    if abs(x) > half - wall or abs(y) > half - wall:
        return False
    return all(float(footprint_distance(f, x, y)) > clearance for f in footprints)


def candidates(rng, truth, footprints, half, n=4, min_gap=1.0, view_dist=(0.8, 1.5), clearance=0.35, wall=0.45,
               reachable=None):
    """`n` locations [{"x", "y", "p", "truth"}]: the true target's viewing point plus decoys, in a random order.

    The viewing point is `view_dist` from the target in open space (so arriving there does not reach the target
    by itself: the robot has to see it). Decoys are free places at least `min_gap` from every other location.
    Probabilities are random and sum to 1."""
    ok = (lambda x, y: True) if reachable is None else reachable
    points = []
    for _ in range(2000):
        a, d = rng.uniform(-math.pi, math.pi), rng.uniform(*view_dist)
        x, y = truth[0] + d * math.cos(a), truth[1] + d * math.sin(a)
        if _free(x, y, footprints, half, clearance, wall) and ok(x, y):
            points.append((x, y, True))
            break
    else:
        points.append((float(truth[0]), float(truth[1]), True))
    for _ in range(5000):
        if len(points) >= n:
            break
        x, y = rng.uniform(-half + wall, half - wall, size=2)
        if (_free(x, y, footprints, half, clearance, wall) and ok(x, y)
                and all(math.hypot(x - px, y - py) >= min_gap for px, py, _ in points)):
            points.append((float(x), float(y), False))
    p = rng.dirichlet(np.ones(len(points)))
    order = rng.permutation(len(points))
    return [{"x": float(points[i][0]), "y": float(points[i][1]), "p": float(p[k]), "truth": bool(points[i][2])}
            for k, i in enumerate(order)]


class RandomSchedule:
    """A random order of the locations, fixed per episode by the seed; probabilities are ignored."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.order = None

    def next(self, locations, checked, robot_pose):
        if self.order is None or len(self.order) != len(locations):
            self.order = [int(i) for i in self.rng.permutation(len(locations))]
        for i in self.order:
            if not checked[i]:
                return i
        return None


class ProbabilitySchedule:
    """The most likely unchecked location first (ties: the nearest)."""

    def __init__(self, seed=None):
        pass

    def next(self, locations, checked, robot_pose):
        free = [i for i in range(len(locations)) if not checked[i]]
        if not free:
            return None
        x, y = robot_pose[:2]
        return min(free, key=lambda i: (-locations[i]["p"], math.hypot(locations[i]["x"] - x, locations[i]["y"] - y)))


SCHEDULES = {"random": RandomSchedule, "probability": ProbabilitySchedule}


def make_schedule(name, seed):
    if name not in SCHEDULES:
        raise SystemExit("unknown target.schedule {!r} (known: {})".format(name, sorted(SCHEDULES)))
    return SCHEDULES[name](seed)


# ---- the goal estimate from detections ---------------------------------------------------------------------------

class Estimate:
    """Where the target is, from detections over steps: a weighted running mean (weight = similarity / range,
    halved at the image edge; older detections fade by `fuse_decay`). Confirmed after `confirm` detections within
    `fuse_radius`; once confirmed, detections within `max_jump` are fused, farther ones need `confirm` agreeing
    detections (within `window` steps) to move it. An unconfirmed estimate not seen again for `stale_steps` is
    dropped. `reject()` forgets the estimate and ignores later detections near it (a spot that turned out empty).
    Used by TargetSearch (the arena test) and by app/spotter.py (the app)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.rejected = []
        self.reset()

    def reset(self):
        """Forget the estimate (the rejected spots stay; `clear()` forgets those too)."""
        self.est, self.w, self.hits, self.step, self.confirmed, self.jump = None, 0.0, 0, -1, False, []

    def clear(self):
        self.reset()
        self.rejected = []

    def reject(self):
        if self.est is not None:
            self.rejected.append(self.est)
        self.reset()

    def add(self, d, step):
        """Fuse detection `d` (dict with x, y, sim, range, edge) seen at `step`. True when it counted for the
        estimate (fused, or it moved it after a confirmed jump); False when it was ignored or is waiting."""
        cfg = self.cfg
        p = (d["x"], d["y"])
        if any(_dist(p, r) < cfg["fuse_radius"] for r in self.rejected):
            return False
        w = d["sim"] / max(d["range"], 0.3) * (0.5 if d["edge"] else 1.0)
        if self.est is not None and not self.confirmed and step - self.step > cfg["stale_steps"]:
            self.est, self.hits = None, 0  # an unconfirmed estimate that was not seen again
        if self.est is None:
            self.est, self.w, self.hits, self.step = p, w, 1, step
        elif _dist(p, self.est) <= (cfg["max_jump"] if self.confirmed else cfg["fuse_radius"]):
            keep = self.w * cfg["fuse_decay"]
            self.est = ((self.est[0] * keep + p[0] * w) / (keep + w), (self.est[1] * keep + p[1] * w) / (keep + w))
            self.w, self.step = keep + w, step
            self.hits += 1
            self.jump = []
        else:
            self.jump = [j for j in self.jump if step - j[2] <= cfg["window"]] + [(p[0], p[1], step, w)]
            if len(self.jump) < cfg["confirm"]:
                return False
            total = sum(j[3] for j in self.jump)
            mx = sum(j[0] * j[3] for j in self.jump) / total
            my = sum(j[1] * j[3] for j in self.jump) / total
            if not all(_dist((j[0], j[1]), (mx, my)) <= cfg["fuse_radius"] for j in self.jump):
                return False
            self.est, self.w, self.step, self.hits = (mx, my), total, step, len(self.jump)
            self.jump = []
        if not self.confirmed and self.hits >= cfg["confirm"]:
            self.confirmed = True
        return True


# ---- the search: goal estimate, look-arounds, locations --------------------------------------------------------

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class TargetSearch:
    """Per robot and episode. Call `update(band_row, depth, lidar, pose)` once per step, then use `goal` (where
    the student should go) and `command` (a scripted (stride, turn) for the look-around, or None: the student
    drives). `truth` is used only for logging (errors), never for decisions."""

    def __init__(self, cfg, detector, locations, schedule, truth=None):
        self.cfg, self.detector, self.schedule, self.truth = cfg, detector, schedule, truth
        self.locations = [dict(loc) for loc in locations]
        total = sum(loc["p"] for loc in self.locations) or 1.0
        for loc in self.locations:
            loc["p"] = loc["p"] / total if total > 0 else 1.0 / len(self.locations)
        self.prior_p = [loc["p"] for loc in self.locations]
        self.checked = [False] * len(self.locations)
        self.visits = []          # location indices in the order they were looked at
        self.current = None
        self.phase = "walking"    # walking | looking | reposition | to_target | not_found
        self.look_at = None       # "location" | "estimate"
        self.fusion = Estimate(cfg)  # the goal estimate from detections (est, confirmed, hits, rejected)
        self.hits = deque(maxlen=cfg["window"])
        self.fresh = 0            # confirmed-looking detections during the current look-around
        self.spin_turned, self.spin_yaw, self.repos_tries, self.repos_left, self.repos_goal = 0.0, None, 0, 0, None
        self.near = 0             # steps within near_radius of the estimate
        self.passes = 1
        self.step = 0
        self.goal, self.command, self.last, self.best_sim = None, None, None, float("nan")
        self.found_step = self.detect_dist = None
        self.false_positives = 0
        self.looks = 0

    # -- detections

    def _fuse(self, d):
        if not self.fusion.add(d, self.step):
            return
        if self.phase == "looking":
            self.fresh += 1
        if self.fusion.confirmed and self.found_step is None:
            self.found_step = self.step
            self.detect_dist = d["range"]

    # the estimate lives in self.fusion (an Estimate); these keep TargetSearch's own names
    est = property(lambda self: self.fusion.est, lambda self, v: setattr(self.fusion, "est", v))
    confirmed = property(lambda self: self.fusion.confirmed, lambda self, v: setattr(self.fusion, "confirmed", v))
    est_hits = property(lambda self: self.fusion.hits, lambda self, v: setattr(self.fusion, "hits", v))
    rejected = property(lambda self: self.fusion.rejected)

    # -- locations

    def _mark_covered(self, pose):
        for i, loc in enumerate(self.locations):
            if i == self.current or _dist(pose, (loc["x"], loc["y"])) <= self.cfg["detect_range"]:
                self.checked[i] = True
        for i, loc in enumerate(self.locations):
            if self.checked[i]:
                loc["p"] = self.cfg["checked_p"]
        total = sum(loc["p"] for loc in self.locations)
        for loc in self.locations:
            loc["p"] /= total

    def _next_location(self, pose):
        i = self.schedule.next(self.locations, self.checked, pose)
        if i is None and self.passes < self.cfg["passes"]:
            self.passes += 1
            self.checked = [False] * len(self.locations)
            i = self.schedule.next(self.locations, self.checked, pose)
        self.current = i
        self.phase = "walking" if i is not None else "not_found"

    def _start_look(self, at, pose):
        self.phase, self.look_at = "looking", at
        self.spin_turned, self.spin_yaw, self.repos_tries, self.fresh = 0.0, pose[2], 0, 0
        self.looks += 1
        if at == "location" and self.current is not None:
            self.visits.append(self.current)

    def _finish_look(self, pose):
        if self.look_at == "estimate":
            self.false_positives += 1
            self.fusion.reject()
        self._mark_covered(pose)
        self._next_location(pose)

    # -- one step

    def update(self, band_row, depth, lidar, pose):
        cfg = self.cfg
        self.command = None
        best_sim, d = self.detector.detect(band_row, depth, lidar, pose)
        self.best_sim, self.last = best_sim, d
        self.hits.append(d is not None)
        was_confirmed = self.confirmed
        if d is not None and sum(self.hits) >= cfg["min_hits"]:
            self._fuse(d)
        if self.current is None and self.phase == "walking":
            self._next_location(pose)
        if self.confirmed and not was_confirmed and self.phase in ("walking", "reposition"):
            self.phase = "to_target"
        # arrivals first, so a look-around that starts this step goes through the room-to-turn check below
        if self.phase == "reposition":
            self.repos_left -= 1
            if self.repos_left < 0:
                self.phase, self.spin_yaw = "looking", pose[2]
        elif self.phase == "walking" and self.current is not None:
            loc = self.locations[self.current]
            if _dist(pose, (loc["x"], loc["y"])) < cfg["arrive_radius"]:
                self._start_look("location", pose)
        elif self.phase == "to_target" and self.est is not None:
            gap = _dist(pose, self.est)
            self.near = self.near + 1 if gap < cfg["near_radius"] else 0
            if gap < cfg["estimate_radius"] or self.near >= cfg["near_steps"]:
                self.near = 0
                self._start_look("estimate", pose)
        if self.phase == "looking":
            if self.confirmed and (self.look_at == "location" or self.fresh >= cfg["confirm"]):
                self.phase = "to_target"
            else:
                yaw = pose[2]
                self.spin_turned += abs(math.atan2(math.sin(yaw - self.spin_yaw), math.cos(yaw - self.spin_yaw)))
                self.spin_yaw = yaw
                if self.spin_turned >= cfg["spin_angle"]:
                    self._finish_look(pose)
                elif lidar is not None and float(np.min(lidar)) < cfg["spin_clearance"]:
                    if self.repos_tries >= cfg["reposition_tries"]:
                        self._finish_look(pose)  # could not get room to turn: count what was seen
                    else:
                        self.repos_tries += 1
                        a = pose[2] + 2 * math.pi * int(np.argmax(lidar)) / len(lidar)
                        r = cfg["reposition_dist"]
                        self.repos_goal = (pose[0] + r * math.cos(a), pose[1] + r * math.sin(a))
                        self.repos_left, self.phase = cfg["reposition_steps"], "reposition"
                else:
                    self.command = (0.0, cfg["spin_turn"])
        if self.phase == "not_found":
            self.command = (0.0, 0.0)
        self.goal = self._goal(pose)
        self.step += 1
        return self.goal, self.command

    def _goal(self, pose):
        if self.phase == "reposition" and self.repos_goal is not None:
            return self.repos_goal
        if self.confirmed and self.est is not None and self.phase in ("to_target", "looking"):
            return self.est
        if self.current is not None:
            loc = self.locations[self.current]
            return (loc["x"], loc["y"])
        return self.goal or (pose[0], pose[1])

    @property
    def source(self):
        if self.phase == "reposition":
            return "reposition"
        if self.confirmed and self.est is not None and self.phase in ("to_target", "looking"):
            return "detected"
        return "location" if self.current is not None else "none"

    def row(self):
        """Columns logged with every step."""
        d, nan = self.last, float("nan")
        err = _dist(self.goal, self.truth) if self.truth is not None and self.goal is not None else nan
        det_err = _dist((d["x"], d["y"]), self.truth) if d is not None and self.truth is not None else nan
        return {"goal_est_x": self.goal[0], "goal_est_y": self.goal[1], "goal_source": self.source,
                "search_phase": self.phase, "search_location": -1 if self.current is None else self.current,
                "scripted": self.command is not None, "target_sim": self.best_sim, "target_hit": d is not None,
                "target_bearing": d["bearing"] if d else nan, "target_range": d["range"] if d else nan,
                "target_column": d["column"] if d else -1,
                "target_body_bearing": d["body_bearing"] if d else nan,
                "checked_mask": int(sum(1 << i for i, c in enumerate(self.checked) if c)),
                "target_det_x": d["x"] if d else nan, "target_det_y": d["y"] if d else nan,
                "target_det_error_m": det_err, "target_error_m": err, "locations_checked": int(sum(self.checked))}

    def summary(self):
        return {"candidates": [{"x": round(loc["x"], 3), "y": round(loc["y"], 3), "p": round(p, 4),
                                "truth": loc["truth"]} for loc, p in zip(self.locations, self.prior_p)],
                "visits": self.visits, "looks": self.looks, "found_step": self.found_step,
                "detect_range_m": None if self.detect_dist is None else round(self.detect_dist, 3),
                "false_positives": self.false_positives, "search_end": self.phase,
                "final_goal_error": None if self.truth is None or self.goal is None
                else round(_dist(self.goal, self.truth), 3)}


def make_search(cfg, detector, env, scene_id, seed):
    """A TargetSearch for one episode of `env` (after reset): candidates and schedule seeded by scene and seed."""
    rng = np.random.default_rng([int(scene_id), int(seed), 11])
    state = env.state()
    locs = candidates(rng, env.goal, state["footprints"], state["half"], cfg["candidates"], cfg["min_gap"],
                      tuple(cfg["view_dist"]), cfg["clearance"], cfg["wall"])
    schedule = make_schedule(cfg["schedule"], [int(scene_id), int(seed), 12])
    return TargetSearch(cfg, detector, locs, schedule, truth=tuple(env.goal))

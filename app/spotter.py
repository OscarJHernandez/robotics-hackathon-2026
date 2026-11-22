"""The robot looks for Margarete with its own camera: the person detector of the app.

Every gait cycle while the robot walks, looks around or watches, `Spotter.see(sim)` takes the robot's camera
picture and depth of the dressed flat (what the "eye" view shows), runs DINOv2 on it, and matches the horizon
band (patch rows 11-14, 46 columns) to Margarete's signature (`just person-data`, then
`python vision/target_signature.py person-v2 --band --name margarete`): nav/dagger/target.py's `Detector`.
A match is ranged with the depth picture and turned into a floor position; `Estimate` fuses matches over cycles
and confirms her after `confirm` agreeing ones. Her truth (where the app put her) is never used here.

Runs on the simulator thread only (MuJoCo rendering is single-threaded); `state()` is what LiveRobot publishes.
The lidar skips her (her geoms have no collisions), so a match the lidar also sees at the same range is
furniture, not her (`lidar_sees_target: false`, from the target search's look-alike check).
"""

import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent
REPO = APP.parent
for p in (REPO, REPO / "vision"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SIGNATURE = "person-v2/signature-margarete.npz"  # in data/vision
LIDAR = (36, 0.15, 8.0)  # rays, min and max range: the same scan as the student
# Detector and estimate settings for her (nav/dagger/target.py keys). She is 1.6 m long lying down, so matches on
# different parts of her body spread more than on the 6 cm pillar: a wider fuse radius and jump.
PERSON = {"signature": SIGNATURE, "range_source": "depth", "lidar_sees_target": False, "offset": 0.10,
          "max_range": 4.0, "margin": 0.05, "window": 3, "min_hits": 1, "confirm": 2, "fuse_radius": 0.5,
          "max_jump": 1.0, "fuse_decay": 0.7, "stale_steps": 6}
# The close-up check counts matches within this many metres of the spot it went to check. Lying, she is 1.6 m long:
# the first sighting may be her feet or head, and up close the camera matches her middle (a live search at 0.5 m
# called her a false alarm twice, 2026-11-22).
CHECK_RADIUS = 1.0


class Spotter:
    """status: off | loading | ready | failed. `see(sim)` once per gait cycle; `state()` for the web page."""

    def __init__(self, signature=None, cfg=None, device=None, detector=None, extract=None):
        """`detector` and `extract` (band from a picture) are for tests; normally both are loaded by `load()`."""
        from nav.dagger.target import Estimate, target_config
        self.cfg = target_config(dict(PERSON, **(cfg or {})))
        self.signature = signature
        self.device = device
        self.detector, self._extract = detector, extract
        self.estimate = Estimate(self.cfg)
        self.status = "ready" if detector is not None and extract is not None else "off"
        self.reason = ""
        self.cycles = 0
        self.last = None          # the last detection (dict) or None
        self.best = float("nan")  # best band similarity in the last picture
        self.seconds = 0.0
        self.marked = None        # the spot (x, y) mark() was called for
        self.watch_hits = 0       # detections near the estimate since mark()

    # ---- loading ----

    def load(self):
        """DINOv2 and the signature. Call on the simulator thread (or before it starts)."""
        if self.status == "ready":
            return self
        self.status = "loading"
        try:
            from dataset import data_dir
            from features import extract, load_dino
            from nav.dagger.featurize import BAND_ROWS
            from nav.dagger.target import Detector, load_signature
            path = Path(self.signature) if self.signature else data_dir() / self.cfg["signature"]
            if not path.is_file():
                raise FileNotFoundError("no signature at {} (just person-data, then vision/target_signature.py "
                                        "person-v2 --band --name margarete)".format(path))
            self.detector = Detector(load_signature(path), self.cfg)
            dino = load_dino(self.device)
            self._extract = lambda rgb: extract(dino, rgb[None])[0, BAND_ROWS[0]:BAND_ROWS[1]].mean(axis=0)
            self.status = "ready"
        except Exception as error:  # noqa: BLE001 - no torch, no signature: the search uses the mock look
            self.status, self.reason = "failed", str(error)
            print("[app] the person detector is off: {}".format(error), flush=True)
        return self

    def load_async(self):
        threading.Thread(target=self.load, name="spotter-load", daemon=True).start()
        return self

    @property
    def ready(self):
        return self.status == "ready"

    # ---- one gait cycle ----

    def see(self, sim):
        """Look once: picture, depth, lidar, pose -> detection -> estimate. Returns the detection or None."""
        started = time.monotonic()
        rgb = sim.render()
        depth = sim.render(depth=True)
        lidar = sim.lidar(*LIDAR)
        x, y, _ = sim.body_position()
        pose = (float(x), float(y), float(sim.body_yaw()))
        band = self._extract(rgb)
        self.best, d = self.detector.detect(band, depth, lidar, pose)
        self.last = d
        self.cycles += 1
        if d is not None:
            self.estimate.add(d, self.cycles)
            if self.marked is not None and math.hypot(d["x"] - self.marked[0], d["y"] - self.marked[1]) <= CHECK_RADIUS:
                self.watch_hits += 1
        self.seconds = 0.8 * self.seconds + 0.2 * (time.monotonic() - started) if self.cycles > 1 else \
            time.monotonic() - started
        return d

    # ---- what the search asks for ----

    def reset(self):
        """A new search: forget the estimate and the rejected spots."""
        self.estimate.clear()
        self.marked, self.watch_hits = None, 0

    def reject(self):
        """The estimate turned out empty (a look-alike): forget it and ignore matches near it from now on."""
        self.estimate.reject()
        self.marked, self.watch_hits = None, 0

    def mark(self):
        """Count matches within CHECK_RADIUS of the current estimate from now on (the close-up check before
        calling her found)."""
        est = self.estimate.est
        self.marked, self.watch_hits = (tuple(est) if est is not None else None), 0

    def state(self):
        est, d = self.estimate.est, self.last
        out = {"status": self.status, "confirmed": bool(self.estimate.confirmed and est is not None),
               "hits": int(self.estimate.hits), "rejected": [[round(r[0], 2), round(r[1], 2)]
                                                             for r in self.estimate.rejected],
               "best": None if not math.isfinite(self.best) else round(float(self.best), 3),
               "threshold": None if self.detector is None else round(float(self.detector.threshold), 3),
               "seen": d is not None, "watch_hits": self.watch_hits, "cycles": self.cycles,
               "seconds": round(self.seconds, 3)}
        if self.reason:
            out["reason"] = self.reason
        if est is not None:
            out["x"], out["y"] = round(est[0], 3), round(est[1], 3)
        if d is not None:
            out["match"] = {"x": round(d["x"], 3), "y": round(d["y"], 3), "sim": round(d["sim"], 3),
                            "range": round(d["range"], 2), "bearing_deg": round(math.degrees(d["body_bearing"]), 1),
                            "column": int(d["column"])}
        return out

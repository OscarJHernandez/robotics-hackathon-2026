"""The search with the robot's own eyes: app/spotter.py, the look-around in app/robot.py and the search loop in
app/apartment.py when the person detector is on. Fakes only (no DINOv2, no TabPFN); the LiveRobot test needs
MuJoCo rendering (skipped without OpenGL)."""

import math
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "app", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import apartment  # noqa: E402
from test_apartment import FakeRobot, finish, make  # noqa: E402

from nav.dagger import target as T  # noqa: E402


def det(x, y, sim=0.9, rng=1.5):
    return {"x": x, "y": y, "sim": sim, "range": rng, "bearing": 0.0, "body_bearing": 0.0, "source": "depth",
            "edge": False, "column": 23}


class EstimateTest(unittest.TestCase):
    def test_confirm_reject_and_ignore_the_rejected_spot(self):
        e = T.Estimate(T.target_config({"fuse_radius": 0.5, "confirm": 2}))
        self.assertTrue(e.add(det(1.0, 1.0), 1))
        self.assertFalse(e.confirmed)
        self.assertTrue(e.add(det(1.2, 1.0), 2))
        self.assertTrue(e.confirmed)
        e.reject()
        self.assertIsNone(e.est)
        self.assertFalse(e.add(det(1.1, 1.0), 3), "a rejected spot is ignored")
        self.assertTrue(e.add(det(3.0, -1.0), 4))
        e.clear()
        self.assertEqual(e.rejected, [])

    def test_a_person_filling_the_view_is_one_match(self):
        """Close up she covers most of the 46 columns: the margin is only checked outside the matched stretch."""
        sig = {"vectors": np.eye(8, dtype=np.float32)[:1], "center": np.zeros(8, np.float32), "threshold": 0.8,
               "name": "t"}
        band = np.random.default_rng(0).normal(size=(46, 8)).astype(np.float32)
        band[:, 0] = 0.0
        band[4:42] = 0.0
        band[4:42, 0] = 1.0
        depth = np.full((360, 640), 0.6, np.float32)
        _, d = T.Detector(sig, T.target_config(None)).detect(band, depth, None, (0.0, 0.0, 0.0))
        self.assertIsNotNone(d)


def floor_depth(cols, z, top):
    """A depth picture: the floor everywhere below the horizon, far wall above, and an object at depth `z` in band
    columns `cols` reaching `top` metres high."""
    rows = np.arange(T.H) + 0.5
    below = rows > T.HORIZON
    d = np.full((T.H, T.W), 8.0, np.float32)
    d[below] = np.minimum(8.0, T.CAMERA_HEIGHT * T.FOCAL / (rows[below] - T.HORIZON))[:, None]
    top_row = int(T.HORIZON - (top - T.CAMERA_HEIGHT) * T.FOCAL / z)
    floor_row = int(T.HORIZON + T.CAMERA_HEIGHT * T.FOCAL / z)
    for c in cols:
        p0, p1 = T.column_pixels(c, c)
        d[max(0, top_row):floor_row, p0:p1] = z
    return d


class HeightTest(unittest.TestCase):
    def setUp(self):
        self.sig = {"vectors": np.eye(8, dtype=np.float32)[:1], "center": np.zeros(8, np.float32), "threshold": 0.8,
                    "name": "t"}
        band = np.random.default_rng(0).normal(size=(46, 8)).astype(np.float32)
        band[:, 0] = 0.0
        band[20:23] = 0.0
        band[20:23, 0] = 1.0
        self.band = band

    def detect(self, top, min_height=0.2):
        cfg = T.target_config({"min_height": min_height})
        return T.Detector(self.sig, cfg).detect(self.band, floor_depth(range(20, 23), 1.5, top), None, (0, 0, 0))[1]

    def test_something_low_on_the_floor_is_not_her(self):
        self.assertIsNone(self.detect(0.10))

    def test_lying_or_standing_she_is_high_enough(self):
        d = self.detect(0.27)
        self.assertIsNotNone(d)
        self.assertAlmostEqual(d["range"], 1.5 + T.CAM_X + 0.03, delta=0.1)
        self.assertIsNotNone(self.detect(1.6))

    def test_without_the_check_the_pillar_rule_is_unchanged(self):
        """No min_height: the arena's rule (nearest pixel at or above the horizon), no height test."""
        d = self.detect(0.30, min_height=None)
        self.assertIsNotNone(d)
        self.assertAlmostEqual(d["range"], 1.5 + T.CAM_X + 0.03, delta=0.1)


class FakeSim:
    def __init__(self):
        self.x = 0.0

    def render(self, depth=False):
        return np.zeros((360, 640), np.float32) if depth else np.zeros((360, 640, 3), np.uint8)

    def lidar(self, *args):
        return np.full(36, 3.0, np.float32)

    def body_position(self):
        return (self.x, 0.0, 0.08)

    def body_yaw(self):
        return 0.0


class ScriptedDetector:
    threshold = 0.7

    def __init__(self, script):
        self.script = list(script)

    def detect(self, band, depth, lidar, pose):
        d = self.script.pop(0) if self.script else None
        return (0.9 if d else 0.1), d


class SpotterTest(unittest.TestCase):
    def test_see_confirm_mark_watch_and_reject(self):
        from spotter import Spotter
        s = Spotter(detector=ScriptedDetector([det(2, 1), det(2.1, 1), None, det(2, 1.1), det(2.05, 1)]),
                    extract=lambda rgb: np.zeros((46, 8)))
        self.assertTrue(s.ready)
        sim = FakeSim()
        s.see(sim)
        self.assertFalse(s.state()["confirmed"])
        s.see(sim)
        st = s.state()
        self.assertTrue(st["confirmed"])
        self.assertAlmostEqual(st["x"], 2.05, places=1)
        s.mark()
        for _ in range(3):
            s.see(sim)
        self.assertEqual(s.state()["watch_hits"], 2)
        s.reject()
        self.assertFalse(s.state()["confirmed"])
        self.assertEqual(len(s.state()["rejected"]), 1)
        s.reset()
        self.assertEqual(s.state()["rejected"], [])


class DetectRobot(FakeRobot):
    """FakeRobot with a person detector: a look-around confirms her when she is within `sight` metres (or confirms
    `decoy`, a look-alike, on the first look); the first estimate of her is `off` metres beyond her (ranged on the
    furniture behind). `approach` walks up to the estimate, re-ranges her on the way (the estimate becomes her) and
    counts `approach_hits` close-up matches; `lost`: she is lost on the way. The watch counts hits only near her."""

    def __init__(self, sight=2.5, decoy=None, on_the_way=False, off=0.0, approach_hits=3, lost=False):
        super().__init__()
        self.her, self.sight, self.decoy, self.on_the_way = None, sight, decoy, on_the_way
        self.off, self.approach_hits, self.lost = off, approach_hits, lost
        self.spot = {"status": "ready", "confirmed": False, "watch_hits": 0}
        self.look = {"running": False}
        self.approach = {"running": False}

    def person(self, x=None, y=None, yaw_deg=0.0, pose="lying"):
        super().person(x, y, yaw_deg, pose)
        self.her = None if x is None else (x, y)

    def _confirm(self, p):
        if p == self.her and self.off:
            d = math.hypot(p[0] - self.x, p[1] - self.y) or 1.0
            p = (p[0] + self.off * (p[0] - self.x) / d, p[1] + self.off * (p[1] - self.y) / d)
        self.spot.update(confirmed=True, x=p[0], y=p[1], match={"sim": 0.9})

    def command(self, text):
        word, *args = text.split()
        if word == "approach":
            self.commands.append(text)
            stop = float(args[0])
            est = (self.spot["x"], self.spot["y"])
            if self.lost:
                self.approach = {"running": False, "reason": "lost", "estimate": list(est), "goal": None,
                                 "updates": 0}
                return text
            is_her = self.her is not None and math.hypot(est[0] - self.her[0], est[1] - self.her[1]) < 1.5
            if is_her:  # closer up, the camera ranges her, not the furniture behind her
                est = self.her
                self.spot.update(x=est[0], y=est[1])
            d = math.hypot(est[0] - self.x, est[1] - self.y) or 1.0
            self.x, self.y = est[0] - stop * (est[0] - self.x) / d, est[1] - stop * (est[1] - self.y) / d
            self.spot["watch_hits"] = self.approach_hits if is_her else 0
            self.approach = {"running": False, "reason": "arrived", "estimate": list(est),
                             "goal": [self.x, self.y], "updates": 2 if self.off else 0}
            self._mode = "goto"
            return text
        if word == "look":
            self.commands.append(text)
            self._mode = "look"
            self.look = {"running": False, "turned_deg": 360.0, "blocked": False}
            if self.decoy is not None:
                self._confirm(self.decoy)
            elif self.her is not None and math.hypot(self.x - self.her[0], self.y - self.her[1]) <= self.sight:
                self._confirm(self.her)
            return text
        if word == "watch":
            self.commands.append(text)
            self._mode = "watch"
            near = self.her is not None and self.spot.get("confirmed") and math.hypot(
                self.spot["x"] - self.her[0], self.spot["y"] - self.her[1]) < 0.6
            self.spot["watch_hits"] += 3 if near else 0
            return text
        if word == "spot":
            self.commands.append(text)
            if args[0] in ("reset", "reject"):
                self.spot.update(confirmed=False, watch_hits=0)
                self.spot.pop("x", None)
                if args[0] == "reject":
                    self.decoy = None
            else:
                self.spot["watch_hits"] = 0
            return text
        if word == "stop":
            self.commands.append(text)
            self._mode = "idle"
            return text
        out = super().command(text)
        if word == "goto" and self.on_the_way and self.her is not None:
            self.on_the_way = False
            self._confirm(self.her)
        return out

    def state(self):
        s = super().state()
        s["spotter"], s["look"], s["approach"] = dict(self.spot), dict(self.look), dict(self.approach)
        return s


class DetectorSearchTest(unittest.TestCase):
    def test_found_by_looking_around_in_her_zone(self):
        robot = DetectRobot()
        a = make(robot=robot)
        a.start("kitchen", pose="standing")
        finish(a)
        s = a.state()["search"]
        self.assertEqual(s["look_by"], "detector")
        self.assertEqual(s["miss_rate"], apartment.DETECT_MISS_RATE)
        self.assertEqual(s["status"], "found")
        self.assertEqual(a.search.steps[-1]["zone"], "kitchen")
        self.assertEqual(robot.poses[-1], "standing")
        self.assertIn("spot reset", robot.commands)
        self.assertIn("look", robot.commands)
        self.assertIn("approach {:.2f}".format(apartment.APPROACH_STOP), robot.commands)
        self.assertNotIn("watch {}".format(apartment.CHECK_CYCLES), robot.commands)  # checked on the way in
        self.assertLess(s["found_error"], 0.01)
        self.assertAlmostEqual(s["robot_to_person"], apartment.APPROACH_STOP, places=2)

    def test_the_approach_re_ranges_her_and_stops_near_her(self):
        robot = DetectRobot(off=0.9)  # the first range lands 0.9 m behind her (on the furniture)
        a = make(robot=robot)
        a.start("kitchen", pose="lying")
        finish(a)
        s = a.state()["search"]
        self.assertEqual(s["status"], "found")
        self.assertLess(s["found_error"], 0.01)  # the final estimate, not the first one
        self.assertAlmostEqual(s["robot_to_person"], apartment.APPROACH_STOP, places=2)
        self.assertEqual(s["approach"]["reason"], "arrived")
        self.assertGreater(s["approach"]["updates"], 0)
        self.assertIn("approach", [e["kind"] for e in a.state()["log"]])

    def test_too_few_hits_on_the_way_then_the_watch_decides(self):
        robot = DetectRobot(approach_hits=0)
        a = make(robot=robot)
        a.start("kitchen")
        finish(a)
        self.assertIn("watch {}".format(apartment.CHECK_CYCLES), robot.commands)
        self.assertEqual(a.state()["search"]["status"], "found")

    def test_lost_on_the_way_is_a_false_alarm(self):
        robot = DetectRobot(lost=True)
        a = make(robot=robot)
        a.start("kitchen")
        finish(a)
        s = a.state()["search"]
        self.assertGreaterEqual(s["false_alarms"], 1)
        self.assertIn("spot reject", robot.commands)
        self.assertNotIn("watch {}".format(apartment.CHECK_CYCLES), robot.commands)

    def test_a_look_alike_is_a_false_alarm_and_the_search_goes_on(self):
        robot = DetectRobot(decoy=(-3.0, 3.0))
        a = make(robot=robot)
        a.start("kitchen")
        finish(a)
        s = a.state()["search"]
        self.assertEqual(s["false_alarms"], 1)
        self.assertIn("spot reject", robot.commands)
        self.assertEqual(s["status"], "found")
        self.assertEqual(a.search.steps[-1]["zone"], "kitchen")
        self.assertIn("false_alarm", [e["kind"] for e in a.state()["log"]])

    def test_seen_on_the_way_needs_no_look_around(self):
        robot = DetectRobot(on_the_way=True)
        a = make(robot=robot)
        a.start("bedroom")
        finish(a)
        s = a.state()["search"]
        self.assertEqual(s["status"], "found")
        self.assertEqual(s["sightings"][0]["while"], "walking")
        self.assertNotIn("look", robot.commands)
        self.assertEqual(a.search.steps[-1]["zone"], "bedroom")

    def test_nobody_home_never_confirms_anyone(self):
        robot = DetectRobot()
        a = make(robot=robot)
        a.start("out")
        finish(a)
        s = a.state()["search"]
        self.assertIn(s["status"], ("nobody_home", "gave_up"))
        self.assertEqual(s["false_alarms"], 0)
        self.assertEqual(robot.commands.count("look"), len(a.search.steps))
        self.assertTrue(all(not step["seen"] for step in a.search.steps))

    def test_without_a_detector_the_look_is_the_mock(self):
        robot = FakeRobot()
        a = make(robot=robot)
        a.start("kitchen", miss_rate=0.0)
        finish(a)
        self.assertEqual(a.state()["search"]["look_by"], "mock")
        self.assertNotIn("look", robot.commands)


class FakeSpotter:
    """Confirms her after `after` cycles (never with None)."""

    status = "ready"
    ready = True

    def __init__(self, after=None):
        self.after, self.cycles = after, 0

        class E:
            confirmed = False
        self.estimate = E()

    def see(self, sim):
        self.cycles += 1
        if self.after is not None and self.cycles >= self.after:
            self.estimate.confirmed = True

    def state(self):
        return {"status": "ready", "confirmed": self.estimate.confirmed, "cycles": self.cycles}


def gl_available():
    try:
        import mujoco
        sys.path.insert(0, str(ROOT / "sim"))
        from jethexa_sim import JetHexaSim  # noqa: F401
        r = mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 8, 8)
        r.close()
        return True
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(gl_available(), "needs MuJoCo rendering")
class LookModeTest(unittest.TestCase):
    def run_look(self, spotter):
        from robot import LiveRobot
        robot = LiveRobot("apartment", "rule", realtime=False, spotter=spotter)
        robot.wait_ready(120)
        robot.command("look")
        deadline = time.time() + 120
        while time.time() < deadline:
            time.sleep(0.05)
            st = robot.state()
            if st.get("mode") == "idle" and not st["look"].get("running") and st["look"].get("kind") == "look":
                return st
        self.fail("the look-around did not end")

    def test_one_full_turn_with_the_detector_every_cycle(self):
        spotter = FakeSpotter()
        st = self.run_look(spotter)
        self.assertGreaterEqual(st["look"]["turned_deg"], 360.0)
        self.assertLess(st["look"]["turned_deg"], 420.0)
        self.assertFalse(st["look"]["seen"])
        self.assertGreaterEqual(spotter.cycles, 15)
        self.assertEqual(st["spotter"]["status"], "ready")

    def test_stops_turning_when_she_is_confirmed(self):
        st = self.run_look(FakeSpotter(after=4))
        self.assertTrue(st["look"]["seen"])
        self.assertLess(st["look"]["turned_deg"], 180.0)


class ApproachPointTest(unittest.TestCase):
    def test_on_the_line_when_free_else_around_her(self):
        from robot import approach_point, clearance
        half = 3.5
        p = approach_point((0.0, 0.0), (3.0, 0.0), 0.6, [], half)
        self.assertAlmostEqual(p[0], 0.6)
        self.assertAlmostEqual(p[1], 0.0)
        box = {"kind": "box", "x": 0.8, "y": 0.0, "yaw": 0.0, "hx": 0.3, "hy": 0.3}  # a chest between them
        p = approach_point((0.0, 0.0), (3.0, 0.0), 0.6, [box], half)
        self.assertGreaterEqual(clearance([box], half, p), 0.30)
        self.assertAlmostEqual(math.hypot(*p), 0.6, places=5)
        self.assertGreater(abs(p[1]), 0.1)  # moved round her, off the line
        self.assertIsNone(approach_point((3.3, 3.3), (0.0, 0.0), 0.6,
                                         [{"kind": "circle", "x": 3.3, "y": 3.3, "r": 1.5}], half))


class MovingSpotter:
    """Sees her every cycle at `est`, which a test can move; `last` is a match while `seeing`."""

    status = "ready"
    ready = True

    def __init__(self, est):
        class E:
            confirmed = True
        self.estimate = E()
        self.estimate.est = est
        self.seeing, self.cycles, self.on = True, 0, []
        self.last = {"sim": 0.9}

    def see(self, sim):
        self.cycles += 1
        self.last = {"sim": 0.9} if self.seeing else None

    def approach(self, on):
        self.on.append(on)

    def state(self):
        return {"status": "ready", "confirmed": True, "cycles": self.cycles}


@unittest.skipUnless(gl_available(), "needs MuJoCo rendering")
class ApproachModeTest(unittest.TestCase):
    def start(self, spotter):
        from robot import LiveRobot
        robot = LiveRobot("apartment", "rule", realtime=False, spotter=spotter)
        robot.wait_ready(120)
        robot.command("reset")
        time.sleep(0.3)
        robot.command("approach")
        return robot

    def wait(self, robot, seconds=120, until=None):
        deadline = time.time() + seconds
        while time.time() < deadline:
            time.sleep(0.05)
            st = robot.state()
            ap = st.get("approach") or {}
            if until is not None and until(st):
                return st
            if until is None and not ap.get("running") and ap.get("reason"):
                return st
        self.fail("the approach did not end")

    def test_follows_her_moving_estimate_stops_short_and_faces_her(self):
        spotter = MovingSpotter((0.0, 1.8))
        robot = self.start(spotter)
        st = self.wait(robot, until=lambda s: (s.get("approach") or {}).get("cycles", 0) >= 3)
        spotter.estimate.est = (0.9, 1.8)  # a closer match ranges her 0.9 m further right
        st = self.wait(robot)
        ap = st["approach"]
        self.assertEqual(ap["reason"], "arrived")
        self.assertGreaterEqual(ap["updates"], 1)
        self.assertEqual(ap["estimate"], [0.9, 1.8])
        dist = math.hypot(st["x"] - 0.9, st["y"] - 1.8)
        self.assertLess(dist, 0.6 + 0.35)
        face = math.degrees(math.atan2(1.8 - st["y"], 0.9 - st["x"]))
        self.assertLess(abs((face - st["yaw_deg"] + 180) % 360 - 180), 15.0)
        self.assertEqual(spotter.on, [True, False])

    def test_ends_lost_when_she_is_no_longer_seen(self):
        import robot as robot_module
        spotter = MovingSpotter((0.0, 2.5))
        spotter.seeing = False
        lost, robot_module.APPROACH_LOST = robot_module.APPROACH_LOST, 4  # before it could get there (~15 cycles)
        try:
            st = self.wait(self.start(spotter))
        finally:
            robot_module.APPROACH_LOST = lost
        self.assertEqual(st["approach"]["reason"], "lost")
        self.assertEqual(st["mode"], "idle")
        self.assertGreater(math.hypot(st["x"], st["y"] - 2.5), 1.0)  # stopped where it was


class ParseTest(unittest.TestCase):
    def test_new_commands(self):
        from robot import parse_command
        self.assertEqual(parse_command("look"), ("look",))
        self.assertEqual(parse_command("watch 4"), ("watch", 4))
        self.assertEqual(parse_command("spot reject"), ("spot", "reject"))
        self.assertEqual(parse_command("approach"), ("approach", 0.6))
        self.assertEqual(parse_command("approach 0.8"), ("approach", 0.8))
        for bad in ("watch 0", "watch 99", "spot now", "look 3", "approach 0.1", "approach 3"):
            with self.assertRaises(ValueError):
                parse_command(bad)


if __name__ == "__main__":
    unittest.main()

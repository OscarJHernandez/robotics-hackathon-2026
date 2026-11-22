"""Target search (nav/dagger/target.py): detector, candidates, schedule and the goal estimate, on synthetic data.

No simulator and no DINOv2: band features, depth pictures and lidar scans are made up so every edge case is exact.
"""

import math
import unittest

import numpy as np

try:
    import pandas as pd

    from nav.dagger import target as T
    HAVE = True
except ImportError:  # pragma: no cover
    HAVE = False

DIM = 8


def signature():
    v = np.zeros((1, DIM), np.float32)
    v[0, 0] = 1.0
    return {"vectors": v, "center": np.zeros(DIM, np.float32), "threshold": 0.8, "name": "test"}


def band(target_cols=(), lookalike_cols=(), seed=0):
    """46 background columns (no component along the signature) with the target in `target_cols`."""
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(T.COLS, DIM)).astype(np.float32)
    b[:, 0] = 0.0
    for c in target_cols:
        b[c] = 0.0
        b[c, 0], b[c, 1] = 1.0, 0.1
    for c in lookalike_cols:
        b[c] = 0.0
        b[c, 0], b[c, 1] = 1.0, 0.12
    return b


def depth(cols=(), z=2.0, background=6.0, fill=1.0):
    """A depth picture: `background` everywhere, `z` in the band's upper rows of band columns `cols`.
    `fill` < 1 puts the target in only part of each column's pixels (a thin target)."""
    d = np.full((T.H, T.W), background, np.float32)
    r0, r1 = T.band_pixel_rows()
    for c in cols:
        p0, p1 = T.column_pixels(c, c)
        d[r0:r1, p0:p0 + max(1, int((p1 - p0) * fill))] = z
    return d


def cfg(**kw):
    out = T.target_config(None)
    out.update(kw)
    return out


def expected_xy(pose, col, z, offset=0.03):
    b = float(T.column_bearings()[col])
    r = z / math.cos(b) + offset
    fx, fy = T.CAM_X + r * math.cos(b), r * math.sin(b)
    x, y, yaw = pose
    return x + math.cos(yaw) * fx - math.sin(yaw) * fy, y + math.sin(yaw) * fx + math.cos(yaw) * fy


@unittest.skipUnless(HAVE, "needs numpy and pandas")
class DetectorTest(unittest.TestCase):
    def test_not_in_view(self):
        sim, det = T.Detector(signature(), cfg()).detect(band(), depth(), None, (0, 0, 0))
        self.assertIsNone(det)
        self.assertLess(sim, 0.8)

    def test_position_from_bearing_and_depth(self):
        pose = (1.0, -0.5, 0.7)
        _, det = T.Detector(signature(), cfg()).detect(band([10]), depth([10], z=2.0), None, pose)
        self.assertIsNotNone(det)
        self.assertEqual(det["source"], "depth")
        self.assertAlmostEqual(det["bearing"], float(T.column_bearings()[10]), places=6)
        ex, ey = expected_xy(pose, 10, 2.0)
        self.assertAlmostEqual(det["x"], ex, places=3)
        self.assertAlmostEqual(det["y"], ey, places=3)

    def test_lookalike_far_away_is_not_a_detection(self):
        _, det = T.Detector(signature(), cfg()).detect(band([10], lookalike_cols=[35]), depth([10, 35]), None,
                                                      (0, 0, 0))
        self.assertIsNone(det, "two equally good places: no detection (margin)")

    def test_two_columns_refine_the_bearing(self):
        _, det = T.Detector(signature(), cfg()).detect(band([20, 21]), depth([20, 21]), None, (0, 0, 0))
        mid = float(T.column_bearings()[20:22].mean())
        self.assertAlmostEqual(det["bearing"], mid, places=3)

    def test_thin_target_uses_the_nearest_pixel(self):
        _, det = T.Detector(signature(), cfg()).detect(band([10]), depth([10], z=2.0, fill=0.3), None, (0, 0, 0))
        b = float(T.column_bearings()[10])
        self.assertAlmostEqual(det["range"], math.hypot(T.CAM_X + (2.0 / math.cos(b) + 0.03) * math.cos(b),
                                                        (2.0 / math.cos(b) + 0.03) * math.sin(b)), places=3)

    def test_too_far_is_ignored(self):
        _, det = T.Detector(signature(), cfg(max_range=1.5)).detect(band([23]), depth([23], z=3.0), None, (0, 0, 0))
        self.assertIsNone(det)

    def _lidar_for(self, det_bearing, r, background=6.0):
        scan = np.full(36, background, np.float32)
        a = det_bearing % (2 * math.pi)
        scan[int(round(a / math.radians(10))) % 36] = r
        return scan

    def test_lidar_gap_falls_back_to_depth(self):
        """The nearest beam passes beside the target and returns the wall: lidar disagrees with depth."""
        lidar = np.full(36, 6.0, np.float32)
        _, auto = T.Detector(signature(), cfg(range_source="auto")).detect(band([23]), depth([23], z=2.0), lidar,
                                                                          (0, 0, 0))
        self.assertEqual(auto["source"], "depth")
        _, only = T.Detector(signature(), cfg(range_source="lidar")).detect(band([23]), depth([23], z=2.0), lidar,
                                                                           (0, 0, 0))
        self.assertIsNone(only, "lidar-only ranging refuses a range the depth camera does not confirm")

    def test_lidar_used_when_it_agrees(self):
        lidar = np.full(36, 6.0, np.float32)
        lidar[0] = lidar[1] = lidar[35] = 2.0 + T.CAM_X - T.LIDAR_X
        _, det = T.Detector(signature(), cfg(range_source="auto")).detect(band([23]), depth([23], z=2.0), lidar,
                                                                         (0, 0, 0))
        self.assertEqual(det["source"], "lidar")

    def test_furniture_beside_the_target_on_the_lidar(self):
        """Furniture in the same 10 degree beam is nearer than the target: depth wins."""
        lidar = np.full(36, 6.0, np.float32)
        lidar[0] = lidar[1] = lidar[35] = 0.9
        _, det = T.Detector(signature(), cfg(range_source="auto")).detect(band([23]), depth([23], z=2.0), lidar,
                                                                         (0, 0, 0))
        self.assertEqual(det["source"], "depth")
        self.assertGreater(det["range"], 1.9)


class StubDetector:
    """Returns the queued detections (None = nothing seen), one per update."""

    def __init__(self):
        self.queue = []

    def detect(self, band_row, depth, lidar, pose):
        d = self.queue.pop(0) if self.queue else None
        return (0.9 if d else 0.1), d


def det(x, y, rng=1.5, sim=0.9):
    return {"x": x, "y": y, "sim": sim, "range": rng, "bearing": 0.0, "body_bearing": 0.0, "source": "depth",
            "edge": False, "column": 23}


class OrderSchedule:
    def __init__(self, order):
        self.order = order

    def next(self, locations, checked, pose):
        return next((i for i in self.order if not checked[i]), None)


def search(locations, order=None, **kw):
    detector = StubDetector()
    s = T.TargetSearch(cfg(**kw), detector, [{"x": x, "y": y, "p": p, "truth": False} for x, y, p in locations],
                       OrderSchedule(order or list(range(len(locations)))), truth=(0.0, 0.0))
    return s, detector


OPEN = np.full(36, 3.0, np.float32)


def spin(s, pose, steps=14, lidar=OPEN):
    x, y, yaw = pose
    for _ in range(steps):
        s.update(np.zeros((46, DIM)), None, lidar, (x, y, yaw))
        if s.command is None or s.command == (0.0, 0.0):
            return
        yaw += 0.53


@unittest.skipUnless(HAVE, "needs numpy and pandas")
class SearchTest(unittest.TestCase):
    def test_walks_to_the_first_location_then_looks_around(self):
        s, _ = search([(2.0, 0.0, 0.5), (-2.0, 0.0, 0.5)])
        goal, cmd = s.update(None, None, OPEN, (0.0, 0.0, 0.0))
        self.assertEqual(goal, (2.0, 0.0))
        self.assertIsNone(cmd)
        _, cmd = s.update(None, None, OPEN, (1.8, 0.0, 0.0))
        self.assertEqual(s.phase, "looking")
        self.assertEqual(cmd, (0.0, s.cfg["spin_turn"]))

    def test_seen_on_the_way_switches_at_once_after_confirmation(self):
        s, d = search([(3.0, 0.0, 1.0)], min_hits=1)
        d.queue = [det(1.0, 1.0), det(1.05, 1.0)]
        s.update(None, None, OPEN, (0.0, 0.0, 0.0))
        self.assertEqual(s.source, "location", "one sighting is not enough")
        goal, _ = s.update(None, None, OPEN, (0.1, 0.0, 0.0))
        self.assertEqual(s.phase, "to_target")
        self.assertLess(math.hypot(goal[0] - 1.02, goal[1] - 1.0), 0.05)
        self.assertEqual(s.source, "detected")

    def test_debounce_ignores_a_single_flash(self):
        s, d = search([(3.0, 0.0, 1.0)], window=3, min_hits=2)
        d.queue = [det(1.0, 1.0), None, None, det(1.0, 1.0), None, None]
        for _ in range(6):
            s.update(None, None, OPEN, (0.0, 0.0, 0.0))
        self.assertIsNone(s.est)

    def test_jump_rejected_until_confirmed(self):
        s, d = search([(3.0, 0.0, 1.0)], min_hits=1)
        d.queue = [det(1.0, 1.0), det(1.0, 1.0), det(2.5, -1.0)]
        for _ in range(3):
            s.update(None, None, OPEN, (0.0, 0.0, 0.0))
        self.assertLess(math.hypot(s.goal[0] - 1.0, s.goal[1] - 1.0), 0.01, "one far sighting does not move it")
        d.queue = [det(2.5, -1.0)]
        s.update(None, None, OPEN, (0.0, 0.0, 0.0))
        self.assertLess(math.hypot(s.goal[0] - 2.5, s.goal[1] + 1.0), 0.01, "confirmed twice: the goal moves")

    def test_lost_target_keeps_the_estimate(self):
        s, d = search([(3.0, 0.0, 1.0)], min_hits=1)
        d.queue = [det(1.0, 1.0), det(1.0, 1.0)]
        for _ in range(2):
            s.update(None, None, OPEN, (0.0, 0.0, 0.0))
        for _ in range(10):
            goal, _ = s.update(None, None, OPEN, (0.2, 0.2, 0.0))
        self.assertEqual(goal, s.est)
        self.assertEqual(s.source, "detected")

    def test_reached_but_empty_is_a_false_positive(self):
        s, d = search([(3.0, 0.0, 0.5), (-3.0, 0.0, 0.5)], min_hits=1)
        d.queue = [det(1.0, 1.0), det(1.0, 1.0)]
        for _ in range(2):
            s.update(None, None, OPEN, (0.0, 0.0, 0.0))
        s.update(None, None, OPEN, (1.0, 0.7, 0.0))
        self.assertEqual(s.phase, "to_target", "0.3 m away: keep driving")
        for _ in range(7):
            s.update(None, None, OPEN, (1.0, 0.7, 0.0))
        self.assertEqual((s.phase, s.look_at), ("looking", "estimate"), "close for 8 steps without reaching it")
        spin(s, (1.0, 0.7, 0.0))
        self.assertEqual(s.false_positives, 1)
        self.assertIsNone(s.est)
        self.assertEqual(s.phase, "walking")
        d.queue = [det(1.0, 1.0), det(1.0, 1.0)]  # the same false spot again: ignored
        s.update(None, None, OPEN, (1.0, 0.9, 0.0))
        s.update(None, None, OPEN, (1.0, 0.9, 0.0))
        self.assertIsNone(s.est)

    def test_one_look_around_covers_close_locations(self):
        s, _ = search([(2.0, 0.0, 0.4), (2.5, 1.0, 0.3), (-3.0, 0.0, 0.3)], detect_range=2.5)
        s.update(None, None, OPEN, (0.0, 0.0, 0.0))
        s.update(None, None, OPEN, (1.9, 0.0, 0.0))
        spin(s, (1.9, 0.0, 0.0))
        self.assertEqual(s.checked, [True, True, False])
        self.assertEqual(s.current, 2)
        self.assertAlmostEqual(sum(loc["p"] for loc in s.locations), 1.0, places=6)
        self.assertGreater(s.locations[2]["p"], 0.85)

    def test_all_checked_is_not_found(self):
        s, _ = search([(1.0, 0.0, 1.0)])
        s.update(None, None, OPEN, (0.8, 0.0, 0.0))
        spin(s, (0.8, 0.0, 0.0))
        self.assertEqual(s.phase, "not_found")
        self.assertEqual(s.command, (0.0, 0.0))
        self.assertEqual(s.summary()["search_end"], "not_found")

    def test_found_while_looking_around(self):
        s, d = search([(1.0, 0.0, 1.0)], min_hits=1)
        s.update(None, None, OPEN, (0.8, 0.0, 0.0))
        d.queue = [None, None, det(2.0, 1.5), det(2.0, 1.5)]
        spin(s, (0.8, 0.0, 0.0))
        self.assertEqual(s.phase, "to_target")
        self.assertIsNone(s.command)
        self.assertEqual(s.goal, s.est)

    def test_no_room_to_turn_moves_into_open_space_first(self):
        s, _ = search([(1.0, 0.0, 1.0)])
        tight = np.full(36, 2.0, np.float32)
        tight[9] = 0.2  # something close on the left
        tight[27] = 4.0  # open to the right
        s.update(None, None, tight, (0.8, 0.0, 0.0))
        s.update(None, None, tight, (0.8, 0.0, 0.0))
        self.assertEqual(s.phase, "reposition")
        self.assertIsNone(s.command, "the student walks to the open space")
        self.assertLess(s.goal[1], -0.5, "towards the open side (right = -y)")


@unittest.skipUnless(HAVE, "needs numpy and pandas")
class CandidateTest(unittest.TestCase):
    def test_candidates(self):
        foot = [{"kind": "circle", "x": 1.0, "y": 1.0, "r": 0.5}]
        for seed in range(20):
            locs = T.candidates(np.random.default_rng(seed), (-1.0, -1.0), foot, 3.5, n=4, min_gap=1.0)
            self.assertEqual(len(locs), 4)
            self.assertAlmostEqual(sum(loc["p"] for loc in locs), 1.0, places=6)
            truth = [loc for loc in locs if loc["truth"]]
            self.assertEqual(len(truth), 1)
            d = math.hypot(truth[0]["x"] + 1.0, truth[0]["y"] + 1.0)
            self.assertTrue(0.8 <= d <= 1.5, d)
            for i, a in enumerate(locs):
                self.assertGreater(math.hypot(a["x"] - 1.0, a["y"] - 1.0), 0.85, "not inside furniture")
                for b in locs[i + 1:]:
                    self.assertGreaterEqual(math.hypot(a["x"] - b["x"], a["y"] - b["y"]), 1.0 - 1e-9)

    def test_truth_is_not_always_first(self):
        firsts = []
        for seed in range(30):
            locs = T.candidates(np.random.default_rng(seed), (0.0, 0.0), [], 3.5)
            order = T.RandomSchedule(seed)
            firsts.append(locs[order.next(locs, [False] * 4, (0, 0, 0))]["truth"])
        self.assertTrue(0 < sum(firsts) < 30)

    def test_random_schedule_repeats_with_the_seed(self):
        locs = [{"x": float(i), "y": 0.0, "p": 0.25} for i in range(4)]

        def order(seed):
            sched, checked, out = T.RandomSchedule(seed), [False] * 4, []
            while (i := sched.next(locs, checked, (0, 0, 0))) is not None:
                out.append(i)
                checked[i] = True
            return out

        self.assertEqual(order([3, 4]), order([3, 4]))
        self.assertEqual(sorted(order(1)), [0, 1, 2, 3])

    def test_probability_schedule(self):
        locs = [{"x": 0.0, "y": 0.0, "p": 0.2}, {"x": 1.0, "y": 0.0, "p": 0.7}, {"x": 2.0, "y": 0.0, "p": 0.1}]
        self.assertEqual(T.make_schedule("probability", 0).next(locs, [False, False, False], (0, 0, 0)), 1)

    def test_unknown_keys_fail(self):
        with self.assertRaises(SystemExit):
            T.target_config({"signatur": "x"})


# ---- run_v2 with fakes: the search's goal reaches the goal columns and the planner; the truth stays for the rest --

class FakeEnv:
    goal = (3.0, 0.0)

    def __init__(self):
        self.steps, self.done, self.path_length = 0, False, 0.0
        self.start, self.start_dist = (0.0, 0.0, 0.0), 3.0
        self.x = 0.0

    def obs(self):
        return {"robot_pose": (self.x, 0.0, 0.0), "goal_point": self.goal, "depth": np.zeros((4, 4)),
                "lidar": np.full(36, 3.0, np.float32)}

    def reset(self, scene_id, seed):
        self.steps, self.x = 0, 0.0
        return self.obs()

    def pose(self):
        return (self.x, 0.0, 0.0)

    def state(self):
        return {}

    def step(self, cmd):
        self.steps += 1
        self.x += 0.1
        done = self.steps >= 3
        return (None if done else self.obs()), {"collision": False, "done": done,
                                                 "done_reason": "timeout" if done else "", "stride": cmd[0],
                                                 "turn": cmd[1]}


class FakeFeaturizer:
    last_band = None

    def base_rows(self, observations, prev):
        self.last_band = np.zeros((len(observations), 46, DIM), np.float16)
        rows = []
        for o in observations:
            dx = o["goal_point"][0] - o["robot_pose"][0]
            rows.append({"goal_dist": dx, "goal_sin": 0.0, "goal_cos": 1.0, "x": o["robot_pose"][0]})
        return pd.DataFrame(rows)


class FakeExpert:
    nopath, look_bearing = False, 0.0

    def act(self, state):
        return (0.2, 0.0)


class FakeFeatures:
    cfg = {"lidar_plan": True}

    def __init__(self):
        self.seen = []

    def live(self, base, band, hist):
        self.seen.append(base[["goal_dist", "goal_sin", "goal_cos"]].to_numpy().copy())
        return np.zeros((len(base), 1), np.float32), list(hist)


class FakePolicy:
    def act_batch(self, X):
        return np.tile([0.1, 0.0], (len(X), 1))


class FakePlanner:
    goals = []

    def reset(self):
        pass

    def step(self, pose, goal, ranges):
        FakePlanner.goals.append(tuple(goal))
        return (0.0, 1.0)


class FixedSearch:
    """Goal (0, 2) to the left; a scripted turn on the second step."""
    phase, source = "walking", "location"

    def __init__(self):
        self.n, self.command = 0, None

    def update(self, band_row, depth, lidar, pose):
        self.n += 1
        self.command = (0.0, 0.25) if self.n == 2 else None
        return (0.0, 2.0), self.command

    def row(self):
        return {"goal_est_x": 0.0, "goal_est_y": 2.0, "scripted": self.command is not None}

    def summary(self):
        return {"looks": 0}


@unittest.skipUnless(HAVE, "needs numpy and pandas")
class RolloutTest(unittest.TestCase):
    def test_search_goal_reaches_student_and_planner_only(self):
        from nav.dagger import lidarmap, rollout
        original = lidarmap.LidarPlanner
        lidarmap.LidarPlanner = FakePlanner
        FakePlanner.goals = []
        try:
            features, env = FakeFeatures(), FakeEnv()
            collected = []
            rollout.run_v2([env], FakeFeaturizer(), FakeExpert(), [{"episode": 0, "scene_id": 1, "seed": 2}],
                           np.random.default_rng(0), ["goal_dist"], policy=FakePolicy(), features=features,
                           beta=0.0, on_chunk=lambda out: collected.append(out.take()),
                           search=lambda e, ep: FixedSearch())
        finally:
            lidarmap.LidarPlanner = original
        self.assertEqual(FakePlanner.goals[0], (0.0, 2.0), "the planner gets the search's goal")
        dist, sin, cos = features.seen[0][0]
        self.assertAlmostEqual(dist, 2.0, places=5)
        self.assertAlmostEqual(sin, 1.0, places=5)  # straight to the left
        base, meta, _, episodes = collected[-1]
        self.assertAlmostEqual(float(base[0][0]), 3.0, "stored rows keep the true goal")
        self.assertEqual(meta[0]["goal_x"], 3.0)
        self.assertEqual(meta[1]["executed_turn"], 0.25, "the look-around command is executed")
        self.assertTrue(meta[1]["scripted"])
        self.assertEqual(meta[0]["y_stride"], 0.2, "labels stay the expert's")
        self.assertIn("looks", episodes[0])


if __name__ == "__main__":
    unittest.main()

"""The student as a driver in the app (app/student.py, app/robot.py), with fakes: no TabPFN, no API calls, no credits.

- Student: a new goto empties the lidar map, the command is clipped, one policy call per decision, a used-up budget
  turns into status no_credits; the credit helpers read the API's usage.
- LiveRobot with a fake student: it walks, the A* route is only shown, contacts are counted, getting stuck hands
  the goto to A* + the a-to-b driver (recorded), and a student that failed to load leaves the a-to-b driver walking.
- The v29 export (nav/dagger/export.py), when it is on this machine: the drive-time path with the saved PCA gives
  the exported rows back.
"""

import math
import sys
import time
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "app", REPO / "homes", REPO / "sim", REPO / "vision"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:  # pragma: no cover
    HAVE_PANDAS = False


def _rendering():
    try:
        from jethexa_sim import JetHexaSim
        JetHexaSim().render()
        return None
    except Exception as error:
        return str(error)


NO_GL = _rendering()


class FakePolicy:
    def __init__(self, command=(0.5, -0.9)):
        self.command, self.calls, self.rows = command, 0, []

    def act_batch(self, X):
        self.calls += 1
        self.rows.append(np.asarray(X))
        return np.array([self.command], float)


class FakeFeaturizer:
    last_band = None

    def base_rows(self, observations, prev):
        obs = observations[0]
        self.last_band = np.zeros((1, 46, 384), np.float16)
        return pd.DataFrame({"goal_dist": [math.hypot(obs["goal_point"][0] - obs["robot_pose"][0],
                                                      obs["goal_point"][1] - obs["robot_pose"][1])],
                             "lr_0": [float(obs["lidar"][0])]})


class FakeFeatures:
    columns = ["goal_dist", "lr_0", "lp_sin", "lp_cos"]

    def live(self, base, band, prev):
        return base[self.columns].to_numpy(np.float32), prev


class FakeTwin:
    def __init__(self):
        self.synced = []

    def sync(self, model, data, goal):
        self.synced.append(tuple(goal))

    def observe(self):
        return {"rgb": np.zeros((360, 640, 3), np.uint8), "depth": np.ones((360, 640), np.float32)}

    def check_compatible(self, model):
        return True


class FakeSim:
    """Enough of JetHexaSim for Student.act."""

    def __init__(self):
        self.model = self.data = None

    def body_position(self):
        return (0.0, 0.0, 0.1)

    def body_yaw(self):
        return 0.0

    def lidar(self, rays, lo, hi):
        return np.full(rays, 2.0, np.float32)


@unittest.skipUnless(HAVE_PANDAS, "needs pandas")
class StudentTest(unittest.TestCase):
    def make(self, policy=None):
        from student import Student
        return Student(policy=policy or FakePolicy(), features=FakeFeatures(), featurizer=FakeFeaturizer(),
                       twin=FakeTwin())

    def test_a_goto_resets_the_map_and_the_command_is_clipped(self):
        student = self.make()
        self.assertEqual(student.status, "ready")
        student.start((1.0, 0.0))
        cmd = student.act(FakeSim(), (1.0, 0.0))
        self.assertEqual(cmd, (0.3, -0.25))  # clipped to STRIDE_RANGE and TURN_RANGE
        self.assertTrue(student.planner.hits.any())  # the lidar scan went on the map
        self.assertEqual(student.policy.calls, 1)  # one call per decision
        self.assertEqual(student.twin.synced[-1], (1.0, 0.0))  # the twin's pillar at the goal
        row = student.policy.rows[-1][0]
        self.assertAlmostEqual(row[0], 1.0, places=5)  # goal_dist
        self.assertAlmostEqual(row[2] ** 2 + row[3] ** 2, 1.0, places=5)  # the plan bearing as sin, cos
        student.start((-1.0, 2.0))
        self.assertFalse(student.planner.hits.any())  # a new goto: an empty map
        self.assertEqual(student.info()["decisions"], 1)

    def test_a_used_up_budget_stops_the_student(self):
        from nav.dagger.policy import CreditBudget, OutOfCredits

        class Budgeted(FakePolicy):
            budget = CreditBudget(1)

            def act_batch(self, X):
                self.budget.spend()
                return super().act_batch(X)

        student = self.make(Budgeted((0.1, 0.0)))
        student.start((1.0, 0.0))
        student.act(FakeSim(), (1.0, 0.0))
        with self.assertRaises(OutOfCredits):
            student.act(FakeSim(), (1.0, 0.0))
        self.assertEqual(student.status, "no_credits")

    def test_an_api_usage_limit_pauses_the_student_until_it_resets(self):
        class Limited(FakePolicy):
            def act_batch(self, X):
                raise RuntimeError("Fail to call predict: [HTTP 429] Daily usage limit reached. Your daily limit is "
                                   "35000000 tokens. Resets at 2026-11-23 00:00:00 UTC.")

        student = self.make(Limited((0.1, 0.0)))
        student.start((1.0, 0.0))
        with self.assertRaises(RuntimeError):
            student.act(FakeSim(), (1.0, 0.0))
        self.assertEqual(student.status, "no_credits")
        self.assertIn("2026-11-23 00:00 UTC", student.info()["reason"])
        student.retry_at = time.time() + 60
        self.assertFalse(student.ready)  # not before the reset: no retry on every walk
        student.retry_at = time.time() - 1
        self.assertTrue(student.ready)  # after it: the student drives again

    def test_other_api_errors_do_not_pause_the_student(self):
        class Flaky(FakePolicy):
            def act_batch(self, X):
                raise RuntimeError("connection reset")

        student = self.make(Flaky((0.1, 0.0)))
        student.start((1.0, 0.0))
        with self.assertRaises(RuntimeError):
            student.act(FakeSim(), (1.0, 0.0))
        self.assertTrue(student.ready)


class CreditsTest(unittest.TestCase):
    def test_usage_is_read_from_the_dict_and_the_sentence(self):
        from nav.dagger.policy import parse_usage
        self.assertEqual(parse_usage({"current_usage": 47800000, "usage_limit": 50000000,
                                      "reset_time": "2026-12-01"})["left"], 2200000)
        self.assertIsNone(parse_usage({"current_usage": 5, "usage_limit": -1})["left"])
        text = ("Currently, you have used 47800000 of the allowed limit of 50000000 credits. "
                "The limit will reset at 2026-12-01 00:00:00 UTC.")
        self.assertEqual(parse_usage(text)["left"], 2200000)
        self.assertIsNone(parse_usage("Currently, you have used 3 of the allowed limit of Unlimited credits.")["left"])

    def test_the_budget_raises_before_it_is_exceeded(self):
        from nav.dagger.policy import CreditBudget, OutOfCredits
        budget = CreditBudget(2)
        budget.spend()
        budget.spend()
        self.assertEqual((budget.left, budget.credits()), (0, 20000))
        with self.assertRaises(OutOfCredits):
            budget.spend()
        self.assertEqual(budget.calls, 2)


@unittest.skipUnless(HAVE_PANDAS, "needs pandas")
class ClientPolicyTest(unittest.TestCase):
    def test_one_stacked_regressor_one_call_for_both_commands(self):
        from nav.dagger.policy import CreditBudget, OutOfCredits, make_student

        class FakeRegressor:
            fits, predicts = [], []

            def fit(self, X, y):
                self.fits.append((X.shape, y.shape))
                self.X, self.y = X, y

            def predict(self, X):
                self.predicts.append(X.shape)
                return np.where(X[:, -1] == 0, 9.0, -9.0)  # stride rows, then turn rows

        policy = make_student({"kind": "tabpfn", "backend": "client", "n_estimators": 4, "model_version": "v3.5"},
                              ["a", "b"], [], budget=CreditBudget(1))
        policy._estimator = FakeRegressor
        table = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 1.0, 0.0], "y_stride": [0.1, 0.2, 0.3],
                              "y_turn": [-0.1, 0.0, 0.1]})
        policy.fit(table)
        self.assertEqual(FakeRegressor.fits[-1], ((6, 3), (6,)))  # rows twice, plus the `output` column
        np.testing.assert_array_equal(policy.model.X[:, -1], [0, 0, 0, 1, 1, 1])
        np.testing.assert_allclose(policy.model.y, [0.1, 0.2, 0.3, -0.1, 0.0, 0.1], atol=1e-6)
        out = policy.act_batch(table[["a", "b"]].iloc[:2])
        self.assertEqual(FakeRegressor.predicts, [(4, 3)])  # one call, 2n rows
        np.testing.assert_allclose(out, [[0.3, -0.25], [0.3, -0.25]])  # clipped
        with self.assertRaises(OutOfCredits):
            policy.act_batch(table[["a", "b"]].iloc[:1])
        self.assertEqual(len(FakeRegressor.predicts), 1)  # refused before the call
        with self.assertRaises(ValueError):
            make_student({"kind": "knn", "backend": "client"}, ["a"], [])


class StuckAndContactTest(unittest.TestCase):
    def test_the_trainers_stuck_rule(self):
        from student import Progress
        p = Progress(cycles=3, gain=0.2)
        p.reset(2.0)
        self.assertFalse(p.update(1.75))  # a new best by more than 0.2 m
        self.assertFalse(p.update(1.7))
        self.assertFalse(p.update(1.65))
        self.assertTrue(p.update(1.6))  # 3 cycles without a 0.2 m gain

    def test_touches_of_walls_and_furniture_are_counted_once_each(self):
        import mujoco

        from robot import ApartmentScene
        from student import ContactCounter
        scene = ApartmentScene(simple=True)
        counter = ContactCounter(scene.sim.model)
        scene.place((0.0, 0.0, 0.0))
        self.assertFalse(counter.update(scene.sim.data, (0.0, 0.0)))
        scene.place((scene.half - 0.05, 0.0, 0.0))  # legs through the east wall
        mujoco.mj_forward(scene.sim.model, scene.sim.data)
        self.assertTrue(counter.update(scene.sim.data, (scene.half - 0.05, 0.0)))
        self.assertFalse(counter.update(scene.sim.data, (scene.half - 0.05, 0.0)))  # the same touch
        scene.place((0.0, 0.0, 0.0))
        counter.update(scene.sim.data)  # clear for one step only: a foot tapping the wall again
        scene.place((scene.half - 0.05, 0.0, 0.0))
        self.assertFalse(counter.update(scene.sim.data))
        scene.place((0.0, 0.0, 0.0))
        for _ in range(counter.clear):  # clear for a second
            counter.update(scene.sim.data)
        scene.place((scene.half - 0.05, 0.0, 0.0))
        self.assertTrue(counter.update(scene.sim.data, (1.0, 2.0)))
        self.assertEqual(counter.count, 2)
        self.assertEqual(counter.points[-1], (1.0, 2.0))


class LiveStudent:
    """A fake Student for LiveRobot: a fixed command, records what it is asked."""

    def __init__(self, command=(0.3, 0.0), status="ready"):
        self.command, self.status, self.reason = command, status, ""
        self.twin = FakeTwin()
        self.starts, self.acts = [], 0

    @property
    def ready(self):
        return self.status == "ready"

    def start(self, goal):
        self.starts.append(tuple(goal))

    def act(self, sim, goal):
        self.acts += 1
        return self.command

    def info(self):
        return {"status": self.status, "decisions": self.acts}


@unittest.skipIf(NO_GL is not None, "rendering is not available: {}".format(NO_GL))
class LiveRobotStudentTest(unittest.TestCase):
    def robot(self, student, **kw):
        import flats as F
        from robot import LiveRobot
        robot = LiveRobot("apartment", "student", student=student, ab="rule", realtime=False, **kw)
        self.assertTrue(robot.wait_ready(120))
        flat = F.random_flat(np.random.default_rng([0, 0]), 0)
        robot.load(flat)
        return robot, flat

    def wait(self, robot, until, seconds=60):
        end = time.time() + seconds
        while time.time() < end:
            state = robot.state()
            if until(state):
                return state
            time.sleep(0.05)
        self.fail("timed out; last state {}".format({k: robot.state().get(k) for k in
                                                     ("mode", "driver", "fallbacks", "x", "y", "student")}))

    def test_the_student_walks_to_the_end_point_and_the_route_is_only_shown(self):
        student = LiveStudent((0.3, 0.0))
        robot, flat = self.robot(student)
        self.wait(robot, lambda s: s.get("x") is not None and math.hypot(s["x"] - flat["dock"]["x"],
                                                                        s["y"] - flat["dock"]["y"]) < 0.05)
        s = robot.state()
        x, y, yaw = s["x"], s["y"], math.radians(s["yaw_deg"])
        goal = (x + 1.0 * math.cos(yaw), y + 1.0 * math.sin(yaw))  # straight ahead, into the room
        robot.command("goto {:.2f} {:.2f}".format(*goal))
        s = self.wait(robot, lambda s: s.get("mode") == "goto" and student.acts >= 2)
        self.assertEqual(s["driver"], "student")
        self.assertEqual(student.starts[-1], (round(goal[0], 2), round(goal[1], 2)))  # a new goto: start()
        self.assertFalse(s["route_followed"])  # the A* route is shown, not walked
        self.assertEqual(s["student"]["status"], "ready")
        self.assertIn("contacts", s)
        s = self.wait(robot, lambda s: s.get("mode") == "idle")
        self.assertLess(math.hypot(s["x"] - goal[0], s["y"] - goal[1]), 0.45)
        self.assertEqual(s["fallbacks"], 0)

    def test_stuck_hands_the_goto_to_astar_and_the_ab_driver(self):
        student = LiveStudent((0.0, 0.0))  # stands still: no progress
        robot, flat = self.robot(student, stuck_cycles=3)
        zone = flat["zones"][0]
        robot.command("goto {:.2f} {:.2f}".format(zone["view_x"], zone["view_y"]))
        s = self.wait(robot, lambda s: s.get("fallbacks", 0) >= 1)
        self.assertEqual(s["fallback"]["driver"], "rule")
        self.assertIn("stuck", s["fallback"]["reason"])
        s = self.wait(robot, lambda s: s.get("mode") == "goto" and s.get("driver") == "rule")
        self.assertTrue(s["route_followed"])
        acts = student.acts
        time.sleep(0.5)
        self.assertEqual(student.acts, acts)  # the student is not asked again for this goto

    def test_a_student_that_failed_to_load_leaves_the_ab_driver_walking(self):
        student = LiveStudent(status="failed")
        robot, flat = self.robot(student)
        zone = flat["zones"][0]
        robot.command("goto {:.2f} {:.2f}".format(zone["view_x"], zone["view_y"]))
        s = self.wait(robot, lambda s: s.get("mode") == "goto" and s.get("driver") == "rule")
        self.assertEqual(s["student"]["status"], "failed")
        self.assertEqual(student.acts, 0)
        self.assertEqual(robot.state()["ab_driver"], "rule")


class FlatBenchTest(unittest.TestCase):
    def test_legs_follow_the_search_and_the_summary_has_intervals(self):
        from nav.dagger.flatbench import make_legs, summarize
        legs = make_legs(2)
        self.assertEqual(len(legs), 2 * 2 * 5)  # flats x {home, scattered} x five zones
        for a, b in zip(legs, legs[1:]):
            if (a["flat_id"], a["condition"]) == (b["flat_id"], b["condition"]):
                self.assertEqual(b["start"][:2], a["goal"])  # the next leg starts where this one ends
                self.assertEqual(b["from"], a["to"])
        self.assertEqual(legs[0]["from"], "dock")
        self.assertEqual([l["to"] for l in legs[:5]], [l["to"] for l in legs[5:10]])  # same zone order, both conditions
        self.assertTrue(legs[5]["flat"].get("scattered"))
        out = [{"driver": "student", "condition": "home", "reason": r, "cycles": 10} for r in ("goal", "goal", "stuck")]
        s = summarize(out)["student/home"]
        self.assertEqual((s["legs"], s["goal"], s["stuck"]), (3, 2, 1))
        self.assertLess(s["goal_ci"][0], 2 / 3)
        self.assertGreater(s["goal_ci"][1], 2 / 3)


@unittest.skipUnless(HAVE_PANDAS, "needs pandas")
class ExportTest(unittest.TestCase):
    def test_the_drive_time_path_gives_the_exported_rows_back(self):
        from nav.dagger.export import load_export
        from student import export_dir
        folder = export_dir()
        if not (folder / "sample.parquet").exists():
            self.skipTest("no student export at {} (nav/dagger/export.py)".format(folder))
        features, table, meta = load_export(folder)
        self.assertEqual(meta["rows"], len(table))
        self.assertEqual(len(features.columns), 116)
        self.assertEqual(features.columns[:3], ["goal_dist", "goal_sin", "goal_cos"])
        self.assertEqual(features.columns[-2:], ["lp_sin", "lp_cos"])
        self.assertEqual(list(table.columns[-2:]), ["y_stride", "y_turn"])
        sample = pd.read_parquet(folder / "sample.parquet")
        band = np.load(folder / "sample_band.npy")
        for i in range(len(sample)):
            X, _ = features.live(sample.iloc[[i]].reset_index(drop=True), band[i:i + 1], [None])
            expected = table.iloc[int(sample["_export_row"][i])][features.columns].to_numpy(np.float32)
            np.testing.assert_allclose(X[0], expected, atol=1e-4)


if __name__ == "__main__":
    unittest.main()

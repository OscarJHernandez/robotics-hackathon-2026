"""The speed-up of the live simulation (app/robot.py LiveRobot.set_speed, POST /api/speed).

- The speed changes the pacing, not the physics: the same commands give the same pose after the same number of
  steps at x1 and x5 (realtime=False).
- With a virtual clock, x5 runs the steps five times faster in wall time, speed_actual says so, and the pictures
  keep their rate per wall second (rendered every VIEWS[view] x speed steps).
- Only x1, x2, x3 and x5 are taken; POST /api/speed and /api/health carry the speed.

No TabPFN. The LiveRobot tests skip without OpenGL, like the other camera tests.
"""

import http.client
import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "app", REPO / "homes", REPO / "sim"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _rendering():
    try:
        from jethexa_sim import JetHexaSim
        JetHexaSim().render()
        return None
    except Exception as error:  # no OpenGL on this machine (for example a CI runner)
        return str(error)


NO_GL = _rendering()


class VirtualClock:
    """Time that moves only when the loop sleeps: every step then waits exactly until its tick."""

    def __init__(self):
        self.now = 0.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.now

    def sleep(self, seconds):
        with self.lock:
            self.now += seconds


@unittest.skipIf(NO_GL is not None, "rendering is not available: {}".format(NO_GL))
class SpeedTest(unittest.TestCase):
    def run_robot(self, speed, steps, **kw):
        from robot import LiveRobot
        robot = LiveRobot("floor", max_steps=steps, **kw)
        robot.set_speed(speed)  # before the loop starts: every step at this speed
        robot.command("forward 1")
        self.assertTrue(robot.wait_ready(60))
        self.assertTrue(robot.finished.wait(120))
        return robot, robot.state()

    def test_the_speed_changes_the_pacing_and_not_the_physics(self):
        poses = {}
        for speed in (1, 5):
            _, state = self.run_robot(speed, 200, realtime=False)
            self.assertEqual(state["speed"], speed)
            self.assertAlmostEqual(state["sim_time"], 200 * 0.02, places=6)
            poses[speed] = (state["x"], state["y"], state["yaw_deg"], state["mode"])
        self.assertEqual(poses[1], poses[5])
        self.assertGreater(poses[1][0], 0.2)  # it did walk forward

    def test_x5_runs_five_times_faster_and_the_pictures_keep_their_rate(self):
        import robot as R
        steps = 240
        rates, actual = {}, {}
        with mock.patch.object(R, "WATCHED", 1e9):  # every view counts as watched for the whole run
            for speed in (1, 5):
                clock = VirtualClock()
                robot, state = self.run_robot(speed, steps, clock=clock, sleep=clock.sleep)
                wall = clock()
                self.assertAlmostEqual(wall, (steps - 1) * R.CONTROL_DT / speed, places=6)  # no wait after the last step
                self.assertAlmostEqual(state["sim_time"], steps * R.CONTROL_DT, places=6)
                actual[speed] = state["speed_actual"]
                rates[speed] = {view: n / wall for view, n in robot.renders.items()}
        for speed in (1, 5):
            self.assertAlmostEqual(actual[speed], speed, delta=0.1 * speed)
        for view, every in R.VIEWS.items():
            expected = 1.0 / (every * R.CONTROL_DT)  # pictures per wall second at x1
            for speed in (1, 5):
                self.assertAlmostEqual(rates[speed][view], expected, delta=0.15 * expected, msg=(view, speed))

    def test_only_the_offered_speeds_are_taken(self):
        from robot import SPEEDS, LiveRobot
        self.assertEqual(SPEEDS, (1, 2, 3, 5))
        robot = LiveRobot("floor", realtime=False, max_steps=2)
        self.assertTrue(robot.wait_ready(60))
        for good in (1, 2, 3, 5, "3", 2.0):
            self.assertEqual(robot.set_speed(good), int(float(good)))
        for bad in (0, 4, 10, -1, 1.5, None, "fast", float("nan")):
            with self.assertRaises(ValueError):
                robot.set_speed(bad)
        self.assertEqual(robot.speed, 2)


class FakeRobot:
    def __init__(self):
        self.speed = 1

    def set_speed(self, speed):
        if speed not in (1, 2, 3, 5):
            raise ValueError("speed must be one of: x1, x2, x3, x5")
        self.speed = speed
        return speed

    def state(self):
        return {"speed": self.speed, "speed_actual": 0.97 * self.speed, "driver": "rule"}


class SpeedRouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import server
        cls.server = server
        cls.patch = mock.patch.object(server, "ROBOT", FakeRobot())
        cls.patch.start()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.httpd.daemon_threads = True
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.patch.stop()

    def request(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=10)
        try:
            conn.request(method, path, body=json.dumps(body) if body is not None else None,
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            return response.status, json.loads(response.read() or b"{}")
        finally:
            conn.close()

    def test_post_speed_and_health(self):
        self.assertIn("speed", self.server.LIMITS)
        status, body = self.request("POST", "/api/speed", {"speed": 3})
        self.assertEqual((status, body), (200, {"ok": True, "speed": 3}))
        status, body = self.request("POST", "/api/speed", {"speed": 4})
        self.assertEqual(status, 400)
        self.assertIn("x5", body["error"])
        status, body = self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["speed"], 3)
        self.assertAlmostEqual(body["speed_actual"], 2.91)


class RouteMarginTest(unittest.TestCase):
    def test_flats_keep_the_apps_route_margin(self):
        import flats as F
        import robot
        self.assertEqual(F.ROUTE_MARGIN, robot.ROUTE_MARGIN)


if __name__ == "__main__":
    unittest.main()

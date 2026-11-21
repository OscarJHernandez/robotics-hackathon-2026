"""Tests for Knock (app/knock.py): the check-in that calls and finds Margarete, and its HTTP routes.

No sim and no network: a fake robot walks instantly, a fake clock moves only when the check-in sleeps, and the
check-in runs on the test's own thread.
"""

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import call  # noqa: E402
import knock  # noqa: E402


class FakeRobot:
    def __init__(self, x=0.9, y=-0.2):
        self.x, self.y, self.yaw, self.mode = x, y, 0.0, "idle"
        self.commands = []
        self.on_goto = None

    def command(self, text):
        self.commands.append(text)
        word, *args = text.split()
        if word == "goto":
            self.x, self.y = float(args[0]), float(args[1])
            if self.on_goto:
                self.on_goto()
        elif word == "turn":
            self.yaw = knock.wrap(self.yaw + float(args[0]))
        return text

    def state(self):
        return {"mode": self.mode, "x": self.x, "y": self.y, "yaw_deg": self.yaw}

    def frame(self, view):
        return b"JPEG"

    def visits(self):
        return [c for c in self.commands if c.startswith("goto")]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def make(where="couch", speak=lambda text: b"MP3", robot=None):
    hub, clock = call.Hub(), Clock()
    robot = robot or FakeRobot()
    k = knock.Knock(hub, robot, speak, clock=clock, sleep=clock.sleep, spawn=lambda fn: fn(),
                    stamp=lambda: "10:35", where=lambda: where)
    return k, hub, robot


def cards(hub):
    return [e for e in hub.since(0, timeout=0)[0] if e["type"] == "knock"]


class HelpersTest(unittest.TestCase):
    def test_after_miss_shares_out(self):
        probs = knock.after_miss(knock.PRIORS, "kitchen")
        self.assertEqual(probs["kitchen"], 0.0)
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertAlmostEqual(probs["couch"], 0.42 / 0.82)  # 51%, as in the mockup

    def test_next_zone_weighs_distance(self):
        # From the dock by the entrance the kitchen is near, so it goes first though the couch is likelier.
        self.assertEqual(knock.next_zone(knock.PRIORS, [], 0.9, -0.2), "kitchen")
        self.assertEqual(knock.next_zone(knock.PRIORS, [], -1.0, -0.3), "couch")
        self.assertIsNone(knock.next_zone(knock.PRIORS, list(knock.ZONES), 0, 0))

    def test_wrap(self):
        self.assertEqual(knock.wrap(270), -90)
        self.assertEqual(knock.wrap(-190), 170)


class KnockTest(unittest.TestCase):
    def test_find_walks_until_found(self):
        k, hub, robot = make(where="couch")
        self.assertEqual(k.find(), "started")
        self.assertEqual(robot.visits(), ["goto 1.00 0.25", "goto -1.00 -0.25"])  # kitchen, then couch
        last = cards(hub)[-1]
        self.assertEqual((last["state"], last["zone"], last["checked"]), ("found", "couch", ["kitchen"]))
        self.assertEqual([s["text"] for s in last["steps"]], ["Kitchen: not there", "Found her in front of the couch"])
        self.assertAlmostEqual(robot.yaw, -90.0)  # faces the couch
        finding = [c for c in cards(hub) if c["state"] == "finding" and c["next"] == "couch"][0]
        self.assertTrue(finding["text"].startswith("Not in the kitchen. Walking to the couch (51%)"))
        self.assertIsNone(k.run)
        self.assertEqual(k.latest()["state"], "found")

    def test_not_home(self):
        k, hub, robot = make(where=None)
        k.find()
        self.assertEqual(len(robot.visits()), 6)
        last = cards(hub)[-1]
        self.assertEqual(last["state"], "not_home")
        self.assertEqual(len(last["checked"]), 6)

    def test_call_then_search(self):
        k, hub, robot = make(where="kitchen")
        self.assertEqual(k.call(), "started")
        states = [c["state"] for c in cards(hub)]
        self.assertEqual(states[:3], ["calling", "calling", "no_answer"])
        with_audio = [c for c in cards(hub) if "audio" in c]
        self.assertEqual(len(with_audio), 1)
        self.assertEqual(with_audio[0]["listen_s"], knock.LISTEN)
        sound = with_audio[0]["audio"].rsplit("/", 1)[1][:-4]
        self.assertEqual(k.audio(sound), b"MP3")
        self.assertEqual(cards(hub)[-1]["state"], "found")
        self.assertEqual(cards(hub)[-1]["steps"][0]["text"], "Called her name, no answer")

    def test_voice_off_still_searches(self):
        def broken(text):
            raise TimeoutError("voice service is busy")
        k, hub, robot = make(speak=broken)
        k.call()
        self.assertFalse(any("audio" in c for c in cards(hub)))
        self.assertEqual(cards(hub)[-1]["state"], "found")

    def test_one_at_a_time_and_stop(self):
        robot = FakeRobot()
        k, hub, robot = make(robot=robot)
        answers = []

        def during_walk():
            robot.on_goto = None
            answers.append(k.find())  # a search runs: refused
            answers.append(k.stop())
        robot.on_goto = during_walk
        k.find()
        self.assertEqual(answers, [None, True])
        self.assertEqual(robot.commands[-1], "stop")
        self.assertEqual(len(robot.visits()), 1)  # no second zone after stop
        self.assertEqual(cards(hub)[-1]["state"], "idle")
        self.assertIsNone(k.run)

    def test_call_again_during_search(self):
        robot = FakeRobot()
        k, hub, robot = make(robot=robot)
        answers = []

        def during_walk():
            robot.on_goto = None
            answers.append(k.call())
        robot.on_goto = during_walk
        k.find()
        self.assertEqual(answers, ["again"])
        self.assertIn("Called her name again", [s["text"] for s in cards(hub)[-1]["steps"]])

    def test_choices(self):
        k, hub, robot = make()
        self.assertFalse(k.choose("call_112"))  # nothing found yet
        k.find()
        self.assertTrue(k.choose("call_112"))
        self.assertEqual(cards(hub)[-1]["choice"], "call_112")
        self.assertTrue(k.choose("talk"))  # ends the check-in, the talk bar is back
        self.assertEqual(cards(hub)[-1]["state"], "idle")
        with self.assertRaises(ValueError):
            k.choose("launch")

    def test_fallen_robot_ends_the_search(self):
        robot = FakeRobot()
        robot.on_goto = lambda: setattr(robot, "mode", "fallen")
        k, hub, robot = make(robot=robot)
        k.find()
        last = cards(hub)[-1]
        self.assertEqual(last["state"], "not_home")
        self.assertIn("fell", last["text"])


class RoutesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import server
        cls.server = server
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.httpd.daemon_threads = True
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = "http://127.0.0.1:{}".format(cls.httpd.server_address[1])

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def setUp(self):
        self.k, self.hub, self.robot = make()
        server = self.server
        for p in [mock.patch.object(server, "HUB", self.hub), mock.patch.object(server, "KNOCK", self.k),
                  mock.patch.object(server, "HITS", server.defaultdict(server.deque))]:
            p.start()
            self.addCleanup(p.stop)

    def request(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, response.headers.get("Content-Type"), response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers.get("Content-Type"), error.read()

    def test_call_find_choice_stop(self):
        status, _, body = self.request("/api/knock/call", {})
        self.assertEqual((status, json.loads(body)), (200, {"ok": "started"}))
        audio = [c["audio"] for c in cards(self.hub) if "audio" in c][0]
        self.assertEqual(self.request(audio)[1:], ("audio/mpeg", b"MP3"))
        self.assertEqual(self.request("/api/knock/audio/nope.mp3")[0], 404)
        self.assertEqual(self.request("/api/knock/choice", {"choice": "call_neighbour"})[0], 200)
        self.assertEqual(self.request("/api/knock/choice", {"choice": "x"})[0], 400)
        self.assertEqual(json.loads(self.request("/api/knock/stop", {})[2]), {"ok": True})
        self.assertEqual(json.loads(self.request("/api/knock/find", {})[2]), {"ok": "started"})

    def test_busy_is_409(self):
        self.k.run = {"stop": False, "skip": False}
        self.k.view = dict(self.k.view, state="found")
        self.assertEqual(self.request("/api/knock/find", {})[0], 409)
        self.assertEqual(self.request("/api/knock/call", {})[0], 409)

    def test_not_running(self):
        with mock.patch.object(self.server, "KNOCK", None):
            self.assertEqual(self.request("/api/knock/find", {})[0], 503)

    def test_live_starts_with_the_card(self):
        self.k.find()
        response = urllib.request.urlopen(self.base + "/api/live", timeout=10)
        try:
            lines = [response.readline() for _ in range(4)]
        finally:
            response.close()
        first = json.loads([line for line in lines if line.startswith(b"data:")][0][5:])
        self.assertEqual((first["type"], first["state"]), ("knock", "found"))


if __name__ == "__main__":
    unittest.main()

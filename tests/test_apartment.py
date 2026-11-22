"""Tests for the apartment (app/apartment.py): the flat, the journal, the belief and the search loop.

No simulator and no waiting: without a robot a marker walks, and the sleep is a no-op. One test uses a fake robot
to check what the search tells a real one (app/robot.py) to do.
"""

import datetime
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import apartment  # noqa: E402

NOW = datetime.datetime(2026, 11, 17, 10, 35)  # a Tuesday morning


def make(**kw):
    return apartment.Apartment(seed=0, now=lambda: NOW, sleep=lambda s: None, walk=0.0, look=0.0, **kw)


def finish(a):
    a._thread.join(timeout=20)
    assert not a._thread.is_alive()


class FakeRobot:
    """Walks instantly: after a goto or turn it is idle at the new pose."""

    def __init__(self):
        self.x, self.y, self.yaw, self.commands, self.loaded, self.people, self.painted = 0.0, 0.0, 0.0, [], [], [], []
        self._mode = "idle"

    def command(self, text):
        self.commands.append(text)
        word, *args = text.split()
        if word == "goto":
            self.x, self.y, self._mode = float(args[0]), float(args[1]), "goto"
        elif word == "turn":
            self.yaw, self._mode = self.yaw + float(args[0]), "turn"
        return text

    def state(self):
        mode, self._mode = self._mode, "idle"  # seen walking once, then done
        return {"mode": mode, "x": self.x, "y": self.y, "yaw_deg": self.yaw, "driver": "fake"}

    def load(self, flat):
        self.loaded.append(flat["flat_id"])
        self.x, self.y = flat["dock"]["x"], flat["dock"]["y"]

    def person(self, x=None, y=None, yaw_deg=0.0):
        self.people.append(None if x is None else (x, y))

    def paint(self, belief):
        self.painted.append(dict(belief))


class ApartmentTest(unittest.TestCase):
    def test_state_has_the_flat_the_journal_and_a_belief(self):
        s = make().state()
        self.assertEqual([z["name"] for z in s["flat"]["zones"]], list(apartment.F.ZONE_NAMES))
        self.assertEqual(len(s["flat"]["obstacles"]), 8)
        self.assertEqual(s["journal_total"], apartment.JOURNAL_DAYS * 3)
        self.assertEqual((s["now"]["weekday"], s["now"]["time"], s["now"]["slot"]), (1, "10:35", 0))
        self.assertEqual(list(s["stages"]), ["population", "habits", "journal", "last_heard", "counting"])
        for belief in list(s["stages"].values()) + [s["belief"]]:
            self.assertAlmostEqual(sum(belief.values()), 1.0)
        self.assertEqual(sorted(s["plan"]), sorted(apartment.F.ZONE_NAMES))
        self.assertEqual(s["search"]["status"], "idle")
        self.assertEqual(s["driver"], "marker")
        self.assertEqual(s["journal"][0]["day"], "2026-11-16")  # newest first: yesterday evening

    def test_a_journal_entry_changes_the_belief_and_can_be_removed(self):
        a = make()
        before = a.state()["belief"]["kitchen"]
        entry = a.add_entry("kitchen", "10:20")  # heard in the kitchen 15 minutes ago
        s = a.state()
        self.assertEqual((s["last_heard"]["zone"], s["last_heard"]["minutes"]), ("kitchen", 15))
        self.assertGreater(s["belief"]["kitchen"], max(before, 0.6))
        self.assertEqual(s["journal"][0]["source"], "entered")
        self.assertTrue(a.remove_entry(entry["id"]))
        self.assertAlmostEqual(a.state()["belief"]["kitchen"], before)
        for bad in (("garage", None, 0), ("kitchen", "25:00", 0), ("kitchen", None, 99)):
            with self.assertRaises(ValueError):
                a.add_entry(*bad)

    def test_a_pasted_journal_replaces_the_prior(self):
        a = make()
        a.set_clock(1, "14:30")  # Tuesday afternoon
        text = "day, time, zone\n# two weeks of afternoons in the dining area\n" + "\n".join(
            "{} 14:00 Dining Area".format(d) for d in ("Mon", "Tue", "Wed", "Thu", "Fri") * 2) + "\nSat 10:00 out"
        result = a.import_journal(text)
        self.assertEqual((result["added"], result["problems"]), (11, []))
        s = a.state()
        self.assertEqual(s["journal_total"], 11)
        self.assertEqual({e["source"] for e in s["journal"]}, {"pasted"})
        self.assertEqual(max((z for z in s["belief"] if z != "out"), key=s["belief"].get), "dining_area")
        self.assertEqual(s["plan"][0], "dining_area")  # the search would start there
        days = sorted({e["day"] for e in s["journal"]})
        # counted back from the check-in (Tue 17 Nov), last line first: Sat 14, then Fri 13 ... Mon 2
        self.assertEqual((days[0], days[-1]), ("2026-11-02", "2026-11-14"))
        self.assertTrue(all((e["day"], e["minute"]) <= ("2026-11-17", 870) for e in s["journal"]))
        # what is copied out can be pasted back in
        again = make()
        again.set_clock(1, "14:30")
        again.import_journal(a.journal_text())
        self.assertEqual([(e["day"], e["time"], e["zone"]) for e in again.journal],
                         [(e["day"], e["time"], e["zone"]) for e in a.journal])

    def test_a_pasted_journal_can_be_added_and_bad_lines_are_reported(self):
        a = make()
        before = a.state()["journal_total"]
        result = a.import_journal("yesterday 21:10 bedroom\n2026-11-16;08:05;kitchen\nMon 9:99 kitchen\nMon 10:00 garage\n"
                                  "just words", replace=False)
        self.assertEqual(result["added"], 2)
        self.assertEqual(len(result["problems"]), 3)
        self.assertEqual(a.state()["journal_total"], before + 2)
        self.assertEqual(a.import_journal('[{"day": "Mon", "time": "07:30", "zone": "kitchen"}]', replace=False)["added"], 1)
        with self.assertRaises(ValueError):
            a.import_journal("nothing useful here")

    def test_custom_inputs_as_json_replace_routines_and_journal(self):
        import json
        a = make()
        data = a.inputs()
        self.assertEqual(set(data), {"routines", "journal"})
        self.assertEqual(len(data["journal"]), apartment.JOURNAL_DAYS * 3)
        self.assertEqual(set(data["routines"]["weekdays"]), {"morning", "afternoon", "evening"})
        custom = {"routines": {"weekdays": {"morning": "Hallway", "afternoon": "kitchen", "evening": "bedroom"},
                               "weekends": ["bedroom", "living room", "living room"],
                               "outings": [{"day": "Tue", "part": "afternoon"}, ["Sat", "morning"]]},
                  "journal": [{"day": "Mon", "time": "08:10", "zone": "hallway"},
                              {"day": "Mon", "time": "20:00", "zone": "bedroom"}]}
        result = a.set_inputs(json.dumps(custom))
        self.assertEqual((result["routines"], result["added"]), (True, 2))
        s = a.state()
        self.assertEqual(s["habits"]["usual"]["0"], ["hallway", "kitchen", "bedroom"])
        self.assertEqual(s["habits"]["outings"], [[1, 1], [5, 0]])
        self.assertEqual(s["journal_total"], 2)
        self.assertGreater(s["stages"]["habits"]["hallway"], s["stages"]["population"]["hallway"])  # Tue morning
        self.assertEqual(a.inputs()["routines"]["outings"], [{"day": "Tue", "part": "afternoon"}, {"day": "Sat", "part": "morning"}])
        a.set_inputs({"routines": a.inputs()["routines"]})  # routines alone: the journal stays
        self.assertEqual(a.state()["journal_total"], 2)
        for bad in ("not json", "{}", {"routines": {"weekdays": ["kitchen"]}},
                    {"routines": {"weekdays": ["out", "kitchen", "kitchen"], "weekends": ["kitchen"] * 3}},
                    {"routines": dict(custom["routines"], outings=[{"day": "Someday", "part": "noon"}])}):
            with self.assertRaises(ValueError):
                a.set_inputs(bad)
        self.assertEqual(a.state()["habits"]["usual"]["0"], ["hallway", "kitchen", "bedroom"])  # bad input changed nothing

    def test_the_check_in_time_can_be_set_and_entries_after_it_do_not_count(self):
        a = make()
        a.add_entry("kitchen", "10:20")
        a.set_clock(1, "09:00")  # the same Tuesday, before that entry
        s = a.state()
        self.assertEqual((s["now"]["time"], s["now"]["real"]), ("09:00", False))
        self.assertNotEqual(s["journal"][0]["time"], "10:20")
        a.set_clock(5, "20:00")  # the Saturday before
        self.assertEqual(a.state()["now"]["day"], "2026-11-14")
        a.set_clock()
        self.assertTrue(a.state()["now"]["real"])

    def test_the_search_finds_her_logs_every_step_and_writes_the_journal(self):
        a = make()
        plan = a.state()["plan"]  # before the search: finding her adds a journal entry, which changes it
        self.assertTrue(a.start("kitchen", miss_rate=0.0))
        finish(a)
        s = a.state()
        self.assertEqual(s["search"]["status"], "found")
        self.assertEqual(s["search"]["steps"][-1], dict(s["search"]["steps"][-1], zone="kitchen", seen=True))
        self.assertEqual([step["zone"] for step in s["search"]["steps"]],
                         plan[:plan.index("kitchen") + 1])  # most likely zone first
        kinds = [line["kind"] for line in s["log"]]
        self.assertEqual(kinds.count("proposal"), len(s["search"]["steps"]))
        self.assertEqual(kinds.count("miss"), len(s["search"]["steps"]) - 1)
        self.assertIn("found", kinds)
        self.assertEqual((s["journal"][0]["zone"], s["journal"][0]["source"]), ("kitchen", "search"))
        self.assertAlmostEqual(s["belief"]["kitchen"], 1.0)

    def test_random_puts_her_in_any_zone_at_one_of_its_spots(self):
        zones, spots = set(), set()
        for _ in range(120):  # enough runs that "every zone and both living-room spots turn up" cannot fail by chance
            a = make()
            a.start("random", miss_rate=0.0)
            finish(a)
            s = a.state()["search"]
            self.assertEqual(s["status"], "found")
            zones.add(s["truth"])
            spots.add(tuple(s["person"]))
            zone = apartment.F.zone(a.flat, s["truth"])
            self.assertIn(tuple(s["person"]), {(sp["person_x"], sp["person_y"]) for sp in zone["spots"]})
        self.assertGreaterEqual(len(zones), 4)  # not always the same place
        self.assertGreater(len(spots), len(zones))  # the living room has two spots
        with self.assertRaises(ValueError):
            make().start("garage")

    def test_nobody_home_ends_the_search(self):
        a = make()
        a.start("out")
        finish(a)
        s = a.state()
        self.assertIn(s["search"]["status"], ("nobody_home", "gave_up"))
        self.assertFalse(any(step["seen"] for step in s["search"]["steps"]))

    def test_with_a_robot_the_search_walks_it_to_each_zone_and_turns_it_to_look(self):
        robot = FakeRobot()
        a = make(robot=robot)
        self.assertEqual(robot.loaded, [0])  # the flat is in the simulator
        a.start("kitchen", miss_rate=0.0)
        finish(a)
        s = a.state()
        self.assertEqual(s["search"]["status"], "found")
        kitchen = apartment.F.zone(a.flat, "kitchen")
        self.assertEqual(robot.people[-1], (kitchen["person_x"], kitchen["person_y"]))  # she lies where she is
        gotos = [c for c in robot.commands if c.startswith("goto")]
        self.assertEqual(len(gotos), len(s["search"]["steps"]))
        self.assertEqual(gotos[-1], "goto {:.2f} {:.2f}".format(kitchen["view_x"], kitchen["view_y"]))
        self.assertAlmostEqual(robot.painted[-1]["kitchen"], 1.0)  # the zones are shaded by the belief
        self.assertEqual(s["driver"], "fake")

    def test_with_a_model_its_belief_is_the_one_the_search_starts_from(self):
        class Model:
            version, rows, calls = "v3.5", 2000, []

            def belief(self, hour, weekday, habits, journal, last):
                self.calls.append((hour, weekday, len(journal), last))
                return {"bedroom": 0.05, "living_room": 0.05, "dining_area": 0.05, "kitchen": 0.05, "hallway": 0.7,
                        "out": 0.1}

        model = Model()
        a = make(model=model)
        s = a.state()
        self.assertEqual(s["model"]["name"], "tabpfn")
        self.assertEqual(s["model"]["label"], "TabPFN 3.5")
        self.assertAlmostEqual(s["belief"]["hallway"], 0.7)
        self.assertEqual(s["plan"][0], "hallway")
        self.assertNotAlmostEqual(s["stages"]["counting"]["hallway"], 0.7)  # the counting belief is kept beside it
        a.state()
        self.assertEqual(len(model.calls), 1)  # predicted once for the same time and journal
        hour, weekday, records, last = model.calls[0]
        self.assertEqual((round(hour, 2), weekday, records), (10.58, 1, apartment.JOURNAL_DAYS * 3))
        a.add_entry("kitchen", "10:20")
        a.state()
        self.assertEqual(model.calls[-1][3], ("kitchen", 15))  # a new entry reaches the model as "last heard"
        a.start("hallway", miss_rate=0.0)
        finish(a)
        self.assertEqual([step["zone"] for step in a.state()["search"]["steps"]], ["hallway"])
        self.assertEqual(make().state()["model"]["name"], "counting")

    def test_clear_forgets_the_search_and_sends_the_robot_home(self):
        robot = FakeRobot()
        a = make(robot=robot)
        a.start("kitchen", miss_rate=0.0)
        finish(a)
        total = a.state()["journal_total"]
        a.clear()
        s = a.state()
        self.assertEqual((s["search"]["status"], s["search"]["steps"], s["search"]["truth"]), ("idle", [], None))
        self.assertEqual(robot.people[-1], None)  # Margarete is out of the room
        self.assertEqual(robot.commands[-1], "reset")  # the robot goes back to its dock
        self.assertEqual(s["journal_total"], total)  # the journal stays
        self.assertAlmostEqual(sum(s["belief"].values()), 1.0)
        self.assertLess(s["belief"]["kitchen"], 1.0)  # no longer "found in the kitchen"

    def test_only_one_search_at_a_time_and_stop_ends_it(self):
        import threading
        gate = threading.Event()
        a = apartment.Apartment(seed=0, now=lambda: NOW, sleep=lambda s: gate.wait(5), walk=1.0, look=0.0)
        self.assertTrue(a.start("kitchen"))
        self.assertFalse(a.start("kitchen"))
        self.assertTrue(a.state()["search"]["running"])
        gate.set()
        self.assertTrue(a.stop() or a.state()["search"]["status"] != "walking")
        finish(a)
        self.assertFalse(a.state()["search"]["running"])


if __name__ == "__main__":
    unittest.main()

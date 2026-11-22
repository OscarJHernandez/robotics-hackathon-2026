"""The apartment view: the flat's zones and obstacles, the journal, the belief, and the search with its log.

One apartment is shared by everyone who opens the page, like the robot. It holds:

- the flat: the MuJoCo arena with five named zones and the arena's furniture as obstacles (homes/flats.py);
- the resident: her stated habits and a journal of where she was heard; viewers can add journal entries;
- the belief: where she probably is now, from habits + journal + last heard (homes/search.py);
- the search: the loop "most likely zone -> walk there -> look -> update -> next zone" and a log of every step.

With a robot (app/robot.py, scene "apartment") the search drives the simulated JetHexa: the flat's furniture and
zones are loaded into MuJoCo, Margarete lies where she really is, each zone is shaded by the belief, and for every
proposal the robot walks to the zone's viewing pose and turns to face it. Without a robot (tests, demo) a marker
moves in a straight line instead. The look is still a mock: it misses her 20% of the time (search.MISS_RATE).

`Apartment.state()` is what GET /api/apartment returns; the page polls it.
"""

import datetime
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "homes"))

import flats as F  # noqa: E402
import residents as R  # noqa: E402
import search as S  # noqa: E402

JOURNAL_DAYS = 14  # days of journal the sample resident starts with
JOURNAL_LIMIT = 400  # entries kept
LOG_LIMIT = 200  # log lines kept
WALK = 0.45  # seconds the marker takes per metre (the mock walk)
POLL = 0.25  # seconds between looks at the robot while it walks
WALK_TIMEOUT = 40.0  # seconds a walk may take per metre of straight line, plus a minute
TURN_TIMEOUT = 40.0
LOOK = 0.9  # seconds a look takes
TICK = 0.1  # seconds between marker positions
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
NAMES = dict(F.LABELS, out="Out")
# Where she really is, for the simulation: drawn from her routine, a random zone (and spot in it), a zone, or out.
WHERE = ("routine", "random") + F.CLASSES


def population():
    """P(class | weekday type, part of the day) of the synthetic population, from homes/data/cold_prior.json
    (written by homes/prior.py). None if the file is missing or from other zones: every class equally likely."""
    try:
        data = json.loads((REPO / "homes" / "data" / "cold_prior.json").read_text())
        if tuple(data["classes"]) != F.CLASSES:
            return None
        counts = np.array(data["pooled"], dtype=float) + 0.5
        return counts / counts.sum(axis=2, keepdims=True)
    except (OSError, ValueError, KeyError):
        return None


def hhmm(minute):
    return "{:02d}:{:02d}".format(int(minute) // 60 % 24, int(minute) % 60)


def parse_time(text):
    """'09:30' -> 570 minutes; raises ValueError."""
    try:
        hours, minutes = str(text).strip().split(":")
        hours, minutes = int(hours), int(minutes)
    except ValueError:
        raise ValueError("time must look like 09:30")
    if not (0 <= hours < 24 and 0 <= minutes < 60):
        raise ValueError("time must look like 09:30")
    return hours * 60 + minutes


class Apartment:
    def __init__(self, seed=0, name="Margarete", now=datetime.datetime.now, sleep=time.sleep, walk=WALK, look=LOOK,
                 robot=None):
        self._now, self._sleep, self._walk, self._look = now, sleep, walk, look
        self._robot = robot  # LiveRobot in the apartment scene, or None: a marker walks instead
        self._lock = threading.RLock()
        self._population = population()
        self._thread = None
        self._stop = threading.Event()
        self._ids = 0
        self.name = name
        self.clock = None  # {"weekday", "minute"}: the check-in time when it is not the real time
        self.log = []
        self.reset(seed)

    # ---- the flat, the resident and her journal ------------------------------------------------------

    def reset(self, seed=0):
        """A new random flat and resident with a fresh journal. Stops a running search."""
        self.stop()
        with self._lock:
            self.seed = int(seed)
            self.flat = F.random_flat(np.random.default_rng([self.seed, 0]), self.seed)
            self.persona = R.sample_persona(np.random.default_rng([self.seed, 1]), 0, self.flat)
            self.habits = R.stated_habits(self.persona)
            self.journal = []
            rng = np.random.default_rng([self.seed, 2])
            today = self._now().date()
            for back in range(JOURNAL_DAYS, 0, -1):
                day = today - datetime.timedelta(days=back)
                codes = self._draw(day.weekday(), rng)
                for slot, (a, b) in enumerate(R.SLOT_WINDOWS):
                    self._add(day, int(rng.uniform(a, b)), F.CLASSES[codes[slot]], "robot")
            self.robot = dict(self.flat["dock"])
            self.search = None
            self.run = {"status": "idle", "goal": None, "truth": None, "miss_rate": S.MISS_RATE, "path": []}
            self.log = []
            if self._robot is not None:
                self._robot.load(self.flat)
                self._robot.person()
            self._paint()
            self._say("New flat {} for {}: {} days of journal".format(self.seed, self.name, JOURNAL_DAYS), "info")

    def _draw(self, weekday, rng):
        """Her zone in each part of a day of this weekday, from her routine."""
        table = self.persona["tables"]["1" if weekday >= 5 else "0"]
        codes = []
        for slot in range(len(table)):
            if rng.random() < self.persona["out_p"][weekday][slot]:
                codes.append(R.OUT)
            else:
                names = list(table[slot])
                probs = np.array([table[slot][n] for n in names])
                codes.append(R.CODE[names[int(rng.choice(len(names), p=probs / probs.sum()))]])
        return codes

    def _add(self, day, minute, zone, source):
        self._ids += 1
        self.journal.append({"id": self._ids, "day": day.isoformat(), "weekday": day.weekday(), "minute": int(minute),
                             "time": hhmm(minute), "slot": R.slot_of(int(minute)), "zone": zone, "source": source})
        self.journal.sort(key=lambda e: (e["day"], e["minute"], e["id"]))
        del self.journal[:-JOURNAL_LIMIT]
        return self.journal[-1] if self.journal[-1]["id"] == self._ids else next(
            e for e in self.journal if e["id"] == self._ids)

    def add_entry(self, zone, time_text=None, days_ago=0):
        """A journal entry from a viewer: she was in `zone` (or "out") at this time, today or up to 13 days ago."""
        if zone not in F.CLASSES:
            raise ValueError("zone must be one of: " + ", ".join(F.CLASSES))
        days_ago = int(days_ago)
        if not 0 <= days_ago < JOURNAL_DAYS:
            raise ValueError("days_ago must be 0 to {}".format(JOURNAL_DAYS - 1))
        with self._lock:
            day, _, minute = self.when()
            if time_text not in (None, ""):
                minute = parse_time(time_text)
            entry = self._add(day - datetime.timedelta(days=days_ago), minute, zone, "entered")
            self._paint()
            self._say("Journal: {} {} {} {} (entered)".format(DAYS[entry["weekday"]], entry["day"][5:], entry["time"],
                                                             NAMES[zone]), "journal")
            return entry

    def remove_entry(self, entry_id):
        with self._lock:
            before = len(self.journal)
            self.journal = [e for e in self.journal if e["id"] != int(entry_id)]
            if len(self.journal) < before:
                self._say("Journal: entry removed", "journal")
                self._paint()
            return len(self.journal) < before

    # ---- the check-in time ---------------------------------------------------------------------------

    def set_clock(self, weekday=None, time_text=None):
        """The check-in time: a weekday (0 = Monday) and a time, or nothing for the real time."""
        with self._lock:
            if weekday is None and time_text in (None, ""):
                self.clock = None
            else:
                real = self._now()
                wd = real.weekday() if weekday is None else int(weekday)
                if not 0 <= wd <= 6:
                    raise ValueError("weekday must be 0 (Monday) to 6 (Sunday)")
                minute = real.hour * 60 + real.minute if time_text in (None, "") else parse_time(time_text)
                self.clock = {"weekday": wd, "minute": minute}
            self._paint()
            return self.clock

    def when(self):
        """(date, weekday, minute of the day) of the check-in: the real time, or the last day with the chosen
        weekday at the chosen time."""
        real = self._now()
        if self.clock is None:
            return real.date(), real.weekday(), real.hour * 60 + real.minute
        back = (real.weekday() - self.clock["weekday"]) % 7
        return real.date() - datetime.timedelta(days=back), self.clock["weekday"], self.clock["minute"]

    # ---- the belief ----------------------------------------------------------------------------------

    def _known(self):
        """Journal entries up to the check-in time, oldest first."""
        day, _, minute = self.when()
        stamp = (day.isoformat(), minute)
        return [e for e in self.journal if (e["day"], e["minute"]) <= stamp]

    def _last(self):
        known = self._known()
        if not known:
            return None
        day, _, minute = self.when()
        last = known[-1]
        days = (day - datetime.date.fromisoformat(last["day"])).days
        return last, days * 1440 + minute - last["minute"]

    def stages(self):
        """The belief step by step (population, + habits, + journal, + last heard) at the check-in time."""
        _, weekday, minute = self.when()
        journal = [(e["weekday"], e["slot"], e["zone"]) for e in self._known() if e["slot"] is not None]
        last = self._last()
        return S.beliefs(self.flat, minute / 60.0, weekday, self._population, self.habits, journal,
                         (last[0]["zone"], last[1]) if last else None)

    # ---- the search ----------------------------------------------------------------------------------

    def start(self, where="routine", miss_rate=None):
        """Start the search loop. False while one is running."""
        if where not in WHERE:
            raise ValueError("where must be one of: " + ", ".join(WHERE))
        miss = S.MISS_RATE if miss_rate is None else float(miss_rate)
        if not 0.0 <= miss <= 0.9:
            raise ValueError("miss_rate must be 0 to 0.9")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            _, weekday, minute = self.when()
            start = self.stages()["last_heard"]
            truth = self._truth(where, weekday, minute)
            self.search = S.Search(self.flat, start, miss)
            person = None  # where she lies: the zone's main spot, or any of its spots for "random"
            if truth != "out":
                spots = F.zone(self.flat, truth)["spots"]
                spot = spots[int(np.random.default_rng().integers(len(spots)))] if where == "random" else spots[0]
                person = (spot["person_x"], spot["person_y"], spot["heading_deg"] + 90.0)
            if self._robot is None:
                self.robot = dict(self.flat["dock"])
            else:  # Margarete lies where she really is; the robot starts from where it stands
                self._robot.person(*person) if person else self._robot.person()
                self.robot = self._pose()
            self.run = {"status": "starting", "goal": None, "truth": truth, "miss_rate": miss, "walked": 0.0,
                        "person": list(person[:2]) if person else None,
                        "path": [[self.robot["x"], self.robot["y"]]]}
            self._paint(start)
            self._stop.clear()
            self._say("Check-in {} {}: where is {}?".format(DAYS[weekday], hhmm(minute), self.name), "start")
            top = sorted(((p, z) for z, p in start.items()), reverse=True)[:3]
            self._say("Belief: " + ", ".join("{} {:.0f}%".format(NAMES[z], 100 * p) for p, z in top), "belief")
            self._thread = threading.Thread(target=self._loop, args=(truth, miss), daemon=True)
            self._thread.start()
            return True

    def _truth(self, where, weekday, minute):
        """Where she really is, for the mock look."""
        if where == "random":  # anywhere in the flat, every zone equally likely
            return str(np.random.default_rng().choice(F.ZONE_NAMES))
        if where != "routine":
            return where
        slot = R.slot_of(minute)
        last = self._last()
        if slot is None:
            return "bedroom"
        if last and last[1] < 16 * 60 and R.slot_of(int(minute - last[1])) == slot:
            return last[0]["zone"]  # heard in this part of the day: she has not moved
        probs = R.zone_probs(self.persona, weekday, slot)
        names = list(probs)
        rng = np.random.default_rng()
        return names[int(rng.choice(len(names), p=np.array([probs[n] for n in names]) / sum(probs.values())))]

    def stop(self):
        """Stop a running search. True if one was running."""
        thread = self._thread
        if thread is None or not thread.is_alive():
            return False
        self._stop.set()
        if thread is not threading.current_thread():
            thread.join(timeout=5)
        return True

    def _loop(self, truth, miss):
        rng = np.random.default_rng()
        search = self.search
        try:
            while True:
                with self._lock:
                    goal = search.goal()
                    if goal is None:
                        break
                    self.run.update(status="walking", goal=goal["zone"])
                    self._say("Proposal {}: {} ({:.0f}%), the most likely zone".format(
                        len(search.steps) + 1, NAMES[goal["zone"]], 100 * goal["p"]), "proposal")
                    start = (self.robot["x"], self.robot["y"])
                self._move(start, (goal["x"], goal["y"]), goal["heading_deg"])
                with self._lock:
                    self.run["status"] = "looking"
                self._wait(self._look)
                seen = bool(S.observe(goal["zone"], truth, rng, miss))
                with self._lock:
                    before = dict(search.belief)
                    search.report(seen)
                    step = search.steps[-1]
                    self._paint(step["belief_after"])
                    if seen:
                        self._say("Observation: sees {} in the {}".format(self.name, NAMES[goal["zone"]].lower()), "seen")
                    else:
                        self._say("Observation: nobody seen in the {}".format(NAMES[goal["zone"]].lower()), "miss")
                        rise = max((z for z in before if z != goal["zone"]),
                                   key=lambda z: step["belief_after"][z] - before[z])
                        self._say("Update: {} {:.0f}% -> {:.0f}%, {} {:.0f}% -> {:.0f}%".format(
                            NAMES[goal["zone"]], 100 * before[goal["zone"]], 100 * step["belief_after"][goal["zone"]],
                            NAMES[rise], 100 * before[rise], 100 * step["belief_after"][rise]), "update")
            with self._lock:
                self.run.update(status=search.status, goal=None)
                looks = len(search.steps)
                tail = "after {} look{}, {:.1f} m".format(looks, "" if looks == 1 else "s", self.run["walked"])
                if search.status == "found":
                    zone = search.steps[-1]["zone"]
                    self._say("Found {} in the {} {}".format(self.name, NAMES[zone].lower(), tail), "found")
                    day, _, minute = self.when()
                    self._add(day, minute, zone, "search")
                    self._say("Journal: {} {} (seen by the robot)".format(hhmm(minute), NAMES[zone]), "journal")
                elif search.status == "nobody_home":
                    self._say("Nobody home ({:.0f}% sure) {}".format(100 * search.belief.get("out", 0.0), tail), "out")
                else:
                    self._say("Gave up {}".format(tail), "out")
        except _Stopped:
            if self._robot is not None:
                self._command("stop")
            with self._lock:
                self.run.update(status="stopped", goal=None)
                self._say("Search stopped", "info")
        except _Failed as error:
            with self._lock:
                self.run.update(status="failed", goal=None)
                self._say("Search ended: {}".format(error), "out")

    def _move(self, start, end, heading_deg=None):
        """Get the robot to `end` and face `heading_deg`. With a LiveRobot it really walks there; without one a
        marker goes in a straight line."""
        if self._robot is not None:
            self._drive(end, heading_deg)
            return
        seconds = self._walk * math.hypot(end[0] - start[0], end[1] - start[1])
        steps = max(1, int(seconds / TICK))
        for i in range(1, steps + 1):
            self._wait(seconds / steps)
            with self._lock:
                t = i / steps
                self.robot = {"x": start[0] + t * (end[0] - start[0]), "y": start[1] + t * (end[1] - start[1])}
        with self._lock:
            self.run["path"].append([end[0], end[1]])
            self.run["walked"] += math.hypot(end[0] - start[0], end[1] - start[1])

    def _pose(self):
        state = self._robot.state()
        return {"x": state.get("x", 0.0), "y": state.get("y", 0.0), "yaw_deg": state.get("yaw_deg", 0.0)}

    def _command(self, text):
        try:
            self._robot.command(text)
        except ValueError as error:
            raise _Failed(str(error))

    def _drive(self, end, heading_deg):
        """The real walk: goto, wait until the robot stands near `end`, then turn to face the zone."""
        self._command("goto {:.2f} {:.2f}".format(*end))
        self._settle(WALK_TIMEOUT * math.hypot(end[0] - self.robot["x"], end[1] - self.robot["y"]) + 60.0, near=end)
        if heading_deg is not None:
            turn = (heading_deg - self._pose()["yaw_deg"] + 180.0) % 360.0 - 180.0
            if abs(turn) > 8.0:
                self._command("turn {:.1f}".format(turn))
                self._settle(TURN_TIMEOUT)

    def _settle(self, timeout, near=None):
        """Wait until the robot has finished its command; follow it on the plan meanwhile."""
        waited, started = 0.0, False
        while True:
            self._wait(POLL)
            waited += POLL
            state = self._robot.state()
            started = started or state.get("mode") in ("goto", "turn")
            with self._lock:
                last = self.robot
                self.robot = self._pose()
                moved = math.hypot(self.robot["x"] - last["x"], self.robot["y"] - last["y"])
                self.run["walked"] += moved
                tail = self.run["path"][-1]
                if math.hypot(self.robot["x"] - tail[0], self.robot["y"] - tail[1]) > 0.15:
                    self.run["path"].append([self.robot["x"], self.robot["y"]])
            if state.get("mode") == "fallen":
                raise _Failed("the robot fell over")
            # "idle" means done only once the robot was seen walking (the simulator may take a moment to pick the
            # command up), or when it stands at the place already.
            if state.get("mode") == "idle" and waited >= 2 * POLL:
                there = near is None or math.hypot(state["x"] - near[0], state["y"] - near[1]) < 0.45
                if there and (started or near is not None or waited > 3.0):
                    return
                if started or waited > 5.0:
                    raise _Failed("the robot stopped before it got there (another command?)")
            if waited > timeout:
                self._command("stop")
                raise _Failed("the robot did not get there in time (something is in the way)")

    def _paint(self, belief=None):
        """Shade the zones on the simulator's floor by the belief."""
        if self._robot is not None:
            self._robot.paint(belief if belief is not None else self.stages()["last_heard"])

    def _wait(self, seconds):
        if self._stop.is_set():
            raise _Stopped()
        if seconds > 0:
            self._sleep(seconds)
        if self._stop.is_set():
            raise _Stopped()

    def _say(self, text, kind):
        with self._lock:
            self.log.append({"id": len(self.log) + 1 if not self.log else self.log[-1]["id"] + 1,
                             "time": self._now().strftime("%H:%M:%S"), "text": text, "kind": kind})
            del self.log[:-LOG_LIMIT]

    # ---- what the page gets ---------------------------------------------------------------------------

    def state(self):
        with self._lock:
            day, weekday, minute = self.when()
            stages = self.stages()
            search = self.search
            running = self.run["status"] in ("starting", "walking", "looking")
            belief = dict(search.belief) if search is not None and self.run["status"] != "idle" else stages["last_heard"]
            if sum(belief.values()) <= 0:
                belief = stages["last_heard"]
            last = self._last()
            flat = self.flat
            return {
                "seed": self.seed, "name": self.name,
                "flat": {"half": flat["width"] / 2, "door": flat["door"], "dock": flat["dock"],
                         "zones": [{k: z[k] for k in ("name", "label", "x", "y", "size_x", "size_y", "view_x", "view_y",
                                                    "person_x", "person_y")}
                                   for z in flat["zones"]],
                         "obstacles": [{k: f[k] for k in ("name", "zone", "x", "y", "size_x", "size_y")}
                                       for f in flat["furniture"]]},
                "habits": {"usual": self.habits["usual"], "outings": [list(o) for o in self.habits["outings"]]},
                "now": {"weekday": weekday, "minute": minute, "time": hhmm(minute), "day": day.isoformat(),
                        "slot": R.slot_of(minute), "real": self.clock is None},
                "journal": list(reversed(self._known()))[:120],
                "journal_total": len(self.journal),
                "last_heard": ({"zone": last[0]["zone"], "minutes": last[1]} if last else None),
                "stages": stages, "belief": belief, "plan": S.plan(flat, stages["last_heard"])[0],
                "search": {"status": self.run["status"], "running": running, "goal": self.run["goal"],
                           "truth": self.run["truth"], "person": self.run.get("person"),
                           "miss_rate": self.run["miss_rate"], "path": self.run["path"],
                           "metres": self.run.get("walked", 0.0),
                           "steps": [{"zone": s["zone"], "p": s["p_before"], "seen": s["seen"], "metres": s["metres"]}
                                     for s in (search.steps if search else [])]},
                "robot": dict(self._pose() if self._robot is not None else self.robot),
                "driver": (self._robot.state().get("driver") if self._robot is not None else "marker"),
                "log": list(self.log),
            }


class _Stopped(Exception):
    pass


class _Failed(Exception):
    pass

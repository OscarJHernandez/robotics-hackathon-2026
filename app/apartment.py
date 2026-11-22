"""The apartment view: the flat's zones and obstacles, the journal, the belief, and the search with its log.

One apartment is shared by everyone who opens the page, like the robot. It holds:

- the flat: the MuJoCo arena with five named zones and the arena's furniture as obstacles (homes/flats.py);
- the resident: her stated habits and a journal of where she was heard; viewers can add journal entries;
- the belief: where she probably is now, from habits + journal + last heard. By default TabPFN gives it
  (homes/tabpfn_belief.py: fit() on synthetic check-ins, predict_proba for now); without TabPFN, or at night, the
  counting belief of homes/search.py does. The steps "population, + habits, + journal" shown beside it are always
  the counting ones;
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
import re
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


# How a pasted journal may name a zone (lower case, spaces as underscores).
ZONE_WORDS = {"bedroom": "bedroom", "bed": "bedroom", "living_room": "living_room", "livingroom": "living_room",
              "living": "living_room", "lounge": "living_room", "couch": "living_room", "sofa": "living_room",
              "dining_area": "dining_area", "dining_room": "dining_area", "dining": "dining_area",
              "table": "dining_area", "kitchen": "kitchen", "hallway": "hallway", "hall": "hallway",
              "entrance": "hallway", "door": "hallway", "out": "out", "away": "out", "not_home": "out"}
WEEKDAYS = {name: i for i, name in enumerate(("mon", "tue", "wed", "thu", "fri", "sat", "sun"))}
JOURNAL_EXAMPLE = "Mon 08:10 kitchen\nMon 14:30 living room\nMon 20:05 bedroom\nTue 09:00 out"


def parse_journal(text, day, minute):
    """A pasted journal -> ([(date, minute of the day, zone)], [problems]).

    One record per line, in the order they happened: a day, a time and a zone, separated by spaces, commas, tabs
    or semicolons. The day is a weekday (Mon, Tuesday), a date (2026-11-20), `today` or `yesterday`; the zone is
    bedroom, living room, dining area, kitchen, hallway or out. Lines starting with # and a header line are
    skipped. A JSON list of {"day", "time", "zone"} works too.

    Weekdays are counted back from the check-in time (`day`, `minute`), last line first, so "Mon ... Sun, Mon ...
    Sun" is two weeks ending at the check-in."""
    rows, problems = [], []
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            items = json.loads(stripped)
            lines = ["{} {} {}".format(i.get("day", i.get("date", i.get("weekday", ""))), i.get("time", ""),
                                       i.get("zone", "")) for i in items]
        except (ValueError, AttributeError, TypeError):
            return [], ["the text starts with [ but is not a JSON list of records"]
    else:
        lines = stripped.splitlines()
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words = [w for w in re.split(r"[,;|\t ]+", line.lower()) if w]
        time_word = next((w for w in words if re.fullmatch(r"\d{1,2}[:.]\d{2}", w)), None)
        if time_word is None:
            if "zone" not in words:  # a header line is skipped without a word
                problems.append("line {}: no time like 09:30".format(number))
            continue
        words.remove(time_word)
        when = None
        for w in list(words):
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", w):
                try:
                    when = datetime.date.fromisoformat(w)
                except ValueError:
                    pass
            elif w[:3] in WEEKDAYS and (w in WEEKDAYS or w.endswith("day")):
                when = WEEKDAYS[w[:3]]
            elif w in ("today", "yesterday"):
                when = day - datetime.timedelta(days=1 if w == "yesterday" else 0)
            else:
                continue
            words.remove(w)
            break
        zone = ZONE_WORDS.get("_".join(words).replace("-", "_"))
        try:
            at = parse_time(time_word.replace(".", ":"))
        except ValueError:
            problems.append("line {}: '{}' is not a time".format(number, time_word))
            continue
        if when is None:
            problems.append("line {}: no day (a weekday, a date, today or yesterday)".format(number))
        elif zone is None:
            problems.append("line {}: unknown zone '{}'".format(number, " ".join(words)))
        else:
            rows.append((when, at, zone))
    # Weekdays become dates, counted back from the check-in, last line first.
    out, bound = [], (day, minute)
    for when, at, zone in reversed(rows):
        if isinstance(when, int):
            date = bound[0] - datetime.timedelta(days=(bound[0].weekday() - when) % 7)
            if date == bound[0] and at > bound[1]:
                date -= datetime.timedelta(days=7)
            when = date
        if (when, at) < bound:
            bound = (when, at)
        out.append((when, at, zone))
    return list(reversed(out)), problems


PARTS = ("morning", "afternoon", "evening")


def parse_routines(data):
    """Pasted routines -> stated habits {"usual": {"0": [3 zones], "1": [3 zones]}, "outings": [(weekday, slot)]}.

    {"weekdays": {"morning": "kitchen", "afternoon": "living room", "evening": "living room"},
     "weekends": {...}, "outings": [{"day": "Tue", "part": "afternoon"}]}
    A list of three zones works in place of the {morning, afternoon, evening} object, and ["Tue", "afternoon"] in
    place of an outing object. Raises ValueError with what is wrong."""
    if not isinstance(data, dict):
        raise ValueError("routines must be an object with weekdays, weekends and outings")
    usual = {}
    for key, name in (("0", "weekdays"), ("1", "weekends")):
        value = data.get(name)
        if isinstance(value, dict):
            value = [value.get(part) for part in PARTS]
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError("routines.{} needs morning, afternoon and evening".format(name))
        zones = [ZONE_WORDS.get(str(z).strip().lower().replace(" ", "_").replace("-", "_")) for z in value]
        if any(z is None or z == "out" for z in zones):
            raise ValueError("routines.{}: each part of the day needs one of: {}".format(
                name, ", ".join(NAMES[z].lower() for z in F.ZONE_NAMES)))
        usual[key] = zones
    outings = []
    for item in data.get("outings") or []:
        day, part = (item.get("day"), item.get("part")) if isinstance(item, dict) else (list(item) + [None, None])[:2]
        weekday = WEEKDAYS.get(str(day).strip().lower()[:3])
        if weekday is None or str(part).strip().lower() not in PARTS:
            raise ValueError("routines.outings: each needs a day (Mon ... Sun) and a part (morning, afternoon, evening)")
        outings.append((weekday, PARTS.index(str(part).strip().lower())))
    return {"usual": usual, "outings": sorted(set(outings))}


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
                 robot=None, model=None):
        self._now, self._sleep, self._walk, self._look = now, sleep, walk, look
        self._robot = robot  # LiveRobot in the apartment scene, or None: a marker walks instead
        self._model = model  # tabpfn_belief.TabPFNBelief, or None: the counting belief
        self._predicted = (None, None)  # the last TabPFN belief and what it was for
        self._habits_version = 0
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
            self._habits_version = getattr(self, "_habits_version", 0) + 1
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

    def import_journal(self, text, replace=True):
        """Take a pasted journal (see parse_journal). With `replace` it becomes her whole journal, so the belief is
        built from it alone (plus the population and her stated habits); otherwise the records are added.
        Returns {"added": n, "problems": [...]}; raises ValueError when no line could be read."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ValueError("wait for the search to end before changing the journal")
            day, _, minute = self.when()
            rows, problems = parse_journal(str(text), day, minute)
            if not rows:
                raise ValueError("no journal records found. " + ("; ".join(problems[:3]) if problems
                                                                 else "One per line, like: Mon 09:30 kitchen"))
            if len(rows) > JOURNAL_LIMIT:
                raise ValueError("at most {} records".format(JOURNAL_LIMIT))
            if replace:
                self.journal = []
            for date, at, zone in rows:
                self._add(date, at, zone, "pasted")
            self._say("Journal: {} records pasted{}".format(len(rows), ", replacing the old journal" if replace else ""),
                      "journal")
            if problems:
                self._say("Journal: {} line{} skipped ({})".format(len(problems), "" if len(problems) == 1 else "s",
                                                                    problems[0]), "info")
            self._paint()
            return {"added": len(rows), "problems": problems[:20]}

    def inputs(self):
        """The two inputs the belief is built from, as one object to copy, edit and paste back (set_inputs)."""
        with self._lock:
            return {
                "routines": {
                    "weekdays": {part: NAMES[z].lower() for part, z in zip(PARTS, self.habits["usual"]["0"])},
                    "weekends": {part: NAMES[z].lower() for part, z in zip(PARTS, self.habits["usual"]["1"])},
                    "outings": [{"day": DAYS[wd], "part": PARTS[slot]} for wd, slot in self.habits["outings"]]},
                "journal": [{"day": e["day"], "time": e["time"], "zone": NAMES[e["zone"]].lower()} for e in self.journal],
            }

    def set_inputs(self, data):
        """Pasted inputs: {"routines": {...}, "journal": [...]}; either may be left out. Routines replace her
        stated habits (parse_routines); a journal replaces her journal (parse_journal). The belief follows at
        once: both are features of the row TabPFN predicts. Returns what changed."""
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError as error:
                raise ValueError("not valid JSON: {}".format(error))
        if not isinstance(data, dict) or not ({"routines", "journal"} & set(data)):
            raise ValueError('expected an object with "routines" and/or "journal"')
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ValueError("wait for the search to end before changing the inputs")
            habits = parse_routines(data["routines"]) if "routines" in data else None  # check before changing anything
            result = {"routines": habits is not None, "added": 0, "problems": []}
            if "journal" in data:
                if not isinstance(data["journal"], list):
                    raise ValueError("journal must be a list of {day, time, zone}")
                result.update(self.import_journal(json.dumps(data["journal"]), replace=True))
            if habits is not None:
                self.habits = habits
                self._habits_version += 1
                self._say("Routines replaced: weekdays {}, weekends {}, {} outing{}".format(
                    " / ".join(NAMES[z].lower() for z in habits["usual"]["0"]),
                    " / ".join(NAMES[z].lower() for z in habits["usual"]["1"]), len(habits["outings"]),
                    "" if len(habits["outings"]) == 1 else "s"), "journal")
                self._paint()
            return result

    def journal_text(self):
        """Her journal as text that import_journal reads back: one record per line, oldest first."""
        with self._lock:
            return "\n".join("{} {} {}".format(e["day"], e["time"], NAMES[e["zone"]].lower()) for e in self.journal)

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
        """The belief step by step at the check-in time: population, + habits, + journal (counting), and
        "last_heard", the final one: TabPFN's when there is a model (and it is day), else the counting one."""
        day, weekday, minute = self.when()
        known = self._known()
        journal = [(e["weekday"], e["slot"], e["zone"]) for e in known if e["slot"] is not None]
        last = self._last()
        last = (last[0]["zone"], last[1]) if last else None
        out = S.beliefs(self.flat, minute / 60.0, weekday, self._population, self.habits, journal, last)
        out["counting"] = out["last_heard"]
        if self._model is not None:
            key = (day, minute, len(known), known[-1]["id"] if known else 0, self.seed, self._habits_version)
            if self._predicted[0] != key:
                records = [(e["weekday"], e["slot"], e["zone"]) for e in known
                           if e["slot"] is not None and e["day"] < day.isoformat()]  # earlier days, as in the examples
                try:
                    self._predicted = (key, self._model.belief(minute / 60.0, weekday, self.habits, records, last))
                except Exception as error:  # the search must not die with the model
                    self._say("TabPFN belief failed ({}), using the counting belief".format(error), "info")
                    self._model = None
                    return out
            if self._predicted[1] is not None:
                out["last_heard"] = self._predicted[1]
        return out

    def model_info(self):
        """Which model gives the belief, for the panel."""
        if self._model is None:
            return {"name": "counting", "label": "Counting (Bayes)", "detail": "population + habits + journal + last heard"}
        return {"name": "tabpfn", "label": "TabPFN {}".format(self._model.version.lstrip("v")),
                "detail": "fit() on {:,} synthetic check-ins".format(self._model.rows)}

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
            self._say("Belief ({}): {}".format(self.model_info()["label"], ", ".join(
                "{} {:.0f}%".format(NAMES[z], 100 * p) for p, z in top)), "belief")
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

    def clear(self):
        """Ready for another run: ends the search, takes Margarete out of the room, puts the robot back on its
        dock and forgets the last search. The flat and the journal stay."""
        self.stop()
        with self._lock:
            self.search = None
            self.run = {"status": "idle", "goal": None, "truth": None, "miss_rate": S.MISS_RATE, "path": []}
            self.robot = dict(self.flat["dock"])
            if self._robot is not None:
                self._robot.person()
                try:
                    self._robot.command("reset")
                except ValueError:
                    pass
            self._say("Reset: {} is out of the room, the robot is on its dock".format(self.name), "info")
            self._paint()

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
                "model": self.model_info(),
                "log": list(self.log),
            }


class _Stopped(Exception):
    pass


class _Failed(Exception):
    pass

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
# With the robot's person detector (app/spotter.py): how often a look-around overlooks her (the Bayesian update's
# miss rate; measured with just search-bench), how long a look-around may take, and the close-up check.
DETECT_MISS_RATE = 0.1
LOOK_TIMEOUT = 90.0
APPROACH_STOP = 0.6   # metres from her the robot stops (its approach keeps re-ranging her on the way)
CHECK_CYCLES = 4      # gait cycles it looks at her from there
CHECK_HITS = 2        # matches near the estimate in those cycles that make it "found"
POSES = ("lying", "standing")
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
                 robot=None, model=None, on_event=None):
        self._now, self._sleep, self._walk, self._look = now, sleep, walk, look
        self._on_event = on_event  # called with (kind, facts) as the search goes (app/narrate.py); must not block
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
            self.home_flat = self.flat  # the flat as built; shuffle() moves its loose pieces and drops clutter
            self.obstacles = "home"  # or the mess level of the last shuffle: clean, medium, messy
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

    def shuffle(self, level="medium", rng=None):
        """The room at a mess level (homes/flats.py scatter): "clean" (nothing loose on the floor: the stool,
        armchair, crate and plant taken out), "medium" (those four dropped at random spots on the open floor plus
        4 clutter items) or "messy" (the four plus up to 12 clutter items), to watch the robot avoid things it was
        not told about. A new arrangement every time, also at the same level. The zones, the journal and the
        resident stay; the robot goes back to its dock. Refused while a search runs. Returns the level."""
        level = "medium" if level in (None, "") else str(level)
        if level not in F.LEVELS:
            raise ValueError("level must be one of: " + ", ".join(F.LEVELS))
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ValueError("wait for the search to end before moving the furniture")
            self.flat = F.scatter(self.home_flat, rng or np.random.default_rng(), level)
            self.obstacles = level
            self.robot = dict(self.flat["dock"])
            self.search = None
            self.run = {"status": "idle", "goal": None, "truth": None, "miss_rate": S.MISS_RATE, "path": []}
            if self._robot is not None:
                self._robot.load(self.flat)
                self._robot.person()
            self._paint()
            n = F.loose_count(self.flat)
            title = {"clean": "Clean room", "medium": "Medium messy room", "messy": "Messy room"}[level]
            wanted = F.LEVELS[level]["clutter"] + (len(F.LOOSE) if F.LEVELS[level]["pieces"] else 0)
            if n == 0:
                self._say("{}: nothing loose on the floor, only the fixed furniture".format(title), "info")
            else:
                self._say("{}: {} thing{} on the floor{}".format(
                    title, n, "" if n == 1 else "s", " ({} of {} fitted)".format(n, wanted) if n < wanted else ""),
                    "info")
            return self.obstacles

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

    def randomize(self, what="both"):
        """A new random example: "routines" (a new resident's routine, so new stated habits), "journal" (2 weeks
        drawn from her routine) or "both". The flat stays. Returns the new inputs (as inputs())."""
        if what not in ("routines", "journal", "both"):
            raise ValueError("what must be routines, journal or both")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ValueError("wait for the search to end before changing the inputs")
            rng = np.random.default_rng()
            if what in ("routines", "both"):
                self.persona = R.sample_persona(rng, 0, self.flat)
                self.habits = R.stated_habits(dict(self.persona, resident_id=int(rng.integers(1 << 30))))
                self._habits_version += 1
            if what in ("journal", "both"):
                day, _, minute = self.when()
                self.journal = []
                for back in range(JOURNAL_DAYS, -1, -1):
                    date = day - datetime.timedelta(days=back)
                    codes = self._draw(date.weekday(), rng)
                    for slot, (lo, hi) in enumerate(R.SLOT_WINDOWS):
                        at = int(rng.uniform(lo, hi))
                        if back == 0 and at >= minute:
                            break  # nothing after the check-in time
                        self._add(date, at, F.CLASSES[codes[slot]], "robot")
            kind = {"routines": "routines", "journal": "journal", "both": "routines and journal"}[what]
            self._say("Random {} ({} type){}".format(kind, self.persona["archetype"].replace("_", " "),
                                                     ", journal drawn from it" if what == "both" else ""), "journal")
            self._paint()
            return self.inputs()

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

    def detector_ready(self):
        """True when the robot has its person detector (app/spotter.py) loaded: then looks are real."""
        if self._robot is None:
            return False
        return (self._robot.state().get("spotter") or {}).get("status") == "ready"

    def start(self, where="routine", miss_rate=None, pose="random"):
        """Start the search loop. False while one is running. `pose`: Margarete "lying", "standing" or "random"."""
        if where not in WHERE:
            raise ValueError("where must be one of: " + ", ".join(WHERE))
        if pose not in POSES + ("random",):
            raise ValueError("pose must be lying, standing or random")
        detect = self.detector_ready()
        miss = (DETECT_MISS_RATE if detect else S.MISS_RATE) if miss_rate is None else float(miss_rate)
        if not 0.0 <= miss <= 0.9:
            raise ValueError("miss_rate must be 0 to 0.9")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            _, weekday, minute = self.when()
            start = self.stages()["last_heard"]
            truth = self._truth(where, weekday, minute)
            self.search = S.Search(self.flat, start, miss)
            person = None  # where she is: the zone's main spot, or any of its spots for "random"
            if pose == "random":
                pose = POSES[int(np.random.default_rng().integers(len(POSES)))]
            if truth != "out":
                spots = F.zone(self.flat, truth)["spots"]
                spot = spots[int(np.random.default_rng().integers(len(spots)))] if where == "random" else spots[0]
                person = (spot["person_x"], spot["person_y"], spot["heading_deg"] + 90.0)
            if self._robot is None:
                self.robot = dict(self.flat["dock"])
            else:  # Margarete is where she really is; the robot starts from where it stands
                self._robot.person(*person, pose=pose) if person else self._robot.person()
                self.robot = self._pose()
            self.run = {"status": "starting", "goal": None, "truth": truth, "miss_rate": miss, "walked": 0.0,
                        "fallbacks": 0, "contacts": 0, "pose": pose if person else None,
                        "look_by": "detector" if detect else "mock", "sightings": [], "false_alarms": 0,
                        "found_at": None, "found_error": None, "robot_to_person": None, "approach": None,
                        "person": list(person[:2]) if person else None,
                        "path": [[self.robot["x"], self.robot["y"]]]}
            self._paint(start)
            self._stop.clear()
            self._say("Check-in {} {}: where is {}?".format(DAYS[weekday], hhmm(minute), self.name), "start")
            top = sorted(((p, z) for z, p in start.items()), reverse=True)[:3]
            self._say("Belief ({}): {}".format(self.model_info()["label"], ", ".join(
                "{} {:.0f}%".format(NAMES[z], 100 * p) for p, z in top)), "belief")
            self._emit("start", name=self.name, top=[[z, float(p)] for p, z in top])
            self._thread = threading.Thread(target=self._loop, args=(truth, miss, detect), daemon=True)
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

    def _loop(self, truth, miss, detect=False):
        """Propose the most likely zone, walk there, look, update; until found, nobody home or gave up.
        `detect`: the robot's person detector looks (walking, then a look-around at the zone, then a close-up
        check of anything it saw); else the mock look (S.observe, `miss` of the looks overlook her)."""
        rng = np.random.default_rng()
        search = self.search
        try:
            if detect:
                self._command("spot reset")
            while True:
                with self._lock:
                    goal = search.goal()
                    if goal is None:
                        break
                    self.run.update(status="walking", goal=goal["zone"])
                    self._say("Proposal {}: {} ({:.0f}%), the most likely zone".format(
                        len(search.steps) + 1, NAMES[goal["zone"]], 100 * goal["p"]), "proposal")
                    self._emit("heading", zone=goal["zone"], p=float(goal["p"]), look=len(search.steps) + 1)
                    start = (self.robot["x"], self.robot["y"])
                if not detect:
                    self._move(start, (goal["x"], goal["y"]), goal["heading_deg"])
                    with self._lock:
                        self.run["status"] = "looking"
                    self._wait(self._look)
                    self._report(search, goal, bool(S.observe(goal["zone"], truth, rng, miss)))
                    continue
                spotted = self._drive((goal["x"], goal["y"]), goal["heading_deg"], watch=True)
                on_the_way = spotted is not None
                if spotted is None:
                    with self._lock:
                        self.run["status"] = "looking"
                        self._say("Looking around in the {}".format(NAMES[goal["zone"]].lower()), "look")
                    spotted = self._look_around()
                if spotted is None:
                    self._report(search, goal, False)
                    continue
                where = self._check(spotted, goal["zone"])
                if where is not None:
                    zone = F.zone_at(self.flat, where[0], where[1])
                    with self._lock:
                        search.report_found(zone)
                        self._paint(search.steps[-1]["belief_after"])
                        person = self.run.get("person")
                        error = math.hypot(where[0] - person[0], where[1] - person[1]) if person else None
                        # for the log and the bench only (never decides anything): the robot's true distance to her
                        near = math.hypot(self.robot["x"] - person[0], self.robot["y"] - person[1]) if person else None
                        self.run.update(found_at=[round(where[0], 2), round(where[1], 2)],
                                        found_error=None if error is None else round(error, 2),
                                        robot_to_person=None if near is None else round(near, 2))
                        self._say("Observation: sees {} in the {} ({:.1f} m from the robot{})".format(
                            self.name, NAMES[zone].lower(), math.hypot(where[0] - self.robot["x"],
                                                                         where[1] - self.robot["y"]),
                            "" if error is None else "; {:.2f} m from where she really is".format(error)), "seen")
                    break
                with self._lock:  # the close-up showed nobody: a look-alike
                    self.run["false_alarms"] = self.run.get("false_alarms", 0) + 1
                    self._say("False alarm: nobody there on a closer look; the robot ignores that spot now",
                              "false_alarm")
                self._emit("false_alarm", zone=goal["zone"])
                self._command("spot reject")
                if not on_the_way:  # it was the look-around of this zone: she was not seen there
                    self._report(search, goal, False)
            with self._lock:
                self.run.update(status=search.status, goal=None)
                looks = len(search.steps)
                tail = "after {} look{}, {:.1f} m".format(looks, "" if looks == 1 else "s", self.run["walked"])
                if search.status == "found":
                    zone = search.steps[-1]["zone"]
                    self._say("Found {} in the {} {}".format(self.name, NAMES[zone].lower(), tail), "found")
                    self._emit("found", name=self.name, zone=zone, looks=looks, metres=float(self.run["walked"]))
                    day, _, minute = self.when()
                    self._add(day, minute, zone, "search")
                    self._say("Journal: {} {} (seen by the robot)".format(hhmm(minute), NAMES[zone]), "journal")
                elif search.status == "nobody_home":
                    self._say("Nobody home ({:.0f}% sure) {}".format(100 * search.belief.get("out", 0.0), tail), "out")
                    self._emit("nobody_home", name=self.name, looks=looks, metres=float(self.run["walked"]),
                               p_out=float(search.belief.get("out", 0.0)))
                else:
                    self._say("Gave up {}".format(tail), "out")
                    self._emit("gave_up", name=self.name, looks=looks, metres=float(self.run["walked"]))
        except _Stopped:
            if self._robot is not None:
                self._command("stop")
            with self._lock:
                self.run.update(status="stopped", goal=None)
                self._say("Search stopped", "info")
                self._emit("stopped", name=self.name)
        except _Failed as error:
            with self._lock:
                self.run.update(status="failed", goal=None)
                self._say("Search ended: {}".format(error), "out")
                self._emit("failed", name=self.name, reason=str(error))

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

    def _report(self, search, goal, seen):
        """Tell the Bayesian search what the look at `goal` showed; log and narrate the update."""
        with self._lock:
            before = dict(search.belief)
            search.report(seen)
            step = search.steps[-1]
            self._paint(step["belief_after"])
            if seen:
                self._say("Observation: sees {} in the {}".format(self.name, NAMES[goal["zone"]].lower()), "seen")
                return
            self._say("Observation: nobody seen in the {}".format(NAMES[goal["zone"]].lower()), "miss")
            rise = max((z for z in before if z != goal["zone"]), key=lambda z: step["belief_after"][z] - before[z])
            self._say("Update: {} {:.0f}% -> {:.0f}%, {} {:.0f}% -> {:.0f}%".format(
                NAMES[goal["zone"]], 100 * before[goal["zone"]], 100 * step["belief_after"][goal["zone"]],
                NAMES[rise], 100 * before[rise], 100 * step["belief_after"][rise]), "update")
            upcoming = search.goal()  # the zone the next proposal picks (None when the search ends)
            self._emit("miss", zone=goal["zone"], belief={z: float(p) for z, p in step["belief_after"].items()},
                       next=upcoming["zone"] if upcoming else None,
                       next_p=float(upcoming["p"]) if upcoming else None)

    def _spotter(self, state=None):
        return ((state or self._robot.state()).get("spotter") or {})

    def _drive(self, end, heading_deg, watch=False):
        """The real walk: goto, wait until the robot stands near `end`, then turn to face the zone. `watch`: stop
        as soon as the person detector confirms her; returns where (x, y), else None."""
        state = self._robot.state()
        self._seen = {"fallbacks": state.get("fallbacks", 0) or 0, "contacts": state.get("contacts", 0) or 0}
        self._command("goto {:.2f} {:.2f}".format(*end))
        spotted = self._settle(WALK_TIMEOUT * math.hypot(end[0] - self.robot["x"], end[1] - self.robot["y"]) + 60.0,
                               near=end, watch=watch)
        if spotted is not None:
            return spotted
        if heading_deg is not None:
            turn = (heading_deg - self._pose()["yaw_deg"] + 180.0) % 360.0 - 180.0
            if abs(turn) > 8.0:
                self._command("turn {:.1f}".format(turn))
                self._settle(TURN_TIMEOUT)
        return None

    def _sighted(self, spot):
        """The detector confirmed her: log it once, narrate it; returns (x, y)."""
        where = (spot["x"], spot["y"])
        with self._lock:
            zone = F.zone_at(self.flat, where[0], where[1])
            dist = math.hypot(where[0] - self.robot["x"], where[1] - self.robot["y"])
            self.run["sightings"].append({"x": round(where[0], 2), "y": round(where[1], 2), "zone": zone,
                                          "dist": round(dist, 2), "while": self.run["status"]})
            self.run["status"] = "approaching"
            self._say("Spotted someone {:.1f} m away, in the {} (camera match {:.2f}); going closer to check".format(
                dist, NAMES[zone].lower(), (spot.get("match") or {}).get("sim", spot.get("best") or 0.0)), "spotted")
        self._emit("spotted", zone=zone, metres=float(dist))
        return where

    def _look_around(self):
        """Turn once on the spot with the detector looking; (x, y) when it confirms her, else None."""
        self._command("look")
        waited = 0.0
        while True:
            self._wait(POLL)
            waited += POLL
            state = self._robot.state()
            with self._lock:
                self.robot = self._pose()
            spot = self._spotter(state)
            if spot.get("confirmed"):
                self._command("stop")
                return self._sighted(spot)
            look = state.get("look") or {}
            if state.get("mode") == "idle" and not look.get("running") and waited >= 2 * POLL:
                if look.get("blocked"):
                    with self._lock:
                        self._say("No room to turn all the way round here (something within {:.0f} cm); looked "
                                  "{:.0f} degrees".format(100 * 0.33, look.get("turned_deg", 0.0)), "info")
                return None
            if state.get("mode") == "fallen":
                raise _Failed("the robot fell over")
            if waited > LOOK_TIMEOUT:
                self._command("stop")
                return None

    def _check(self, where, zone):
        """The close-up: the robot walks up to her (its `approach`: every gait cycle the destination is recomputed
        from the detector's estimate, APPROACH_STOP short of her) and faces her; the strong matches from up close on
        the way count for the check. Too few by then: it stands and looks CHECK_CYCLES more cycles. (x, y) of the
        final estimate when CHECK_HITS matches agree, else None (a false alarm)."""
        with self._lock:
            self.run["status"] = "approaching"
            self.run["approach"] = {"goal": None, "estimate": [round(where[0], 2), round(where[1], 2)],
                                    "updates": 0, "reason": None}
        self._command("spot mark")
        reason = self._approach(where)
        with self._lock:
            self.run["status"] = "checking"
        spot = self._spotter()
        if reason == "arrived" and spot.get("watch_hits", 0) < CHECK_HITS:
            self._command("watch {}".format(CHECK_CYCLES))
            self._settle(CHECK_CYCLES * 3.0 + 20.0, idle_modes=("watch",))
            spot = self._spotter()
        if spot.get("watch_hits", 0) >= CHECK_HITS and spot.get("x") is not None:
            return (spot["x"], spot["y"])
        return None

    def _approach(self, where):
        """Send `approach` and follow it on the plan until it ends; log the destination when it moves (her range was
        measured again). Returns how it ended: arrived, lost, interrupted, fallen or timeout."""
        state = self._robot.state()
        self._seen = {"fallbacks": state.get("fallbacks", 0) or 0, "contacts": state.get("contacts", 0) or 0}
        start = (self.robot["x"], self.robot["y"])
        with self._lock:
            self._say("Walking up to {}: {:.1f} m away, to stop {:.1f} m from her".format(
                self.name, math.hypot(where[0] - start[0], where[1] - start[1]), APPROACH_STOP), "approach")
        self._command("approach {:.2f}".format(APPROACH_STOP))
        timeout = WALK_TIMEOUT * math.hypot(where[0] - start[0], where[1] - start[1]) + 60.0
        waited, said = 0.0, None
        while True:
            self._wait(POLL)
            waited += POLL
            state = self._robot.state()
            ap = state.get("approach") or {}
            with self._lock:
                last = self.robot
                self.robot = self._pose()
                self.run["walked"] += math.hypot(self.robot["x"] - last["x"], self.robot["y"] - last["y"])
                tail = self.run["path"][-1]
                if math.hypot(self.robot["x"] - tail[0], self.robot["y"] - tail[1]) > 0.15:
                    self.run["path"].append([self.robot["x"], self.robot["y"]])
                self.run["approach"] = {"goal": ap.get("goal"), "estimate": ap.get("estimate"),
                                        "updates": ap.get("updates", 0), "reason": ap.get("reason")}
                goal = ap.get("goal")
                if goal is not None and said is not None and math.hypot(goal[0] - said[0], goal[1] - said[1]) > 0.3:
                    self._say("Destination updated: the camera ranges {} at {:.1f} m; the robot now heads for "
                              "({:.1f}, {:.1f})".format(self.name, ap.get("range") or 0.0, goal[0], goal[1]),
                              "approach")
                if goal is not None and (said is None or math.hypot(goal[0] - said[0], goal[1] - said[1]) > 0.3):
                    said = goal
            self._watch(state)
            if state.get("mode") == "fallen":
                raise _Failed("the robot fell over")
            if not ap.get("running") and waited >= 2 * POLL:
                reason = ap.get("reason") or "lost"
                with self._lock:
                    est = ap.get("estimate")
                    if reason == "arrived" and est is not None:
                        self._say("Standing {:.1f} m from {}, facing her".format(
                            math.hypot(est[0] - self.robot["x"], est[1] - self.robot["y"]), self.name), "approach")
                    elif reason == "lost":
                        self._say("Lost sight of whoever it was on the way", "approach")
                return reason
            if waited > timeout:
                self._command("stop")
                return "timeout"

    def _settle(self, timeout, near=None, watch=False, idle_modes=()):
        """Wait until the robot has finished its command; follow it on the plan meanwhile. `watch`: return (x, y)
        as soon as the person detector confirms her (and stop the walk); otherwise returns None. `idle_modes`:
        more modes that count as "the command has started" (watch)."""
        waited, started = 0.0, False
        while True:
            self._wait(POLL)
            waited += POLL
            state = self._robot.state()
            started = started or state.get("mode") in ("goto", "turn") + tuple(idle_modes)
            with self._lock:
                last = self.robot
                self.robot = self._pose()
                moved = math.hypot(self.robot["x"] - last["x"], self.robot["y"] - last["y"])
                self.run["walked"] += moved
                tail = self.run["path"][-1]
                if math.hypot(self.robot["x"] - tail[0], self.robot["y"] - tail[1]) > 0.15:
                    self.run["path"].append([self.robot["x"], self.robot["y"]])
            self._watch(state)
            if watch and self._spotter(state).get("confirmed"):
                self._command("stop")
                return self._sighted(self._spotter(state))
            if state.get("mode") == "fallen":
                raise _Failed("the robot fell over")
            # "idle" means done only once the robot was seen walking (the simulator may take a moment to pick the
            # command up), or when it stands at the place already.
            if state.get("mode") == "idle" and waited >= 2 * POLL:
                there = near is None or math.hypot(state["x"] - near[0], state["y"] - near[1]) < 0.45
                if there and (started or near is not None or waited > 3.0):
                    return None
                if started or waited > 5.0:
                    raise _Failed("the robot stopped before it got there (another command?)")
            if waited > timeout:
                self._command("stop")
                raise _Failed("the robot did not get there in time (something is in the way)")

    def _watch(self, state):
        """Log (and narrate) what the robot reports while it walks: the student handing a walk to A* + the
        a-to-b driver (a normal part of walking, not a failure), and touches of walls and furniture."""
        seen = getattr(self, "_seen", None)
        if seen is None:
            return
        fallbacks, contacts = state.get("fallbacks", 0) or 0, state.get("contacts", 0) or 0
        if fallbacks > seen["fallbacks"]:
            seen["fallbacks"] = fallbacks
            info = state.get("fallback") or {}
            with self._lock:
                self.run["fallbacks"] = self.run.get("fallbacks", 0) + 1
                zone = self.run.get("goal")
                self._say("Fallback: {}; the mapped route (A* + {}) walks the rest of the way".format(
                    info.get("reason", "handed over"), info.get("driver", "a-to-b")), "fallback")
            self._emit("fallback", zone=zone, reason=str(info.get("reason", "")))
        if contacts > seen["contacts"]:
            with self._lock:
                self.run["contacts"] = self.run.get("contacts", 0) + contacts - seen["contacts"]
                self._say("Contact: the robot touched a wall or a piece of furniture ({} this search)".format(
                    self.run["contacts"]), "contact")
            seen["contacts"] = contacts

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

    def _emit(self, kind, **facts):
        """Tell on_event what just happened. Never lets it stop the search."""
        if self._on_event is None:
            return
        try:
            self._on_event(kind, facts)
        except Exception as error:  # noqa: BLE001 - the narration must never break the search
            print("[apartment] on_event failed: {}".format(error), flush=True)

    def note(self, text, kind="metacon"):
        """A line from outside the search loop in the log (metacon's narration)."""
        self._say(text, kind)

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
            running = self.run["status"] in ("starting", "walking", "looking", "approaching", "checking")
            belief = dict(search.belief) if search is not None and self.run["status"] != "idle" else stages["last_heard"]
            if sum(belief.values()) <= 0:
                belief = stages["last_heard"]
            last = self._last()
            flat = self.flat
            robot_state = self._robot.state() if self._robot is not None else {}
            return {
                "seed": self.seed, "name": self.name,
                "flat": {"half": flat["width"] / 2, "door": flat["door"], "dock": flat["dock"],
                         "zones": [{k: z[k] for k in ("name", "label", "x", "y", "size_x", "size_y", "view_x", "view_y",
                                                    "person_x", "person_y")}
                                   for z in flat["zones"]],
                         "obstacles": [dict({k: f[k] for k in ("name", "zone", "x", "y", "size_x", "size_y",
                                                                      "yaw_deg")},
                                            length=2 * F.shape(f["name"])["half"][0],
                                            width=2 * F.shape(f["name"])["half"][1],
                                            round=F.shape(f["name"])["shape"] == "circle",
                                            moved=bool(f.get("moved")), clutter=bool(f.get("clutter")),
                                            loose=f["name"] in F.LOOSE or bool(f.get("clutter")))
                                       for f in F.on_floor(flat)]},
                "obstacles": self.obstacles,
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
                           "metres": self.run.get("walked", 0.0), "fallbacks": self.run.get("fallbacks", 0),
                           "contacts": self.run.get("contacts", 0), "pose": self.run.get("pose"),
                           "look_by": self.run.get("look_by", "detector" if self.detector_ready() else "mock"),
                           "sightings": self.run.get("sightings", []), "false_alarms": self.run.get("false_alarms", 0),
                           "found_at": self.run.get("found_at"), "found_error": self.run.get("found_error"),
                           "robot_to_person": self.run.get("robot_to_person"), "approach": self.run.get("approach"),
                           "steps": [{"zone": s["zone"], "p": s["p_before"], "seen": s["seen"], "metres": s["metres"]}
                                     for s in (search.steps if search else [])]},
                "robot": dict(self._pose() if self._robot is not None else self.robot),
                "driver": robot_state.get("driver", "marker"),
                "student": robot_state.get("student", {"status": "off"}),
                "spotter": robot_state.get("spotter", {"status": "off"}),
                "look": robot_state.get("look", {"running": False}),
                "ab_driver": robot_state.get("ab_driver"),
                "contact_points": robot_state.get("contact_points", []),
                "route": robot_state.get("route", []) if robot_state.get("mode") == "goto" else [],
                "route_followed": robot_state.get("route_followed", True),
                "model": self.model_info(),
                "log": list(self.log),
            }


class _Stopped(Exception):
    pass


class _Failed(Exception):
    pass

"""Knock: the robot checks on Margarete. Spec: docs/knock-ui.md; her day: docs/knock-routines.md.

Call Margarete: the robot says a message in its own voice and listens for LISTEN seconds. Nobody answers yet
(Margarete is not in the simulation), so the search starts by itself. Find Margarete: the robot walks to the zone
where she most likely is (probability / walking distance), looks, and goes on to the next zone until it finds her.

One check-in at a time. Every change is published on the Hub as {"type": "knock", "state": ...} with the whole
card in it, so every viewer's page just shows the latest one:

    idle       the Margarete card with Find and Call
    calling    "text" spoken (with "audio": a URL, once the speech is ready), "listen_s" seconds left to answer
    finding    "probs" per zone, "checked" zones, "next" zone, "text" (what the robot is doing)
    found      "zone", "pose", "seen", "heard"; "choice" once the person picked call_112 or call_neighbour
    not_home   every zone checked, nobody found
Always: "steps" [{"text", "time"}], "last_heard" {"zone", "time"}, "source" ("placeholder" until the zone model).

The zone probabilities and where she is are placeholders until Krishna's zone model and the person detector
exist: PRIORS below (Tuesday 10:35 in knock-routines.md), and $KNOCK_FOUND_ZONE (default couch; "none" = not home).
"""

import math
import os
import secrets
import threading
import time
from collections import OrderedDict

LISTEN = 20.0  # seconds the robot waits for an answer after calling her
LOOK = 2.0  # seconds it looks at a zone
WALK_TIMEOUT = 90.0
TURN_TIMEOUT = 15.0
POLL = 0.1
NEAR = 0.2  # metres: close enough to the viewing pose
KEEP_AUDIO = 4
MESSAGE = "Margarete, it's Thomas. Are you all right?"

# Zone: label, viewing pose (x, y) in the open aisle, heading to face the zone (degrees, 0 = +x), where she lies.
# Cells from knock-routines.md (2 x 3 grid of 1 m cells); move them when the demo room (pull request #12) is agreed.
ZONES = OrderedDict([
    ("bed", ("Bed", (-1.0, 0.25), 90.0, "by the bed")),
    ("bathroom", ("Bathroom", (0.0, 0.25), 90.0, "on the bathroom floor")),
    ("kitchen", ("Kitchen", (1.0, 0.25), 90.0, "by the kitchen table")),
    ("couch", ("Couch", (-1.0, -0.25), -90.0, "in front of the couch")),
    ("armchair", ("Armchair", (0.0, -0.25), -90.0, "by the armchair")),
    ("entrance", ("Entrance", (1.0, -0.25), -90.0, "by the front door")),
])
PRIORS = {"couch": 0.42, "kitchen": 0.18, "armchair": 0.16, "bed": 0.13, "bathroom": 0.09, "entrance": 0.02}
LAST_HEARD = {"zone": "kitchen", "time": "09:10"}


def found_zone():
    zone = os.environ.get("KNOCK_FOUND_ZONE", "couch").strip().lower()
    return zone if zone in ZONES else None


def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def next_zone(probs, checked, x, y):
    """The unchecked zone with the highest probability per metre of walking (near zones first when close)."""
    best, best_score = None, -1.0
    for zone, (_, (zx, zy), _, _) in ZONES.items():
        if zone in checked:
            continue
        score = probs.get(zone, 0.0) / max(math.hypot(zx - x, zy - y), 0.5)
        if score > best_score:
            best, best_score = zone, score
    return best


def after_miss(probs, zone):
    """Probabilities once zone is checked and she was not there: it drops to 0, the others share its part."""
    rest = {z: p for z, p in probs.items() if z != zone}
    total = sum(rest.values())
    out = {z: (p / total if total > 0 else 0.0) for z, p in rest.items()}
    out[zone] = 0.0
    return out


class Stopped(Exception):
    pass


class Knock:
    def __init__(self, hub, robot, speak, clock=time.monotonic, sleep=time.sleep, spawn=None,
                 stamp=lambda: time.strftime("%H:%M"), listen=LISTEN, look=LOOK, where=found_zone):
        self.hub, self.robot, self.speak = hub, robot, speak
        self.clock, self.sleep, self.stamp, self.listen, self.look, self.where = clock, sleep, stamp, listen, look, where
        self.spawn = spawn or (lambda fn: threading.Thread(target=fn, name="knock", daemon=True).start())
        self.lock = threading.Lock()
        self.run = None  # {"stop": bool, "skip": bool} while a check-in runs
        self.sounds = OrderedDict()
        self.listen_end = None
        self.view = self._idle()

    # --- what the server calls ---

    def latest(self):
        """The current card, for a page that opens in the middle of a check-in (without old audio)."""
        with self.lock:
            view = dict(self.view)
            if view["state"] == "calling" and self.listen_end is not None:
                view["listen_s"] = max(0.0, round(self.listen_end - self.clock(), 1))
        return dict(view, type="knock")

    def audio(self, sound_id):
        with self.lock:
            return self.sounds.get(sound_id)

    def call(self):
        """Start a check-in with a call-out. "started", "again" (called once more during a search), or None if busy."""
        with self.lock:
            again = self.run is not None
            if again and self.view["state"] != "finding":
                return None
            run = self.run if again else {"stop": False, "skip": False}
            self.run = run
        if again:
            self.spawn(lambda: self._call_again(run))
            return "again"
        self.spawn(lambda: self._check_in(run, call_first=True))
        return "started"

    def find(self):
        """Start a search. "started", "skip" (stop listening and search now), or None if a search runs."""
        with self.lock:
            if self.run:
                if self.view["state"] != "calling":
                    return None
                self.run["skip"] = True
                return "skip"
            run = self.run = {"stop": False, "skip": False}
        self.spawn(lambda: self._check_in(run, call_first=False))
        return "started"

    def stop(self):
        """Stop the check-in (the robot stops too) or close a found / not-home card."""
        with self.lock:
            run, self.run = self.run, None
            if run:
                run["stop"] = True
            was = self.view["state"]
        if run:
            self._robot("stop")
        if run or was != "idle":
            self._publish(self._idle())
            return True
        return False

    def choose(self, choice):
        """The person's choice on the found card. In the demo the calls are simulated."""
        if choice not in ("call_112", "call_neighbour", "talk"):
            raise ValueError("choice must be call_112, call_neighbour or talk")
        with self.lock:
            if self.view["state"] != "found":
                return False
        if choice == "talk":
            self.stop()
            return True
        text = "Calling 112… (simulated)" if choice == "call_112" else "Calling Frau Weber… (simulated)"
        self._update(choice=choice, steps=self._steps(text))
        return True

    # --- the check-in, on its own thread ---

    def _check_in(self, run, call_first):
        try:
            if call_first:
                self._call_out(run)
            self._search(run)
        except Stopped:
            pass
        except Exception as error:  # never leave the card stuck on "finding"
            self._end(run, state="not_home", text="The search stopped: {}".format(error))
        finally:
            with self.lock:
                if self.run is run:
                    self.run = None

    def _call_out(self, run):
        self._publish(dict(self._idle(), state="calling", text=MESSAGE, listen_s=None), run)
        sound = self._say(MESSAGE)
        with self.lock:
            self._check(run)
            self.listen_end = self.clock() + self.listen
        self._update(run, audio=sound, listen_s=self.listen)
        end = self.listen_end
        while self.clock() < end:
            self._check(run)
            if run["skip"]:
                break
            self.sleep(POLL)
        self.listen_end = None
        if not run["skip"]:
            self._update(run, state="no_answer", listen_s=None)
            self._wait(run, 1.5)  # long enough for every viewer to read "No answer"
        self._update(run, steps=self._steps("Called her name, no answer"))

    def _call_again(self, run):
        sound = self._say(MESSAGE)
        if self.run is run and not run["stop"]:
            self._update(run, audio=sound, steps=self._steps("Called her name again"))

    def _search(self, run):
        probs, checked = dict(PRIORS), []
        she = self.where()
        while True:
            state = self._robot_state()
            zone = next_zone(probs, checked, state.get("x", 0.0), state.get("y", 0.0))
            if zone is None:
                self._end(run, state="not_home", text="She doesn't seem to be home.", probs=probs, checked=checked,
                          next=None, steps=self._steps("Every zone checked: nobody there"))
                return
            label = ZONES[zone][0].lower()
            text = "Walking to the {} ({:.0f}%).".format(label, probs[zone] * 100)
            if checked:
                text = "Not in the {}. {}".format(ZONES[checked[-1]][0].lower(), text)
            self._update(run, state="finding", text=text, probs=probs, checked=list(checked), next=zone,
                         listen_s=None)
            self._go(run, zone)
            self._update(run, text="Looking at the {}…".format(label))
            self._wait(run, self.look, watch=True)
            if zone == she:
                where = ZONES[zone][3]
                self._end(run, state="found", zone=zone, pose="lying", next=None, probs=probs, checked=checked,
                          text="On the floor {}. She moved her arm but didn't answer clearly.".format(where),
                          seen="moved her arm", heard="a sound, no words", time=self.stamp(),
                          steps=self._steps("Found her {}".format(where)))
                return
            checked.append(zone)
            probs = after_miss(probs, zone)
            self._update(run, steps=self._steps("{}: not there".format(ZONES[zone][0])))

    def _go(self, run, zone):
        _, (x, y), face, _ = ZONES[zone]
        self._robot("goto {:.2f} {:.2f}".format(x, y))
        self._settle(run, WALK_TIMEOUT, near=(x, y))
        state = self._robot_state()
        turn = wrap(face - state.get("yaw_deg", face))
        if abs(turn) > 5.0:
            self._robot("turn {:.1f}".format(turn))
            self._settle(run, TURN_TIMEOUT)

    def _settle(self, run, timeout, near=None):
        """Wait until the robot has finished its command (idle again, and at the pose for a goto)."""
        start = self.clock()
        while True:
            self.sleep(POLL)
            self._check(run)
            state = self._robot_state()
            if state.get("mode") == "fallen":
                raise RuntimeError("the robot fell over")
            elapsed = self.clock() - start
            if state.get("mode") == "idle" and elapsed >= 2 * POLL:
                if near is None or math.hypot(state["x"] - near[0], state["y"] - near[1]) < NEAR:
                    return
                if elapsed > 1.0:
                    raise RuntimeError("the robot was given another command")
            if elapsed > timeout:
                raise RuntimeError("the robot did not get there in time")

    def _wait(self, run, seconds, watch=False):
        end = self.clock() + seconds
        while self.clock() < end:
            self._check(run)
            if watch:
                self.robot.frame("eye")  # keeps the robot-eye view rendered for the found photo
            self.sleep(POLL)

    # --- helpers ---

    def _idle(self):
        return {"state": "idle", "probs": dict(PRIORS), "checked": [], "next": None, "steps": [],
                "last_heard": dict(LAST_HEARD), "source": "placeholder", "text": ""}

    def _check(self, run):
        if run["stop"] or self.run is not run:
            raise Stopped()

    def _robot(self, text):
        if self.robot is not None:
            self.robot.command(text)

    def _robot_state(self):
        return self.robot.state() if self.robot is not None else {"mode": "idle", "x": 0.0, "y": 0.0, "yaw_deg": 0.0}

    def _say(self, text):
        """The message as MP3 under a new id, or None if the voice service is off or fails (the check-in goes on)."""
        try:
            mp3 = self.speak(text)
        except Exception:
            mp3 = None
        if not mp3:
            return None
        sound_id = secrets.token_hex(8)
        with self.lock:
            self.sounds[sound_id] = mp3
            while len(self.sounds) > KEEP_AUDIO:
                self.sounds.popitem(last=False)
        return "/api/knock/audio/{}.mp3".format(sound_id)

    def _steps(self, text):
        with self.lock:
            return self.view["steps"] + [{"text": text, "time": self.stamp()}]

    def _publish(self, view, run=None):
        with self.lock:
            if run is not None:
                self._check(run)
            self.view = view
        self.hub.publish(dict(view, type="knock"))

    def _update(self, run=None, **changes):
        """Change the card. "audio" goes only in this one event, so each sound plays once."""
        audio = changes.pop("audio", None)
        with self.lock:
            if run is not None:
                self._check(run)
            self.view = dict(self.view, **changes)
            view = self.view
        self.hub.publish(dict(view, type="knock", **({"audio": audio} if audio else {})))

    def _end(self, run, **changes):
        with self.lock:
            if self.run is run:
                self.run = None
        self._update(None, **changes)

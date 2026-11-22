"""metacon narrates the search for Margarete: one short spoken line at each step, on every viewer's page.

The search (app/apartment.py) calls `Narrator.event(kind, facts)` with plain facts; that only puts them on a queue,
so the search never waits. One worker thread then, for each event:

1. builds a template sentence from the facts (always correct, first person, round percentages);
2. asks metacon to say the same in one short spoken sentence (llm.complete, 6 s at most). The reply is used only if
   it is short, names the same zones as the template and no other, keeps its numbers and does not change the
   outcome. Otherwise, or on any error, or when the gateway is busy with chats (not waited for), the template is used;
3. publishes {"type": "narration", "id", "kind", "text", "source": "metacon" | "template"} on the Hub at once;
4. on a second thread (so text never waits for speech), when the voice service is up and nobody is on a voice
   call, gets the MP3 (short wait for the service), keeps the last few, and publishes the same id again with
   "audio": "/api/narration/audio/<id>.mp3". When lines wait for speech, the oldest that is not an ending is skipped.

Kinds and their facts:
    start        name, top [[zone, p], ...] (the three most likely)
    heading      zone, p, look (1 for the first look)
    miss         zone, belief {class: p} after the update, next (zone or None), next_p
    found        name, zone, looks, metres
    nobody_home  name, looks, metres, p_out
    gave_up      name, looks, metres
    stopped      name
    failed       name, reason
    fallback     zone (where the robot is walking), reason: the student handed the walk to A* + the a-to-b driver

If the worker falls behind, the oldest waiting "heading" lines are dropped; the others are never dropped, and while
more lines wait it uses the template without asking metacon, to catch up.
"""

import re
import secrets
import threading
import time
from collections import OrderedDict, deque

LABELS = {"bedroom": "bedroom", "living_room": "living room", "dining_area": "dining area", "kitchen": "kitchen",
          "hallway": "hallway"}
KINDS = ("start", "heading", "miss", "found", "nobody_home", "gave_up", "stopped", "failed", "fallback")
DROPPABLE = ("heading",)
FINAL = ("found", "nobody_home", "gave_up", "failed")  # always spoken
MAX_UNSPOKEN = 2
MAX_PENDING = 3  # waiting lines before the oldest droppable one goes
MAX_WORDS = 30  # a longer metacon reply is not used
LLM_TIMEOUT = 6.0
VOICE_WAIT = 3.0  # seconds to wait for the voice service when it is busy
KEEP_AUDIO = 6
REPLAY = 120.0  # seconds a line is shown to a page that opens after it
NEGATION = re.compile(r"\b(not|no|nobody|isn't|wasn't|isnt|wasnt|empty|nowhere|couldn't|can't|cannot|didn't)\b")

# Without "do not copy the draft" qwen3:8b returned the draft unchanged; without the "Not in the kitchen" hint it said
# "I'm not in the bedroom" (live check, 2026-11-22).
SYSTEM = """You are the voice of JetHexa, a small home-care robot searching a flat for {name}, an elderly woman, to \
check she is all right. You get what just happened and a plain draft sentence. Rewrite the draft in your own words as \
ONE short, warm, natural spoken sentence, at most 25 words. Do not copy the draft word for word. Speak as the robot \
(I, me) and call {name} she or by her name: "Not in the kitchen" means {name} is not in the kitchen. Keep every room \
and every number from the draft; add no other room, number, guess or advice. Plain words only, no lists, no emoji, \
no quotes. Answer with the sentence alone.
/no_think"""


def pct(p):
    return "{:.0f}%".format(100 * float(p))


def room(zone):
    return LABELS.get(zone, str(zone).replace("_", " "))


def template(kind, facts):
    """The sentence for an event, from its facts alone. None for an event that needs no line."""
    name = facts.get("name") or "Margarete"
    if kind == "start":
        top = [(z, p) for z, p in facts.get("top", []) if z in LABELS or z == "out"]
        top = top[:1] + [(z, p) for z, p in top[1:] if p >= 0.05]  # "the dining area (0%)" is no help
        if not top:
            return "I'm starting to look for {}.".format(name)
        parts = ["the {} ({})".format(room(z), pct(p)) if z != "out" else "out ({})".format(pct(p)) for z, p in top[:3]]
        where = ("in " + parts[0]) if top[0][0] != "out" else parts[0]
        rest = ", then " + " or ".join(parts[1:]) if len(parts) > 1 else ""
        return "I'm looking for {}. She is most likely {}{}.".format(name, where, rest)
    if kind == "heading":
        if int(facts.get("look", 1)) <= 1:
            return "I'm heading to the {} first.".format(room(facts["zone"]))
        return "I'm heading to the {}, {} likely.".format(room(facts["zone"]), pct(facts["p"]))
    if kind == "miss":
        p = facts.get("next_p")
        if facts.get("next") and p is None:
            p = (facts.get("belief") or {}).get(facts["next"], 0.0)
        if facts.get("next") == facts["zone"]:  # the look can miss her, so the same zone can still be the best bet
            return "I didn't see her in the {}, but it is still the most likely at {}, so I'll look again.".format(
                room(facts["zone"]), pct(p))
        text = "Not in the {}.".format(room(facts["zone"]))
        if facts.get("next"):
            text += " The {} is now most likely at {}, so I'm heading there.".format(room(facts["next"]), pct(p))
        return text
    if kind == "found":
        looks, metres = int(facts.get("looks", 1)), float(facts.get("metres", 0.0))
        walked = " and about {:.0f} metre{}".format(metres, "" if round(metres) == 1 else "s") if metres >= 0.5 else ""
        return "I found {} in the {}, after {} look{}{}.".format(
            name, room(facts["zone"]), looks, "" if looks == 1 else "s", walked)
    if kind == "nobody_home":
        return "I've looked in the likely rooms and nobody is home: I'm {} sure {} is out.".format(
            pct(facts.get("p_out", 0.0)), name)
    if kind == "gave_up":
        looks = int(facts.get("looks", 0))
        return "I've looked {} time{} and haven't found {}, so I'm stopping here.".format(
            looks, "" if looks == 1 else "s", name)
    if kind == "stopped":
        return "The search is stopped."
    if kind == "fallback":
        where = " to the {}".format(room(facts["zone"])) if facts.get("zone") in LABELS else ""
        if "stuck" in str(facts.get("reason", "")):
            return "Something is in my way, so I'm taking the mapped route{} instead.".format(where)
        return "I'm taking the mapped route{} for the rest of this walk.".format(where)
    if kind == "failed":
        reason = str(facts.get("reason") or "something went wrong").rstrip(".")
        return "I had to stop the search: {}.".format(reason)
    return None


def zones_in(text):
    """The zones a sentence names (by their labels), as a set."""
    low = " " + re.sub(r"[^a-z ]+", " ", text.lower()) + " "
    low = re.sub(r"\s+", " ", low)
    return {z for z, label in LABELS.items() if " {} ".format(label) in low or " {}s ".format(label) in low}


def numbers_in(text):
    return set(re.findall(r"\d+", text))


def acceptable(kind, reply, template_text):
    """True if metacon's reply says the same as the template: short, the same zones and no others, no new numbers,
    and the same outcome (found / not found)."""
    if not reply or len(reply.split()) > MAX_WORDS or "\n" in reply.strip():
        return False
    if re.search(r"\bi(?:'m| am) not in\b", reply.lower()):
        return False  # the robot is not the one missing
    if zones_in(reply) != zones_in(template_text):
        return False
    if not numbers_in(reply) <= numbers_in(template_text):
        return False
    low = reply.lower()
    if kind == "found":
        return not NEGATION.search(low) and ("found" in low or "here" in low or "see" in low)
    for match in re.finditer(r"\bfound\b|\bthere she is\b", low):
        before = low[max(0, match.start() - 8):match.start()]
        if "not " not in before and "n't " not in before:
            return False  # says she was found when she was not
    if kind == "miss":
        return bool(NEGATION.search(low))
    return True


def clean(reply):
    """A reply as one spoken line: no quotes, markdown or surrounding whitespace."""
    text = re.sub(r"[*_#`>|\"“”]+", "", reply or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Narrator:
    """One worker thread turns search events into lines (see the module docstring). Every outside call is injected:

    hub        call.Hub (publish)
    complete   complete(system, user) -> str: metacon (llm.complete with the key bound), or None for templates only
    speak      speak(text) -> MP3 bytes (raises or returns None when there is no voice), or None for no voice
    voice_up   () -> bool: the voice service is up
    call_busy  () -> bool: someone is on a voice call (then no speech: the call has the voice)
    chats      a semaphore shared with the chat (try-acquired, never waited for), or None
    on_line    on_line(text, source): also note the line somewhere (the Apartment log), or None
    spawn      starts the workers (text, voice); the tests pass None and call run_pending() themselves
    """

    def __init__(self, hub, complete=None, speak=None, voice_up=lambda: True, call_busy=lambda: False, chats=None,
                 on_line=None, clock=time.monotonic, spawn="thread"):
        self.hub, self.complete, self.speak = hub, complete, speak
        self.voice_up, self.call_busy, self.chats, self.on_line, self.clock = voice_up, call_busy, chats, on_line, clock
        self.cond = threading.Condition()
        self.pending = deque()
        self.unspoken = deque()  # lines waiting for the voice (its own thread, so text never waits for speech)
        self.sounds = OrderedDict()
        self.last = None  # (time, event) of the latest line, for pages that open after it
        self.announced = None  # the zone the last "miss" line said it is heading to
        self.stats = {"metacon": 0, "template": 0, "dropped": 0, "audio": 0}
        if spawn == "thread":
            threading.Thread(target=self._work, name="narrator", daemon=True).start()
            threading.Thread(target=self._speak_loop, name="narrator-voice", daemon=True).start()
        elif spawn is not None:
            spawn(self._work)
            spawn(self._speak_loop)

    # --- what the search and the server call ---

    def event(self, kind, facts):
        """Queue an event. Never blocks, never raises."""
        try:
            facts = dict(facts or {})
            with self.cond:
                if kind == "heading" and int(facts.get("look", 1)) > 1 and facts.get("zone") == self.announced:
                    self.announced = None  # the "not in the kitchen ... heading to the bedroom" line said it
                    return
                if kind == "miss":
                    self.announced = facts.get("next")
                elif kind in ("start", "found", "nobody_home", "gave_up", "stopped", "failed"):
                    self.announced = None
                self.pending.append((kind, facts))
                while len(self.pending) > MAX_PENDING:
                    old = next((i for i, (k, _) in enumerate(self.pending) if k in DROPPABLE), None)
                    if old is None:
                        break
                    del self.pending[old]
                    self.stats["dropped"] += 1
                self.cond.notify_all()
        except Exception as error:  # noqa: BLE001
            print("[narrate] event dropped: {}".format(error), flush=True)

    def audio(self, sound_id):
        with self.cond:
            return self.sounds.get(sound_id)

    def latest(self):
        """The latest line (text only, no audio) if it is recent, for a page that opens after it; else None."""
        with self.cond:
            if self.last is None or self.clock() - self.last[0] > REPLAY:
                return None
            return dict(self.last[1])

    # --- the worker ---

    def _work(self):
        while True:
            with self.cond:
                while not self.pending:
                    self.cond.wait()
                kind, facts = self.pending.popleft()
                behind = bool(self.pending)
            self._say(kind, facts, ask=not behind)

    def _speak_loop(self):
        while True:
            with self.cond:
                while not self.unspoken:
                    self.cond.wait()
                line = self.unspoken.popleft()
            self._voice(line)

    def run_pending(self):
        """Handle every waiting event, and speak its line, on this thread (tests, and spawn=None)."""
        while True:
            with self.cond:
                if not self.pending:
                    return
                kind, facts = self.pending.popleft()
                behind = bool(self.pending)
            self._say(kind, facts, ask=not behind)
            while True:
                with self.cond:
                    if not self.unspoken:
                        break
                    line = self.unspoken.popleft()
                self._voice(line)

    def _say(self, kind, facts, ask=True):
        try:
            base = template(kind, facts)
        except Exception as error:  # noqa: BLE001 - a fact missing: say nothing rather than stop the worker
            print("[narrate] no template for {}: {}".format(kind, error), flush=True)
            return
        if not base:
            return
        text, source = base, "template"
        if ask:
            reply = self._ask(kind, facts, base)
            if reply and reply.lower().rstrip(".!") != base.lower().rstrip(".!"):
                text, source = reply, "metacon"  # an unchanged draft still counts as the template
        self.stats[source] += 1
        line = {"type": "narration", "id": secrets.token_hex(8), "kind": kind, "text": text, "source": source}
        with self.cond:
            self.last = (self.clock(), line)
        self.hub.publish(line)
        if self.on_line is not None:
            try:
                self.on_line(text, source)
            except Exception as error:  # noqa: BLE001
                print("[narrate] on_line failed: {}".format(error), flush=True)
        if self.speak is not None:
            with self.cond:
                self.unspoken.append(line)
                # Speech is slower than text: when lines wait, the oldest one that is not an ending goes unspoken.
                while len(self.unspoken) > MAX_UNSPOKEN:
                    old = next((i for i, l in enumerate(self.unspoken) if l["kind"] not in FINAL), None)
                    if old is None:
                        break
                    del self.unspoken[old]
                self.cond.notify_all()

    def _ask(self, kind, facts, base):
        """metacon's wording of `base`, or None (no metacon, busy, error, timeout or a reply that does not fit)."""
        if self.complete is None:
            return None
        if self.chats is not None and not self.chats.acquire(blocking=False):
            return None  # the chat is using metacon: do not queue behind it
        try:
            user = "What happened: {} {}\nSentence: {}".format(kind.replace("_", " "), self._facts_text(kind, facts), base)
            reply = clean(self.complete(SYSTEM.format(name=facts.get("name") or "Margarete"), user))
        except Exception as error:  # noqa: BLE001 - any failure: the template is used
            print("[narrate] metacon not used ({})".format(error), flush=True)
            return None
        finally:
            if self.chats is not None:
                self.chats.release()
        return reply if acceptable(kind, reply, base) else None

    @staticmethod
    def _facts_text(kind, facts):
        """The facts as short plain text for the prompt (rooms by name, probabilities as percentages)."""
        out = []
        for key, value in facts.items():
            if key == "belief":
                value = ", ".join("{} {}".format(room(z), pct(p)) for z, p in sorted(value.items(), key=lambda i: -i[1])
                                  if z != "out" or p >= 0.01)
            elif key == "top":
                value = ", ".join("{} {}".format(room(z), pct(p)) for z, p in value)
            elif key in ("zone", "next") and value:
                value = room(value)
            elif key in ("p", "next_p", "p_out") and value is not None:
                value = pct(value)
            elif key == "metres":
                value = "{:.0f} m".format(value)
            out.append("{}: {}".format(key.replace("_", " "), value))
        return "(" + "; ".join(out) + ")"

    def _voice(self, line):
        if self.speak is None:
            return
        try:
            if not self.voice_up() or self.call_busy():
                return
            mp3 = self.speak(line["text"].replace("%", " percent"))
        except Exception as error:  # noqa: BLE001 - no voice: the text is already out
            print("[narrate] no voice for this line ({})".format(error), flush=True)
            return
        if not mp3:
            return
        with self.cond:
            self.sounds[line["id"]] = mp3
            while len(self.sounds) > KEEP_AUDIO:
                self.sounds.popitem(last=False)
        self.stats["audio"] += 1
        self.hub.publish(dict(line, audio="/api/narration/audio/{}.mp3".format(line["id"])))

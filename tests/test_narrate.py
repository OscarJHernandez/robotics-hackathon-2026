"""Tests for metacon's narration of the search (app/narrate.py) and its audio route.

No network: metacon and the voice service are fakes, and the narrator runs on the test's own thread
(spawn=None, then run_pending()).
"""

import datetime
import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import apartment  # noqa: E402
import call  # noqa: E402
import narrate  # noqa: E402

NOW = datetime.datetime(2026, 11, 17, 10, 35)  # a Tuesday morning

FACTS = {
    "start": {"name": "Margarete", "top": [["kitchen", 0.412], ["living_room", 0.22], ["bedroom", 0.151]]},
    "heading": {"zone": "kitchen", "p": 0.412, "look": 1},
    "miss": {"zone": "kitchen", "belief": {"kitchen": 0.12, "bedroom": 0.35, "out": 0.05}, "next": "bedroom",
             "next_p": 0.348},
    "found": {"name": "Margarete", "zone": "bedroom", "looks": 2, "metres": 4.3},
    "nobody_home": {"name": "Margarete", "looks": 5, "metres": 12.0, "p_out": 0.962},
    "gave_up": {"name": "Margarete", "looks": 12, "metres": 30.0},
    "stopped": {"name": "Margarete"},
    "failed": {"name": "Margarete", "reason": "the robot fell over"},
    "fallback": {"zone": "kitchen", "reason": "stuck"},
}

EXPECTED = {
    "start": "I'm looking for Margarete. She is most likely in the kitchen (41%), then the living room (22%) or the "
             "bedroom (15%).",
    "heading": "I'm heading to the kitchen first.",
    "miss": "Not in the kitchen. The bedroom is now most likely at 35%, so I'm heading there.",
    "found": "I found Margarete in the bedroom, after 2 looks and about 4 metres.",
    "nobody_home": "I've looked in the likely rooms and nobody is home: I'm 96% sure Margarete is out.",
    "gave_up": "I've looked 12 times and haven't found Margarete, so I'm stopping here.",
    "stopped": "The search is stopped.",
    "failed": "I had to stop the search: the robot fell over.",
    "fallback": "Something is in my way, so I'm taking the mapped route to the kitchen instead.",
}


def lines(hub):
    return [e for e in hub.events if e["type"] == "narration"]


def make(complete=None, speak=None, voice_up=True, call_busy=False, chats=None, on_line=None):
    hub = call.Hub()
    n = narrate.Narrator(hub, complete=complete, speak=speak, voice_up=lambda: voice_up,
                         call_busy=lambda: call_busy, chats=chats, on_line=on_line, spawn=None)
    return n, hub


def say(n, kind, facts=None):
    n.event(kind, FACTS[kind] if facts is None else facts)
    n.run_pending()


def boom(*args, **kwargs):
    raise OSError("down")


class TemplateTest(unittest.TestCase):
    def test_every_kind_has_its_sentence(self):
        for kind in narrate.KINDS:
            self.assertEqual(narrate.template(kind, FACTS[kind]), EXPECTED[kind], kind)

    def test_a_fallback_only_blames_an_obstacle_when_the_student_was_stuck(self):
        line = narrate.template("fallback", {"zone": "kitchen",
                                             "reason": "the student is unavailable (TabPFN API budget used up)"})
        self.assertEqual(line, "I'm taking the mapped route to the kitchen for the rest of this walk.")

    def test_later_heading_and_a_miss_with_no_next_zone(self):
        self.assertEqual(narrate.template("heading", {"zone": "dining_area", "p": 0.25, "look": 3}),
                         "I'm heading to the dining area, 25% likely.")
        self.assertEqual(narrate.template("miss", {"zone": "hallway", "belief": {}, "next": None}),
                         "Not in the hallway.")
        self.assertEqual(narrate.template("miss", {"zone": "bedroom", "belief": {}, "next": "bedroom", "next_p": 0.81}),
                         "I didn't see her in the bedroom, but it is still the most likely at 81%, so I'll look again.")
        self.assertEqual(narrate.template("found", {"zone": "kitchen", "looks": 1, "metres": 0.2}),
                         "I found Margarete in the kitchen, after 1 look.")  # it was there already
        self.assertEqual(narrate.template("start", {"top": [["bedroom", 0.95], ["kitchen", 0.04], ["hallway", 0.01]]}),
                         "I'm looking for Margarete. She is most likely in the bedroom (95%).")

    def test_acceptable_checks_zones_numbers_length_and_outcome(self):
        base = EXPECTED["miss"]
        self.assertTrue(narrate.acceptable("miss", "She's not in the kitchen; the bedroom looks likeliest now at "
                                                   "35%, so I'm going there.", base))
        self.assertFalse(narrate.acceptable("miss", "Not in the kitchen, I'm off to the hallway at 35%.", base))
        self.assertFalse(narrate.acceptable("miss", "Not in the kitchen, the bedroom is 40% now.", base))  # new number
        self.assertFalse(narrate.acceptable("miss", "I found her in the kitchen and the bedroom.", base))
        self.assertFalse(narrate.acceptable("miss", " ".join(["kitchen bedroom not"] * 12), base))  # too long
        self.assertFalse(narrate.acceptable("found", "She is not in the bedroom.", EXPECTED["found"]))
        self.assertTrue(narrate.acceptable("found", "Good news, I found Margarete in the bedroom after 2 looks.",
                                           EXPECTED["found"]))
        self.assertTrue(narrate.acceptable("gave_up", "After 12 looks I haven't found Margarete, so I'll stop.",
                                           EXPECTED["gave_up"]))


class NarratorTest(unittest.TestCase):
    def test_template_only_and_text_published_at_once(self):
        n, hub = make()
        for kind in narrate.KINDS:
            say(n, kind)
        got = lines(hub)
        self.assertEqual([e["kind"] for e in got], list(narrate.KINDS))
        self.assertEqual([e["text"] for e in got], [EXPECTED[k] for k in narrate.KINDS])
        self.assertTrue(all(e["source"] == "template" and "audio" not in e for e in got))

    def test_a_good_metacon_reply_is_used(self):
        asked = []

        def complete(system, user):
            asked.append(user)
            return "Not in the kitchen, so I'm off to the bedroom, now 35% likely."
        n, hub = make(complete=complete)
        say(n, "miss")
        self.assertEqual((lines(hub)[0]["source"], lines(hub)[0]["text"]),
                         ("metacon", "Not in the kitchen, so I'm off to the bedroom, now 35% likely."))
        self.assertIn(EXPECTED["miss"], asked[0])
        self.assertIn("bedroom 35%", asked[0])

    def test_wrong_long_or_failing_replies_fall_back_to_the_template(self):
        for reply in ["Not in the kitchen, heading to the hallway now.", "Not in the kitchen. " + "Hmm " * 40,
                      "", "Found her in the kitchen!", "I'm not in the kitchen, so the bedroom at 35% is next.",
                      EXPECTED["miss"]]:
            n, hub = make(complete=lambda s, u, r=reply: r)
            say(n, "miss")
            self.assertEqual((lines(hub)[0]["source"], lines(hub)[0]["text"]), ("template", EXPECTED["miss"]), reply)
        n, hub = make(complete=boom)
        say(n, "found")
        self.assertEqual((lines(hub)[0]["source"], lines(hub)[0]["text"]), ("template", EXPECTED["found"]))

    def test_a_busy_gateway_is_not_waited_for(self):
        chats = threading.BoundedSemaphore(1)
        chats.acquire()  # a chat holds metacon
        n, hub = make(complete=lambda s, u: self.fail("metacon asked while busy"), chats=chats)
        say(n, "heading")
        self.assertEqual(lines(hub)[0]["source"], "template")
        chats.release()
        n, hub = make(complete=lambda s, u: "I'm going to the kitchen first.", chats=chats)
        say(n, "heading")
        self.assertEqual(lines(hub)[0]["source"], "metacon")
        self.assertTrue(chats.acquire(blocking=False))  # released again

    def test_audio_follows_with_the_same_id(self):
        spoken = []
        n, hub = make(speak=lambda text: spoken.append(text) or b"MP3")
        say(n, "miss")
        first, second = lines(hub)
        self.assertEqual(first["id"], second["id"])
        self.assertNotIn("audio", first)
        self.assertEqual(second["audio"], "/api/narration/audio/{}.mp3".format(first["id"]))
        self.assertEqual(n.audio(first["id"]), b"MP3")
        self.assertIn("35 percent", spoken[0])

    def test_voice_down_or_a_call_gives_text_only(self):
        for kwargs in ({"voice_up": False}, {"call_busy": True}, {"speak": boom}, {"speak": lambda t: None}):
            n, hub = make(**dict({"speak": lambda t: b"MP3"}, **kwargs))
            say(n, "found")
            self.assertEqual(len(lines(hub)), 1, kwargs)
            self.assertNotIn("audio", lines(hub)[0])

    def test_only_the_last_few_sounds_are_kept(self):
        n, hub = make(speak=lambda t: b"MP3")
        for _ in range(narrate.KEEP_AUDIO + 2):
            say(n, "stopped")
        ids = [e["id"] for e in lines(hub) if "audio" in e]
        self.assertIsNone(n.audio(ids[0]))
        self.assertEqual(n.audio(ids[-1]), b"MP3")

    def test_falling_behind_drops_old_heading_lines_but_never_found(self):
        n, hub = make()
        for look in range(1, 6):
            n.event("heading", {"zone": "kitchen", "p": 0.4, "look": 1})
        n.event("found", FACTS["found"])
        n.event("nobody_home", FACTS["nobody_home"])
        n.run_pending()
        kinds = [e["kind"] for e in lines(hub)]
        self.assertIn("found", kinds)
        self.assertIn("nobody_home", kinds)
        self.assertLessEqual(len(kinds), narrate.MAX_PENDING)
        self.assertGreater(n.stats["dropped"], 0)

    def test_while_behind_metacon_is_not_asked(self):
        asked = []
        n, hub = make(complete=lambda s, u: asked.append(u) or "I found Margarete in the bedroom after 2 looks.")
        n.event("miss", FACTS["miss"])
        n.event("found", FACTS["found"])
        n.run_pending()
        self.assertEqual([e["source"] for e in lines(hub)], ["template", "metacon"])
        self.assertEqual(len(asked), 1)

    def test_heading_after_a_miss_that_named_it_is_not_repeated(self):
        n, hub = make()
        n.event("miss", FACTS["miss"])
        n.event("heading", {"zone": "bedroom", "p": 0.35, "look": 2})
        n.event("heading", {"zone": "hallway", "p": 0.2, "look": 3})  # not announced: said
        n.run_pending()
        self.assertEqual([e["kind"] for e in lines(hub)], ["miss", "heading"])

    def test_speech_runs_on_its_own_thread_and_skips_old_lines_but_never_endings(self):
        entered, gate = threading.Event(), threading.Event()

        def speak(text):
            entered.set()
            gate.wait(5)
            return b"MP3"
        n = narrate.Narrator(call.Hub(), speak=speak)  # real threads
        n.event("start", FACTS["start"])
        self.assertTrue(entered.wait(5))  # the voice is busy with the first line...
        for kind in ("miss", "stopped", "found"):
            n.event(kind, FACTS[kind])
        for _ in range(100):  # ...and every text still comes out
            if len(lines(n.hub)) >= 4:
                break
            time.sleep(0.05)
        self.assertEqual([e["kind"] for e in lines(n.hub)], ["start", "miss", "stopped", "found"])
        gate.set()
        for _ in range(100):
            if any("audio" in e and e["kind"] == "found" for e in lines(n.hub)):
                break
            time.sleep(0.05)
        # three lines waited for the voice: the oldest that is not an ending was skipped
        self.assertEqual([e["kind"] for e in lines(n.hub) if "audio" in e], ["start", "stopped", "found"])

    def test_latest_is_the_recent_line_without_audio(self):
        n, hub = make(speak=lambda t: b"MP3")
        self.assertIsNone(n.latest())
        say(n, "found")
        self.assertEqual(n.latest()["text"], EXPECTED["found"])
        self.assertNotIn("audio", n.latest())
        n.clock = lambda: 1e9
        self.assertIsNone(n.latest())

    def test_event_never_raises(self):
        n, hub = make()
        n.event("heading", None)
        n.event("found", {"zone": "bedroom"})  # missing facts get defaults
        n.event("miss", {})  # no zone: the template fails, nothing is said
        n.run_pending()
        self.assertEqual([e["kind"] for e in lines(hub)], ["found"])


class SearchTest(unittest.TestCase):
    def test_the_search_finishes_when_metacon_and_voice_both_fail(self):
        n, hub = make(complete=boom, speak=boom)
        a = apartment.Apartment(seed=0, now=lambda: NOW, sleep=lambda s: None, walk=0.0, look=0.0, on_event=n.event)
        n.on_line = lambda text, source: a.note(text)
        self.assertTrue(a.start("bedroom", miss_rate=0.0))
        a._thread.join(timeout=20)
        self.assertEqual(a.state()["search"]["status"], "found")
        n.run_pending()
        got = lines(hub)
        self.assertEqual(got[0]["kind"], "start")
        self.assertEqual(got[-1]["kind"], "found")
        self.assertIn("bedroom", got[-1]["text"])
        self.assertTrue(all(e["source"] == "template" and "audio" not in e for e in got))
        self.assertEqual([l["text"] for l in a.state()["log"] if l["kind"] == "metacon"], [e["text"] for e in got])

    def test_an_on_event_that_raises_does_not_stop_the_search(self):
        a = apartment.Apartment(seed=0, now=lambda: NOW, sleep=lambda s: None, walk=0.0, look=0.0, on_event=boom)
        a.start("kitchen", miss_rate=0.0)
        a._thread.join(timeout=20)
        self.assertEqual(a.state()["search"]["status"], "found")


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
        self.n, self.hub = make(speak=lambda t: b"MP3")
        for p in [mock.patch.object(self.server, "HUB", self.hub), mock.patch.object(self.server, "NARRATOR", self.n),
                  mock.patch.object(self.server, "KNOCK", None)]:
            p.start()
            self.addCleanup(p.stop)

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as response:
                return response.status, response.headers.get("Content-Type"), response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers.get("Content-Type"), error.read()

    def test_audio_route_serves_the_cached_mp3(self):
        say(self.n, "found")
        audio = [e["audio"] for e in lines(self.hub) if "audio" in e][0]
        self.assertEqual(self.get(audio), (200, "audio/mpeg", b"MP3"))
        self.assertEqual(self.get("/api/narration/audio/nope.mp3")[0], 404)
        with mock.patch.object(self.server, "NARRATOR", None):
            self.assertEqual(self.get(audio)[0], 404)

    def test_health_says_whether_narration_is_on(self):
        self.assertTrue(json.loads(self.get("/api/health")[2])["narration"])
        with mock.patch.object(self.server, "NARRATOR", None):
            self.assertFalse(json.loads(self.get("/api/health")[2])["narration"])

    def test_live_starts_with_the_latest_line(self):
        say(self.n, "miss")
        response = urllib.request.urlopen(self.base + "/api/live", timeout=10)
        try:
            data = [response.readline() for _ in range(4)]
        finally:
            response.close()
        first = json.loads([d for d in data if d.startswith(b"data:")][0][5:])
        self.assertEqual((first["type"], first["text"]), ("narration", EXPECTED["miss"]))
        self.assertNotIn("audio", first)


if __name__ == "__main__":
    unittest.main()

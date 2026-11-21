"""Tests for voice calls in the web app: the Voice Control client, the call line and hub, and the HTTP routes.

No network: urlopen and the voice functions are mocked, and the server runs on a free local port.
"""

import io
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
import llm  # noqa: E402
import server  # noqa: E402
import voice  # noqa: E402


class Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class VoiceClientTest(unittest.TestCase):
    def test_transcribe_sends_multipart_with_key(self):
        sent = {}

        def urlopen(request, timeout):
            sent["request"] = request
            return Reply(b'{"text": "  walk to the pillar "}')

        with mock.patch.dict("os.environ", {"VOICE_CONTROL_API_KEY": "k"}), \
                mock.patch("urllib.request.urlopen", urlopen):
            text = voice.transcribe(b"AUDIO", "audio/webm;codecs=opus")
        self.assertEqual(text, "walk to the pillar")
        request = sent["request"]
        self.assertTrue(request.full_url.endswith("/v1/audio/transcriptions"))
        self.assertEqual(request.get_header("Authorization"), "Bearer k")
        self.assertIn("multipart/form-data; boundary=", request.get_header("Content-type"))
        self.assertIn(b'filename="speech.webm"', request.data)
        self.assertIn(b'name="language"', request.data)
        self.assertIn(b"AUDIO", request.data)

    def test_no_key_no_header(self):
        sent = {}

        def urlopen(request, timeout):
            sent["request"] = request
            return Reply(b'{"text": "hi"}')

        with mock.patch.dict("os.environ", {"VOICE_CONTROL_API_KEY": ""}), \
                mock.patch("urllib.request.urlopen", urlopen):
            voice.transcribe(b"x", "audio/mp4")
        self.assertIsNone(sent["request"].get_header("Authorization"))
        self.assertIn(b'filename="speech.m4a"', sent["request"].data)

    def test_unknown_type(self):
        for ctype in ("", "text/plain", "video/mp4"):
            with self.assertRaises(ValueError):
                voice.extension(ctype)

    def test_busy_live_caption_is_skipped(self):
        with voice.LOCK, mock.patch("urllib.request.urlopen") as urlopen:
            self.assertIsNone(voice.transcribe(b"x", "audio/webm", wait=False))
            urlopen.assert_not_called()

    def test_speak_uses_named_voice_and_cuts_text(self):
        sent = []

        def urlopen(request, timeout):
            sent.append(request)
            if request.full_url.endswith("/v1/voices"):
                return Reply(json.dumps([{"voice_id": "aiden", "display_name": "Aiden"},
                                         {"voice_id": "voice-123", "display_name": "Glados-portal"}]).encode())
            return Reply(b"MP3")

        with mock.patch.dict(voice._voice, {"resolved": False, "id": None}), \
                mock.patch.object(voice, "VOICE", "glados-portal"), mock.patch("urllib.request.urlopen", urlopen):
            self.assertEqual(voice.speak("x" * 2000), b"MP3")
        payload = json.loads(sent[-1].data)
        self.assertEqual(payload["voice"], "voice-123")
        self.assertEqual(len(payload["input"]), voice.MAX_SPEAK)
        self.assertEqual(payload["response_format"], "mp3")
        self.assertNotIn("speed", payload)

    def test_errors(self):
        def http(code):
            return urllib.error.HTTPError("u", code, "x", {}, None)

        self.assertIn("understand", voice.describe_error(http(400)))
        self.assertIn("key", voice.describe_error(http(401)))
        self.assertIn("busy", voice.describe_error(http(503)))
        self.assertIn("reach", voice.describe_error(urllib.error.URLError("down")))

    def test_spoken_text(self):
        self.assertEqual(llm.spoken("Going **now**.\n[[goto 1 0.5]]\n"), "Going now.")
        self.assertIn("by voice", llm.system_prompt({}, voice=True))
        self.assertNotIn("by voice", llm.system_prompt({}, voice=False))


class LineTest(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.hub = call.Hub()
        self.line = call.Line(self.hub, clock=lambda: self.now)

    def test_one_caller_at_a_time(self):
        first = self.line.claim()
        self.assertTrue(first)
        self.assertIsNone(self.line.claim())
        self.assertTrue(self.line.active(first))
        self.assertFalse(self.line.active("someone-else"))
        self.assertTrue(self.line.end(first))
        self.assertTrue(self.line.claim())

    def test_idle_call_expires_and_viewers_see_it_end(self):
        first = self.line.claim()
        self.now += call.IDLE + 1
        self.assertFalse(self.line.busy())
        events, _ = self.hub.since(0, timeout=0)
        self.assertEqual([(e["type"], e.get("state")) for e in events], [("call", "start"), ("call", "end")])
        self.assertFalse(self.line.active(first))

    def test_live_captions_are_capped(self):
        first = self.line.claim()
        for _ in range(call.MAX_INTERIM):
            self.assertTrue(self.line.active(first, interim=True))
        self.assertFalse(self.line.active(first, interim=True))
        self.assertTrue(self.line.active(first))  # the final transcription still goes through

    def test_late_viewer_gets_the_call_so_far(self):
        self.hub.publish({"type": "call", "state": "start", "call": "old"})
        self.hub.publish({"type": "call", "state": "end", "call": "old"})
        self.assertEqual(self.hub.current()[0], [])
        self.hub.publish({"type": "call", "state": "start", "call": "c"})
        self.hub.publish({"type": "heard", "call": "c", "text": "hi", "final": True})
        events, seq = self.hub.current()
        self.assertEqual([e["type"] for e in events], ["call", "heard"])
        self.assertEqual(seq, 4)


class RoutesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.httpd.daemon_threads = True
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = "http://127.0.0.1:{}".format(cls.httpd.server_address[1])

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def setUp(self):
        hub = call.Hub()
        patches = [mock.patch.object(server, "HUB", hub), mock.patch.object(server, "LINE", call.Line(hub)),
                   mock.patch.object(server, "HITS", server.defaultdict(server.deque)),
                   mock.patch.object(server.voice, "available", return_value=True),
                   mock.patch.object(server.voice, "transcribe", return_value="turn around"),
                   mock.patch.object(server.voice, "speak", return_value=b"MP3DATA")]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.hub = hub

    def post(self, path, body, ctype="application/json"):
        data = json.dumps(body).encode() if ctype == "application/json" else body
        request = urllib.request.Request(self.base + path, data=data, method="POST", headers={"Content-Type": ctype})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.headers.get("Content-Type"), response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers.get("Content-Type"), error.read()

    def test_call_round_trip(self):
        status, _, body = self.post("/api/call/start", {})
        self.assertEqual(status, 200)
        call_id = json.loads(body)["call"]
        self.assertEqual(self.post("/api/call/start", {})[0], 409)

        status, _, body = self.post("/api/transcribe?call={}&final=1".format(call_id), b"AUDIO", "audio/webm")
        self.assertEqual((status, json.loads(body)), (200, {"text": "turn around"}))

        status, ctype, body = self.post("/api/speak", {"text": "Turning. [[turn 180]]", "call": call_id})
        self.assertEqual((status, ctype, body), (200, "audio/mpeg", b"MP3DATA"))
        server.voice.speak.assert_called_with("Turning.")

        self.assertEqual(self.post("/api/call/end", {"call": call_id})[0], 200)
        events, _ = self.hub.since(0, timeout=0)
        self.assertEqual([e["type"] for e in events], ["call", "heard", "speaking", "call"])
        self.assertEqual(events[-1]["state"], "end")

    def test_needs_a_call(self):
        self.assertEqual(self.post("/api/transcribe?call=nope&final=1", b"AUDIO", "audio/webm")[0], 409)
        self.assertEqual(self.post("/api/speak", {"text": "hi", "call": "nope"})[0], 409)

    def test_bad_audio(self):
        call_id = json.loads(self.post("/api/call/start", {})[2])["call"]
        url = "/api/transcribe?call={}&final=1".format(call_id)
        self.assertEqual(self.post(url, b"x" * (server.MAX_AUDIO + 1), "audio/webm")[0], 400)
        self.assertEqual(self.post(url, b"AUDIO", "text/plain")[0], 400)
        with mock.patch.object(server.voice, "transcribe", return_value=""):
            self.assertEqual(self.post(url, b"AUDIO", "audio/webm")[0], 400)

    def test_voice_offline(self):
        with mock.patch.object(server.voice, "available", return_value=False):
            self.assertEqual(self.post("/api/call/start", {})[0], 503)

    def test_rate_limit(self):
        count, _ = server.LIMITS["call"]
        for _ in range(count):
            status, _, body = self.post("/api/call/start", {})
            if status == 200:
                self.post("/api/call/end", {"call": json.loads(body)["call"]})
        self.assertEqual(self.post("/api/call/start", {})[0], 429)


if __name__ == "__main__":
    unittest.main()

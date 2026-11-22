"""Web app: the live JetHexa simulation plus a chat with a metacon LLM that can drive it.

    uv run --group app python app/server.py [--port 8790]

One simulated robot is shared by everyone who opens the page. The front end (app/web, built into
app/web/dist with `just app-build`) is served from /, the API from /api:

    GET  /api/health                 {"ok": true, "model": ..., "llm": true/false}
    GET  /api/state                  robot position, heading, mode and target
    GET  /api/views                  {"views": ["chase", "orbit", "overhead", "eye"]}
    GET  /api/stream/<view>          MJPEG stream (use as <img src>): chase (behind the robot), orbit
                                     (three-quarter view), overhead (whole floor), eye (robot camera)
    GET  /api/frame/<view>.jpg       one JPEG
    POST /api/command  {"text": "goto 1 0.5"}            -> {"ok": true, "command": "goto 1.00 0.50"}
    POST /api/chat     {"messages": [{"role", "content"}], "call": id?} -> text/event-stream, events:
         {"type": "token", "text": "..."}  piece of the answer
         {"type": "command", "text": "goto 1 0.5", "ok": true, "error": null}  a command the model ran
         {"type": "error", "text": "..."}  {"type": "done"}
         With the id of the current voice call, the model answers for speech and everyone sees it on /api/live.

Voice calls (app/call.py; speech by Voice Control through app/voice.py), one caller at a time:
    POST /api/call/start                          -> {"call": id}, or 409 while someone else is on the line
    POST /api/call/end    {"call": id}
    POST /api/transcribe?call=ID&final=0|1        body: the recording (audio/webm, audio/mp4, ...) -> {"text"}
                          final=0 is a live caption of the recording so far; skipped ({"text": null}) when busy
    POST /api/speak       {"text": ..., "call": id} -> audio/mpeg in the robot's voice
    GET  /api/live        text/event-stream of the call for every viewer (events listed in app/call.py),
                          and of the Knock check-in (app/knock.py)

Knock, the robot checks on Margarete (app/knock.py, docs/knock-ui.md), one check-in at a time:
    POST /api/knock/call    {}  call her through the robot, then search  -> {"ok": "started" | "again"}, or 409
    POST /api/knock/find    {}  search now                               -> {"ok": "started" | "skip"}, or 409
    POST /api/knock/stop    {}  stop (the robot stops too), or close the found card
    POST /api/knock/choice  {"choice": "call_112" | "call_neighbour" | "talk"}  simulated calls
    GET  /api/knock/audio/<id>.mp3  the call-out in the robot's voice, for every viewer

The apartment (app/apartment.py): the flat's zones and obstacles, the journal, the belief and the search loop:
    GET  /api/apartment                 everything the Apartment panel shows (polled): flat, habits, journal,
                                        belief step by step, search steps, robot pose, log
    POST /api/apartment/journal         {"zone": "kitchen", "time": "09:30"?, "days_ago": 0?}  add an entry
    POST /api/apartment/journal/remove  {"id": 12}
    POST /api/apartment/journal/import  {"text": "Mon 09:30 kitchen\\n...", "replace": true}  a pasted journal:
                                        one record per line (day, time, zone); see apartment.parse_journal
    GET  /api/apartment/journal.txt     the journal as text in that format (to copy, edit and paste back)
    GET  /api/apartment/inputs.json     both inputs of the belief as one object: {"routines": {...}, "journal": [...]}
    POST /api/apartment/inputs          {"text": "<that JSON, edited>"}  routines replace her stated habits, a
                                        journal replaces her journal; the belief follows at once
    POST /api/apartment/clock           {"weekday": 0-6?, "time": "10:35"?}  the check-in time; {} = the real time
    POST /api/apartment/search          {"where": "routine" | "random" | zone | "out", "miss_rate": 0.2?}  start, or 409
                                        ("random": she is in a random zone, at a random spot in it)
    POST /api/apartment/stop            {}
    POST /api/apartment/clear           {}  ready for another run: no search, Margarete out, the robot on its dock
    POST /api/apartment/reset           {"seed": 3?}  a new random flat and resident

    --scene apartment (default): the robot lives in the furnished flat and the search walks it from zone to zone.
    --scene floor: the robot alone on its floor, as before; the search then moves a marker on the plan only.
    --driver tabpfn (default) | planner | rule: who walks each leg in the apartment (app/robot.py).
    --belief tabpfn (default) | counting: who says where she probably is (homes/tabpfn_belief.py, homes/search.py).
"""

import argparse
import json
import mimetypes
import sys
import threading
import time
import urllib.parse
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm  # noqa: E402
import voice  # noqa: E402
from call import Hub, Line  # noqa: E402
from apartment import Apartment  # noqa: E402
from knock import Knock  # noqa: E402
from robot import HELP, VIEWS, LiveRobot  # noqa: E402

WEB = Path(__file__).resolve().parent / "web" / "dist"
# Requests per window (seconds), per visitor. Transcriptions are also capped per call (call.MAX_INTERIM).
LIMITS = {"chat": (20, 600), "command": (60, 60), "call": (20, 600), "transcribe": (300, 600), "speak": (30, 600),
          "knock": (60, 600), "apartment": (240, 60)}
MAX_STREAMS = 40  # open picture streams across all visitors
MAX_LISTENERS = 200  # open /api/live streams across all visitors
MAX_BODY = 64 * 1024
MAX_AUDIO = 1536 * 1024  # bytes of one recording: about 15 s at the bit rates browsers use, with room to spare
PING = 15.0  # seconds between keep-alive comments on /api/live (the tunnel drops silent connections)

ROBOT = None
KEY = ""
HUB = Hub()
LINE = Line(HUB)
KNOCK = None  # Knock, made in main() once the robot runs
APARTMENT = None  # Apartment, made in main()
SCENE = "floor"
HITS = defaultdict(deque)
HITS_LOCK = threading.Lock()
STREAMS = threading.BoundedSemaphore(MAX_STREAMS)
LISTENERS = threading.BoundedSemaphore(MAX_LISTENERS)
CHATS = threading.BoundedSemaphore(2)  # metacon runs one model; do not queue up many answers
POST_ROUTES = {"/api/command": "command", "/api/chat": "chat", "/api/call/start": "call", "/api/call/end": None,
               "/api/transcribe": "transcribe", "/api/speak": "speak", "/api/knock/call": "knock",
               "/api/knock/find": "knock", "/api/knock/stop": None, "/api/knock/choice": "knock",
               "/api/apartment/journal": "apartment", "/api/apartment/journal/remove": "apartment",
               "/api/apartment/journal/import": "apartment", "/api/apartment/inputs": "apartment",
               "/api/apartment/clock": "apartment", "/api/apartment/search": "apartment",
               "/api/apartment/stop": None, "/api/apartment/reset": "apartment", "/api/apartment/clear": "apartment"}


def sse(payload):
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def allowed(kind, who):
    count, window = LIMITS[kind]
    now = time.monotonic()
    with HITS_LOCK:
        hits = HITS[(kind, who)]
        while hits and now - hits[0] > window:
            hits.popleft()
        if len(hits) >= count:
            return False
        hits.append(now)
        return True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "jethexa-app"

    def visitor(self):
        # Behind the Cloudflare tunnel every request comes from 127.0.0.1; Cloudflare names the real one.
        return self.headers.get("Cf-Connecting-Ip") or self.client_address[0]

    def send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_body(self, limit):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > limit:
            if limit < length <= 4 * limit:  # read it anyway, so the sender gets the answer, not a broken pipe
                while length > 0:
                    chunk = self.rfile.read(min(length, 65536))
                    if not chunk:
                        break
                    length -= len(chunk)
            elif length > limit:
                self.close_connection = True  # the unread body must not be taken for the next request
            raise ValueError("body missing or too large")
        return self.rfile.read(length)

    def read_json(self):
        body = json.loads(self.read_body(MAX_BODY))
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
        return body

    def send_events_head(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/health":
            self.send_json(200, {"ok": True, "model": llm.MODEL, "llm": bool(KEY), "commands": HELP,
                                 "views": list(VIEWS), "voice": voice.available(), "scene": SCENE,
                                 "apartment": APARTMENT is not None})
        elif path == "/api/apartment/inputs.json" and APARTMENT is not None:
            self.send_json(200, APARTMENT.inputs())
        elif path == "/api/apartment/journal.txt" and APARTMENT is not None:
            body = APARTMENT.journal_text().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/apartment":
            if APARTMENT is None:
                self.send_json(503, {"error": "the apartment is not running"})
            else:
                self.send_json(200, APARTMENT.state())
        elif path == "/api/live":
            self.live()
        elif path == "/api/views":
            self.send_json(200, {"views": list(VIEWS)})
        elif path == "/api/state":
            self.send_json(200, ROBOT.state())
        elif path.startswith("/api/frame/") and path.endswith(".jpg") and path[11:-4] in VIEWS:
            body = ROBOT.frame(path[11:-4])
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path.startswith("/api/stream/") and path[12:] in VIEWS:
            self.stream(path[12:])
        elif path.startswith("/api/knock/audio/") and path.endswith(".mp3"):
            audio = KNOCK.audio(path[17:-4]) if KNOCK else None
            if audio is None:
                self.send_json(404, {"error": "no such sound"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Cache-Control", "private, max-age=600")
            self.send_header("Content-Length", str(len(audio)))
            self.end_headers()
            self.wfile.write(audio)
        elif path.startswith("/api/"):
            self.send_json(404, {"error": "not found"})
        else:
            self.static(path)

    def do_POST(self):
        path, _, query = self.path.partition("?")
        if path not in POST_ROUTES:
            self.send_json(404, {"error": "not found"})
            return
        kind = POST_ROUTES[path]
        if kind and not allowed(kind, self.visitor()):
            self.close_connection = True
            self.send_json(429, {"error": "too many requests, wait a minute"})
            return
        if path == "/api/transcribe":
            self.transcribe(urllib.parse.parse_qs(query))
            return
        if path == "/api/call/start":
            if not voice.available():
                self.send_json(503, {"error": "voice is offline"})
                return
            call_id = LINE.claim()
            if call_id:
                self.send_json(200, {"call": call_id})
            else:
                self.send_json(409, {"error": "someone else is on a call with the robot, wait a moment"})
            return
        try:
            body = self.read_json()
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
            return
        if path == "/api/call/end":
            self.send_json(200, {"ok": LINE.end(body.get("call"))})
        elif path == "/api/speak":
            self.speak(body)
        elif path.startswith("/api/knock/"):
            self.knock(path[11:], body)
        elif path.startswith("/api/apartment/"):
            self.apartment(path[15:], body)
        elif kind == "command":
            try:
                self.send_json(200, {"ok": True, "command": ROBOT.command(str(body.get("text", "")))})
            except ValueError as error:
                self.send_json(400, {"ok": False, "error": str(error)})
        else:
            self.chat(body)

    def transcribe(self, query):
        call_id = (query.get("call") or [""])[0]
        final = (query.get("final") or ["1"])[0] != "0"
        try:
            audio = self.read_body(MAX_AUDIO)
            voice.extension(self.headers.get("Content-Type"))
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
            return
        if not LINE.active(call_id, interim=not final):
            self.send_json(409, {"error": "no call in progress"})
            return
        try:
            text = voice.transcribe(audio, self.headers.get("Content-Type"), wait=final)
        except Exception as error:
            if final:
                self.send_json(502, {"error": voice.describe_error(error)})
            else:  # a live caption that fails is simply skipped
                self.send_json(200, {"text": None})
            return
        if text is None or (not final and not text):
            self.send_json(200, {"text": None})
            return
        LINE.active(call_id)
        if text:
            HUB.publish({"type": "heard", "call": call_id, "text": text, "final": final})
        if not text:
            self.send_json(400, {"error": "heard nothing, hold the button while you speak"})
        else:
            self.send_json(200, {"text": text})

    def speak(self, body):
        call_id = body.get("call")
        text = llm.spoken(str(body.get("text") or ""))
        if not text:
            self.send_json(400, {"error": "nothing to say"})
            return
        if not LINE.active(call_id):
            self.send_json(409, {"error": "no call in progress"})
            return
        HUB.publish({"type": "speaking", "call": call_id, "text": text[:voice.MAX_SPEAK]})
        try:
            audio = voice.speak(text)
        except Exception as error:
            self.send_json(502, {"error": voice.describe_error(error)})
            return
        LINE.active(call_id)
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)

    def knock(self, action, body):
        if KNOCK is None:
            self.send_json(503, {"error": "the robot is not running"})
            return
        if action == "choice":
            try:
                self.send_json(200, {"ok": KNOCK.choose(str(body.get("choice", "")))})
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
            return
        if action == "stop":
            self.send_json(200, {"ok": KNOCK.stop()})
            return
        result = KNOCK.call() if action == "call" else KNOCK.find()
        if result:
            self.send_json(200, {"ok": result})
        else:
            self.send_json(409, {"error": "a check-in is already running"})

    def apartment(self, action, body):
        if APARTMENT is None:
            self.send_json(503, {"error": "the apartment is not running"})
            return
        try:
            if action == "journal":
                entry = APARTMENT.add_entry(str(body.get("zone", "")), body.get("time"), body.get("days_ago", 0))
                self.send_json(200, {"ok": True, "entry": entry})
            elif action == "journal/import":
                self.send_json(200, dict(APARTMENT.import_journal(body.get("text", ""), bool(body.get("replace", True))),
                                         ok=True))
            elif action == "inputs":
                self.send_json(200, dict(APARTMENT.set_inputs(body.get("text", body)), ok=True))
            elif action == "journal/remove":
                self.send_json(200, {"ok": APARTMENT.remove_entry(body.get("id", -1))})
            elif action == "clock":
                self.send_json(200, {"ok": True, "clock": APARTMENT.set_clock(body.get("weekday"), body.get("time"))})
            elif action == "search":
                if APARTMENT.start(str(body.get("where", "routine")), body.get("miss_rate")):
                    self.send_json(200, {"ok": True})
                else:
                    self.send_json(409, {"error": "a search is already running"})
            elif action == "stop":
                self.send_json(200, {"ok": APARTMENT.stop()})
            elif action == "clear":
                APARTMENT.clear()
                self.send_json(200, {"ok": True})
            else:  # reset
                APARTMENT.reset(int(body.get("seed", 0)))
                self.send_json(200, {"ok": True})
        except (ValueError, TypeError) as error:
            self.send_json(400, {"error": str(error)})

    def live(self):
        if not LISTENERS.acquire(blocking=False):
            self.send_json(503, {"error": "too many viewers"})
            return
        try:
            self.send_events_head()
            card = KNOCK.latest() if KNOCK else None  # the Knock card as it is now (before the replay's seq)
            events, seq = HUB.current()  # a page that opens mid-call sees the call so far
            events = [e for e in events if e["type"] != "knock"] + ([card] if card else [])
            self.wfile.write(b"retry: 3000\n\n" + b"".join(sse(e) for e in events))
            self.wfile.flush()
            while True:
                events, seq_now = HUB.since(seq, PING)
                self.wfile.write(b"".join(sse(e) for e in events) if events else b": ping\n\n")
                self.wfile.flush()
                seq = seq_now
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        finally:
            LISTENERS.release()

    def stream(self, camera):
        if not STREAMS.acquire(blocking=False):
            self.send_json(503, {"error": "too many viewers"})
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            last = None
            while True:
                frame = ROBOT.frame(camera)
                if frame and frame is not last:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                    self.wfile.flush()
                    last = frame
                time.sleep(0.02 * VIEWS[camera])
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        finally:
            STREAMS.release()

    def chat(self, body):
        messages = llm.clean_history(body.get("messages"))
        if not messages or messages[-1]["role"] != "user":
            self.send_json(400, {"error": "messages must end with a user message"})
            return
        call_id = body.get("call")
        on_call = bool(call_id) and LINE.active(call_id)
        self.send_events_head()

        def event(payload):
            if on_call and payload["type"] in ("token", "command", "error"):
                LINE.active(call_id)  # an answer in progress keeps the call alive
                HUB.publish(dict(payload, call=call_id))
            self.wfile.write(sse(payload))
            self.wfile.flush()

        try:
            if not KEY:
                event({"type": "error", "text": "no metacon API key on the server"})
            elif not CHATS.acquire(timeout=30):
                event({"type": "error", "text": "metacon is busy with other chats, try again shortly"})
            else:
                try:
                    scanner = llm.CommandScanner()
                    for piece in llm.stream_chat(messages, ROBOT.state(), KEY, voice=on_call):
                        event({"type": "token", "text": piece})
                        for text in scanner.feed(piece):
                            try:
                                event({"type": "command", "text": ROBOT.command(text), "ok": True, "error": None})
                            except ValueError as error:
                                event({"type": "command", "text": text, "ok": False, "error": str(error)})
                except (BrokenPipeError, ConnectionResetError):
                    raise
                except Exception as error:
                    event({"type": "error", "text": llm.describe_error(error)})
                finally:
                    CHATS.release()
            event({"type": "done"})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def static(self, path):
        if not WEB.is_dir():
            self.send_json(404, {"error": "front end not built: run `just app-build`"})
            return
        target = (WEB / path.lstrip("/")).resolve()
        if WEB.resolve() not in target.parents or not target.is_file():
            target = WEB / "index.html"  # single-page app: unknown paths get the page
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        cache = "no-cache" if target.name == "index.html" else "public, max-age=31536000, immutable"
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if not any(quiet in self.path for quiet in ("/api/frame", "/api/state", "/api/health", "final=0")):
            print("[app] {} {}".format(self.visitor(), fmt % args), flush=True)


def main():
    global ROBOT, KEY, KNOCK, APARTMENT, SCENE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--scene", default="apartment", choices=("apartment", "floor"),
                        help="apartment: the furnished flat with zones (default); floor: the robot alone")
    parser.add_argument("--driver", default="tabpfn", choices=("tabpfn", "planner", "rule"),
                        help="who walks each leg in the apartment: the TabPFN a-to-b controller (needs the ml "
                             "packages and a controller built with `just controller`), the nav/dagger planner, "
                             "or the steer-to-bearing rule")
    parser.add_argument("--controller", default="default", help="name of the TabPFN controller (data/nav/controllers)")
    parser.add_argument("--flat", type=int, default=0, help="seed of the apartment's layout and resident")
    parser.add_argument("--belief", default="tabpfn", choices=("tabpfn", "counting"),
                        help="who gives the belief the search starts from: TabPFN (fit() on synthetic check-ins; "
                             "3.5 with TABPFN_TOKEN or the repo's .env, else v2) or the counting Bayes model")
    args = parser.parse_args()

    KEY = llm.api_key()
    if not KEY:
        print("warning: no metacon key ($METACON_LLM_KEY or Keychain 'metacon-llm'), chat is off", flush=True)
    if not voice.check():
        print("warning: voice service not reachable at {}, voice calls are off until it is".format(voice.URL),
              flush=True)
    voice.watch()
    LINE.sweep()
    SCENE = args.scene
    ROBOT = LiveRobot(args.scene, args.driver, args.controller)
    ROBOT.wait_ready(240)  # the TabPFN controller's fit() takes a few seconds
    KNOCK = Knock(HUB, ROBOT, lambda text: voice.speak(text) if voice.available() else None)
    model = None
    if args.belief == "tabpfn":
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "homes"))
            import tabpfn_belief
            model = tabpfn_belief.TabPFNBelief()
            print("[app] belief: TabPFN {} with {} synthetic check-ins as examples".format(model.version, model.rows),
                  flush=True)
        except Exception as error:
            print("[app] no TabPFN belief ({}), using the counting belief".format(error), flush=True)
    APARTMENT = Apartment(seed=args.flat, robot=ROBOT if args.scene == "apartment" else None, model=model)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print("JetHexa app on http://{}:{} (model {}, front end {})".format(
        args.host, args.port, llm.MODEL, WEB if WEB.is_dir() else "not built"), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

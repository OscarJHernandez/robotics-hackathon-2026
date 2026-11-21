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
    POST /api/chat     {"messages": [{"role", "content"}]} -> text/event-stream, events:
         {"type": "token", "text": "..."}  piece of the answer
         {"type": "command", "text": "goto 1 0.5", "ok": true, "error": null}  a command the model ran
         {"type": "error", "text": "..."}  {"type": "done"}
"""

import argparse
import json
import mimetypes
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm  # noqa: E402
from robot import HELP, VIEWS, LiveRobot  # noqa: E402

WEB = Path(__file__).resolve().parent / "web" / "dist"
LIMITS = {"chat": (20, 600), "command": (60, 60)}  # requests per window (seconds), per visitor
MAX_STREAMS = 40  # open picture streams across all visitors
MAX_BODY = 64 * 1024

ROBOT = None
KEY = ""
HITS = defaultdict(deque)
HITS_LOCK = threading.Lock()
STREAMS = threading.BoundedSemaphore(MAX_STREAMS)
CHATS = threading.BoundedSemaphore(2)  # metacon runs one model; do not queue up many answers


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

    def read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            raise ValueError("body missing or too large")
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/health":
            self.send_json(200, {"ok": True, "model": llm.MODEL, "llm": bool(KEY), "commands": HELP,
                                 "views": list(VIEWS)})
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
        elif path.startswith("/api/"):
            self.send_json(404, {"error": "not found"})
        else:
            self.static(path)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path not in ("/api/command", "/api/chat"):
            self.send_json(404, {"error": "not found"})
            return
        kind = path.rsplit("/", 1)[1]
        if not allowed(kind, self.visitor()):
            self.send_json(429, {"error": "too many requests, wait a minute"})
            return
        try:
            body = self.read_json()
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
            return
        if kind == "command":
            try:
                self.send_json(200, {"ok": True, "command": ROBOT.command(str(body.get("text", "")))})
            except ValueError as error:
                self.send_json(400, {"ok": False, "error": str(error)})
        else:
            self.chat(body)

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
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def event(payload):
            self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
            self.wfile.flush()

        try:
            if not KEY:
                event({"type": "error", "text": "no metacon API key on the server"})
            elif not CHATS.acquire(timeout=30):
                event({"type": "error", "text": "metacon is busy with other chats, try again shortly"})
            else:
                try:
                    scanner = llm.CommandScanner()
                    for piece in llm.stream_chat(messages, ROBOT.state(), KEY):
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
        if "/api/frame" not in self.path and "/api/state" not in self.path:
            print("[app] {} {}".format(self.visitor(), fmt % args), flush=True)


def main():
    global ROBOT, KEY
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    KEY = llm.api_key()
    if not KEY:
        print("warning: no metacon key ($METACON_LLM_KEY or Keychain 'metacon-llm'), chat is off", flush=True)
    ROBOT = LiveRobot()
    ROBOT.wait_ready()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print("JetHexa app on http://{}:{} (model {}, front end {})".format(
        args.host, args.port, llm.MODEL, WEB if WEB.is_dir() else "not built"), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

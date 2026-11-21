"""Serve the training dashboard live: rebuilds it from the database at most every 30 s.

    uv run --group ml python nav/live_dashboard.py [--port 8770] [--refresh 30]

The page reloads itself every `--refresh` seconds, so running collection, training or
evaluation shows up as it is recorded. The database is only read; if a run is writing at that
moment the server retries, then shows the last good page.
"""

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dashboard  # noqa: E402
from db import db_path, query  # noqa: E402

LAST = {"html": None, "at": 0.0}
LOCK = threading.Lock()
MIN_REBUILD = 30.0  # seconds: read the database at most this often, so training writes are not blocked


def waiting_page(message, refresh):
    return ("<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' content='{r}'>"
            "<title>Navigation training</title><body style='font:16px system-ui;padding:2rem'>"
            "<h1>Navigation training</h1><p>{m}</p><p>This page refreshes every {r} s.</p>").format(m=message, r=refresh)


def render(model, refresh):
    with LOCK:  # one rebuild at a time; serve the cached page if it is fresh
        if LAST["html"] and time.time() - LAST["at"] < MIN_REBUILD:
            return LAST["html"]
        return _render(model, refresh)


def _render(model, refresh):
    if not db_path().exists():
        return waiting_page("No data yet: {} does not exist. Run <code>just collect</code>.".format(db_path()), refresh)
    for attempt in range(5):
        try:
            try:
                name = model or dashboard.default_model()
            except SystemExit:  # nothing trained or evaluated yet
                name = None
            if not name:
                counts = query("SELECT kind, count(*) AS runs FROM runs GROUP BY kind ORDER BY kind")
                rows = ", ".join("{} {}".format(r.runs, r.kind) for r in counts.itertuples()) or "none"
                return waiting_page("No trained model yet. Runs recorded so far: {}.".format(rows), refresh)
            data = dashboard.build(name, db_path().parent / "dashboard.html")
            payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
            html = dashboard.TEMPLATE.read_text().replace("__DASHBOARD_DATA__", payload).replace(
                "__DASHBOARD_TITLE__", "Navigation training: " + name.replace("<", "&lt;"))
            html = html.replace("</head>", "<meta http-equiv='refresh' content='{}'></head>".format(refresh), 1)
            LAST["html"], LAST["at"] = html, time.time()
            return html
        except Exception as error:  # usually the database is locked by a run that is writing
            last_error = error
            time.sleep(0.5 * (attempt + 1))
    return LAST["html"] or waiting_page("Could not read the database: {}".format(last_error), refresh)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--refresh", type=int, default=30, help="seconds between automatic reloads")
    parser.add_argument("--model", help="model to show (default: the most recently trained)")
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] not in ("/", "/index.html"):
                self.send_error(404)
                return
            body = render(args.model, args.refresh).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *a):
            print("[dashboard] " + fmt % a, flush=True)

    print("live dashboard on http://127.0.0.1:{} reading {}".format(args.port, db_path()), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

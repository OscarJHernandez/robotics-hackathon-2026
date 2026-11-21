"""Voice calls: one caller at a time holds the line, and every viewer sees the call live.

The caller's page claims the line (Line.claim), sends the recording as it grows for live captions, then the final
recording, then the chat and the spoken reply, each with the call id; the server publishes what happens on the Hub.
Every page listens on GET /api/live (server-sent events) and shows the call, so people watching the robot see what
the caller says and what metacon answers while it happens.

Events: {"type": "call", "state": "start" | "end"}, {"type": "heard", "text", "final"}, {"type": "token", "text"},
{"type": "command", "text", "ok"}, {"type": "speaking", "text"}; each also carries "call" (the id) and "seq".
"""

import secrets
import threading
import time

IDLE = 30.0  # seconds without activity before a call is dropped and the line is free again
MAX_CALL = 120.0  # seconds: no call lasts longer than this
MAX_INTERIM = 15  # live caption transcriptions per call
KEEP = 400  # events kept for pages that reconnect


class Line:
    """Who is calling right now. Call ids are random secrets: knowing one is what lets a page use the call."""

    def __init__(self, hub, clock=time.monotonic):
        self.hub, self.clock = hub, clock
        self.lock = threading.Lock()
        self.call = None  # {"id", "started", "seen", "interim"}

    def _expire(self, now):
        call = self.call
        if call and (now - call["seen"] > IDLE or now - call["started"] > MAX_CALL):
            self.call = None
            self.hub.publish({"type": "call", "state": "end", "call": call["id"]})

    def claim(self):
        """A new call id, or None while someone else is on the line."""
        with self.lock:
            now = self.clock()
            self._expire(now)
            if self.call:
                return None
            call_id = secrets.token_hex(12)
            self.call = {"id": call_id, "started": now, "seen": now, "interim": 0}
        self.hub.publish({"type": "call", "state": "start", "call": call_id})
        return call_id

    def active(self, call_id, interim=False):
        """True if call_id is the current call (and marks it alive). interim=True also counts a live caption."""
        with self.lock:
            now = self.clock()
            self._expire(now)
            call = self.call
            if not call_id or not call or not secrets.compare_digest(call["id"], str(call_id)):
                return False
            if interim:
                if call["interim"] >= MAX_INTERIM:
                    return False
                call["interim"] += 1
            call["seen"] = now
            return True

    def busy(self):
        with self.lock:
            self._expire(self.clock())
            return self.call is not None

    def sweep(self, every=5.0):
        """End abandoned calls (a caller who closed the page) from a background thread, so viewers see them end."""
        def loop():
            while True:
                time.sleep(every)
                self.busy()
        threading.Thread(target=loop, name="call-sweep", daemon=True).start()

    def end(self, call_id):
        with self.lock:
            call = self.call
            if not call or not secrets.compare_digest(call["id"], str(call_id or "")):
                return False
            self.call = None
        self.hub.publish({"type": "call", "state": "end", "call": call_id})
        return True


class Hub:
    """Events for every listening page, numbered so a page can wait for the ones after the last it saw."""

    def __init__(self):
        self.cond = threading.Condition()
        self.events = []
        self.seq = 0

    def publish(self, event):
        with self.cond:
            self.seq += 1
            self.events.append(dict(event, seq=self.seq))
            del self.events[:-KEEP]
            self.cond.notify_all()

    def since(self, seq, timeout=15.0):
        """Events newer than seq; waits up to timeout for one. Returns (events, last seq)."""
        with self.cond:
            if self.seq <= seq:
                self.cond.wait(timeout)
            return [e for e in self.events if e["seq"] > seq], self.seq

    def current(self):
        """Events of the call in progress (for a page that opens in the middle of one), and the last seq."""
        with self.cond:
            start = None
            for i in range(len(self.events) - 1, -1, -1):
                event = self.events[i]
                if event["type"] == "call":
                    start = i if event["state"] == "start" else None
                    break
            return (self.events[start:] if start is not None else []), self.seq

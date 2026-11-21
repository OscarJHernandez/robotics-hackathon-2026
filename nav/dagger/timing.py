"""Time per phase, and the verbose run log (readable .log plus .jsonl) with heartbeats and ETAs."""

import datetime
import json
import os
import platform
import resource
import socket
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


class Timer:
    """Wall seconds per named phase: `with timer("render_rgb"): ...`."""

    def __init__(self):
        self.total = defaultdict(float)
        self.count = defaultdict(int)

    @contextmanager
    def __call__(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.total[name] += time.perf_counter() - start
            self.count[name] += 1

    def add(self, name, seconds, count=1):
        self.total[name] += seconds
        self.count[name] += count

    def summary(self, per=None):
        """{phase: {"seconds", "calls", "ms_per_call", "ms_per_step"}} sorted by total time."""
        out = {}
        for name in sorted(self.total, key=lambda n: -self.total[n]):
            seconds, calls = self.total[name], self.count[name]
            out[name] = {"seconds": round(seconds, 3), "calls": calls,
                         "ms_per_call": round(1000 * seconds / max(calls, 1), 3)}
            if per:
                out[name]["ms_per_step"] = round(1000 * seconds / per, 3)
        return out


def host_name():
    return os.environ.get("DAGGER_HOST") or socket.gethostname().split(".")[0]


def peak_memory_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)  # bytes on macOS, KiB on Linux


def machine_info():
    info = {"host": host_name(), "platform": platform.platform(), "python": platform.python_version(),
            "cpus": os.cpu_count(), "mujoco_gl": os.environ.get("MUJOCO_GL", "")}
    try:
        import mujoco
        info["mujoco"] = mujoco.__version__
    except ImportError:
        pass
    for name in ("torch", "tabpfn", "numpy", "pandas"):
        try:
            info[name] = __import__(name).__version__
        except ImportError:
            pass
    return info


def _clock(seconds):
    seconds = int(max(0, seconds))
    return "{:d}:{:02d}:{:02d}".format(seconds // 3600, seconds % 3600 // 60, seconds % 60)


class RunLog:
    """One log per process: `<name>.log` for people, `<name>.jsonl` for analysis. Also prints every line."""

    def __init__(self, folder, name, echo=True):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.text = open(folder / (name + ".log"), "a", buffering=1)
        self.jsonl = open(folder / (name + ".jsonl"), "a", buffering=1)
        self.start = time.time()
        self.name = name
        self.echo = echo
        self.last_beat = 0.0

    def event(self, kind, message="", **fields):
        now = time.time()
        record = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "host": host_name(),
                  "pid": os.getpid(), "log": self.name, "event": kind, "elapsed_s": round(now - self.start, 2)}
        record.update(fields)
        self.jsonl.write(json.dumps(record, default=str) + "\n")
        line = "{} [{} {} +{}] {:9s} {}".format(record["time"], record["host"], self.name, _clock(now - self.start),
                                                kind, message)
        self.text.write(line + "\n")
        if self.echo:
            print(line, flush=True)

    def heartbeat(self, done, total, steps, counts, timer=None, force=False, every=30.0):
        """Progress, rate, elapsed time, ETA and projected finish; at most every `every` seconds unless forced."""
        now = time.time()
        if not force and now - self.last_beat < every:
            return
        self.last_beat = now
        elapsed = now - self.start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else None
        finish = (datetime.datetime.now() + datetime.timedelta(seconds=eta)).strftime("%H:%M:%S") if eta else "?"
        fields = {"episodes_done": done, "episodes_total": total, "steps": steps,
                  "steps_per_s": round(steps / elapsed, 3) if elapsed else 0.0,
                  "episodes_per_min": round(60 * rate, 3), "eta_s": round(eta, 1) if eta else None,
                  "finish_at": finish, "outcomes": dict(counts), "peak_mem_mb": peak_memory_mb()}
        if timer is not None:
            fields["phases"] = timer.summary(per=max(steps, 1))
        message = "episodes {}/{}  steps {}  {:.2f} steps/s  elapsed {}  ETA {}  finish ~{}  {}".format(
            done, total, steps, fields["steps_per_s"], _clock(elapsed), _clock(eta) if eta else "?", finish,
            " ".join("{}={}".format(k, v) for k, v in sorted(counts.items())))
        self.event("heartbeat", message, **fields)

    def close(self):
        self.text.close()
        self.jsonl.close()

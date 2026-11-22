"""Keep DAgger jobs from breaking the machine they run on.

- `available_gb()`: memory the system can still hand out (Linux MemAvailable; macOS: the free percentage
  from `memory_pressure`).
- `fit_procs(wanted, ...)`: lower the process count to what fits next to a memory reserve.
- `MemoryGuard`: a thread in the parent that stops every child process when available memory falls
  under the reserve, long before the system's own out-of-memory killer (earlyoom on metacon) starts
  killing other services.
- `die_with_parent()`: run in every child; it exits as soon as its parent is gone, so a killed parent
  never leaves orphan workers holding memory.
"""

import os
import re
import subprocess
import sys
import threading
import time

PER_PROC_GB = 2.5  # measured: one worker with DINOv2 and 16 robots peaks at about 2.4 GB
RESERVE_GB = 6.0  # never let available memory go under this (or RESERVE_SHARE of the total, if larger)
RESERVE_SHARE = 0.12  # metacon 14.5 GB, vendetta 7.7, victory 7.1, a 16 GB laptop 6


def total_gb():
    if sys.platform.startswith("linux"):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1048576
    out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout
    return int(out.strip()) / 1073741824


def available_gb():
    """Memory the system can still hand out, in GB."""
    if sys.platform.startswith("linux"):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576
        return float("nan")
    # macOS's own figure (what Activity Monitor's memory pressure is based on). vm_stat's free + inactive
    # pages leave out compressible and file-backed memory and read far too low.
    out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    match = re.search(r"free percentage:\s*(\d+)%", out)
    if not match:
        return float("nan")
    return total_gb() * int(match.group(1)) / 100


def reserve_gb(total=None, reserve=RESERVE_GB, share=RESERVE_SHARE):
    total = total_gb() if total is None else total
    return max(reserve, share * total)


def fit_procs(wanted, available=None, reserve=None, per_proc=PER_PROC_GB):
    """How many worker processes fit: at most `wanted`, at least 1, leaving `reserve` GB free."""
    available = available_gb() if available is None else available
    reserve = reserve_gb() if reserve is None else reserve
    room = int((available - reserve) // per_proc)
    return max(1, min(wanted, room))


class MemoryGuard:
    """Calls `on_low(available)` once when available memory stays under `reserve` for two checks in a row."""

    def __init__(self, on_low, reserve=None, every=5.0, log=None):
        self.reserve = reserve_gb() if reserve is None else reserve
        self.on_low, self.every, self.log = on_low, every, log
        self.tripped = False
        self.lowest = float("inf")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="memory-guard")

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2 * self.every)

    def _run(self):
        low_checks = 0
        while not self._stop.wait(self.every):
            try:
                available = available_gb()
            except Exception:  # a failed reading is no reason to stop the job
                continue
            self.lowest = min(self.lowest, available)
            low_checks = low_checks + 1 if available < self.reserve else 0
            if low_checks >= 2 and not self.tripped:
                self.tripped = True
                if self.log:
                    self.log("available memory {:.1f} GB < reserve {:.1f} GB: stopping this machine's workers".format(
                        available, self.reserve))
                self.on_low(available)
                return


def die_with_parent(poll=2.0):
    """In a child process: exit when the parent process is gone (no orphans holding memory)."""
    parent = os.getppid()
    if sys.platform.startswith("linux"):
        try:  # the kernel sends SIGKILL to this process when its parent dies
            import ctypes
            import signal
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
        except OSError:
            pass

    def watch():
        while True:
            time.sleep(poll)
            if os.getppid() != parent:
                os._exit(17)

    threading.Thread(target=watch, daemon=True, name="parent-watch").start()

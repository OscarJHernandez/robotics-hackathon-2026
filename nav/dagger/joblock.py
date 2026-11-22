"""One DAgger job at a time per machine: a lock file data/nav/dagger/queue/<host>.lock.

The file is created atomically (O_CREAT | O_EXCL) and holds the experiment id, pid, host and start time. A lock
held by a dead pid on this same host is stale and is taken over (with a log line). A lock that names another host
is never taken over: we cannot tell whether that pid is alive.
"""

import json
import os
import time
from pathlib import Path


def queue_dir():
    from .config import runs_dir
    path = runs_dir() / "queue"
    path.mkdir(parents=True, exist_ok=True)
    return path


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (ValueError, TypeError, OverflowError):
        return False
    return True


class Busy(RuntimeError):
    pass


class JobLock:
    def __init__(self, host=None, directory=None, log=print):
        if host is None:
            from .timing import host_name
            host = host_name()
        self.host, self.log = host, log
        self.path = Path(directory or queue_dir()) / "{}.lock".format(host)
        self.held = False

    def read(self):
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None

    def _stale(self, info):
        """True when the holder is on this host and its pid is dead (or the file stayed empty for a while)."""
        if info is None:  # empty or half-written: the holder may be between create and write
            try:
                return time.time() - self.path.stat().st_mtime > 10
            except OSError:
                return True
        if info.get("host", self.host) != self.host:
            return False  # another machine's lock: never taken over
        return not pid_alive(info.get("pid"))

    def acquire(self, experiment_id):
        body = json.dumps({"pid": os.getpid(), "experiment": experiment_id, "host": self.host,
                           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}).encode()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                info = self.read()
                if attempt == 0 and self._stale(info):
                    self.log("queue: taking over stale lock {} (was {})".format(self.path, info))
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                info = info or {}
                raise Busy("{} is busy: experiment {} (pid {}, host {}, since {}); wait for it, or pass --no-lock".format(
                    self.host, info.get("experiment"), info.get("pid"), info.get("host"), info.get("started")))
            try:
                os.write(fd, body)
                os.fsync(fd)
            finally:
                os.close(fd)
            self.held = True
            return self
        raise Busy("could not take the lock {}".format(self.path))

    def release(self):
        if self.held:
            info = self.read()
            if info is not None and info.get("pid") == os.getpid() and info.get("host", self.host) == self.host:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
            self.held = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def lock(experiment_id, host=None, directory=None, log=print):
    """Context manager: `with lock("run1"): ...` acquires on entry, releases on exit."""
    return _Held(JobLock(host, directory, log), experiment_id)


class _Held:
    def __init__(self, job_lock, experiment_id):
        self.job_lock, self.experiment_id = job_lock, experiment_id

    def __enter__(self):
        return self.job_lock.acquire(self.experiment_id)

    def __exit__(self, *exc):
        self.job_lock.release()

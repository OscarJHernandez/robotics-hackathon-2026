"""One DAgger job at a time per machine: a lock file data/nav/dagger/queue/<host>.lock.

The file holds the pid, the experiment id and the start time. A lock whose pid is no longer alive is stale
and is taken over (with a log line).
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

    def acquire(self, experiment_id):
        body = json.dumps({"pid": os.getpid(), "experiment": experiment_id,
                           "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                info = self.read()
                if info is not None and pid_alive(info.get("pid")):
                    raise Busy("{} is busy: experiment {} (pid {}, since {}); wait for it, or pass --no-lock".format(
                        self.host, info.get("experiment"), info.get("pid"), info.get("started")))
                self.log("queue: taking over stale lock {} (was {})".format(self.path, info))
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(fd, "w") as f:
                f.write(body)
            self.held = True
            return self
        raise Busy("could not take the lock {}".format(self.path))

    def release(self):
        if self.held:
            info = self.read()
            if info is None or info.get("pid") == os.getpid():
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

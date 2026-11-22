"""Experiment records for the DAgger research loop: result.json per experiment, failures.json, incidents, leaderboard.

Layout: <run>/experiments/<id>/result.json (+ failures.json). Writes are atomic (temp file, then rename).
"""

import datetime
import json
import os
import tempfile
from pathlib import Path

RESULT_FIELDS = ["id", "hypothesis", "parent", "change", "kill_criterion", "status", "verdict", "metrics", "cost",
                 "git_commit", "started", "finished", "notes"]
STATUSES = ("planned", "running", "done", "failed")
VERDICTS = ("", "promoted", "not_better", "killed", "failed")
DICT_FIELDS = ("metrics", "cost")


def _atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _validate(fields):
    unknown = set(fields) - set(RESULT_FIELDS)
    if unknown:
        raise ValueError("unknown result field(s): {}".format(", ".join(sorted(unknown))))
    if "status" in fields and fields["status"] not in STATUSES:
        raise ValueError("status must be one of {}, got {!r}".format(STATUSES, fields["status"]))
    if "verdict" in fields and fields["verdict"] not in VERDICTS:
        raise ValueError("verdict must be one of {}, got {!r}".format(VERDICTS, fields["verdict"]))
    for key in DICT_FIELDS:
        if key in fields and not isinstance(fields[key], dict):
            raise ValueError("{} must be a dict".format(key))


def write_result(folder, **fields):
    """Write folder/result.json. Missing fields get defaults (status planned, empty strings and dicts). Raises
    ValueError on an unknown key or a bad status or verdict."""
    _validate(fields)
    result = {"id": "", "hypothesis": "", "parent": "", "change": "", "kill_criterion": "", "status": "planned",
              "verdict": "", "metrics": {}, "cost": {}, "git_commit": "", "started": "", "finished": "",
              "notes": ""}
    result.update(fields)
    _atomic_write(Path(folder) / "result.json", json.dumps(result, indent=1, sort_keys=False))
    return result


def read_result(folder):
    return json.loads((Path(folder) / "result.json").read_text())


def update_result(folder, **changes):
    """Merge changes into folder/result.json (same validation as write_result); returns the new record."""
    _validate(changes)
    result = read_result(folder)
    result.update(changes)
    return write_result(folder, **result)


def write_failures(folder, table):
    """folder/failures.json: the non-goal episodes of an episode table."""
    keep = ["episode", "scene_id", "seed", "failure", "steps", "final_dist"]
    sub = table[~table["goal"].astype(bool)]
    items = [{k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}
             for rec in sub[keep].to_dict("records")]
    _atomic_write(Path(folder) / "failures.json", json.dumps(items, indent=1))
    return items


def append_incident(run_folder, what, cause, fix, commit=""):
    """Append a dated markdown entry to run_folder/incidents.md."""
    path = Path(run_folder) / "incidents.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = "## {}\n\n- What: {}\n- Cause: {}\n- Fix: {}\n".format(stamp, what, cause, fix)
    if commit:
        entry += "- Commit: {}\n".format(commit)
    with open(path, "a") as f:
        f.write(entry + "\n")
    return entry


def _closed_goal_rate(result):
    closed = result.get("metrics", {}).get("closed")
    value = closed.get("goal_rate") if isinstance(closed, dict) else None
    return value if isinstance(value, (int, float)) and value == value else None


def leaderboard(run_folder):
    """Every experiments/*/result.json under run_folder, best closed-loop goal rate first (missing last); also
    written to run_folder/leaderboard.json. Each entry is the result plus `goal_rate` and `folder`."""
    entries = []
    for path in sorted(Path(run_folder).glob("experiments/*/result.json")):
        result = json.loads(path.read_text())
        result["goal_rate"] = _closed_goal_rate(result)
        result["folder"] = path.parent.name
        entries.append(result)
    entries.sort(key=lambda r: (r["goal_rate"] is None, -(r["goal_rate"] or 0.0)))
    _atomic_write(Path(run_folder) / "leaderboard.json", json.dumps(entries, indent=1))
    return entries

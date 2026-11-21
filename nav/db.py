"""All recorded navigation data, in one DuckDB database: data/nav/nav.duckdb.

Tables (query them with SQL; see docs/navigation.md):

    runs      one row per run (collect, eval, demo): kind, policy, model (controller name), settings
    episodes  one row per episode: start, target, outcome (reached, fell, final distance, directness)
    cycles    one row per gait cycle: time, robot sensors, history, location, target, command, outcome
    steps     the body pose every 20 ms, for path plots
    controllers   one row per controller built with `just controller`: kind, examples given to
              TabPFN's fit(), features, outputs, the file holding those examples
    predictions   held-out examples per controller with the label and TabPFN's prediction
              (`model` = controller name; `round` is NULL and `weights` is 'pretrained')
    policy_map    each controller's command over a grid of goals (bearing x distance): its steering map
    columns   the data dictionary: group, unit and meaning of every `cycles` column

Per run, `runs/<run_id>/` next to the database holds the camera images and CSV copies of its rows.
"""

import json
import platform
import time
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from common import data_dir, git_commit

DB_NAME = "nav.duckdb"
LOCK_RETRIES = 240  # 60 s of waiting for another process to release the database

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY, kind VARCHAR, policy VARCHAR, model VARCHAR, round INTEGER,
    started_at TIMESTAMP, finished_at TIMESTAMP, git_commit VARCHAR, settings JSON);
CREATE TABLE IF NOT EXISTS episodes (
    run_id VARCHAR, episode INTEGER, policy VARCHAR, model VARCHAR, round INTEGER, task INTEGER,
    start_x DOUBLE, start_y DOUBLE, start_yaw DOUBLE, target_x DOUBLE, target_y DOUBLE,
    start_dist DOUBLE, cycles INTEGER, seconds DOUBLE, final_x DOUBLE, final_y DOUBLE,
    final_dist DOUBLE, reached BOOLEAN, fell BOOLEAN, path_length DOUBLE, directness DOUBLE);
CREATE TABLE IF NOT EXISTS steps (
    run_id VARCHAR, episode INTEGER, t DOUBLE, x DOUBLE, y DOUBLE, yaw DOUBLE);
CREATE TABLE IF NOT EXISTS controllers (
    name VARCHAR, kind VARCHAR, created_at TIMESTAMP, context_rows INTEGER, total_examples INTEGER,
    val_rows INTEGER, features VARCHAR, outputs VARCHAR, fit_seconds DOUBLE, context_file VARCHAR,
    git_commit VARCHAR, settings JSON);
CREATE TABLE IF NOT EXISTS predictions (
    model VARCHAR, round INTEGER, output VARCHAR, weights VARCHAR, run_id VARCHAR, episode INTEGER,
    cycle INTEGER, goal_dist DOUBLE, goal_sin DOUBLE, goal_cos DOUBLE, horizon DOUBLE,
    actual DOUBLE, predicted DOUBLE);
CREATE TABLE IF NOT EXISTS policy_map (
    model VARCHAR, round INTEGER, output VARCHAR, weights VARCHAR, goal_dist DOUBLE, bearing DOUBLE,
    horizon DOUBLE, predicted DOUBLE);
CREATE TABLE IF NOT EXISTS columns (name VARCHAR PRIMARY KEY, grp VARCHAR, unit VARCHAR, meaning VARCHAR);
"""


def db_path():
    return data_dir() / DB_NAME


def connect(read_only=False):
    path = db_path()
    if read_only and not path.exists():
        raise FileNotFoundError("no navigation data yet: run `just collect` first ({})".format(path))
    # DuckDB lets one process at a time open the file for writing (readers are blocked by a writer
    # and vice versa). Runs open it only briefly, so wait for the lock instead of failing.
    for attempt in range(LOCK_RETRIES):
        try:
            con = duckdb.connect(str(path), read_only=read_only)
            break
        except duckdb.IOException as error:
            if "lock" not in str(error).lower() or attempt == LOCK_RETRIES - 1:
                raise
            time.sleep(0.25)
    if not read_only:
        con.execute(SCHEMA).fetchall()  # DuckDB 2: finish every result before the next statement
    return con


def describe_column(name):
    """(group, unit, meaning) of a `cycles` column."""
    joint = name.split("_", 2)[-1] if name.startswith("j_") else ""
    fixed = {
        "run_id": ("key", "", "run this row belongs to"), "episode": ("key", "", "episode number in the run"),
        "t_rel": ("time", "s", "time since the episode started"),
        "cycle": ("time", "", "gait cycle number in the episode (0 = first)"),
        "gyro_x": ("robot", "rad/s", "IMU angular velocity, body x (forward)"),
        "gyro_y": ("robot", "rad/s", "IMU angular velocity, body y (left)"),
        "gyro_z": ("robot", "rad/s", "IMU angular velocity, body z (up)"),
        "accel_x": ("robot", "m/s^2", "IMU acceleration with gravity, body x"),
        "accel_y": ("robot", "m/s^2", "IMU acceleration with gravity, body y"),
        "accel_z": ("robot", "m/s^2", "IMU acceleration with gravity, body z (about 9.81 at rest)"),
        "roll": ("robot", "rad", "body roll"), "pitch": ("robot", "rad", "body pitch"),
        "prev_stride": ("history", "rad", "stride of the previous cycle"),
        "prev_turn": ("history", "rad", "turn of the previous cycle"),
        "fwd_1": ("history", "m", "forward motion over the last cycle, in the current body frame"),
        "left_1": ("history", "m", "sideways motion over the last cycle (positive = left)"),
        "dyaw_1": ("history", "rad", "heading change over the last cycle"),
        "fwd_3": ("history", "m", "forward motion over the last 3 cycles"),
        "left_3": ("history", "m", "sideways motion over the last 3 cycles"),
        "dyaw_3": ("history", "rad", "heading change over the last 3 cycles"),
        "droll_1": ("history", "rad", "roll change over the last cycle"),
        "dpitch_1": ("history", "rad", "pitch change over the last cycle"),
        "x": ("location", "m", "body x in the world"), "y": ("location", "m", "body y in the world"),
        "sin_yaw": ("location", "", "sine of the heading"), "cos_yaw": ("location", "", "cosine of the heading"),
        "target_x": ("target", "m", "point b, world x"), "target_y": ("target", "m", "point b, world y"),
        "target_dist": ("target", "m", "distance to b"),
        "target_sin": ("target", "", "sine of the bearing to b (positive = b is to the left)"),
        "target_cos": ("target", "", "cosine of the bearing to b"),
        "goal_dist": ("goal", "m", "policy input: distance to b, capped at what `horizon` cycles can reach"),
        "goal_sin": ("goal", "", "policy input: sine of the bearing to b"),
        "goal_cos": ("goal", "", "policy input: cosine of the bearing to b"),
        "horizon": ("goal", "cycles", "policy input: cycles to reach the goal at full speed (1 to 8)"),
        "stride": ("command", "rad", "label: coxa swing used during this cycle"),
        "turn": ("command", "rad", "label: left/right swing difference used during this cycle (+ = left)"),
        "x_next": ("outcome", "m", "body x after the cycle"), "y_next": ("outcome", "m", "body y after the cycle"),
        "yaw_next": ("outcome", "rad", "heading after the cycle"),
        "dist_next": ("outcome", "m", "distance to b after the cycle"),
        "partial": ("outcome", "", "the cycle was cut short because b was reached or the robot fell"),
        "reached": ("outcome", "", "b was reached during this cycle"),
        "fell": ("outcome", "", "the robot fell during this cycle"),
        "hold": ("command", "", "id of the stretch of cycles that used the same command (random runs)"),
        "image": ("image", "", "camera picture at the start of the cycle, relative to the data folder"),
        "source": ("key", "", "how the command was chosen: random, baseline, tabpfn, tabpfn-planner"),
    }
    if name in fixed:
        return fixed[name]
    if name.startswith("j_pos_"):
        return ("robot", "rad", "joint angle, " + joint)
    if name.startswith("j_vel_"):
        return ("robot", "rad/s", "joint speed, " + joint + " (not available on the real robot)")
    if name.startswith("j_tau_"):
        return ("robot", "N m", "servo torque, " + joint + " (not available on the real robot)")
    if name.startswith("foot_"):
        return ("robot", "N", "floor force on foot " + name[5:] + " (0 = in the air)")
    return ("other", "", "")


class Recorder:
    """Records one run: a `runs` row, then episodes with their cycles and steps.

    Each episode is written when it ends, so an interrupted run keeps all finished episodes.
    """

    def __init__(self, kind, policy="", model="", round_=None, settings=None, images=False):
        self.started = datetime.now()
        self.run_id = "{}-{}".format(self.started.strftime("%Y%m%d-%H%M%S"), kind)
        self.folder = data_dir() / "runs" / self.run_id
        suffix = 1
        while self.folder.exists():
            suffix += 1
            self.folder = data_dir() / "runs" / "{}-{}".format(self.run_id, suffix)
        self.run_id = self.folder.name
        self.folder.mkdir(parents=True)
        self.images = images
        if images:
            (self.folder / "images").mkdir()
        self.kind, self.policy, self.model, self.round = kind, policy, model, round_
        settings = dict(settings or {})
        settings.setdefault("python", platform.python_version())
        try:
            import mujoco
            import tabpfn
            settings.setdefault("mujoco", mujoco.__version__)
            settings.setdefault("tabpfn", tabpfn.__version__)
        except ImportError:
            pass
        self.settings = settings
        self.episodes = 0
        with connect() as con:
            con.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                        [self.run_id, kind, policy, model, round_, self.started, git_commit(),
                         json.dumps(settings)])
        (self.folder / "run.json").write_text(json.dumps(
            {"run_id": self.run_id, "kind": kind, "policy": policy, "model": model, "round": round_,
             "started_at": self.started.isoformat(), "git_commit": git_commit(), "settings": settings},
            indent=2))

    def image_path(self, episode, cycle):
        """Relative path (from the data folder) for a cycle's camera picture, or "" without images."""
        if not self.images:
            return ""
        return str(Path("runs") / self.run_id / "images" / "{:04d}_{:03d}.png".format(episode, cycle))

    def add_episode(self, episode, rows, steps, summary, task=None):
        """Store one finished episode: its cycle rows, its 20 ms poses and its summary."""
        frame = pd.DataFrame(rows)
        frame.insert(0, "episode", episode)
        frame.insert(0, "run_id", self.run_id)
        step_frame = pd.DataFrame(steps, columns=["t", "x", "y", "yaw"])
        step_frame.insert(0, "episode", episode)
        step_frame.insert(0, "run_id", self.run_id)
        ep = dict(run_id=self.run_id, episode=episode, policy=self.policy, model=self.model, round=self.round,
                  task=task, **summary)
        ep_frame = pd.DataFrame([ep])
        with connect() as con:
            exists = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'cycles'"
                                 ).fetchall()[0][0]
            con.register("frame", frame)
            if not exists:
                con.execute("CREATE TABLE cycles AS SELECT * FROM frame")
                con.executemany("INSERT OR REPLACE INTO columns VALUES (?, ?, ?, ?)",
                                [[c] + list(describe_column(c)) for c in frame.columns])
            else:
                have = {r[0] for r in con.execute("DESCRIBE cycles").fetchall()}
                for column in frame.columns:
                    if column not in have:
                        kind = "DOUBLE" if pd.api.types.is_numeric_dtype(frame[column]) else "VARCHAR"
                        if pd.api.types.is_bool_dtype(frame[column]):
                            kind = "BOOLEAN"
                        con.execute('ALTER TABLE cycles ADD COLUMN "{}" {}'.format(column, kind))
                        con.execute("INSERT OR REPLACE INTO columns VALUES (?, ?, ?, ?)",
                                    [column] + list(describe_column(column)))
                con.execute("INSERT INTO cycles BY NAME SELECT * FROM frame")
            con.unregister("frame")
            con.register("step_frame", step_frame)
            con.execute("INSERT INTO steps BY NAME SELECT * FROM step_frame")
            con.unregister("step_frame")
            con.register("ep_frame", ep_frame)
            con.execute("INSERT INTO episodes BY NAME SELECT * FROM ep_frame")
            con.unregister("ep_frame")
        self.episodes += 1

    def close(self):
        """Mark the run finished and write CSV copies of its rows into the run folder."""
        with connect() as con:
            con.execute("UPDATE runs SET finished_at = ? WHERE run_id = ?", [datetime.now(), self.run_id])
            for table in ("cycles", "episodes"):
                try:
                    con.execute("COPY (SELECT * FROM {} WHERE run_id = ?) TO '{}' (HEADER)".format(
                        table, self.folder / (table + ".csv")), [self.run_id])
                except duckdb.CatalogException:
                    pass
            con.execute("COPY (SELECT * FROM columns ORDER BY grp, name) TO '{}' (HEADER)".format(
                self.folder / "columns.csv"))
        return self.run_id


def query(sql, params=None):
    """Run SQL on the database (read-only) and return a pandas DataFrame."""
    with connect(read_only=True) as con:
        return con.execute(sql, params or []).df()

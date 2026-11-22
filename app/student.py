"""The obstacle-avoiding TabPFN student (nav/dagger v29, "lidar-map plan") as a driver in the app.

The student was fitted on the plain training arena: grey checker floor, 4.5 m walls, randomly coloured furniture,
the red pillar standing at the goal. The app's apartment looks different (wood floor, low walls, zone patches, rugs,
cushions, a door, Margarete on the floor), and the camera feeds 75 of the student's 116 inputs. So the student never
looks at the apartment: it looks at a **plain twin**.

    PlainTwin     a second MuJoCo model, the plain training arena (vision/scene.py `Arena`, the same mesh/simple
                  choice as the app). Before each decision it gets the robot's joint state (qpos) and the eight
                  furniture poses from the app's model, the red pillar is put at the goal (as in training), and the
                  student's camera picture and depth are rendered there.
    lidar         cast in the app's own scene: rays skip geoms without collisions, so zones, decor and Margarete
                  are invisible to it; tests/test_student_twin.py checks it equals the twin's lidar.
    LidarPlanner  the robot's own map of lidar hits and its planned bearing (nav/dagger/lidarmap.py), reset per goto.
    Featurizer    frozen DINOv2 + patch probe on the twin's picture: the horizon band (nav/dagger/featurize.py).
    Features      the exported v29 pipeline (nav/dagger/export.py: column order and band PCA from fit time).
    policy        TabPFN through the hosted API (tabpfn-client, `ClientTabPFNPolicy`): fit() once on the exported
                  31,586 rows (free), one predict() per decision (10,000 credits each).

`Student` loads all of that on a background thread (status: loading | ready | failed | no_credits, with a reason),
so the server starts at once with the a-to-b driver, and the student takes over when it is ready. Below
MIN_CREDITS credits left, it does not start; its budget stops it before the account drops below that floor.

Also here, used by app/robot.py for every driver: `ContactCounter` (the trainer's collision rule: any contact between
a robot geom and a wall_/obs_ geom; counted, not ended) and `Progress` (the trainer's stuck rule: no new best of
0.2 m towards the goal in 40 gait cycles).
"""

import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "sim", REPO / "vision", REPO / "homes"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LIDAR = (36, 0.15, 8.0)  # rays, min and max range: nav/dagger/dagger.yaml `lidar`
GOAL_RADIUS = 0.25  # metres: the trainer's goal radius
STUCK_CYCLES = 40  # gait cycles without a new best of STUCK_GAIN towards the goal: stuck (nav/dagger/env.py)
STUCK_GAIN = 0.2
MIN_CREDITS = 150_000  # the student is not started (or stops) when the account would drop below this
VARIANT = "v29"
STATUSES = ("off", "loading", "ready", "failed", "no_credits")


def export_dir(variant=VARIANT):
    """data/nav/dagger/v2/student/<variant> in the main checkout (nav.common.data_dir; gitignored)."""
    from nav.common import data_dir
    return data_dir() / "dagger" / "v2" / "student" / variant


def robot_pose(sim):
    x, y, _ = sim.body_position()
    return (x, y, sim.body_yaw())



def api_limit(error):
    """(reason, retry_at epoch seconds) when the TabPFN API refused with a usage limit (HTTP 429), else None.
    The student then pauses until the limit resets, instead of failing again on every walk."""
    import datetime
    import re
    text = str(error)
    if "429" not in text and "usage limit" not in text.lower():
        return None
    found = re.search(r"Resets at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC", text)
    if found:
        when = datetime.datetime.strptime(found.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
        return "TabPFN API usage limit reached; the student is back after {} UTC".format(found.group(1)[:16]), when.timestamp()
    return "TabPFN API usage limit reached; the student retries in an hour", time.time() + 3600

class PlainTwin:
    """The plain training arena, kept in step with the app's scene (see the module doc)."""

    def __init__(self, simple=None):
        from scene import Arena  # vision/scene.py

        import flats as F  # homes/flats.py (only for the furniture colours of a flat)
        self.arena = Arena(simple)
        self.sim = self.arena.sim
        model = self.sim.model
        self._obstacles = [o["name"] for o in self.arena.obstacles]
        self._mocap = {name: int(model.body(name).mocapid[0]) for name in self._obstacles}
        self._target = int(model.body("target").mocapid[0])
        # The furniture's colours: those of the arena file (homes/flats.py PIECES), as homes/sim_bridge.FlatEnv uses.
        colors = {p["body"]: list(p["color"]) for p in F.PIECES.values()}
        layout = {"obstacles": [{"name": o["name"], "x": 0.0, "y": 0.0, "yaw": 0.0, "radius": o["radius"],
                                 "color": colors.get(o["name"], [0.5, 0.5, 0.5])} for o in self.arena.obstacles],
                  "target": (50.0, 50.0)}
        self.arena.apply(layout)

    def check_compatible(self, model):
        """Raise if `model` (the app's) does not carry the same robot (qpos layout) and furniture bodies."""
        twin = self.sim.model
        if model.nq != twin.nq:
            raise RuntimeError("the app's model has {} qpos, the twin {}".format(model.nq, twin.nq))
        for j in range(twin.njnt):
            if model.joint(j).name != twin.joint(j).name:
                raise RuntimeError("joint {} differs: {} vs {}".format(j, model.joint(j).name, twin.joint(j).name))
        for name in self._obstacles:
            model.body(name)  # raises KeyError if missing

    def sync(self, model, data, goal):
        """Copy the robot's joint state and the furniture poses from the app's (model, data); pillar at `goal`."""
        twin, tdata = self.sim.model, self.sim.data
        tdata.qpos[:] = data.qpos[:twin.nq]
        tdata.qvel[:] = 0.0
        for name, m in self._mocap.items():
            src = int(model.body(name).mocapid[0])
            tdata.mocap_pos[m] = data.mocap_pos[src]
            tdata.mocap_quat[m] = data.mocap_quat[src]
        tdata.mocap_pos[self._target] = (goal[0], goal[1], 0.0)
        mujoco.mj_forward(twin, tdata)

    def observe(self):
        """The student's camera picture and depth in the twin: {"rgb", "depth"}."""
        return {"rgb": self.sim.render(), "depth": self.sim.render(depth=True).astype(np.float32)}

    def lidar(self):
        return self.sim.lidar(*LIDAR)

    def close(self):
        self.arena.close()


class ContactCounter:
    """Robot vs wall_/obs_ contacts, the trainer's collision rule (nav/dagger/env.py). `update()` (once per control
    step) returns True when a new touch starts; `count` is the number of touches. A foot tapping the same wall
    step after step is one touch: a new one needs `clear` steps without contact before it (default 50 = 1 s)."""

    def __init__(self, model, clear=50):
        self.clear = clear
        robot = model.body("body").id
        self._robot = np.array([model.body_rootid[model.geom_bodyid[g]] == robot for g in range(model.ngeom)])
        self._solid = np.array([model.geom(g).name.startswith(("wall_", "obs_")) for g in range(model.ngeom)])
        self.count = 0
        self.points = []  # (x, y) where each touch started
        self._free = clear  # control steps since the last contact

    def touching(self, data):
        n = data.ncon
        if n == 0:
            return False
        g1, g2 = data.contact.geom1[:n], data.contact.geom2[:n]
        return bool(np.any((self._robot[g1] & self._solid[g2]) | (self._robot[g2] & self._solid[g1])))

    def update(self, data, at=None):
        now = self.touching(data)
        new = now and self._free >= self.clear
        self._free = 0 if now else self._free + 1
        if new:
            self.count += 1
            if at is not None:
                self.points.append((round(float(at[0]), 3), round(float(at[1]), 3)))
                del self.points[:-50]
        return new

    def reset(self):
        self.count, self.points, self._free = 0, [], self.clear


class Progress:
    """The trainer's stuck rule: stuck after `cycles` gait cycles without getting `gain` m closer than ever before."""

    def __init__(self, cycles=STUCK_CYCLES, gain=STUCK_GAIN):
        self.cycles, self.gain = cycles, gain
        self.best, self.since = None, 0

    def reset(self, dist):
        self.best, self.since = dist, 0

    def update(self, dist):
        """One gait cycle at distance `dist` from the goal. True when stuck."""
        if self.best is None or dist < self.best - self.gain:
            self.best, self.since = dist, 0
        else:
            self.since += 1
        return self.since >= self.cycles


class Student:
    """The v29 student for the app: build() (or load_async()) then, per goto, start() and act() once per cycle.

    Everything it needs can be passed in (tests use fakes); otherwise build() makes it: the export, the client
    policy (fit() on the export), the featurizer (DINOv2 + probe) and the plain twin.
    """

    def __init__(self, policy=None, features=None, featurizer=None, twin=None, simple=None, export=None,
                 min_credits=MIN_CREDITS, max_calls=None, log=print):
        """`max_calls`: at most this many API calls in this process (on top of the MIN_CREDITS floor)."""
        self.policy, self.features, self.featurizer, self.twin = policy, features, featurizer, twin
        self.simple, self.export, self.min_credits, self._log = simple, export, min_credits, log
        self.max_calls = max_calls
        self.status, self.reason = ("ready", "") if policy is not None and features is not None else ("off", "")
        self.planner = None
        self.credits_at_start = None
        self.decisions = 0
        self.seconds = 0.0  # time spent in act(), summed
        self.last_seconds = None
        self.last_command = None

    # ---- loading -------------------------------------------------------------------------------------

    def load_async(self):
        self.status, self.reason = "loading", ""
        threading.Thread(target=self._build_safely, name="student", daemon=True).start()
        return self

    def _build_safely(self):
        try:
            self.build()
        except Exception as error:  # noqa: BLE001 - the app must keep walking with the a-to-b driver
            self.status, self.reason = "failed", "{}: {}".format(type(error).__name__, error)
            self._log("[student] not available: {}".format(self.reason))

    def build(self):
        """Load and fit everything; sets status. Raises on failure (load_async turns that into status failed)."""
        from nav.dagger.config import load_config, vision_dir
        from nav.dagger.export import load_export
        from nav.dagger.featurize import Featurizer
        from nav.dagger.policy import CREDITS_PER_PREDICT, CreditBudget, client_login, credits_left, make_student
        started = time.time()
        self.status = "loading"
        folder = Path(self.export) if self.export else export_dir()
        if not (folder / "features.parquet").exists():
            raise RuntimeError("no student export at {} (nav/dagger/export.py, see docs/navigation.md)".format(folder))
        if self.policy is None:
            if not client_login():
                raise RuntimeError("no TABPFN_TOKEN (environment or the repo's .env)")
            left = credits_left()
            self.credits_at_start = left
            if left is not None and left < self.min_credits:
                self.status = "no_credits"
                self.reason = "{:,} TabPFN API credits left, the student needs at least {:,}".format(left, self.min_credits)
                self._log("[student] " + self.reason)
                return
        features, table, meta = load_export(folder)
        if self.features is None:
            self.features = features
        if self.policy is None:
            calls = None if left is None else max(0, (left - self.min_credits) // CREDITS_PER_PREDICT)
            if self.max_calls is not None:
                calls = self.max_calls if calls is None else min(calls, self.max_calls)
            budget = CreditBudget(calls)
            config = load_config()
            student_cfg = dict(config["student"], kind="tabpfn", backend="client")
            policy = make_student(student_cfg, features.columns, features.categorical, seed=0, budget=budget)
            policy.fit(table)  # uploads the examples; fit() costs no credits
            self.policy = policy
        if self.featurizer is None:
            config = load_config()
            self.featurizer = Featurizer.from_config(config["features"], vision_dir())
        if self.twin is None:
            self.twin = PlainTwin(self.simple)
        self.status, self.reason = "ready", ""
        self._log("[student] ready in {:.0f} s: {} fit() on {:,} examples x {} columns ({})".format(
            time.time() - started, getattr(self.policy, "name", "policy"), len(table), len(features.columns),
            meta.get("variant")))

    @property
    def ready(self):
        retry_at = getattr(self, "retry_at", None)
        if self.status == "no_credits" and retry_at is not None and time.time() >= retry_at:
            self.status, self.reason, self.retry_at = "ready", None, None  # the API's limit has reset
        return self.status == "ready"

    # ---- driving -------------------------------------------------------------------------------------

    def start(self, goal):
        """A new goto: an empty lidar map."""
        from nav.dagger.lidarmap import LidarPlanner
        if self.planner is None:
            self.planner = LidarPlanner(rays=LIDAR[0], max_range=LIDAR[2])
        self.planner.reset()
        self.goal = (float(goal[0]), float(goal[1]))

    def inputs(self, sim, goal, lidar=None):
        """The student's input row for the robot in `sim` (the app's JetHexaSim) walking to `goal`: (X (1, k), obs).

        Lidar from the app's scene; picture and depth from the plain twin; the plan bearing from the lidar map."""
        pose = robot_pose(sim)
        if lidar is None:
            lidar = sim.lidar(*LIDAR)
        lp = self.planner.step(pose, goal, lidar)
        self.twin.sync(sim.model, sim.data, goal)
        obs = self.twin.observe()
        obs.update(robot_pose=pose, goal_point=tuple(goal), lidar=lidar)
        base = self.featurizer.base_rows([obs], [None])
        band = self.featurizer.last_band
        base["lp_sin"], base["lp_cos"] = [lp[0]], [lp[1]]
        X, _ = self.features.live(base, band, [None])
        return X, obs

    def act(self, sim, goal):
        """(stride, turn) for the next gait cycle. Raises when the API fails or the budget is used up (the caller
        then hands the walk to the a-to-b driver)."""
        from nav.dagger.policy import OutOfCredits
        started = time.perf_counter()
        X, _ = self.inputs(sim, goal)
        try:
            cmd = self.policy.act_batch(X)[0]
        except OutOfCredits as error:
            self.status, self.reason = "no_credits", str(error)
            raise
        except Exception as error:  # noqa: BLE001 - the caller falls back; an API limit also pauses the student
            limit = api_limit(error)
            if limit is not None:
                self.status, self.reason, self.retry_at = "no_credits", limit[0], limit[1]
            raise
        self.last_seconds = time.perf_counter() - started
        self.seconds += self.last_seconds
        self.decisions += 1
        from nav.common import STRIDE_RANGE, TURN_RANGE
        self.last_command = (float(np.clip(cmd[0], *STRIDE_RANGE)), float(np.clip(cmd[1], *TURN_RANGE)))
        return self.last_command

    def info(self):
        out = {"status": self.status, "decisions": self.decisions}
        if self.reason:
            out["reason"] = self.reason
        if self.last_seconds is not None:
            out["seconds_per_decision"] = round(self.seconds / max(self.decisions, 1), 2)
        budget = getattr(self.policy, "budget", None)
        if budget is not None and getattr(budget, "max_calls", None) is not None:
            out["calls_left"] = budget.left
        return out

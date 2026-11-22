"""The simulated JetHexa running live in a background thread, for the web app.

One thread owns the simulator: it steps the physics in real time, steers the robot and renders JPEG
pictures (outside views named in VIEWS, and the robot's own camera), only of views someone watches. Other threads only
put commands on a queue and read the latest pictures and state. MuJoCo's renderer must be created
and used on one thread, which is why rendering happens here too.

Commands (the same words the chat understands): goto X Y, forward M, turn DEG, stop, reset, target (walk to the
red pillar), random (moves the red end-point pillar to a random spot; a robot walking to the pillar follows it).

Two scenes: "floor" (the robot alone) and "apartment" (the furnished room with zones; see LiveRobot).
"""

import io
import math
import queue
import random
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))

from fetch_meshes import meshes_available  # noqa: E402
from jethexa_sim import SIM_DIR, JetHexaSim, tripod_targets  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student import ContactCounter, Progress  # noqa: E402

CONTROL_DT = 0.02  # seconds between new joint targets
CYCLE = 1.0  # seconds per gait cycle; the steering command changes once per cycle
PHASE0 = 0.25  # gait clock start: both tripods mid-sweep, so walking starts without a jump
MAX_STRIDE = 0.3  # radians of coxa swing
MAX_TURN = 0.25  # radians of left/right swing difference; positive turns left
MAX_STEP = 0.13  # metres per cycle at full stride
REACHED = 0.10  # metres: close enough to a goto point
YAW_REACHED = 0.06  # radians: close enough to a turn target
ARENA = 2.6  # metres: goto points are kept inside this square (the floor is 3 m each way)
APARTMENT = 3.2  # the same in the apartment scene (walls at 3.54 m)
APARTMENT_REACHED = 0.27  # metres: the planner stops 0.25 m from a goto point in the apartment
WALL_HEIGHT = 0.7  # metres: the apartment's walls are cut down to this, so the outside views can look in
# The home look of the apartment scene (sim/apartment_decor.xml). Colors per piece of furniture (arena body):
FURNITURE_COLORS = {"obs_0": (0.62, 0.49, 0.36), "obs_1": (0.33, 0.41, 0.50), "obs_2": (0.76, 0.75, 0.72),
                    "obs_3": (0.60, 0.48, 0.35), "obs_4": (0.52, 0.39, 0.27), "obs_5": (0.68, 0.54, 0.26),
                    "obs_6": (0.64, 0.54, 0.40), "obs_7": (0.60, 0.38, 0.28)}
WALL_COLOR = (0.80, 0.78, 0.74)
# Decoration: (geom, piece of furniture it belongs to, forward and left offset on that piece in metres)
DECOR = (("decor_duvet", "bed", 0.24, 0.0), ("decor_pillow", "bed", -0.62, 0.0),
         ("decor_cushion_a", "couch", -0.38, 0.05), ("decor_cushion_b", "couch", 0.38, 0.05),
         ("decor_seat", "armchair", 0.04, 0.0), ("decor_worktop", "cabinet", 0.0, 0.0),
         ("decor_leaves_a", "plant", 0.0, 0.0), ("decor_leaves_b", "plant", 0.03, -0.02),
         ("decor_rug_dining", "table", 0.0, 0.0), ("decor_rug_living", "couch", 0.0, 1.1))
PARKED = (50.0, 50.0)  # where the pillar goes when nobody is home
POSES = ("lying", "standing")  # how Margarete is found
NEAR_PLANE = 0.05  # metres: the robot camera's near clipping plane in the apartment
STANDING_HEIGHT = 0.83  # metres: standing, her feet (x = -0.80, radius 0.045 in the decor file) touch the floor
RANDOM_SPAN = 2.0  # metres: random end points lie in this square, so they stay well inside the floor
RANDOM_MIN_DIST = 0.8  # metres: a random end point is at least this far from the robot
FALLEN = 0.5  # body_upright() below this counts as a fall
JPEG_QUALITY = 72
VIEW_SIZE = (1280, 720)  # width, height of the outside views
# Picture views: control steps between pictures (4 = 12.5 per second). "eye" is the robot's own camera.
VIEWS = {"chase": 4, "orbit": 4, "overhead": 6, "eye": 6}
WATCHED = 3.0  # seconds: a view is rendered only while someone asked for it this recently

HELP = ("goto X Y (metres, floor is -{a} to {a}), forward M (metres), turn DEG (positive = left), "
        "stop, reset, target (walk to the red pillar), random (new random end point), "
        "look (turn once on the spot looking for her), watch N (stand N cycles looking)").format(a=ARENA)
LOOK_TURN = 0.15       # turn command of a look-around: about 0.32 rad per gait cycle, 20 cycles a turn (PR 26)
LOOK_CLEARANCE = 0.33  # metres (lidar): closer than this, the corner legs could sweep into something while turning


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def parse_command(text, limit=ARENA):
    """Turn "goto 1 0.5" and friends into a tuple, or raise ValueError with a message for the user.
    goto points are kept inside the square of half-width `limit`."""
    words = text.replace(",", " ").split()
    if not words:
        raise ValueError("empty command")
    name, args = words[0].lower().lstrip("/"), words[1:]
    if name == "spot" and len(args) == 1 and args[0].lower() in ("reset", "reject", "mark"):
        return ("spot", args[0].lower())  # the person detector: new search / the spot was empty / count from now
    try:
        numbers = [float(a) for a in args]
    except ValueError:
        raise ValueError("'{}' needs numbers, got: {}".format(name, " ".join(args)))
    if any(not math.isfinite(n) for n in numbers):
        raise ValueError("numbers must be finite")
    if name == "goto" and len(numbers) == 2:
        return ("goto", max(-limit, min(limit, numbers[0])), max(-limit, min(limit, numbers[1])))
    if name == "forward" and len(numbers) == 1 and 0 < numbers[0] <= 2 * limit:
        return ("forward", numbers[0])
    if name == "turn" and len(numbers) == 1 and abs(numbers[0]) <= 360:
        return ("turn", math.radians(numbers[0]))
    if name in ("stop", "reset", "random", "target", "look") and not numbers:
        return (name,)
    if name == "watch" and len(numbers) == 1 and 1 <= numbers[0] <= 30:
        return ("watch", int(numbers[0]))
    raise ValueError("unknown command '{}'. Commands: {}".format(text.strip(), HELP))


def random_point(state, rng=random):
    """A random end point inside RANDOM_SPAN, at least RANDOM_MIN_DIST from the robot in `state`."""
    rx, ry = state.get("x", 0.0), state.get("y", 0.0)
    for _ in range(100):
        x, y = rng.uniform(-RANDOM_SPAN, RANDOM_SPAN), rng.uniform(-RANDOM_SPAN, RANDOM_SPAN)
        if math.hypot(x - rx, y - ry) >= RANDOM_MIN_DIST:
            break
    return (round(x, 2), round(y, 2))


def to_jpeg(rgb):
    from PIL import Image  # only the app group installs Pillow; the parser works without it

    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


def wood_floor(model):
    """Paint wooden planks into the floor's texture (the robot file's grey checker). Looks only."""
    import numpy as np

    tex = model.tex("grid").id
    w, h = int(model.tex_width[tex]), int(model.tex_height[tex])
    start = int(model.tex_adr[tex])
    rng = np.random.default_rng(7)
    img = np.zeros((h, w, 3), dtype=np.float32)
    planks, base = 8, np.array([0.74, 0.58, 0.42])
    rows = h // planks
    for k in range(planks):
        offset = int(rng.integers(w))
        for part in range(2):  # two boards per row, with the joint at a random place
            shade = base * (0.90 + 0.16 * rng.random()) + (rng.random(3) - 0.5) * 0.02
            cols = (np.arange(w) + offset) % w
            mask = (cols < w // 2) if part == 0 else (cols >= w // 2)
            img[k * rows:(k + 1) * rows, mask] = shade
        grain = (rng.random((rows, 1)) - 0.5) * 0.035  # streaks along the board
        img[k * rows:(k + 1) * rows] += grain[:, :, None]
        img[k * rows:k * rows + 1] *= 0.72  # the seam between two rows
        for joint in (offset % w, (offset + w // 2) % w):
            img[k * rows:(k + 1) * rows, (w - joint) % w] *= 0.72
    data = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    model.tex_data[start:start + w * h * 3] = data.reshape(-1)
    mat = model.geom_matid[model.geom("floor").id]
    model.mat_texrepeat[mat] = (6, 6)
    model.mat_reflectance[mat] = 0.0


def dress(model):
    """The home's colors: furniture (keeping each part's own shading) and walls."""
    for body, color in FURNITURE_COLORS.items():
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(body).id]
        top = max(float(model.geom_rgba[g][:3].max()) for g in geoms)
        for g in geoms:
            shade = float(model.geom_rgba[g][:3].max()) / top
            model.geom_rgba[g][:3] = [c * (0.82 + 0.18 * shade) for c in color]
    for g in range(model.ngeom):
        if model.geom(g).name.startswith("wall_"):
            model.geom_rgba[g][:3] = WALL_COLOR
            model.geom_size[g][2] = model.geom_pos[g][2] = WALL_HEIGHT / 2


def furnish(model, flat):
    """Put the decoration of sim/apartment_decor.xml on and beside the flat's furniture, and the door on its wall."""
    pieces = {f["name"]: f for f in flat["furniture"]}

    def put(name, x, y, yaw):
        g = model.geom(name).id
        model.geom_pos[g][:2] = (x, y)
        model.geom_quat[g] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    half = flat["width"] / 2
    for name, piece, forward, left in DECOR:
        f = pieces[piece]
        yaw = math.radians(f["yaw_deg"])
        x = f["x"] + forward * math.cos(yaw) - left * math.sin(yaw)
        y = f["y"] + forward * math.sin(yaw) + left * math.cos(yaw)
        if name.startswith("decor_rug"):  # a rug stays on the floor of the room
            size = max(model.geom_size[model.geom(name).id][:2])
            x, y = max(-half + size, min(half - size, x)), max(-half + size, min(half - size, y))
        put(name, x, y, yaw)
    door = flat["door"]
    inward = {"xmin": (1.0, 0.0), "xmax": (-1.0, 0.0), "ymin": (0.0, 1.0), "ymax": (0.0, -1.0)}[door["wall"]]
    along = 0.0 if door["wall"] in ("ymin", "ymax") else math.pi / 2
    put("decor_door", door["x"] + 0.03 * inward[0], door["y"] + 0.03 * inward[1], along)
    put("decor_knob", door["x"] + 0.08 * inward[0] + 0.3 * math.cos(along), door["y"] + 0.08 * inward[1] + 0.3 * math.sin(along), 0.0)
    put("decor_mat", door["x"] + 0.36 * inward[0], door["y"] + 0.36 * inward[1], along)


def footprints(model, data, below=0.25):
    """Floor footprints of every wall and obstacle shape that reaches below `below` metres, in the form the
    nav/dagger planner takes (as SimEnv.footprints() in nav/dagger/env.py)."""
    out = []
    for g in range(model.ngeom):
        if not model.geom(g).name.startswith(("wall_", "obs_")):
            continue
        size, pos, kind = model.geom_size[g], data.geom_xpos[g], model.geom_type[g]
        half_height = size[2] if kind == mujoco.mjtGeom.mjGEOM_BOX else size[1]
        if pos[2] - half_height > below:
            continue
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            xmat = data.geom_xmat[g].reshape(3, 3)
            out.append({"kind": "box", "x": float(pos[0]), "y": float(pos[1]),
                        "yaw": float(math.atan2(xmat[1, 0], xmat[0, 0])), "hx": float(size[0]), "hy": float(size[1])})
        else:
            out.append({"kind": "circle", "x": float(pos[0]), "y": float(pos[1]), "r": float(size[0])})
    return out


ROUTE_MARGIN = 0.42  # metres the route keeps the robot's centre away from furniture and walls
WAYPOINT = 0.30  # metres: close enough to a point on the way to turn to the next one
DRIVERS = ("student", "tabpfn", "planner", "rule")
STUDENT_REACHED = 0.25  # metres: the student's goal radius, as in its training (nav/dagger/dagger.yaml)


def load_tabpfn(name):
    """The TabPFN controller `name` built with `just controller` (nav/controller.py): TabPFN's fit() on recorded
    examples, then `act([row])` gives (stride, turn) for the next gait cycle. Needs the ml packages."""
    if str(REPO / "nav") not in sys.path:
        sys.path.insert(0, str(REPO / "nav"))
    import controller as nav_controller
    try:
        return nav_controller.load(name)
    except SystemExit as error:  # no such controller
        raise RuntimeError(str(error))


def nav_row(pose, poses, prev, target):
    """What a nav/ controller reads before a gait cycle (the columns of nav.common.Walker.observe() it uses):
    the last command, how the body moved over the last cycles, and where the target is."""
    from common import make_goal, relative, to_body, wrap as nav_wrap

    row = {"prev_stride": prev[0], "prev_turn": prev[1]}
    for n in (1, 3):
        before = poses[max(0, len(poses) - n)] if poses else pose
        row["fwd_%d" % n], row["left_%d" % n] = to_body(before, pose[0], pose[1])
        row["dyaw_%d" % n] = nav_wrap(pose[2] - before[2])
    dist, bearing = relative(pose, *target)
    row["target_dist"], row["target_sin"], row["target_cos"] = dist, math.sin(bearing), math.cos(bearing)
    row.update(make_goal(dist, bearing))
    return row


def plan_route(shapes, half, start, goal):
    """Straight legs from `start` to `goal` that keep ROUTE_MARGIN away from the furniture: the points to walk
    to, the last one being `goal`. The a-to-b controller walks each leg; it does not see obstacles itself.
    Uses the grid and A* of nav/dagger/expert.py. [goal] when there is no way or no planner."""
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from nav.dagger.expert import Grid, astar
        import numpy as np
    except Exception:
        return [goal]
    grid = Grid(shapes, half, cell=0.05, inflate=ROUTE_MARGIN, goal=goal, goal_radius=0.3)
    # The robot may stand inside the margin (the dock is near the wall): free the floor around it, not the shapes.
    gx, gy = np.meshgrid(grid.xs, grid.xs, indexing="ij")
    grid.blocked &= ~((np.hypot(gx - start[0], gy - start[1]) < 0.3) & (grid.clearance > 0.02))
    path = astar(grid, start, goal)
    if not path:
        return [goal]

    def clear(a, b):
        steps = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.04))
        for k in range(steps + 1):
            t = k / steps
            if grid.blocked[grid.index(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))]:
                return False
        return True

    points, at, i = [], start, 0
    path = path + [goal]
    while i < len(path) - 1:  # from each point, go straight to the farthest point of the path that is in sight
        j = len(path) - 1
        while j > i + 1 and not clear(at, path[j]):
            j -= 1
        at, i = path[j], j
        points.append((float(at[0]), float(at[1])))
    return points


class ApartmentScene:
    """The apartment model with the home look, without a thread or a window: the furnished 50 m² room
    (sim/apartment.xml), wood floor, low walls, recoloured furniture, zone patches, decor and Margarete.
    LiveRobot runs one; tests build one to compare with the student's plain twin (app/student.py)."""

    def __init__(self, simple=None):
        if simple is None:
            simple = not meshes_available()
        self.simple = simple
        self.sim = sim = JetHexaSim(SIM_DIR / ("apartment_simple.xml" if simple else "apartment.xml"))
        model = sim.model
        self.half = math.sqrt(50.0) / 2
        model.geom_size[model.geom("floor").id, :2] = self.half + 0.1  # the robot file's floor is 6 x 6 m
        model.light_active[0] = 0  # the robot file's lamp; the room has its own even lights
        model.mat_reflectance[model.geom_matid[model.geom("floor").id]] = 0.0
        wood_floor(model)
        dress(model)
        # MuJoCo puts the camera's near plane at znear x the model's extent; the decor parked 40-50 m away makes the
        # extent 112 m, so everything closer than 1.1 m vanished from the robot's camera (and its depth). 5 cm instead.
        model.vis.map.znear = NEAR_PLANE / model.stat.extent
        self._person = int(model.body("person").mocapid[0])
        self.zone_geoms = dict((model.geom(g).name[5:], g) for g in range(model.ngeom)
                               if model.geom(g).name.startswith("zone_"))
        mujoco.mj_forward(model, sim.data)
        self.shapes = footprints(model, sim.data)
        self.home = (0.0, 0.0, 0.0)

    def load(self, flat):
        """The flat's furniture, zone patches and decoration; the robot at the dock; nobody home. Returns home."""
        sim, model = self.sim, self.sim.model
        key = model.key("stand").id
        mpos, mquat = model.key_mpos[key].reshape(-1, 3), model.key_mquat[key].reshape(-1, 4)
        for piece in flat["furniture"]:
            m = int(model.body(piece["body"]).mocapid[0])
            yaw_ = math.radians(piece["yaw_deg"])
            mpos[m] = (piece["x"], piece["y"], 0.0)
            mquat[m] = (math.cos(yaw_ / 2), 0.0, 0.0, math.sin(yaw_ / 2))
        for zone in flat["zones"]:
            g = self.zone_geoms[zone["name"]]
            model.geom_pos[g][:2] = (zone["x"], zone["y"])
            model.geom_size[g][:2] = (zone["size_x"] / 2 - 0.02, zone["size_y"] / 2 - 0.02)
        furnish(model, flat)
        dock = flat["dock"]
        self.home = (dock["x"], dock["y"], math.atan2(-dock["y"], -dock["x"]))  # at the dock, facing the room
        self.place(self.home)  # sim.reset() also puts the furniture where the keyframe now says
        sim.set_target(*PARKED)
        self.person()
        self.shapes = footprints(model, sim.data)
        return self.home

    def place(self, pose):
        """Standing at (x, y, yaw)."""
        sim = self.sim
        sim.reset()
        sim.data.qpos[0], sim.data.qpos[1] = pose[0], pose[1]
        sim.data.qpos[3:7] = (math.cos(pose[2] / 2), 0.0, 0.0, math.sin(pose[2] / 2))
        mujoco.mj_forward(sim.model, sim.data)

    def paint(self, belief):
        for name, g in self.zone_geoms.items():
            self.sim.model.geom_rgba[g][3] = 0.04 + 0.36 * max(0.0, min(1.0, belief.get(name, 0.0)))

    def person(self, x=None, y=None, yaw=0.0, pose="lying"):
        """Margarete at (x, y); no point: out of sight. `pose` "lying": on her back, head towards `yaw` (radians);
        "standing": the same figure turned upright (her head-to-feet line, +x in sim/apartment_decor.xml, points
        up), feet on the floor, facing away from `yaw`."""
        data = self.sim.data
        if x is None:
            x, y = PARKED[0], PARKED[1] + 2.0
        yaw_q = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        if pose == "standing":
            pitch = np.array([math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0])  # -90 degrees about y
            quat = np.empty(4)
            mujoco.mju_mulQuat(quat, yaw_q, pitch)
            data.mocap_pos[self._person] = (x, y, STANDING_HEIGHT)
            data.mocap_quat[self._person] = quat
        else:
            data.mocap_pos[self._person] = (x, y, 0.0)
            data.mocap_quat[self._person] = yaw_q


class LiveRobot:
    """scene "floor": the robot alone on its 6 m floor. scene "apartment": the 50 m² room with furniture and
    zone patches (sim/apartment.xml); `load(flat)` arranges it and `paint(belief)` shades the zones.

    Drivers in the apartment, first choice first: "student" (app/student.py: the obstacle-avoiding TabPFN student
    from nav/dagger, through the TabPFN API; it walks straight to the end point with its own lidar map and camera,
    the A* route is only shown), then the a-to-b drivers, which walk a route of straight legs around the known
    furniture (plan_route): "tabpfn" the a-to-b controller built with `just controller`, "planner" the nav/dagger
    expert, "rule" the steer-to-bearing rule. A driver that cannot be loaded falls back to the next one. The
    student loads on a background thread; until it is ready the a-to-b driver walks. When the student gets stuck
    (the trainer's rule) or the API fails, the goto is handed to A* + the a-to-b driver (`fallbacks` in state).
    Every control step counts touches of walls and furniture (`contacts`). state()["driver"] says who walks."""

    def __init__(self, scene="floor", driver="tabpfn", controller="default", student=None, ab=None, realtime=True,
                 stuck_cycles=None, student_calls=None, spotter=None):
        """`student`: an app/student.py Student to use (tests pass a fake); with driver "student" and none given,
        one is made and loaded in the background. `ab`: the first a-to-b driver to try under the student (default
        the next in DRIVERS). `realtime` False runs the physics as fast as it goes (tests). `stuck_cycles`: the
        stuck rule's cycles (default the trainer's 40). `student_calls`: at most this many TabPFN API calls (each
        10,000 credits) for the student made here; None: as many as the credit floor allows. `spotter`: an
        app/spotter.py Spotter (the person detector, apartment only); it sees once per gait cycle while the robot
        walks, looks around (`look`) or watches (`watch N`)."""
        self.scene = scene
        self._driver, self._controller = driver, controller
        self._ab_first = ab
        self._realtime = realtime
        self._stuck_cycles = stuck_cycles
        if scene == "apartment" and driver == "student" and student is None:
            from student import Student
            student = Student(simple=not meshes_available(), max_calls=student_calls).load_async()
        self._student = student if scene == "apartment" else None
        self._spotter = spotter if scene == "apartment" else None
        self._look = {"running": False}
        self._contacts = None
        self._fallbacks, self._fallback = 0, None
        self._ab = None
        self.limit = APARTMENT if scene == "apartment" else ARENA
        self._commands = queue.Queue()
        self._lock = threading.Lock()
        self._frames = dict((view, b"") for view in VIEWS)
        self._wanted = dict((view, 0.0) for view in VIEWS)
        self._state = {}
        self._ready = threading.Event()
        self._error = None
        threading.Thread(target=self._run, name="sim", daemon=True).start()

    # ---- called from web threads ----

    def wait_ready(self, timeout=60):
        self._ready.wait(timeout)
        if self._error:
            raise self._error
        return self._ready.is_set()

    def command(self, text):
        """Queue a command; returns its normal form. Raises ValueError if it is not understood."""
        parsed = parse_command(text, self.limit)
        if parsed[0] == "random":
            parsed = ("target",) + random_point(self.state())
        elif parsed[0] == "target":  # walk to wherever the pillar is now
            state = self.state()
            parsed = ("goto", state.get("target_x", 1.0), state.get("target_y", 0.0))
        self._commands.put(parsed)
        if parsed[0] in ("goto", "forward", "turn", "look", "watch"):
            # Say so at once: the simulator thread may be busy (planning a route, TabPFN predicting) for a moment,
            # and a caller that waits for "idle" must not take the old state for "already done".
            with self._lock:
                self._state["mode"] = parsed[0] if parsed[0] in ("turn", "look", "watch") else "goto"
        if parsed[0] == "target":
            return "random end point {:.2f} {:.2f}".format(parsed[1], parsed[2])
        if parsed[0] == "spot":
            return "spot " + parsed[1]
        return " ".join([parsed[0]] + ["{:.2f}".format(math.degrees(v) if parsed[0] == "turn" else v)
                                       for v in parsed[1:]])

    def frame(self, view):
        """Latest JPEG of a view in VIEWS (b"" before the first one). Asking keeps the view rendered."""
        with self._lock:
            self._wanted[view] = time.monotonic()
            return self._frames.get(view, b"")

    def state(self):
        with self._lock:
            return dict(self._state)

    def load(self, flat):
        """Apartment scene: put the furniture and zone patches where `flat` (homes/flats.py) has them, and the
        robot at the dock."""
        self._commands.put(("layout", flat))

    def paint(self, belief):
        """Apartment scene: shade each zone patch by the belief {zone: probability} that she is there."""
        self._commands.put(("zones", dict(belief)))

    def person(self, x=None, y=None, yaw_deg=0.0, pose="lying"):
        """Apartment scene: Margarete (the figure in sim/apartment_decor.xml) at (x, y), lying on the floor with
        her head towards `yaw_deg`, or standing (`pose="standing"`); with no point, out of sight (nobody home)."""
        if x is None:
            x, y = PARKED[0], PARKED[1] + 2.0
        if pose not in POSES:
            raise ValueError("pose must be one of: " + ", ".join(POSES))
        self._commands.put(("person", float(x), float(y), math.radians(yaw_deg), pose))

    # ---- the simulator thread ----

    def _run(self):
        home = (0.0, 0.0, 0.0)  # where reset puts the robot: x, y, yaw
        expert, policy, driver, layout = None, None, "rule", 0
        legs, poses, prev = [], [], (0.0, 0.0)  # the route's points still to reach; the body pose at each cycle start
        reached = REACHED
        scene = None
        try:
            if self.scene == "apartment":
                scene = ApartmentScene()
                sim = scene.sim
                reached = APARTMENT_REACHED
                first = self._ab_first if self._ab_first in DRIVERS else self._driver
                wanted = DRIVERS[DRIVERS.index(first):] if first in DRIVERS else DRIVERS
                for name in wanted:  # the a-to-b driver: walks while the student loads, and takes over from it
                    if name == "student":
                        continue
                    try:
                        if name == "tabpfn":
                            policy = load_tabpfn(self._controller)
                            reached = 0.15
                        elif name == "planner":
                            if str(REPO) not in sys.path:
                                sys.path.insert(0, str(REPO))
                            from nav.dagger.expert import Expert
                            expert = Expert(goal_radius=0.25)
                        driver = name
                        break
                    except Exception as error:
                        print("[app] driver '{}' is not available ({}), trying the next".format(name, error), flush=True)
                self._contacts = ContactCounter(sim.model)
                print("[app] apartment scene, driver: {}{}".format(
                    "student ({}), then ".format(self._student.status) if self._student is not None else "", driver),
                    flush=True)
            else:
                sim = JetHexaSim()
            sim.set_target(1.0, 0.0)
            sim.model.vis.global_.offwidth, sim.model.vis.global_.offheight = VIEW_SIZE
            renderer = mujoco.Renderer(sim.model, height=VIEW_SIZE[1], width=VIEW_SIZE[0])
            cameras = {}
            apartment = self.scene == "apartment"
            for view, (distance, azimuth, elevation) in (("chase", (0.9, 0, -22)),
                                                          ("orbit", (2.6 if apartment else 1.2, 135, -34 if apartment else -28)),
                                                          ("overhead", (9.6 if apartment else 6.2, 90, -89.9))):
                cameras[view] = camera = mujoco.MjvCamera()
                camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
                camera.lookat[:] = (0.0, 0.0, 0.0) if view == "overhead" else sim.body_position()
            self._views = (renderer, cameras)
            self._ab = driver
            with self._lock:  # first pictures of every view, so a new page never starts empty
                self._wanted = dict((view, time.monotonic()) for view in VIEWS)
            self._publish(sim, 0, "idle", None)
            self._ready.set()
        except Exception as error:  # report a start-up failure to the web thread
            self._error = error
            self._ready.set()
            return

        mode, goal = "idle", None  # goal: (x, y) for "goto", a yaw for "turn"
        gait, cycle_at, stride, turn = PHASE0, -1.0, 0.0, 0.0
        step = 0
        walker = driver  # who walks the current goto: "student" or the a-to-b driver
        progress = Progress(self._stuck_cycles) if self._stuck_cycles else Progress()
        twin_checked = False
        route = []  # the A* route of the current goto (walked by the a-to-b driver, only shown for the student)
        next_tick = time.monotonic()

        def fall_back(reason, x, y):
            """The student hands this goto to A* + the a-to-b driver."""
            nonlocal legs, walker, poses, prev
            legs = plan_route(scene.shapes, scene.half, (x, y), goal)
            walker, poses, prev = driver, [], (0.0, 0.0)
            self._fallbacks += 1
            self._fallback = {"n": self._fallbacks, "reason": reason, "x": round(x, 2), "y": round(y, 2),
                              "goal": [round(goal[0], 2), round(goal[1], 2)], "driver": driver}
            print("[app] student -> {} (A* route): {}".format(driver, reason), flush=True)

        while True:
            while True:
                try:
                    cmd = self._commands.get_nowait()
                except queue.Empty:
                    break
                x, y, _ = sim.body_position()
                yaw = sim.body_yaw()
                if cmd[0] == "layout":  # apartment: a new flat (or the same flat with its loose pieces moved)
                    home = scene.load(cmd[1])
                    layout += 1
                    mode, goal = "idle", None
                elif cmd[0] == "zones":
                    scene.paint(cmd[1])
                    continue
                elif cmd[0] == "person":  # Margarete: where she lies or stands, or out of sight
                    scene.person(cmd[1], cmd[2], cmd[3], cmd[4] if len(cmd) > 4 else "lying")
                    continue
                elif cmd[0] == "reset":
                    if scene is not None:
                        scene.place(home)
                    else:
                        sim.reset()
                        sim.set_target(1.0, 0.0)
                    mode, goal = "idle", None
                elif cmd[0] == "target":  # move the pillar; a walk to the pillar follows it
                    sim.set_target(cmd[1], cmd[2])
                    if mode == "goto" and self.scene != "apartment":  # in the apartment the pillar is the person
                        goal = (cmd[1], cmd[2])
                        legs = [goal]
                    continue  # keep the gait clock: no jump in the legs
                elif cmd[0] == "stop":
                    mode, goal = "idle", None
                    self._look = dict(self._look, running=False)
                elif cmd[0] == "turn":
                    mode, goal = "turn", wrap(yaw + cmd[1])
                elif cmd[0] == "spot":  # the person detector: reset / reject / mark (no change to the walk)
                    spotter = self._spotter
                    if spotter is not None and spotter.ready:
                        getattr(spotter, cmd[1])()
                    continue
                elif cmd[0] in ("look", "watch"):  # turn once on the spot looking, or stand looking N cycles
                    mode, goal = cmd[0], None
                    self._look = {"running": True, "kind": cmd[0], "turned_deg": 0.0, "cycles": 0,
                                  "want": int(cmd[1]) if cmd[0] == "watch" else None, "blocked": False,
                                  "seen": False, "last_yaw": yaw}
                else:
                    if cmd[0] == "forward":
                        tx, ty = x + cmd[1] * math.cos(yaw), y + cmd[1] * math.sin(yaw)
                        tx, ty = max(-self.limit, min(self.limit, tx)), max(-self.limit, min(self.limit, ty))
                    else:
                        tx, ty = cmd[1], cmd[2]
                    # The pillar stays put: only `random` moves it. (Moving it to every goto end point made a
                    # misread "walk to the target", e.g. [[forward 1]], put the pillar right in front of the robot.)
                    mode, goal = "goto", (tx, ty)
                    walker = driver
                    if scene is not None:
                        route = plan_route(scene.shapes, scene.half, (x, y), goal)
                        legs = route
                        student = self._student
                        if student is not None and student.ready and not twin_checked:
                            try:
                                student.twin.check_compatible(sim.model)
                            except Exception as error:  # noqa: BLE001
                                student.status, student.reason = "failed", str(error)
                            twin_checked = True
                        if student is not None and student.ready:
                            walker, legs = "student", [goal]  # the student walks to the end point itself
                            student.start(goal)
                            progress.reset(math.hypot(goal[0] - x, goal[1] - y))
                    else:
                        route, legs = [], [goal]
                    poses, prev = [], (0.0, 0.0)
                if mode not in ("look", "watch") and self._look.get("running"):
                    self._look = dict(self._look, running=False)  # another command ended the look-around
                gait, cycle_at = PHASE0, -1.0

            x, y, _ = sim.body_position()
            yaw = sim.body_yaw()
            if self._contacts is not None and self._contacts.update(sim.data, (x, y)):
                print("[app] contact {} at ({:.2f}, {:.2f})".format(self._contacts.count, x, y), flush=True)
            goal_radius = STUDENT_REACHED if walker == "student" else reached
            if sim.body_upright() < FALLEN:
                mode, goal = "fallen", None
            elif mode == "goto" and math.hypot(goal[0] - x, goal[1] - y) < goal_radius:
                mode, goal, legs = "idle", None, []
            elif mode == "goto" and len(legs) > 1 and math.hypot(legs[0][0] - x, legs[0][1] - y) < WAYPOINT:
                legs = legs[1:]  # on to the next point of the route
            elif mode == "turn" and abs(wrap(goal - yaw)) < YAW_REACHED:
                mode, goal = "idle", None

            if mode in ("goto", "turn", "look", "watch"):
                if gait - cycle_at >= CYCLE:  # new steering command once per gait cycle
                    cycle_at = gait
                    spotter = self._spotter
                    if spotter is not None and spotter.ready and mode != "turn":  # the person detector looks
                        try:
                            spotter.see(sim)
                        except Exception as error:  # noqa: BLE001 - never let the detector stop the robot
                            spotter.status, spotter.reason = "failed", str(error)
                            print("[app] the person detector failed: {}".format(error), flush=True)
                        next_tick = time.monotonic()  # the physics waited while DINOv2 looked
                    if mode in ("look", "watch"):
                        look = self._look
                        look["turned_deg"] += abs(math.degrees(wrap(yaw - look["last_yaw"])))
                        look["last_yaw"] = yaw
                        confirmed = spotter is not None and spotter.ready and spotter.estimate.confirmed
                        look["seen"] = bool(confirmed)
                        stride, turn = 0.0, 0.0
                        if mode == "look":
                            if confirmed or look["turned_deg"] >= 360.0:
                                mode = "idle"
                            elif float(np.min(sim.lidar(36, 0.15, 8.0))) < LOOK_CLEARANCE:
                                look["blocked"], mode = True, "idle"  # no room to turn: report what was seen
                            else:
                                turn = LOOK_TURN
                        elif look["cycles"] >= look["want"]:
                            mode = "idle"
                        look["cycles"] += 1
                        if mode == "idle":
                            look["running"] = False
                            goal = None
                    if mode == "goto" and walker == "student":  # the student: own sensors, own map, the end point
                        if progress.update(math.hypot(goal[0] - x, goal[1] - y)):
                            fall_back("the student was stuck (not {:.1f} m closer in {} cycles)".format(progress.gain, progress.cycles), x, y)
                        else:
                            try:
                                stride, turn = self._student.act(sim, goal)
                            except Exception as error:  # noqa: BLE001 - API down, no credits: A* + a-to-b
                                fall_back("the student is unavailable ({})".format(error), x, y)
                            next_tick = time.monotonic()  # the physics waited while the student decided
                    leg = legs[0] if legs else goal
                    if mode in ("look", "watch", "idle"):
                        pass  # the look-around set stride and turn above (idle: it just ended)
                    elif mode == "goto" and walker == "student":
                        pass
                    elif mode == "goto" and policy is not None:  # the TabPFN a-to-b controller walks this leg
                        stride, turn = (float(v) for v in policy.act([nav_row((x, y, yaw), poses, prev, leg)])[0])
                        poses.append((x, y, yaw))
                        del poses[:-4]
                        prev = (stride, turn)
                        next_tick = time.monotonic()  # the physics waited while TabPFN predicted
                    elif mode == "goto" and expert is not None:  # the nav/dagger planner
                        stride, turn = expert.act({"key": (layout, goal), "footprints": scene.shapes,
                                                   "half": scene.half, "goal": goal, "pose": (x, y, yaw)})
                    elif mode == "goto":
                        dist = math.hypot(leg[0] - x, leg[1] - y)
                        bearing = wrap(math.atan2(leg[1] - y, leg[0] - x) - yaw)
                        turn = max(-MAX_TURN, min(MAX_TURN, 0.5 * bearing))
                        stride = MAX_STRIDE * max(0.0, math.cos(bearing)) ** 2 * min(1.0, dist / (2 * MAX_STEP))
                    else:
                        error = wrap(goal - yaw)
                        stride, turn = 0.0, max(-MAX_TURN, min(MAX_TURN, 1.5 * error))
                        if abs(turn) < 0.05:
                            turn = math.copysign(0.05, error)
                sim.set_targets(tripod_targets(sim.stand, gait, swing=stride, turn=turn))
                gait += CONTROL_DT
            else:
                sim.set_targets(sim.stand)
            sim.step(CONTROL_DT)
            step += 1
            following = mode == "goto" and walker != "student"
            self._route = [list(p) for p in (legs if following else route if mode == "goto" else [])]
            self._route_followed = following
            student = self._student
            if self.scene != "apartment":
                self._driver_name = "rule"
            elif mode == "goto":
                self._driver_name = walker
            else:
                self._driver_name = "student" if student is not None and student.ready else driver
            self._publish(sim, step, mode, goal)

            if not self._realtime:
                time.sleep(0)
                continue
            next_tick += CONTROL_DT
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:  # fell behind (laptop asleep, slow render): do not try to catch up
                next_tick = time.monotonic()

    def _publish(self, sim, step, mode, goal):
        renderer, cameras = self._views
        x, y, z = sim.body_position()
        yaw = math.degrees(sim.body_yaw())
        now = time.monotonic()
        with self._lock:
            watched = [view for view, at in self._wanted.items() if now - at < WATCHED]
        frames = {}
        for view, every in VIEWS.items():
            if view == "overhead":
                continue
            camera = cameras.get(view)
            if camera is not None:  # the outside views follow the robot smoothly, even unwatched
                for i, value in enumerate((x, y, z)):
                    camera.lookat[i] += 0.12 * (value - camera.lookat[i])
                if view == "chase":
                    camera.azimuth += 0.08 * (((yaw - camera.azimuth) + 180.0) % 360.0 - 180.0)  # behind it
        for view in watched:
            if step % VIEWS[view]:
                continue
            if view == "eye":
                frames[view] = to_jpeg(sim.render())
            else:
                renderer.update_scene(sim.data, camera=cameras[view])
                frames[view] = to_jpeg(renderer.render())
        tx, ty = sim.target()
        state = {"mode": mode, "scene": self.scene, "x": round(x, 3), "y": round(y, 3), "height": round(z, 3),
                 "yaw_deg": round(yaw, 1), "target_x": round(tx, 3),
                 "target_y": round(ty, 3), "sim_time": round(sim.time, 2),
                 # where the pillar is seen from the robot: metres, and degrees (positive = to its left)
                 "target_dist": round(math.hypot(tx - x, ty - y), 2),
                 "target_bearing_deg": round(math.degrees(wrap(math.atan2(ty - y, tx - x) - math.radians(yaw))))}
        if mode == "turn":
            state["turn_to_deg"] = round(math.degrees(goal), 1)
        if self.scene == "apartment":
            state["driver"] = getattr(self, "_driver_name", "rule")
            state["ab_driver"] = self._ab
            state["route"] = getattr(self, "_route", [])
            state["route_followed"] = getattr(self, "_route_followed", True)
            state["student"] = self._student.info() if self._student is not None else {"status": "off"}
            if self._contacts is not None:
                state["contacts"] = self._contacts.count
                state["contact_points"] = list(self._contacts.points)
            state["fallbacks"] = self._fallbacks
            state["fallback"] = self._fallback
            state["spotter"] = self._spotter.state() if self._spotter is not None else {"status": "off"}
            look = self._look
            state["look"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in look.items()
                             if k != "last_yaw"}
            if mode == "goto":
                state["goal_x"], state["goal_y"] = round(goal[0], 3), round(goal[1], 3)
        with self._lock:
            self._frames.update(frames)
            self._state = state

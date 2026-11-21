"""The simulated JetHexa running live in a background thread, for the web app.

One thread owns the simulator: it steps the physics in real time, steers the robot and renders JPEG
pictures (outside views named in VIEWS, and the robot's own camera), only of views someone watches. Other threads only
put commands on a queue and read the latest pictures and state. MuJoCo's renderer must be created
and used on one thread, which is why rendering happens here too.

Commands (the same words the chat understands): goto X Y, forward M, turn DEG, stop, reset, target (walk to the
red pillar), random (moves the red end-point pillar to a random spot; a robot walking to the pillar follows it).
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))

from jethexa_sim import JetHexaSim, tripod_targets  # noqa: E402

CONTROL_DT = 0.02  # seconds between new joint targets
CYCLE = 1.0  # seconds per gait cycle; the steering command changes once per cycle
PHASE0 = 0.25  # gait clock start: both tripods mid-sweep, so walking starts without a jump
MAX_STRIDE = 0.3  # radians of coxa swing
MAX_TURN = 0.25  # radians of left/right swing difference; positive turns left
MAX_STEP = 0.13  # metres per cycle at full stride
REACHED = 0.10  # metres: close enough to a goto point
YAW_REACHED = 0.06  # radians: close enough to a turn target
ARENA = 2.6  # metres: goto points are kept inside this square (the floor is 3 m each way)
RANDOM_SPAN = 2.0  # metres: random end points lie in this square, so they stay well inside the floor
RANDOM_MIN_DIST = 0.8  # metres: a random end point is at least this far from the robot
FALLEN = 0.5  # body_upright() below this counts as a fall
JPEG_QUALITY = 72
VIEW_SIZE = (1280, 720)  # width, height of the outside views
# Picture views: control steps between pictures (4 = 12.5 per second). "eye" is the robot's own camera.
VIEWS = {"chase": 4, "orbit": 4, "overhead": 6, "eye": 6}
WATCHED = 3.0  # seconds: a view is rendered only while someone asked for it this recently

HELP = ("goto X Y (metres, floor is -{a} to {a}), forward M (metres), turn DEG (positive = left), "
        "stop, reset, target (walk to the red pillar), random (new random end point)").format(a=ARENA)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def parse_command(text):
    """Turn "goto 1 0.5" and friends into a tuple, or raise ValueError with a message for the user."""
    words = text.replace(",", " ").split()
    if not words:
        raise ValueError("empty command")
    name, args = words[0].lower().lstrip("/"), words[1:]
    try:
        numbers = [float(a) for a in args]
    except ValueError:
        raise ValueError("'{}' needs numbers, got: {}".format(name, " ".join(args)))
    if any(not math.isfinite(n) for n in numbers):
        raise ValueError("numbers must be finite")
    if name == "goto" and len(numbers) == 2:
        return ("goto", max(-ARENA, min(ARENA, numbers[0])), max(-ARENA, min(ARENA, numbers[1])))
    if name == "forward" and len(numbers) == 1 and 0 < numbers[0] <= 2 * ARENA:
        return ("forward", numbers[0])
    if name == "turn" and len(numbers) == 1 and abs(numbers[0]) <= 360:
        return ("turn", math.radians(numbers[0]))
    if name in ("stop", "reset", "random", "target") and not numbers:
        return (name,)
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


class LiveRobot:
    def __init__(self):
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
        parsed = parse_command(text)
        if parsed[0] == "random":
            parsed = ("target",) + random_point(self.state())
        elif parsed[0] == "target":  # walk to wherever the pillar is now
            state = self.state()
            parsed = ("goto", state.get("target_x", 1.0), state.get("target_y", 0.0))
        self._commands.put(parsed)
        if parsed[0] == "target":
            return "random end point {:.2f} {:.2f}".format(parsed[1], parsed[2])
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

    # ---- the simulator thread ----

    def _run(self):
        try:
            sim = JetHexaSim()
            sim.set_target(1.0, 0.0)
            sim.model.vis.global_.offwidth, sim.model.vis.global_.offheight = VIEW_SIZE
            renderer = mujoco.Renderer(sim.model, height=VIEW_SIZE[1], width=VIEW_SIZE[0])
            cameras = {}
            for view, (distance, azimuth, elevation) in (("chase", (0.9, 0, -22)), ("orbit", (1.2, 135, -28)),
                                                          ("overhead", (6.2, 90, -89.9))):
                cameras[view] = camera = mujoco.MjvCamera()
                camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
                camera.lookat[:] = (0.0, 0.0, 0.0) if view == "overhead" else sim.body_position()
            self._views = (renderer, cameras)
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
        next_tick = time.monotonic()
        while True:
            while True:
                try:
                    cmd = self._commands.get_nowait()
                except queue.Empty:
                    break
                x, y, _ = sim.body_position()
                yaw = sim.body_yaw()
                if cmd[0] == "reset":
                    sim.reset()
                    sim.set_target(1.0, 0.0)
                    mode, goal = "idle", None
                elif cmd[0] == "target":  # move the pillar; a walk to the pillar follows it
                    sim.set_target(cmd[1], cmd[2])
                    if mode == "goto":
                        goal = (cmd[1], cmd[2])
                    continue  # keep the gait clock: no jump in the legs
                elif cmd[0] == "stop":
                    mode, goal = "idle", None
                elif cmd[0] == "turn":
                    mode, goal = "turn", wrap(yaw + cmd[1])
                else:
                    if cmd[0] == "forward":
                        tx, ty = x + cmd[1] * math.cos(yaw), y + cmd[1] * math.sin(yaw)
                        tx, ty = max(-ARENA, min(ARENA, tx)), max(-ARENA, min(ARENA, ty))
                    else:
                        tx, ty = cmd[1], cmd[2]
                    sim.set_target(tx, ty)
                    mode, goal = "goto", (tx, ty)
                gait, cycle_at = PHASE0, -1.0

            x, y, _ = sim.body_position()
            yaw = sim.body_yaw()
            if sim.body_upright() < FALLEN:
                mode, goal = "fallen", None
            elif mode == "goto" and math.hypot(goal[0] - x, goal[1] - y) < REACHED:
                mode, goal = "idle", None
            elif mode == "turn" and abs(wrap(goal - yaw)) < YAW_REACHED:
                mode, goal = "idle", None

            if mode in ("goto", "turn"):
                if gait - cycle_at >= CYCLE:  # new steering command once per gait cycle
                    cycle_at = gait
                    if mode == "goto":
                        dist = math.hypot(goal[0] - x, goal[1] - y)
                        bearing = wrap(math.atan2(goal[1] - y, goal[0] - x) - yaw)
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
            self._publish(sim, step, mode, goal)

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
        state = {"mode": mode, "x": round(x, 3), "y": round(y, 3), "height": round(z, 3),
                 "yaw_deg": round(yaw, 1), "target_x": round(tx, 3),
                 "target_y": round(ty, 3), "sim_time": round(sim.time, 2),
                 # where the pillar is seen from the robot: metres, and degrees (positive = to its left)
                 "target_dist": round(math.hypot(tx - x, ty - y), 2),
                 "target_bearing_deg": round(math.degrees(wrap(math.atan2(ty - y, tx - x) - math.radians(yaw))))}
        if mode == "turn":
            state["turn_to_deg"] = round(math.degrees(goal), 1)
        with self._lock:
            self._frames.update(frames)
            self._state = state

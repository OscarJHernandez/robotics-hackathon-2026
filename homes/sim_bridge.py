"""The search as the robot's orchestrator: it says where to go, the simulated JetHexa walks there.

    uv run --group ml python homes/sim_bridge.py                # one search in MuJoCo, with the DAgger expert driving
    uv run --group ml python homes/sim_bridge.py --flat 3 --zone kitchen --miss-rate 0.2

This joins the two halves of the work:

    homes/search.py     Search.goal()  -> the next zone and its viewing pose (x, y)      "where to go"
    nav/dagger/         SimEnv + a driver (the expert, or the TabPFN student)            "how to walk there"

`FlatEnv` is nav/dagger's `SimEnv` with the furniture of a homes/ flat instead of a random scene, the robot starting
at the dock, and a goal that can be changed between walks. `search_in_sim` runs the loop: ask the search for a goal,
walk there with the driver, look (the mock observation for now), report, repeat until found.

A driver is anything with `act(state) -> (stride, turn)`, where `state` is `SimEnv.state()`. The DAgger expert is
one; the TabPFN student needs camera pictures, so make the env with `render=True` and wrap its featurizer.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import flats as F  # noqa: E402
import search as S  # noqa: E402
from nav.common import PHASE0  # noqa: E402
from nav.dagger.env import SimEnv  # noqa: E402
from nav.dagger.expert import Expert  # noqa: E402

AWAY = (50.0, 50.0)  # where the target pillar is parked when nobody is home


class FlatEnv(SimEnv):
    """SimEnv in a homes/ flat. The red target pillar stands in for the person, at `person` (x, y) or parked far
    away when she is out; the goal the robot walks to is set separately with `set_goal`."""

    def __init__(self, flat, person=None, render=False, **kw):
        super().__init__(**kw)
        self.render_pictures = render
        self.load(flat, person)

    def load(self, flat, person=None):
        self.flat = flat
        self.layout = F.arena_layout(flat)
        self.arena.apply(self.layout)
        self._footprints = None
        self.sim.set_target(*(person or AWAY))
        dock = flat["dock"]
        self.start = (dock["x"], dock["y"], math.atan2(-dock["y"], -dock["x"]))  # at the dock, facing the room
        self.arena.place(*self.start)
        self.sim.set_target(*(person or AWAY))  # place() resets the simulation, which moves the pillar back
        self.goal = (dock["x"], dock["y"])
        self.gait, self.steps, self.path_length, self.done = PHASE0, 0, 0.0, False
        self.start_dist, self.best_dist, self.since_gain = 0.0, 0.0, 0  # SimEnv.step's stuck rule needs these
        self.scene_id = "flat-{}".format(flat["flat_id"])

    def set_goal(self, x, y):
        """A new place to walk to, from where the robot stands now."""
        self.goal = (float(x), float(y))
        self.steps, self.done = 0, False
        px, py, _ = self.pose()
        self.start_dist = self.best_dist = math.hypot(px - self.goal[0], py - self.goal[1])
        self.since_gain = 0

    def observe(self):
        if self.render_pictures:
            return super().observe()
        return {"robot_pose": self.pose(), "goal_point": self.goal}  # no camera: enough for the expert


def walk_to(env, driver, x, y):
    """Walk the robot to (x, y) with the driver. Returns (reason, gait cycles, metres): reason is "goal" on arrival,
    else "collision", "fell" or "timeout"."""
    env.set_goal(x, y)
    before = env.path_length
    if math.hypot(env.pose()[0] - x, env.pose()[1] - y) < env.goal_radius:
        return "goal", 0, 0.0
    info = {"done_reason": "timeout"}
    while not env.done:
        _, info = env.step(driver.act(env.state()))
    return info["done_reason"], env.steps, env.path_length - before


def search_in_sim(flat, probs, truth, person=None, miss_rate=S.MISS_RATE, rng=None, driver=None, env=None, log=print):
    """The whole loop in the simulator. `truth`: the zone she is in, or "out"; `person`: her (x, y), for the pillar.
    Returns (search, legs): the finished `Search` and one dict per walk (zone, reason, cycles, metres, seen)."""
    rng = rng or np.random.default_rng(0)
    driver = driver or Expert()
    own = env is None
    env = env or FlatEnv(flat, person)
    search = S.Search(flat, probs, miss_rate)
    legs = []
    try:
        while (goal := search.goal()) is not None:
            reason, cycles, metres = walk_to(env, driver, goal["x"], goal["y"])
            if reason != "goal":
                legs.append({"zone": goal["zone"], "reason": reason, "cycles": cycles, "metres": metres, "seen": None})
                log("  walk to the {} failed: {}".format(goal["zone"], reason))
                break
            seen = bool(S.observe(goal["zone"], truth, rng, miss_rate))  # the mock look; a detector goes here later
            search.report(seen)
            legs.append({"zone": goal["zone"], "reason": reason, "cycles": cycles, "metres": metres, "seen": seen})
            x, y, _ = env.pose()
            log("  look {}: {:<12} {:>3.0f}%  walked {:>4.1f} m in {:>3} s, robot at ({:+.2f}, {:+.2f}) in the {}  ->  {}"
                .format(len(legs), goal["zone"], 100 * goal["p"], metres, cycles, x, y, F.zone_at(flat, x, y),
                        "FOUND" if seen else "nobody seen"))
    finally:
        if own:
            env.close()
    return search, legs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--flat", type=int, default=0, help="which random flat (the seed of its layout)")
    parser.add_argument("--zone", default="kitchen", choices=F.CLASSES, help="where she really is")
    parser.add_argument("--miss-rate", type=float, default=S.MISS_RATE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    flat = F.random_flat(np.random.default_rng([0, args.flat]), args.flat)
    # A belief for the demo: mostly living room, so a search for someone elsewhere needs several looks.
    probs = {"bedroom": 0.15, "living_room": 0.40, "dining_area": 0.12, "kitchen": 0.22, "hallway": 0.06, "out": 0.05}
    person = None if args.zone == "out" else (F.zone(flat, args.zone)["person_x"], F.zone(flat, args.zone)["person_y"])
    print("flat {}: she is {}. The search proposes a zone, the simulated JetHexa walks there, looks, and the belief "
          "is updated.".format(args.flat, "out" if args.zone == "out" else "in the " + args.zone))
    search, legs = search_in_sim(flat, probs, args.zone, person, args.miss_rate, np.random.default_rng(args.seed))
    walked = sum(leg["metres"] for leg in legs)
    print("{} after {} look{}; the robot walked {:.1f} m in {} s of simulated time (straight lines: {:.1f} m)".format(
        search.status, len(search.steps), "" if len(search.steps) == 1 else "s", walked,
        sum(leg["cycles"] for leg in legs), search.metres))


if __name__ == "__main__":
    main()

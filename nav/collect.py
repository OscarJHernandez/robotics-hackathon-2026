"""Walk the robot with random commands and record every gait cycle (the data for round 0).

    uv run --group ml python nav/collect.py --episodes 150 --cycles 30 [--images]

Each episode starts at the origin facing a random heading, with the red marker at a random point.
Commands are random and held for 1 to 8 cycles, so stretches of constant command can later be
relabelled as "this command reaches that point in K cycles" (see hindsight.py).
"""

import argparse
import math
import time

import numpy as np

from common import MAX_HORIZON, STRIDE_RANGE, TURN_RANGE, JetHexaSim, Walker
from db import Recorder
from images import save_camera


def random_episode(sim, rng, recorder, episode, cycles):
    yaw = rng.uniform(-math.pi, math.pi)
    dist, bearing = rng.uniform(0.3, 1.5), rng.uniform(-math.pi, math.pi)
    target = (dist * math.cos(yaw + bearing), dist * math.sin(yaw + bearing))
    walker = Walker(sim, (0.0, 0.0, yaw), target)
    rows, hold, left = [], -1, 0
    while walker.cycle < cycles and not walker.fell:
        if left == 0:
            hold += 1
            left = int(rng.integers(1, MAX_HORIZON + 1))
            command = (rng.uniform(*STRIDE_RANGE), rng.uniform(*TURN_RANGE))
        row = walker.observe()
        row["image"] = save_camera(sim, recorder, episode, walker.cycle)
        row["source"] = "random"
        row["hold"] = hold
        row.update(walker.run_cycle(*command))
        rows.append(row)
        left -= 1
    recorder.add_episode(episode, rows, walker.steps, walker.summary())
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--cycles", type=int, default=30, help="gait cycles (seconds) per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--images", action="store_true", help="also save the camera picture of every cycle")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    sim = JetHexaSim()
    recorder = Recorder("collect", policy="random", settings=vars(args), images=args.images)
    start, total = time.time(), 0
    for episode in range(args.episodes):
        total += random_episode(sim, rng, recorder, episode, args.cycles)
        if (episode + 1) % 25 == 0:
            print("{} episodes, {} rows, {:.0f} s".format(episode + 1, total, time.time() - start), flush=True)
    sim.close()
    print("recorded run", recorder.close(), "with", total, "rows")


if __name__ == "__main__":
    main()

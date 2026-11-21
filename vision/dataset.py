"""Make a labeled camera dataset in the arena: just vision-data --episodes 60 --poses 16

One episode is one random layout (furniture positions and colors, target). In each, the robot is put at
several random poses, half of them followed by a short random walk, and the camera picture is saved with the
class of every pixel (floor, wall, obstacle, target) from the simulator's segmentation.

Output in data/vision/<run>/: images/EEEE_PP.png, labels.npz (the class maps) and meta.json.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from scene import CLASSES, IGNORE, REPO, Arena


def data_dir():
    """Where vision data lives: $VISION_DATA_DIR, else data/vision in the main checkout.

    The main checkout (not a temporary git worktree) is used so data survives across worktrees.
    """
    env = os.environ.get("VISION_DATA_DIR")
    if env:
        path = Path(env)
    else:
        root = REPO
        try:
            common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                    cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
            root = Path(common).parent
        except (OSError, subprocess.CalledProcessError):
            pass
        path = root / "data" / "vision"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(name):
    """The folder of run `name`; "latest" picks the newest run."""
    if name == "latest":
        runs = sorted(p for p in data_dir().iterdir() if (p / "meta.json").exists())
        if not runs:
            raise SystemExit("no vision runs yet: run `just vision-data` first")
        return runs[-1]
    path = data_dir() / name
    if not (path / "meta.json").exists():
        raise SystemExit("no such run: {}".format(path))
    return path


def generate(out, episodes, poses, seed, simple=None, log=print):
    """Write `episodes` x `poses` pictures with class maps into `out`. Returns the meta dict."""
    rng = np.random.default_rng(seed)
    arena = Arena(simple=simple)
    (out / "images").mkdir(parents=True, exist_ok=True)
    frames, layouts, maps = [], [], []
    try:
        for episode in range(episodes):
            layouts.append(arena.randomize(rng))
            for k in range(poses):
                while True:
                    arena.sample_pose(rng)
                    walked = k % 2 == 1
                    if not walked or arena.wander(rng):  # a fall during the walk: draw another pose
                        break
                rgb = arena.sim.render()
                classes = arena.class_map()
                name = "images/{:04d}_{:02d}.png".format(episode, k)
                Image.fromarray(rgb).save(out / name)
                maps.append(classes)
                x, y, yaw = arena.pose()
                frames.append({"image": name, "episode": episode, "pose": [x, y, yaw], "walked": walked})
            if (episode + 1) % 10 == 0 or episode + 1 == episodes:
                log("episode {}/{}".format(episode + 1, episodes))
    finally:
        arena.close()
    maps = np.stack(maps)
    np.savez_compressed(out / "labels.npz", classes=maps)
    share = {name: float((maps == i).mean()) for i, name in enumerate(CLASSES)}
    share["ignore"] = float((maps == IGNORE).mean())
    meta = {"seed": seed, "episodes": episodes, "poses": poses, "classes": list(CLASSES), "ignore": IGNORE,
            "pixel_share": share, "layouts": layouts, "frames": frames}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=60, help="random layouts")
    parser.add_argument("--poses", type=int, default=16, help="pictures per layout")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="run name (default: date and time)")
    args = parser.parse_args()
    out = data_dir() / (args.name or time.strftime("%Y%m%d-%H%M%S"))
    meta = generate(out, args.episodes, args.poses, args.seed)
    print("{} pictures in {}".format(len(meta["frames"]), out))
    print("share of pixels:", ", ".join("{} {:.1%}".format(k, v) for k, v in meta["pixel_share"].items()))


if __name__ == "__main__":
    main()

"""Walk the robot from a to b with a TabPFN controller, record it and make a video.

    uv run --group ml python nav/run.py --controller default --target 1.0 0.6 [--yaw 3.0]

Writes a `demo` run to the database and, in its run folder, `demo.gif` (top view with the robot's own
camera in the corner) and `path.png` (top-down path with the command of every cycle).
"""

import argparse
import math

import numpy as np
import pandas as pd

from common import REACHED, JetHexaSim, Walker
from db import Recorder
from images import follow_view, save_camera, write_gif


def compose(sim):
    """Top view with the robot camera's picture in the top-left corner."""
    frame = follow_view(sim).copy()
    eye = sim.render()[::3, ::3]  # 120 x 214
    h, w = eye.shape[:2]
    frame[8:8 + h + 4, 8:8 + w + 4] = 255
    frame[10:10 + h, 10:10 + w] = eye
    return frame


def demo(policy, start, target, controller_name, max_cycles=40, fps=10):
    sim = JetHexaSim()
    recorder = Recorder("demo", policy=policy.name, model=controller_name,
                        settings={"start": list(start), "target": list(target)}, images=True)
    walker = Walker(sim, start, target)
    frames, rows = [compose(sim)], []
    every = max(1, int(round(1.0 / fps / 0.02)))
    while not walker.done and walker.cycle < max_cycles:
        row = walker.observe()
        row["image"] = save_camera(sim, recorder, 0, walker.cycle)
        row["source"] = policy.name
        row["hold"] = -1
        stride, turn = policy.act([row])[0]
        # run the cycle in slices so the video gets a frame every 0.1 s
        outcome = run_cycle_with_frames(walker, stride, turn, frames, every)
        row.update(outcome)
        rows.append(row)
    summary = walker.summary()
    recorder.add_episode(0, rows, walker.steps, summary, task=-1)
    gif = recorder.folder / "demo.gif"
    write_gif(gif, frames + [frames[-1]] * fps, fps=fps)
    plot = recorder.folder / "path.png"
    plot_path(walker, rows, plot, policy.name)
    sim.close()
    recorder.close()
    return summary, gif, plot, recorder.run_id


def run_cycle_with_frames(walker, stride, turn, frames, every):
    """Like Walker.run_cycle, adding a video frame every `every` control steps."""
    sim = walker.sim
    original_step = sim.step
    count = [0]

    def step_and_film(seconds):
        original_step(seconds)
        count[0] += 1
        if count[0] % every == 0:
            frames.append(compose(sim))

    sim.step = step_and_film
    try:
        return walker.run_cycle(stride, turn)
    finally:
        sim.step = original_step


def plot_path(walker, rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = np.array(walker.steps)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    ax.plot(steps[:, 1], steps[:, 2], color="#2a6fdb", lw=2, label="body path")
    x0, y0, yaw0 = walker.start
    ax.annotate("", xy=(x0 + 0.12 * math.cos(yaw0), y0 + 0.12 * math.sin(yaw0)), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.plot([x0], [y0], "ko", label="a (start, arrow = heading)")
    tx, ty = walker.target
    ax.add_patch(plt.Circle((tx, ty), REACHED, color="#d62728", alpha=0.2))
    ax.plot([tx], [ty], "o", color="#d62728", label="b (target, 0.1 m circle)")
    frame = pd.DataFrame(rows)
    ax.plot(frame["x"], frame["y"], ".", color="#2a6fdb", ms=6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("{}: {} in {} cycles".format(title, "reached b" if walker.reached else "did not reach b",
                                             walker.cycle))
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    bx.step(frame["cycle"], frame["stride"], where="post", label="stride (rad)")
    bx.step(frame["cycle"], frame["turn"], where="post", label="turn (rad, + = left)")
    bx.plot(frame["cycle"], frame["target_dist"], "k--", lw=1, label="distance to b (m)")
    bx.set_xlabel("gait cycle (1 s each)")
    bx.set_title("TabPFN's command every cycle")
    bx.legend(fontsize=8)
    bx.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    import controller
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller", default="default", help="controller name (see `just controller`)")
    parser.add_argument("--target", type=float, nargs=2, default=(1.0, 0.6), metavar=("X", "Y"))
    parser.add_argument("--yaw", type=float, default=0.0, help="start heading in radians (start is at 0, 0)")
    args = parser.parse_args()
    policy = controller.load(args.controller)
    summary, gif, plot, run_id = demo(policy, (0.0, 0.0, args.yaw), tuple(args.target), args.controller)
    print("{}: {} in {} cycles ({:.0f} s), final distance {:.3f} m, path {:.2f} m".format(
        policy.name, "reached b" if summary["reached"] else "did not reach b", summary["cycles"],
        summary["seconds"], summary["final_dist"], summary["path_length"]))
    print("run", run_id)
    print("video", gif)
    print("plot", plot)


if __name__ == "__main__":
    main()

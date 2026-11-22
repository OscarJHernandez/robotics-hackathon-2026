"""Film recorded DAgger episodes: python -m nav.dagger.video --run NAME --round 0 --episodes 22 475

Each episode is replayed exactly from the dataset (same scene, start seed and executed commands; the
physics is deterministic) and checked against the recorded poses. Frames (1280 x 720, 2x real time):

    robot camera with the patch probe's classes | chase camera
    top-down map: furniture, path, target         | step, commands (expert / student / executed), outcome

Writes data/nav/dagger/<run>/videos/round<R>-episode<E>.mp4 (needs ffmpeg).
"""

import argparse
import math
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from .config import load_config, run_dir, vision_dir  # noqa: E402
from .env import SimEnv  # noqa: E402
from .expert import Expert  # noqa: E402
from .worker import round_number  # noqa: E402

from features import extract, load_dino  # noqa: E402
from probe import COLORS, load_probe, predict_probs  # noqa: E402
from scene import HALF  # noqa: E402

W, H = 640, 360
FRAME_EVERY = 4  # control steps (20 ms each) per video frame: 12.5 frames per simulated second
FPS = 25  # so the video plays at 2x real time
REASON_COLOR = {"goal": (40, 160, 70), "collision": (210, 40, 40), "timeout": (200, 140, 0), "fell": (150, 60, 160)}


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class Filmer:
    def __init__(self, env, probe, dino):
        self.env, self.probe, self.dino = env, probe, dino
        model = env.sim.model
        self.chase = mujoco.Renderer(model, height=H, width=W)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = model.body("body").id
        self.camera.distance, self.camera.elevation = 1.6, -28.0
        self.option = mujoco.MjvOption()
        self.path = []

    def eye_with_classes(self):
        rgb = self.env.sim.render()
        classes = predict_probs(self.probe, extract(self.dino, rgb[None])[0]).argmax(-1)
        color = np.asarray(Image.fromarray(COLORS[classes].astype(np.uint8)).resize((W, H), Image.NEAREST), np.float32)
        return (0.6 * rgb + 0.4 * color).astype(np.uint8)

    def chase_view(self):
        self.camera.azimuth = math.degrees(self.env.pose()[2]) + 180.0  # from behind the robot
        self.chase.update_scene(self.env.sim.data, camera=self.camera, scene_option=self.option)
        return self.chase.render().copy()

    def map_view(self, footprints, goal):
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_xlim(-HALF - 0.1, HALF + 0.1)
        ax.set_ylim(-HALF - 0.1, HALF + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(plt.Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, fill=False, lw=2, color="0.3"))
        for f in footprints:
            if f["kind"] == "circle":
                ax.add_patch(plt.Circle((f["x"], f["y"]), f["r"], color="#d08a2a"))
            elif max(abs(f["x"]), abs(f["y"])) < HALF:  # furniture; the walls are the frame
                c, s = math.cos(f["yaw"]), math.sin(f["yaw"])
                corner = (f["x"] - c * f["hx"] + s * f["hy"], f["y"] - s * f["hx"] - c * f["hy"])
                ax.add_patch(plt.Rectangle(corner, 2 * f["hx"], 2 * f["hy"], angle=math.degrees(f["yaw"]),
                                           color="#d08a2a"))
        ax.add_patch(plt.Circle(goal, self.env.goal_radius, color="#e03030", alpha=0.35))
        ax.plot(*goal, marker="o", color="#e03030")
        if self.path:
            xs, ys = zip(*self.path)
            ax.plot(xs, ys, color="#2a6fdb", lw=2)
        x, y, yaw = self.env.pose()
        ax.arrow(x, y, 0.3 * math.cos(yaw), 0.3 * math.sin(yaw), width=0.04, color="#123a80")
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return np.asarray(Image.fromarray(image).resize((W, H)))

    def info_view(self, title, row, step, steps, outcome):
        panel = Image.new("RGB", (W, H), (248, 248, 246))
        draw = ImageDraw.Draw(panel)
        big, small = font(26), font(20)
        draw.text((24, 18), title, fill=(20, 20, 20), font=big)
        lines = [
            "step {} of {}   (one step = one 1 s gait cycle)".format(step + 1, steps),
            "",
            "expert   stride {:.2f}  turn {:+.2f}".format(row["expert_stride"], row["expert_turn"]),
            "student  " + ("-   (round 0: expert only)" if pd.isna(row["student_stride"])
                           else "stride {:.2f}  turn {:+.2f}".format(row["student_stride"], row["student_turn"])),
            "executed stride {:.2f}  turn {:+.2f}  ({})".format(
                row["executed_stride"], row["executed_turn"], "expert" if row["used_expert"] else "student"),
            "label    = the expert's command",
            "",
            "goal {:.2f} m away".format(math.hypot(row["goal_x"] - self.env.pose()[0], row["goal_y"] - self.env.pose()[1])),
        ]
        for i, line in enumerate(lines):
            draw.text((24, 70 + 30 * i), line, fill=(40, 40, 40), font=small)
        if outcome:
            draw.rectangle((24, H - 64, W - 24, H - 20), fill=REASON_COLOR.get(outcome, (90, 90, 90)))
            draw.text((40, H - 58), "episode ends: " + outcome, fill=(255, 255, 255), font=big)
        return np.asarray(panel)

    def frame(self, footprints, goal, title, row, step, steps, outcome=""):
        top = np.concatenate([self.eye_with_classes(), self.chase_view()], axis=1)
        bottom = np.concatenate([self.map_view(footprints, goal), self.info_view(title, row, step, steps, outcome)], axis=1)
        return np.concatenate([top, bottom], axis=0)


def film(run, round_, episode, config, probe, dino, out, table=None, label=None):
    if table is None:
        table = pd.read_parquet(run_dir(run) / "round_{}.parquet".format(round_number(round_)))
    rows = table[table["episode"] == episode].sort_values("step").reset_index(drop=True)
    if rows.empty:
        raise SystemExit("episode {} is not in round {}".format(episode, round_))
    env_cfg = config["env"]
    expert = Expert.from_config(config["expert"], env_cfg["goal_radius"])
    env = SimEnv(simple=env_cfg["simple"], goal_radius=env_cfg["goal_radius"], max_steps=env_cfg["max_steps"],
                 min_start_dist=env_cfg["min_start_dist"], expert=expert,
                 target_clearance=env_cfg.get("target_clearance"), target_wall=env_cfg.get("target_wall"))
    env.reset(int(rows["scene_id"][0]), int(rows["seed"][0]))
    filmer = Filmer(env, probe, dino)
    footprints, goal = env.footprints(), env.goal
    outcome = rows["done_reason"].iloc[-1]
    share = rows["used_expert"].mean()
    driver = ("EXPERT driving" if share == 1 else "STUDENT alone (expert only labels)" if share == 0
              else "mixed: expert {:.0%} of steps".format(share))
    title = "{} ep. {}: {}".format(label or "round {}".format(round_), episode, driver)
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", "{}x{}".format(2 * W, 2 * H), "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                               "-pix_fmt", "yuv420p", "-crf", "23", str(out)], stdin=subprocess.PIPE)
    deviation = 0.0
    counter = {"n": 0}
    for i, row in rows.iterrows():
        x, y, _ = env.pose()
        deviation = max(deviation, math.hypot(x - row["x"], y - row["y"]))  # replay vs recording

        def capture(e, row=row, i=i):
            counter["n"] += 1
            if counter["n"] % 2 == 0:
                filmer.path.append(e.pose()[:2])
            if counter["n"] % FRAME_EVERY == 0:
                ffmpeg.stdin.write(filmer.frame(footprints, goal, title, row, i, len(rows)).tobytes())

        _, info = env.step((row["executed_stride"], row["executed_turn"]), on_control=capture)
        if info["done"]:
            break
    last = rows.iloc[-1]
    for _ in range(2 * FPS):  # hold the last frame for 2 s with the outcome
        ffmpeg.stdin.write(filmer.frame(footprints, goal, title, last, len(rows) - 1, len(rows), outcome).tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    env.close()
    filmer.chase.close()
    return {"episode": episode, "steps": len(rows), "outcome": outcome, "replay_outcome": info["done_reason"],
            "max_replay_deviation_m": round(deviation, 4), "video": str(out)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default="dagger1")
    parser.add_argument("--round", default="0")
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment", default=None, help="v2: film episodes of data/nav/dagger/v2/experiments/<id>")
    args = parser.parse_args()
    config = load_config(args.config)
    probe = load_probe(vision_dir() / config["features"]["probe"])
    dino = load_dino()
    table = None
    if args.experiment:
        from .config import experiment_dir
        folder = experiment_dir(args.experiment) / "videos"
        table = pd.read_parquet(experiment_dir(args.experiment) / "rows.parquet")
    else:
        folder = run_dir(args.run) / "videos"
    folder.mkdir(exist_ok=True)
    for episode in args.episodes:
        name = "episode{}.mp4".format(episode) if args.experiment else "round{}-episode{}.mp4".format(
            round_number(args.round), episode)
        label = "{} test".format(args.experiment.rsplit("-", 1)[-1]) if args.experiment else None
        print(film(args.run, args.round, episode, config, probe, dino, folder / name, table, label), flush=True)


if __name__ == "__main__":
    main()

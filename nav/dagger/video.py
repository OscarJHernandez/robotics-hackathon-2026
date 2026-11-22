"""Film recorded DAgger episodes: python -m nav.dagger.video --run NAME --round 0 --episodes 22 475

Each episode is replayed exactly from the dataset (same scene, start seed and executed commands; the
physics is deterministic) and checked against the recorded poses. Frames (1280 x 720, 2x real time):

    robot camera with the patch probe's classes | chase camera
    top-down map: furniture, path, target, lidar  | step, commands (expert / student / executed), safety, outcome
    lidar ranges as a heatmap around the robot    | DINOv2 horizon strip: its 3 PCA values as R, G, B

Frames are 1280 x 1080. Lidar colours everywhere: red = near, blue = DEPTH_MAX (5 m) or more. The map
draws the live 36 beams in those colours; the lidar panel shows the same 36 ranges around the robot
(heading up, one 10 degree wedge per beam, wedge length = range up to 3 m) and unrolled as a bar from
behind-left through ahead to behind-right. The strip panel colours the 23 horizon cells the student sees
(dp_* columns of the recorded row; R, G, B = PCA components 1, 2, 3, each scaled by its 2nd to 98th
percentile over the round) on a dimmed camera picture, with an enlarged copy of the same strip below it.
The map also shows the legs' reach (0.29 m, grey circle) and the turn-sweep tolerance (0.35 m, dashed).

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


DEPTH_MAX = 5.0  # metres: the depth heatmap's colour scale
STRIP_ROWS = (12, 14)  # the 2 patch rows (of 26) on the horizon that the strip cells average


def strip_scale(table):
    """Per PCA component (2nd, 98th percentile) over a round's dp_* columns, or None without them."""
    cols = [c for c in table.columns if c.startswith("dp_")]
    if not cols:
        return None
    values = table[cols].to_numpy().reshape(len(table), -1, 3)
    return np.percentile(values.reshape(-1, 3), [2, 98], axis=0)


class Filmer:
    def __init__(self, env, probe, dino, scale=None):
        self.env, self.probe, self.dino = env, probe, dino
        self.scale = scale  # strip_scale() of the round, for the PCA colours
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
        ranges = self.env.lidar()
        ox, oy, _ = self.env.sim.data.geom("lidar_geom").xpos
        cmap = plt.get_cmap("turbo_r")
        for k, r in enumerate(ranges):
            a = yaw + k * 2 * math.pi / len(ranges)
            hx, hy = ox + r * math.cos(a), oy + r * math.sin(a)
            colour = cmap(min(1.0, r / DEPTH_MAX))
            hit = r < 9.99
            ax.plot([ox, hx], [oy, hy], color=colour, lw=0.9, alpha=0.7 if hit else 0.2)
            if hit:
                ax.plot(hx, hy, marker="o", ms=3.5, color=colour)
        ax.add_patch(plt.Circle((x, y), 0.29, fill=False, lw=1.2, color="0.35"))  # leg reach
        ax.add_patch(plt.Circle((x, y), 0.35, fill=False, lw=1.0, ls="--", color="0.35"))  # sweep tolerance
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
        if "lc_ahead" in row:
            lines.append("lidar  ahead {:.2f} m   nearest {:.2f} m".format(row["lc_ahead"], row["lc_near"]))
        if "safety_rule" in row and isinstance(row["safety_rule"], str):
            lines.append("safety  " + (row["safety_rule"] or "-"))
        for i, line in enumerate(lines):
            draw.text((24, 62 + 26 * i), line, fill=(40, 40, 40), font=small)
        if outcome:
            draw.rectangle((24, H - 64, W - 24, H - 20), fill=REASON_COLOR.get(outcome, (90, 90, 90)))
            draw.text((40, H - 58), "episode ends: " + outcome, fill=(255, 255, 255), font=big)
        return np.asarray(panel)

    def lidar_view(self):
        """The 36 lidar ranges as a heatmap: wedges around the robot (heading up) and an unrolled bar."""
        ranges = np.asarray(self.env.lidar(), np.float64)
        n = len(ranges)
        cmap = plt.get_cmap("turbo_r")
        colours = cmap(np.clip(ranges / DEPTH_MAX, 0, 1))
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0.07, 0.27, 0.4, 0.64], projection="polar")
        ax.set_theta_zero_location("N")  # ahead is up; beams count counter-clockwise (left)
        angles = np.arange(n) * 2 * np.pi / n
        ax.bar(angles, np.minimum(ranges, 3.0), width=2 * np.pi / n, color=colours, edgecolor="white", lw=0.4)
        ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, 0.29), color="0.2", lw=1.2)  # leg reach
        ax.set_ylim(0, 3.0)
        ax.set_yticks([0.5, 1, 2, 3])
        ax.set_yticklabels(["0.5", "1", "2", "3 m"], fontsize=7)
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax.set_xticklabels(["ahead", "left", "behind", "right"], fontsize=8)
        bar = fig.add_axes([0.04, 0.06, 0.92, 0.1])
        order = [(k + n // 2) % n for k in range(n)][::-1]  # behind via left, ahead in the middle, right, behind
        bar.imshow(colours[order][None, :, :3], aspect="auto")
        bar.set_yticks([])
        bar.set_xticks([0, n // 4, n // 2, 3 * n // 4, n - 1])
        bar.set_xticklabels(["behind", "left", "ahead", "right", "behind"], fontsize=8)
        legend = fig.add_axes([0.58, 0.32, 0.38, 0.06])
        legend.imshow(cmap(np.linspace(0, 1, 100))[None, :, :3], aspect="auto")
        legend.set_yticks([])
        legend.set_xticks([0, 99])
        legend.set_xticklabels(["0 m", "{:.0f} m+".format(DEPTH_MAX)], fontsize=8)
        fig.text(0.58, 0.45, "lidar ranges\nred = near, blue/black = far\nnearest {:.2f} m".format(ranges.min()), fontsize=11)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return np.asarray(Image.fromarray(image).resize((W, H)))

    def strip_view(self, row):
        rgb = self.env.sim.render().astype(np.float32)
        panel = Image.fromarray((0.45 * rgb).astype(np.uint8))
        draw = ImageDraw.Draw(panel)
        cols = sorted((c for c in row.index if isinstance(c, str) and c.startswith("dp_")),
                      key=lambda c: tuple(int(v) for v in c.split("_")[1:]))
        if not cols or self.scale is None:
            draw.text((24, 24), "no DINOv2 strip features in this run", fill=(255, 255, 255), font=font(22))
            return np.asarray(panel)
        values = np.asarray([row[c] for c in cols], np.float64).reshape(-1, 3)
        lo, hi = self.scale
        colours = (np.clip((values - lo) / np.maximum(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
        cells = len(colours)
        top, bottom = (int(round(r * H / 26)) for r in STRIP_ROWS)
        for k, c in enumerate(colours):  # the cells in place on the picture ...
            x0, x1 = int(k * W / cells), int((k + 1) * W / cells)
            draw.rectangle((x0, top, x1 - 1, bottom), fill=tuple(int(v) for v in c))
            draw.rectangle((x0, H - 96, x1 - 1, H - 36), fill=tuple(int(v) for v in c))  # ... and enlarged below
        draw.rectangle((10, 10, 520, 40), fill=(0, 0, 0))
        draw.text((18, 14), "DINOv2 horizon strip (23 cells): PCA 1, 2, 3 = R, G, B", fill=(255, 255, 255), font=font(18))
        draw.text((18, H - 120), "the same 23 cells, enlarged:", fill=(230, 230, 230), font=font(16))
        draw.text((18, H - 30), "left of view", fill=(230, 230, 230), font=font(16))
        draw.text((W - 120, H - 30), "right of view", fill=(230, 230, 230), font=font(16))
        return np.asarray(panel)

    def frame(self, footprints, goal, title, row, step, steps, outcome=""):
        top = np.concatenate([self.eye_with_classes(), self.chase_view()], axis=1)
        middle = np.concatenate([self.map_view(footprints, goal), self.info_view(title, row, step, steps, outcome)], axis=1)
        bottom = np.concatenate([self.lidar_view(), self.strip_view(row)], axis=1)
        return np.concatenate([top, middle, bottom], axis=0)


def film(run, round_, episode, config, probe, dino, out):
    table = pd.read_parquet(run_dir(run) / "round_{}.parquet".format(round_number(round_)))
    rows = table[table["episode"] == episode].sort_values("step").reset_index(drop=True)
    if rows.empty:
        raise SystemExit("episode {} is not in round {}".format(episode, round_))
    env_cfg = config["env"]
    expert = Expert.from_config(config["expert"], env_cfg["goal_radius"])
    env = SimEnv(simple=env_cfg["simple"], goal_radius=env_cfg["goal_radius"], max_steps=env_cfg["max_steps"],
                 min_start_dist=env_cfg["min_start_dist"], expert=expert)
    env.reset(int(rows["scene_id"][0]), int(rows["seed"][0]))
    filmer = Filmer(env, probe, dino, strip_scale(table))
    footprints, goal = env.footprints(), env.goal
    outcome = rows["done_reason"].iloc[-1]
    share = rows["used_expert"].mean()
    driver = ("expert driving" if share == 1 else "student alone (no expert help)" if share == 0
              else "mixed: expert {:.0%} of steps".format(share))
    title = "round {} ep. {}: scene {}, {}".format(round_, episode, int(rows["scene_id"][0]), driver)
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", "{}x{}".format(2 * W, 3 * H), "-r", str(FPS), "-i", "-", "-c:v", "libx264",
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
    parser.add_argument("--run", required=True)
    parser.add_argument("--round", default="0")
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    probe = load_probe(vision_dir() / config["features"]["probe"])
    dino = load_dino()
    folder = run_dir(args.run) / "videos"
    folder.mkdir(exist_ok=True)
    for episode in args.episodes:
        out = folder / "round{}-episode{}.mp4".format(round_number(args.round), episode)
        print(film(args.run, args.round, episode, config, probe, dino, out), flush=True)


if __name__ == "__main__":
    main()

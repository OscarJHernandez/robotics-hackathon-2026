"""Camera pictures, a follow camera for videos, and PNG/GIF writing."""

import math

import mujoco
import numpy as np

from common import data_dir


def save_camera(sim, recorder, episode, cycle):
    """Save the robot camera's picture for this cycle if the run records images; returns its path or ""."""
    path = recorder.image_path(episode, cycle)
    if path:
        from PIL import Image
        Image.fromarray(sim.render()).save(data_dir() / path, optimize=True)
    return path


def follow_view(sim, distance=1.6, elevation=-55.0):
    """A camera above and behind the robot that also keeps point b in view."""
    camera = mujoco.MjvCamera()
    x, y, _ = sim.body_position()
    tx, ty = sim.target()
    camera.lookat[:] = ((x + tx) / 2, (y + ty) / 2, 0.05)
    spread = math.hypot(tx - x, ty - y)
    camera.distance = max(distance, 1.2 * spread + 0.6)
    camera.azimuth = 90.0
    camera.elevation = elevation
    return sim.render(camera=camera)


def write_gif(path, frames, fps=10):
    from PIL import Image
    images = [Image.fromarray(np.asarray(f)).convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0,
                   optimize=True)

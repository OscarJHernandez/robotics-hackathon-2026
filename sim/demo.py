"""Walk the simulated JetHexa forward.

    python sim/demo.py                       # no window: walk 5 s and print where it ended up
    python sim/demo.py --snapshot walk.png   # also save a picture of the robot at the end
    mjpython sim/demo.py --view              # live 3D window (macOS needs mjpython, see docs/simulation.md)
"""

import argparse
import struct
import time
import zlib

import mujoco

from jethexa_sim import JetHexaSim, tripod_targets, walk


def write_png(path, rgb):
    """Save an (H, W, 3) uint8 array as a PNG, without extra dependencies."""
    height, width, _ = rgb.shape
    rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    with open(path, "wb") as out:
        out.write(b"\x89PNG\r\n\x1a\n")
        out.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        out.write(chunk(b"IDAT", zlib.compress(rows)))
        out.write(chunk(b"IEND", b""))


def snapshot(sim, path):
    """A picture of the robot from outside, following the body."""
    camera = mujoco.MjvCamera()
    camera.lookat[:] = sim.body_position()
    camera.distance, camera.azimuth, camera.elevation = 0.8, 135, -25
    write_png(path, sim.render(camera=camera))


def view(sim, seconds):
    import mujoco.viewer

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        start = sim.time
        while viewer.is_running() and sim.time - start < seconds:
            tick = time.monotonic()
            sim.set_targets(tripod_targets(sim.stand, sim.time - start))
            sim.step(0.02)
            viewer.sync()
            time.sleep(max(0.0, 0.02 - (time.monotonic() - tick)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=5.0, help="simulated time to walk (default 5)")
    parser.add_argument("--view", action="store_true", help="open the live 3D window")
    parser.add_argument("--snapshot", metavar="PNG", help="save a picture of the robot at the end")
    args = parser.parse_args()

    sim = JetHexaSim()
    if args.view:
        view(sim, args.seconds)
    else:
        walk(sim, args.seconds)
    x, y, z = sim.body_position()
    print("MuJoCo {}: walked {:.1f} s, body at x={:.3f} m, y={:.3f} m, height={:.3f} m".format(
        mujoco.__version__, sim.time, x, y, z))
    if args.snapshot:
        snapshot(sim, args.snapshot)
        print("saved", args.snapshot)
    sim.close()


if __name__ == "__main__":
    main()

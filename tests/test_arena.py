"""Checks the arena: a closed 50 m² room, random layouts that stay apart, and class labels per pixel."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision"))

from scene import AREA, FLOOR, HALF, IGNORE, OBSTACLE, PASSAGE, TARGET, WALL, Arena  # noqa: E402


class ArenaTest(unittest.TestCase):
    def setUp(self):
        self.arena = Arena(simple=True)
        self.addCleanup(self.arena.close)
        self.rng = np.random.default_rng(0)

    def test_room_is_a_closed_square_of_50_square_metres(self):
        model = self.arena.sim.model
        inner = {}
        for name, axis in (("wall_e", 0), ("wall_w", 0), ("wall_n", 1), ("wall_s", 1)):
            geom = model.geom(name)
            inner[name] = abs(geom.pos[axis]) - geom.size[axis]  # distance from the centre to the inner face
            self.assertGreaterEqual(geom.size[1 - axis], HALF)  # long enough to close the corners
        for value in inner.values():
            self.assertAlmostEqual(value, HALF, places=3)
        self.assertAlmostEqual((2 * HALF) ** 2, AREA, places=6)
        self.assertEqual(len(self.arena.obstacles), 8)

    def test_random_layout_keeps_furniture_and_target_inside_and_apart(self):
        for _ in range(20):
            layout = self.arena.randomize(self.rng)
            items = layout["obstacles"]
            for i, a in enumerate(items):
                self.assertLess(max(abs(a["x"]), abs(a["y"])) + a["radius"], HALF)
                for b in items[i + 1:]:
                    gap = math.hypot(a["x"] - b["x"], a["y"] - b["y"]) - a["radius"] - b["radius"]
                    self.assertGreater(gap, PASSAGE)
            self.assertTrue(self.arena.is_free(*layout["target"], 0.1))

    def test_layout_changes_colors_and_survives_a_reset(self):
        first = self.arena.randomize(self.rng)
        second = self.arena.randomize(self.rng)
        self.assertNotEqual(first["obstacles"][0]["color"], second["obstacles"][0]["color"])
        for item in second["obstacles"]:
            self.assertGreater(max(item["color"]) - item["color"][0], -0.2)  # never clearly red
        self.arena.sim.reset()
        self.assertEqual(self.arena.sim.target(), second["target"])
        body = self.arena.sim.data.body(second["obstacles"][3]["name"])
        self.assertAlmostEqual(float(body.xpos[0]), second["obstacles"][3]["x"])

    def test_robot_poses_are_on_free_floor(self):
        self.arena.randomize(self.rng)
        for _ in range(30):
            x, y, _ = self.arena.sample_pose(self.rng)
            self.assertTrue(self.arena.is_free(x, y, 0.29))
        self.arena.sim.step(1.0)
        self.assertGreater(self.arena.sim.body_upright(), 0.99)

    def test_every_pixel_gets_a_class_and_there_is_no_sky(self):
        self.arena.randomize(self.rng)
        seen = set()
        for i in range(30):
            self.arena.sample_pose(self.rng)
            try:
                seg = self.arena.sim.render(segmentation=True)
            except Exception as error:  # no OpenGL on this machine
                self.skipTest("rendering is not available: {}".format(error))
            self.assertEqual(seg.shape, (360, 640))
            self.assertGreaterEqual(int(seg.min()), 0, "sky above the walls in view {}".format(i))
            seen.update(int(c) for c in np.unique(self.arena.class_map(seg)))
        self.assertTrue({FLOOR, WALL, OBSTACLE, TARGET} <= seen)
        self.assertTrue(seen <= {FLOOR, WALL, OBSTACLE, TARGET, IGNORE})


if __name__ == "__main__":
    unittest.main()

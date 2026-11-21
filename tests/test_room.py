"""Checks that the one-room scene loads, has its 4 obstacles and the robot stands in it."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

from jethexa_sim import SIM_DIR, JetHexaSim  # noqa: E402

OBSTACLES = ("obs_bed", "obs_couch", "obs_armchair", "obs_table_top")


class RoomTest(unittest.TestCase):
    def setUp(self):
        self.sim = JetHexaSim(SIM_DIR / "room_simple.xml")
        self.addCleanup(self.sim.close)

    def test_has_walls_and_four_obstacles(self):
        names = {self.sim.model.geom(i).name for i in range(self.sim.model.ngeom)}
        for name in OBSTACLES + ("wall_back", "wall_left", "wall_right", "wall_front_a", "wall_front_b"):
            self.assertIn(name, names)

    def test_robot_stands_in_the_room(self):
        self.sim.step(2.0)
        x, y, z = self.sim.body_position()
        self.assertLess(abs(x) + abs(y), 0.05)
        self.assertGreater(self.sim.body_upright(), 0.99)


if __name__ == "__main__":
    unittest.main()

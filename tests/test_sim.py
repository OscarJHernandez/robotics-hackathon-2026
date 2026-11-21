"""Checks that the simulated JetHexa loads, stands and walks. Run: python -m unittest discover -s tests -v"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

from jethexa_sim import LEGS, MESH_MODEL_PATH, SIMPLE_MODEL_PATH, JetHexaSim, walk  # noqa: E402
from fetch_meshes import meshes_available  # noqa: E402


class JetHexaSimTest(unittest.TestCase):
    def setUp(self):
        self.sim = JetHexaSim()
        self.addCleanup(self.sim.close)

    def test_model_has_18_servo_joints_and_a_camera(self):
        self.assertEqual(self.sim.model.nu, 18)
        self.assertEqual(len(self.sim.joint_names), 18)
        self.assertEqual(self.sim.model.ncam, 1)

    def test_stands_still_on_six_feet(self):
        self.sim.step(2.0)
        x, y, z = self.sim.body_position()
        self.assertAlmostEqual(x, 0.0, delta=0.005)
        self.assertAlmostEqual(y, 0.0, delta=0.005)
        self.assertAlmostEqual(z, 0.08, delta=0.01)
        self.assertGreater(self.sim.body_upright(), 0.99)
        floor = self.sim.model.geom("floor").id
        touching = set()
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            other = contact.geom2 if contact.geom1 == floor else contact.geom1
            touching.add(self.sim.model.geom(other).name)
        self.assertEqual(touching, set("foot_" + leg for leg in LEGS))

    def test_walks_forward_and_stays_upright(self):
        x, y, _ = walk(self.sim, 3.0)
        self.assertGreater(x, 0.15)
        self.assertLess(abs(y), 0.1)
        self.assertGreater(self.sim.body_upright(), 0.95)

    def test_simple_model_matches_mesh_model(self):
        if not meshes_available():
            self.skipTest("meshes not downloaded (python sim/fetch_meshes.py)")
        mesh, simple = JetHexaSim(MESH_MODEL_PATH), JetHexaSim(SIMPLE_MODEL_PATH)
        self.assertEqual(mesh.joint_names, simple.joint_names)
        self.assertAlmostEqual(float(mesh.model.body_mass.sum()), float(simple.model.body_mass.sum()), places=6)
        self.assertEqual(walk(mesh, 1.0).round(6).tolist(), walk(simple, 1.0).round(6).tolist())

    def test_camera_gives_rgb_and_depth(self):
        try:
            rgb = self.sim.render()
            depth = self.sim.render(depth=True)
        except Exception as error:  # no OpenGL on this machine (for example a CI runner)
            self.skipTest("rendering is not available: {}".format(error))
        self.assertEqual(rgb.shape, (360, 640, 3))
        self.assertEqual(depth.shape, (360, 640))
        self.assertGreater(float(depth.min()), 0.0)


if __name__ == "__main__":
    unittest.main()

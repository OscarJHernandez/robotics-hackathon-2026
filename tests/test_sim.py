"""Checks that the simulated JetHexa loads, stands and walks. Run: python -m unittest discover -s tests -v"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

from jethexa_sim import LEGS, MESH_MODEL_PATH, SIMPLE_MODEL_PATH, JetHexaSim, tripod_targets, walk  # noqa: E402
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
        self.assertEqual(mesh.model.nsensor, simple.model.nsensor)
        self.assertEqual(mesh.target(), simple.target())

    def test_turn_steers_left_and_right(self):
        walk(self.sim, 3.0, turn=0.1)
        self.assertGreater(self.sim.body_yaw(), 0.3)
        self.assertGreater(self.sim.body_upright(), 0.95)
        other = JetHexaSim()
        self.addCleanup(other.close)
        walk(other, 3.0, turn=-0.1)
        self.assertLess(other.body_yaw(), -0.3)

    def test_zero_turn_is_the_plain_gait(self):
        plain = tripod_targets(self.sim.stand, 0.3)
        self.assertEqual(plain, tripod_targets(self.sim.stand, 0.3, turn=0.0))
        self.assertNotEqual(plain, tripod_targets(self.sim.stand, 0.3, turn=0.1))

    def test_sensors_at_rest(self):
        self.sim.step(2.0)
        imu = self.sim.imu()
        self.assertAlmostEqual(imu["accel"][2], 9.81, delta=0.05)
        for value in imu["gyro"] + (imu["roll"], imu["pitch"], self.sim.body_yaw()):
            self.assertAlmostEqual(value, 0.0, delta=0.01)
        feet = self.sim.foot_contacts()
        self.assertEqual(set(feet), set(LEGS))
        weight = float(self.sim.model.body_mass.sum()) * 9.81
        self.assertAlmostEqual(sum(feet.values()), weight, delta=0.05 * weight)
        for readings in (self.sim.joint_velocities(), self.sim.actuator_forces()):
            self.assertEqual(list(readings), self.sim.joint_names)
        self.assertLess(max(abs(v) for v in self.sim.joint_velocities().values()), 0.01)

    def test_feet_lift_while_walking(self):
        in_air = []
        walk(self.sim, 1.0, on_step=lambda sim: in_air.append(sum(f == 0.0 for f in sim.foot_contacts().values())))
        self.assertGreaterEqual(max(in_air), 3)

    def test_target_marker_moves_and_does_not_block(self):
        self.assertEqual(self.sim.target(), (1.0, 0.0))
        self.sim.set_target(0.2, 0.0)
        self.assertEqual(self.sim.target(), (0.2, 0.0))
        x, _, _ = walk(self.sim, 3.0)  # walks straight through the marker
        self.assertGreater(x, 0.25)

    def test_target_marker_is_in_the_camera_picture(self):
        try:
            self.sim.set_target(0.5, 0.0)
            rgb = self.sim.render().astype(int)
        except Exception as error:
            self.skipTest("rendering is not available: {}".format(error))
        red = (rgb[..., 0] > 150) & (rgb[..., 1] < 80) & (rgb[..., 2] < 80)
        self.assertGreater(int(red.sum()), 500)

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

"""The student's plain twin (app/student.py): the app's dressing must not reach the student's inputs.

The student (nav/dagger v29) was fitted in the plain training arena. In the app it looks at a plain twin of the
apartment: the same robot pose and furniture, the red pillar at the goal, none of the home look. These tests check
that the inputs it gets through the app equal those of the plain homes/sim_bridge.FlatEnv at the same layout and
pose, and that painting the zones, moving Margarete and moving the decor change none of them.

No TabPFN and no API calls. Uses the mesh models when sim/meshes is there (as the app does), else the simple ones;
skips cleanly without OpenGL, like the other camera tests.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "app", REPO / "homes", REPO / "sim", REPO / "vision"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fetch_meshes import meshes_available  # noqa: E402

SIMPLE = not meshes_available()
LAYOUTS = 20


def _rendering():
    try:
        from jethexa_sim import JetHexaSim
        sim = JetHexaSim()
        sim.render()
        sim.close() if hasattr(sim, "close") else None
        return None
    except Exception as error:  # no OpenGL on this machine (for example a CI runner)
        return str(error)


NO_GL = _rendering()


def _free_pose(env, rng):
    """A standing pose on free floor in the FlatEnv's layout, any heading."""
    from scene import ROBOT_RADIUS
    for _ in range(5000):
        x, y = rng.uniform(-3.1, 3.1, size=2)
        if env.arena.is_free(x, y, ROBOT_RADIUS + 0.05):
            return (float(x), float(y), float(rng.uniform(-math.pi, math.pi)))
    raise RuntimeError("no free floor")


@unittest.skipIf(NO_GL is not None, "rendering is not available: {}".format(NO_GL))
class PlainTwinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import flats as F
        from robot import ApartmentScene
        from sim_bridge import FlatEnv
        from student import PlainTwin
        cls.F = F
        cls.scene = ApartmentScene(simple=SIMPLE)
        cls.twin = PlainTwin(simple=SIMPLE)
        cls.env = FlatEnv(F.random_flat(np.random.default_rng([0, 0]), 0), render=True, simple=SIMPLE,
                          lidar_rays=36, lidar_range=(0.15, 8.0))

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        cls.twin.close()

    def flat(self, k):
        return self.F.random_flat(np.random.default_rng([k, 0]), k)

    def app_inputs(self, pose, goal, place=True):
        """What the student gets in the app: twin picture and depth, lidar cast in the dressed scene."""
        sim = self.scene.sim
        if place:
            self.scene.place(pose)
        self.twin.sync(sim.model, sim.data, goal)
        obs = self.twin.observe()
        obs["lidar"] = sim.lidar(36, 0.15, 8.0)
        return obs

    def plain_inputs(self, flat, pose, goal):
        """The same in the plain FlatEnv (the training arena with the flat's furniture)."""
        env = self.env
        env.load(flat)
        env.arena.place(*pose)
        env.sim.set_target(*goal)
        env.goal = goal
        return env.observe()

    def assertSameInputs(self, a, b, what):
        np.testing.assert_allclose(a["lidar"], b["lidar"], atol=1e-4, err_msg="lidar " + what)
        np.testing.assert_allclose(a["depth"], b["depth"], atol=1e-3, err_msg="depth " + what)
        band = slice(11 * 14 * 360 // (26 * 14), 15 * 14 * 360 // (26 * 14))  # the horizon band's pixel rows
        np.testing.assert_allclose(a["depth"][band], b["depth"][band], atol=1e-3, err_msg="depth band " + what)
        diff = np.abs(a["rgb"].astype(int) - b["rgb"].astype(int))
        self.assertLessEqual(diff.max(), 2, "picture " + what)

    def test_the_twin_is_the_plain_training_arena(self):
        from robot import WALL_HEIGHT
        from scene import Arena
        flat = self.flat(3)
        self.scene.load(flat)
        goal = (flat["zones"][0]["view_x"], flat["zones"][0]["view_y"])
        self.app_inputs((flat["dock"]["x"], flat["dock"]["y"], 0.4), goal)
        twin = self.twin.sim.model
        names = [twin.geom(g).name for g in range(twin.ngeom)]
        self.assertFalse([n for n in names if n.startswith(("zone_", "decor_", "person"))])
        app_names = [self.scene.sim.model.geom(g).name for g in range(self.scene.sim.model.ngeom)]
        self.assertTrue(any(n.startswith("zone_") for n in app_names))  # the app's scene does have them
        for g, name in enumerate(names):
            if name.startswith("wall_"):
                self.assertAlmostEqual(2 * twin.geom_size[g][2], 4.5, places=3)  # full-height walls
        app = self.scene.sim.model
        self.assertAlmostEqual(2 * app.geom_size[app.geom("wall_n").id][2], WALL_HEIGHT, places=3)
        fresh = Arena(SIMPLE)
        try:  # the grey checker floor, not the app's wood
            tex = twin.tex("grid").id
            a, n = twin.tex_adr[tex], twin.tex_width[tex] * twin.tex_height[tex] * 3
            np.testing.assert_array_equal(twin.tex_data[a:a + n], fresh.sim.model.tex_data[a:a + n])
            self.assertFalse(np.array_equal(app.tex_data[a:a + n], twin.tex_data[a:a + n]))
        finally:
            fresh.close()
        # furniture poses and robot joints are the app's; the pillar is at the goal
        sim, tdata = self.scene.sim, self.twin.sim.data
        for piece in flat["furniture"]:
            src = int(app.body(piece["body"]).mocapid[0])
            dst = int(twin.body(piece["body"]).mocapid[0])
            np.testing.assert_allclose(tdata.mocap_pos[dst], sim.data.mocap_pos[src])
            np.testing.assert_allclose(tdata.mocap_quat[dst], sim.data.mocap_quat[src])
        np.testing.assert_allclose(tdata.qpos, sim.data.qpos[:twin.nq])
        self.assertEqual(tuple(round(v, 6) for v in self.twin.sim.target()), tuple(round(v, 6) for v in goal))
        self.twin.check_compatible(app)

    def test_app_inputs_equal_the_plain_flat_env_and_ignore_the_dressing(self):
        rng = np.random.default_rng(11)
        sim = self.scene.sim
        changed_app_picture = 0
        for k in range(LAYOUTS):
            flat = self.flat(100 + k)
            if k % 2:  # half of them with the loose pieces scattered
                flat = self.F.scatter(flat, np.random.default_rng([k, 9]))
            self.scene.load(flat)
            self.env.load(flat)
            pose = _free_pose(self.env, rng)
            zone = flat["zones"][int(rng.integers(len(flat["zones"])))]
            goal = (zone["view_x"], zone["view_y"])
            app = self.app_inputs(pose, goal)
            plain = self.plain_inputs(flat, pose, goal)
            self.assertSameInputs(app, plain, "app vs FlatEnv, layout {}".format(k))
            # the lidar cast in the dressed scene equals the twin's own lidar
            np.testing.assert_allclose(app["lidar"], self.twin.lidar(), atol=1e-4)

            # Dress the app's scene differently: zones painted, Margarete right in front of the robot, decor moved.
            before_eye = sim.render()
            self.scene.paint({z["name"]: float(rng.random()) for z in flat["zones"]})
            x, y, yaw = pose
            self.scene.person(x + 0.7 * math.cos(yaw), y + 0.7 * math.sin(yaw), yaw + 1.0)
            import mujoco
            for g in range(sim.model.ngeom):
                if sim.model.geom(g).name.startswith("decor_"):
                    sim.model.geom_pos[g][:2] = (x + rng.uniform(-1, 1), y + rng.uniform(-1, 1))
            mujoco.mj_forward(sim.model, sim.data)
            changed_app_picture += int(not np.array_equal(before_eye, sim.render()))
            again = self.app_inputs(pose, goal, place=False)
            self.assertSameInputs(again, app, "after dressing, layout {}".format(k))
            self.scene.load(flat)  # put the decor back for the next layout
        # The dressing does change the app's own camera picture: without the twin it would reach the student.
        self.assertGreater(changed_app_picture, LAYOUTS // 2)


if __name__ == "__main__":
    unittest.main()

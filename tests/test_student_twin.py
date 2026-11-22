"""The student's plain twin (app/student.py): the app's dressing must not reach the student's inputs.

The student (nav/dagger v29) was fitted in the plain training arena. In the app it looks at a plain twin of the
apartment: the same robot pose and furniture, the red pillar at the goal, none of the home look. These tests check
that the inputs it gets through the app equal those of the plain homes/sim_bridge.FlatEnv at the same layout and
pose, and that painting the zones, moving Margarete and moving the decor change none of them. The twin carries the
twelve clutter bodies too (sim/arena_clutter.xml), so the app's Medium and Messy rooms reach its camera as well as
its lidar; the training arena (sim/arena.xml) stays without them.

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
        self.assertMirrored()
        np.testing.assert_allclose(tdata.qpos, sim.data.qpos[:twin.nq])
        self.assertEqual(tuple(round(v, 6) for v in self.twin.sim.target()), tuple(round(v, 6) for v in goal))
        self.twin.check_compatible(app)

    def assertMirrored(self):
        """Every obs_ mocap body of the app (furniture and clutter, parked or not) stands where the twin has it."""
        app, twin = self.scene.sim.model, self.twin.sim.model
        bodies = [app.body(b).name for b in range(app.nbody)
                  if app.body(b).name.startswith("obs_") and app.body_mocapid[b] >= 0]
        self.assertEqual(len(bodies), 8 + len(self.F.CLUTTER))
        for name in bodies:
            src, dst = int(app.body(name).mocapid[0]), int(twin.body(name).mocapid[0])
            np.testing.assert_allclose(self.twin.sim.data.mocap_pos[dst], self.scene.sim.data.mocap_pos[src], err_msg=name)
            np.testing.assert_allclose(self.twin.sim.data.mocap_quat[dst], self.scene.sim.data.mocap_quat[src], err_msg=name)

    def test_the_twin_mirrors_the_clutter(self):
        flat = self.flat(5)
        for level in ("messy", "medium", "clean"):
            out = self.F.scatter(flat, np.random.default_rng([5, 1]), level)
            self.scene.load(out)
            goal = (out["zones"][1]["view_x"], out["zones"][1]["view_y"])
            self.app_inputs((out["dock"]["x"], out["dock"]["y"], 0.3), goal)
            self.assertMirrored()
            sim = self.scene.sim
            inside = [p for p in out["furniture"] if p.get("clutter")]
            for p in inside:  # the clutter on the floor is where the flat says, in both
                m = int(sim.model.body(p["body"]).mocapid[0])
                np.testing.assert_allclose(sim.data.mocap_pos[m][:2], (p["x"], p["y"]), atol=1e-9)
            parked = [b for b in self.F.PARK if b.startswith("obs_c") and b not in {p["body"] for p in inside}]
            for b in parked:  # the rest stays outside the room
                m = int(sim.model.body(b).mocapid[0])
                self.assertGreater(max(abs(sim.data.mocap_pos[m][0]), abs(sim.data.mocap_pos[m][1])), 10.0)
            self.assertEqual(len(inside), {"clean": 0, "medium": 4, "messy": len(inside)}[level])
        # The clutter is in the student's camera, not only in its lidar: looking at an item, the twin's depth
        # changes when the item is taken away.
        messy = self.F.scatter(flat, np.random.default_rng([5, 1]), "messy")
        item = next(p for p in messy["furniture"] if p.get("clutter"))
        tidy = dict(messy, furniture=[p for p in messy["furniture"] if p is not item])
        seen = 0
        for k in range(16):
            a = 2 * math.pi * k / 16
            pose = (item["x"] + 0.9 * math.cos(a), item["y"] + 0.9 * math.sin(a), a + math.pi)
            if max(abs(pose[0]), abs(pose[1])) > 3.1:
                continue
            depth = []
            for f in (messy, tidy):
                self.scene.load(f)
                depth.append(self.app_inputs(pose, (0.0, 0.0))["depth"])
            seen += int(np.abs(depth[0] - depth[1]).max() > 0.1)
        self.assertGreater(seen, 0)

    def test_app_inputs_equal_the_plain_flat_env_and_ignore_the_dressing(self):
        rng = np.random.default_rng(11)
        sim = self.scene.sim
        changed_app_picture = 0
        for k in range(LAYOUTS):
            flat = self.flat(100 + k)
            level = (None, "medium", "messy", "clean")[k % 4]  # as built, and each mess level (with clutter)
            if level:
                flat = self.F.scatter(flat, np.random.default_rng([k, 9]), level)
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


class TrainingArenaTest(unittest.TestCase):
    """The clutter must not reach the training arena: its files have no obs_c bodies, and Arena.randomize (which
    seeds every recorded DAgger layout) gives the same layout as before the clutter was added."""

    # Arena().randomize(np.random.default_rng(1234)) before sim/clutter.xml existed (same with and without meshes)
    BEFORE = [("obs_0", 2.025, -0.5089, 2.6593), ("obs_1", -1.2408, -0.8719, 2.9159), ("obs_2", 0.2204, 1.4058, -0.321),
              ("obs_3", 3.2304, -3.266, -1.8769), ("obs_4", 2.091, 2.0918, 1.0988), ("obs_5", -1.917, 2.1654, -2.7637),
              ("obs_6", -2.9764, 0.0786, 0.7087), ("obs_7", -2.7656, -1.8668, 1.1384)]
    TARGET = [-2.7415, -3.0168]

    def test_the_training_arena_files_have_no_clutter(self):
        for name in ("arena.xml", "arena_simple.xml", "arena_furniture.xml"):
            text = (REPO / "sim" / name).read_text()
            self.assertNotIn("obs_c", text, name)
            self.assertNotIn("clutter.xml", text, name)
        for name in ("arena_clutter.xml", "arena_clutter_simple.xml", "apartment.xml", "apartment_simple.xml"):
            self.assertIn('<include file="clutter.xml"/>', (REPO / "sim" / name).read_text(), name)

    def test_randomize_gives_the_same_layout_as_before(self):
        from scene import Arena
        arena = Arena(SIMPLE)
        try:
            self.assertFalse([o for o in arena.obstacles if o["name"].startswith("obs_c")])
            self.assertEqual(len(arena.obstacles), 8)
            layout = arena.randomize(np.random.default_rng(1234))
            got = [(o["name"], round(o["x"], 4), round(o["y"], 4), round(o["yaw"], 4)) for o in layout["obstacles"]]
            self.assertEqual(got, self.BEFORE)
            self.assertEqual([round(v, 4) for v in layout["target"]], self.TARGET)
        finally:
            arena.close()

    def test_the_clutter_arena_has_the_twelve_bodies_parked_outside(self):
        from scene import Arena
        import flats as F
        arena = Arena(SIMPLE, clutter=True)
        try:
            names = [o["name"] for o in arena.obstacles]
            self.assertEqual(len(names), 8 + 12)
            for c in F.CLUTTER.values():
                o = next(o for o in arena.obstacles if o["name"] == c["body"])
                self.assertAlmostEqual(o["radius"], c["radius"], places=3)
                x, y = arena.sim.data.body(c["body"]).xpos[:2]
                self.assertEqual((round(float(x), 6), round(float(y), 6)), F.PARK[c["body"]])
        finally:
            arena.close()


if __name__ == "__main__":
    unittest.main()

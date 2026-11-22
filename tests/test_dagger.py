"""Tests for nav/dagger: scenes, expert, features, rollout mixing, student wrapper, one real arena step.

No GPU, no TabPFN key and no downloads: TabPFN and DINOv2 are replaced by fakes.
"""

import math
import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pandas  # noqa: F401
    import yaml  # noqa: F401
    HAVE_ML = True
except ImportError:
    HAVE_ML = False


HAVE_MUJOCO_GL = bool(os.environ.get("MUJOCO_GL")) or sys.platform == "darwin"


def _inside(fp, px, py):
    """Is (px, py) inside a footprint from SimEnv.footprints()?"""
    if fp["kind"] == "circle":
        return math.hypot(px - fp["x"], py - fp["y"]) < fp["r"]
    c, s = math.cos(-fp["yaw"]), math.sin(-fp["yaw"])
    lx = c * (px - fp["x"]) - s * (py - fp["y"])
    ly = s * (px - fp["x"]) + c * (py - fp["y"])
    return abs(lx) < fp["hx"] and abs(ly) < fp["hy"]


@unittest.skipUnless(HAVE_ML, "needs the ml packages: uv run --group ml --group vision")
class ScenesTest(unittest.TestCase):
    def test_plans(self):
        from nav.dagger.config import episode_plan, heldout_ids, load_config, split_plan, train_ids
        config = load_config()
        self.assertFalse(set(train_ids(config)) & set(heldout_ids(config)))
        plan0 = episode_plan(config, 0)
        self.assertEqual(len(plan0), config["rounds"][0]["episodes"])
        self.assertEqual(len({e["scene_id"] for e in plan0}), min(len(plan0), config["scenes"]["train"]))
        self.assertEqual(episode_plan(config, 2), episode_plan(config, 2))  # same plan every time
        self.assertTrue(all(e["scene_id"] in set(train_ids(config)) for e in episode_plan(config, 3)))
        held = episode_plan(config, "heldout")
        self.assertTrue(all(e["scene_id"] in set(heldout_ids(config)) for e in held))
        shares = split_plan(plan0, {"a": 1, "b": 3})
        self.assertEqual(sum(len(v) for v in shares.values()), len(plan0))
        self.assertAlmostEqual(len(shares["b"]) / len(plan0), 0.75, delta=0.01)
        ids = [e["episode"] for v in shares.values() for e in v]
        self.assertEqual(len(ids), len(set(ids)))

    def test_layout_is_repeatable(self):
        from nav.dagger.env import scene_layout
        from scene import Arena
        arena = Arena(simple=True)
        a = scene_layout(arena, 17)
        b = scene_layout(arena, 17)
        c = scene_layout(arena, 18)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        arena.close()


class ExpertTest(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    def test_path_around_a_block_and_no_path_when_walled_in(self):
        from nav.dagger.expert import Grid, astar, cost_field
        block = {"kind": "box", "x": 0.0, "y": 0.0, "yaw": 0.0, "hx": 0.2, "hy": 1.0}
        grid = Grid([block], half=2.0, cell=0.05, inflate=0.2)
        path = astar(grid, (-1.0, 0.0), (1.0, 0.0))
        self.assertIsNotNone(path)
        self.assertTrue(any(abs(y) > 1.0 for _, y in path))  # it goes around the end of the block
        cost = cost_field(grid, (1.0, 0.0))
        self.assertTrue(np.isfinite(cost[grid.index(-1.0, 0.0)]))
        self.assertGreater(cost[grid.index(-1.0, 0.0)], 2.0)  # longer than the straight 2 m
        ring = [{"kind": "box", "x": 0.0, "y": 0.0, "yaw": math.radians(a), "hx": 0.05, "hy": 0.8} for a in (0, 90)]
        walls = [{"kind": "box", "x": x, "y": y, "yaw": 0.0, "hx": hx, "hy": hy}
                 for x, y, hx, hy in ((-1.5, 0, 0.05, 2), (1.5, 0, 0.05, 2), (0, -1.5, 2, 0.05), (0, 1.5, 2, 0.05))]
        boxed = Grid(walls + ring, half=3.0, cell=0.05, inflate=0.1)
        self.assertIsNone(astar(boxed, (-2.5, -2.5), (0.5, 0.5)))
        self.assertFalse(np.isfinite(cost_field(boxed, (0.5, 0.5))[boxed.index(-2.5, -2.5)]))

    def test_expert_commands(self):
        from nav.dagger.expert import Expert, pursuit_command
        stride, turn = pursuit_command((0, 0, 0), (1.0, 0.3), goal_dist=2.0)
        self.assertGreater(stride, 0.1)
        self.assertGreater(turn, 0.0)  # point ahead-left: steer left
        stride, turn = pursuit_command((0, 0, 0), (-1.0, -0.1), goal_dist=2.0)
        self.assertEqual(stride, 0.0)  # point behind: turn on the spot
        self.assertLess(turn, 0.0)  # behind and slightly right: turn right
        self.assertEqual(pursuit_command((0, 0, 0), (0.1, 0.0), goal_dist=0.1), (0.0, 0.0))
        expert = Expert(inflate=0.2)
        block = {"kind": "box", "x": 0.0, "y": 0.0, "yaw": 0.0, "hx": 0.2, "hy": 1.0}
        state = {"key": "t", "footprints": [block], "half": 2.0, "goal": (1.2, 0.0), "pose": (-1.2, 0.0, 0.0)}
        stride, turn = expert.act(state)
        self.assertFalse(expert.nopath)
        self.assertNotEqual(turn, 0.0)  # the block is straight ahead: the path bends round it


@unittest.skipUnless(HAVE_ML, "needs the ml packages")
class FeaturesTest(unittest.TestCase):
    def test_columns(self):
        from nav.dagger.featurize import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, columns
        self.assertEqual(len(FEATURE_COLUMNS), 26 * 46 * 2 + 8)
        self.assertEqual(len(CATEGORICAL_COLUMNS), 26 * 46)
        self.assertEqual(FEATURE_COLUMNS[0], "pc_0_0")
        self.assertEqual(FEATURE_COLUMNS[26 * 46], "pd_0_0")
        self.assertEqual(columns(1, True)[0][-2:], ["prev_stride", "prev_turn"])
        self.assertEqual(len(columns(2)[1]), 13 * 23)

    def test_patch_depth_and_obstacle_summary(self):
        from nav.dagger.featurize import obstacle_summary, patch_depth, pool_grid
        from scene import FLOOR, OBSTACLE
        depth = np.full((360, 640), 4.0, np.float32)
        depth[:, :70] = 1.0  # a near obstacle on the far left of the picture
        patches = patch_depth(depth)
        self.assertEqual(patches.shape, (26, 46))
        self.assertAlmostEqual(float(patches[10, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(patches[10, 30]), 4.0, places=4)
        classes = np.full((26, 46), FLOOR, np.int8)
        classes[:, :5] = OBSTACLE
        summary = obstacle_summary(classes, patches, cap=10.0)
        self.assertAlmostEqual(summary["near_dist"], 1.0, places=4)
        self.assertGreater(summary["near_bearing"], 0.4)  # left of centre, near the edge of a 62 degree view
        self.assertAlmostEqual(summary["free_left"], 1.0, places=4)
        self.assertEqual(summary["free_ahead"], 10.0)  # nothing solid ahead
        pooled_c, pooled_d = pool_grid(classes, patches, 2)
        self.assertEqual(pooled_c.shape, (13, 23))
        self.assertEqual(int(pooled_c[0, 0]), OBSTACLE)

    def test_depth_obstacles_history_and_safety(self):
        from nav.dagger.featurize import DEPTH, Featurizer, columns, depth_obstacles
        from nav.dagger.rollout import safety_command
        # a level camera 0.15 m up at the robot's centre, looking along +x; a flat wall 1.0 m ahead
        camera = {"pos": np.array([0.0, 0.0, 0.15]), "mat": np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)}
        depth = np.full((360, 640), 1.0, np.float32)
        d = depth_obstacles(depth, camera, (0.0, 0.0, 0.0), goal=(5.0, 5.0))
        self.assertAlmostEqual(d["dc_ahead"], 1.0, places=3)
        self.assertGreater(d["dc_left"], 1.0)  # the wall's left part is farther than straight ahead
        self.assertAlmostEqual(d["dc_near"], 1.0, places=2)
        # the same wall turned away (robot yaw 180 degrees) is behind the robot: nothing in the corridor ahead
        self.assertEqual(depth_obstacles(depth, camera, (0.0, 0.0, math.pi))["dc_ahead"], 10.0)
        # the target pillar is not an obstacle
        self.assertEqual(depth_obstacles(depth, camera, (0.0, 0.0, 0.0), goal=(1.0, 0.0))["dc_ahead"], 10.0)

        feature, _ = columns(2, False, True, 2)
        self.assertEqual(len(feature), 299 * 2 + 8 + 4 + 2 * 9)
        self.assertEqual(feature[-1], "h2_dc_near")
        f = Featurizer("truth", patch_pool=2, depth_summary=True, history=2)
        obs = {"seg": np.zeros((360, 640), np.int32), "depth": depth, "camera": camera,
               "robot_pose": (0.0, 0.0, 0.0), "goal_point": (3.0, 0.0)}
        m1, s1 = f.build_batch([obs], [None])
        row1 = dict(zip(f.columns, m1[0]))
        self.assertAlmostEqual(row1["h1_dc_ahead"], row1["dc_ahead"])  # first step: history repeats the present
        far = dict(obs, depth=np.full((360, 640), 2.0, np.float32))
        m2, s2 = f.build_batch([far], s1)
        row2 = dict(zip(f.columns, m2[0]))
        self.assertAlmostEqual(row2["dc_ahead"], 2.0, places=3)
        self.assertAlmostEqual(row2["h1_dc_ahead"], 1.0, places=3)
        self.assertAlmostEqual(row2["h2_dc_ahead"], 1.0, places=3)
        self.assertEqual(set(s2[0]["depth"]), set(DEPTH))

        safety = {"stop_distance": 0.45, "turn": 0.25}
        near = {"dc_ahead": 0.3, "dc_left": 0.4, "dc_right": 1.5, "dc_near": 0.3}
        self.assertEqual(safety_command((0.3, 0.05), near, safety), ((0.0, -0.25), True))  # right is freer
        self.assertEqual(safety_command((0.3, 0.05), dict(near, dc_ahead=0.8), safety), ((0.3, 0.05), False))
        self.assertEqual(safety_command((0.3, 0.05), near, None), ((0.3, 0.05), False))

    def test_depth_only_scan_and_column_groups(self):
        from nav.dagger.config import load_config
        from nav.dagger.featurize import Featurizer, select_columns
        from nav.dagger.policy import TabPFNPolicy
        camera = {"pos": np.array([0.0, 0.0, 0.15]), "mat": np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)}
        depth = np.full((360, 640), 3.0, np.float32)
        depth[:, :160] = 1.0  # something close on the left quarter of the picture
        f = Featurizer("none", patch_pool=2, depth_summary=True, history=2, depth_scan=15)
        self.assertEqual(len(f.columns), 299 + 3 + 4 + 15 + 8)
        self.assertFalse(any(c.startswith("pc_") or c in ("near_dist", "free_ahead") for c in f.columns))
        obs = {"depth": depth, "camera": camera, "robot_pose": (0.0, 0.0, 0.0), "goal_point": (5.0, 0.0)}
        m, _ = f.build_batch([obs], [None])
        row = dict(zip(f.columns, m[0]))
        self.assertLess(row["ds_0"], 1.3)  # leftmost direction: the close thing
        self.assertGreater(row["ds_14"], 2.9)  # rightmost: the far wall
        self.assertAlmostEqual(row["dc_ahead"], 3.0, places=2)
        self.assertEqual(len(select_columns(f.columns, ["goal", "depth", "history"])), 15)
        self.assertEqual(len(select_columns(f.columns, ["goal", "depth", "history", "scan"])), 30)
        self.assertEqual(len(select_columns(f.columns, ["goal", "depth", "history", "patch_depth"])), 314)
        cfg = load_config(str(Path(__file__).resolve().parent.parent / "nav" / "dagger" / "depth.yaml"))
        self.assertEqual(Featurizer.from_config(cfg["features"], Path("."), None).columns, f.columns)
        cols = select_columns(f.columns, cfg["student"]["column_groups"])
        policy = TabPFNPolicy(cols, []).use_source_columns(f.columns)
        self.assertEqual([f.columns[i] for i in policy.source_index], cols)

    def test_lidar_strip_columns_and_api_timeout(self):
        from nav.dagger.config import load_config
        from nav.dagger.featurize import Featurizer, lidar_summary, select_columns
        from nav.dagger.policy import call_api
        ranges = np.full(36, 10.0, np.float32)
        ranges[0] = 0.8   # straight ahead
        ranges[9] = 0.5   # 90 degrees: left
        ranges[18] = 2.0  # behind
        s = lidar_summary(ranges)
        self.assertAlmostEqual(s["lc_ahead"], 0.8, places=5)
        self.assertAlmostEqual(s["lc_left"], 0.5, places=5)
        self.assertEqual(s["lc_right"], 10.0)
        self.assertAlmostEqual(s["lc_rear"], 2.0, places=5)
        self.assertAlmostEqual(s["lc_near"], 0.5, places=5)

        rng = np.random.default_rng(1)
        pca = {"mean": np.zeros(384, np.float32), "components": rng.normal(size=(3, 384)).astype(np.float32),
               "band": 6, "cells": 23, "fake": True}
        f = Featurizer("strip", patch_pool=2, patch_depths=False, history=2, lidar_beams=36, strip_pca=pca)
        self.assertEqual(len(f.columns), 3 + 36 + 5 + 69 + 10)
        self.assertFalse(any(c.startswith(("pc_", "pd_", "dc_", "ds_")) for c in f.columns))
        camera = {"pos": np.array([0.0, 0.0, 0.15]), "mat": np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)}
        band = np.zeros((6, 46, 384), np.float32)
        band[2:4, :2] = 1.0  # the horizon rows, first 2 columns = cell 0
        obs = {"depth": np.full((360, 640), 3.0, np.float32), "camera": camera, "robot_pose": (0.0, 0.0, 0.0),
               "goal_point": (2.0, 0.0), "lidar": ranges, "band": band}
        m, st = f.build_batch([obs], [None])
        row = dict(zip(f.columns, m[0]))
        self.assertAlmostEqual(row["lr_9"], 0.5, places=5)
        self.assertAlmostEqual(row["h1_lc_left"], 0.5, places=5)  # first step: history repeats the present
        expected = pca["components"] @ np.ones(384, np.float32)
        self.assertAlmostEqual(row["dp_0_0"], float(expected[0]), places=3)
        self.assertAlmostEqual(row["dp_1_0"], 0.0, places=5)
        self.assertIn("dc_ahead", st[0]["depth"])  # the safety stop still gets the depth camera's distance
        cfg = load_config(str(Path(__file__).resolve().parent.parent / "nav" / "dagger" / "lidar.yaml"))
        self.assertEqual(len(select_columns(f.columns, cfg["student"]["column_groups"])), 123)

        calls = {"n": 0}

        def slow_then_fast():
            calls["n"] += 1
            if calls["n"] == 1:
                time.sleep(1.0)
            return "ok"

        self.assertEqual(call_api(slow_then_fast, state={"last": 0.0}, timeout=0.2), "ok")
        self.assertEqual(calls["n"], 2)

    def test_leg_aware_lidar_safety(self):
        from nav.dagger.rollout import lidar_safety
        safety = {"stop_distance": 0.48, "corridor": 0.32, "side_clearance": 0.32, "nudge": 0.1,
                  "sweep_radius": 0.35, "turn": 0.25}

        def beams(**hits):  # degrees -> range in metres, everything else free
            r = np.full(36, 10.0, np.float32)
            for deg, dist in hits.items():
                r[int(deg[1:]) // 10] = dist
            return r

        self.assertEqual(lidar_safety((0.3, 0.0), beams(), safety), ((0.3, 0.0), ""))
        # something 0.4 m ahead: no stride; it turns on the spot toward the freer side
        (stride, turn), rule = lidar_safety((0.3, 0.0), beams(a0=0.4, a90=0.6), safety)
        self.assertEqual(stride, 0.0)
        self.assertLess(turn, 0)  # left is closer (0.6 m at 90 degrees), so it turns right
        self.assertIn("stop", rule)
        # the corridor covers the legs: a hit at 30 degrees, 0.5 m away, is 0.25 m to the side -> still a stop
        self.assertIn("stop", lidar_safety((0.3, 0.0), beams(a30=0.5), safety)[1])
        # walking past something 0.28 m to the left: turn away (right) at least the nudge
        (stride, turn), rule = lidar_safety((0.3, 0.05), beams(a90=0.28), safety)
        self.assertEqual(stride, 0.3)
        self.assertLessEqual(turn, -0.1)
        self.assertIn("side", rule)
        # a left turn with an obstacle 0.3 m front-left would swing the front legs into it: flipped right
        (stride, turn), rule = lidar_safety((0.0, 0.2), beams(a40=0.3), safety)
        self.assertLess(turn, 0)
        self.assertIn("sweep", rule)
        # rear-right is also bad for a left turn (the rear legs swing that way)
        self.assertLess(lidar_safety((0.0, 0.2), beams(a220=0.3), safety)[0][1], 0)
        # a right turn there is fine
        self.assertEqual(lidar_safety((0.0, -0.2), beams(a40=0.3), safety), ((0.0, -0.2), ""))

    def test_edge_recovery_and_backward_strides(self):
        from nav.dagger import STRIDE
        from nav.dagger.expert import Expert, recovery_command
        from nav.dagger.policy import choose_context
        from nav.dagger.rollout import lidar_safety
        import pandas as pd
        box = {"kind": "box", "x": 1.0, "y": 0.0, "yaw": 0.0, "hx": 0.2, "hy": 0.5}  # front face at x = 0.8
        state = {"key": "t", "footprints": [box], "half": 3.5, "goal": (2.5, 2.0), "pose": (0.55, 0.1, 0.0)}
        # 0.25 m from the box, which is ahead and a little to the right: back up, turn left (away from it)
        stride, turn = recovery_command(state)
        self.assertLess(stride, 0)
        self.assertGreater(turn, 0)
        # clear of everything: no recovery
        self.assertIsNone(recovery_command(dict(state, pose=(0.0, 0.0, 0.0))))
        # the box behind the robot: no recovery (normal path following moves away from it)
        self.assertIsNone(recovery_command(dict(state, pose=(0.55, 0.1, math.pi))))
        # the box beside the robot, 0.27 m away, heading 30 degrees into it (box on the right): step back
        # slightly and turn left, away from it
        stride, turn = recovery_command(dict(state, pose=(0.53, 0.1, math.pi / 2 - math.radians(30))))
        self.assertLess(stride, 0)
        self.assertGreater(stride, -0.1)
        self.assertGreater(turn, 0)
        # the same spot walking alongside it (parallel), as the plain expert passes edges: no recovery
        self.assertIsNone(recovery_command(dict(state, pose=(0.53, 0.1, math.pi / 2))))
        expert = Expert(recover={"clearance": 0.30, "back_stride": -0.12, "turn": 0.2, "cone": 45})
        stride, turn = expert.act(state)
        self.assertTrue(expert.recovering)
        self.assertEqual(stride, -0.12)
        expert.act(dict(state, pose=(-1.0, -1.0, 0.0)))
        self.assertFalse(expert.recovering)
        self.assertEqual(STRIDE[0], -0.15)
        # backing into something 0.35 m behind: the rear rule stops it
        r = np.full(36, 10.0, np.float32)
        r[18] = 0.33  # behind (the lidar is 2.3 cm behind the centre: 0.353 m from the centre)
        safety = {"stop_distance": 0.48, "corridor": 0.32, "rear_distance": 0.40, "sweep_radius": 0.35}
        (stride, _), rule = lidar_safety((-0.12, 0.0), r, safety)
        self.assertEqual(stride, 0.0)
        self.assertIn("rear", rule)
        self.assertEqual(lidar_safety((-0.12, 0.0), np.full(36, 10.0, np.float32), safety)[0][0], -0.12)
        # the balanced context keeps the rare backing-up rows in their own bin
        table = pd.DataFrame({"y_stride": [0.3] * 900 + [0.0] * 90 + [-0.12] * 10, "y_turn": [0.0] * 1000})
        self.assertEqual(int((choose_context(table, 100).y_stride < 0).sum()), 10)

    @unittest.skipUnless(HAVE_MUJOCO_GL, "needs MuJoCo rendering")
    def test_replay_adds_the_same_depth_columns(self):
        import pandas as pd
        from nav.dagger.augment import new_columns, replay_episode
        from nav.dagger.env import SimEnv
        from nav.dagger.expert import Expert
        from nav.dagger.featurize import Featurizer
        f = Featurizer("none", patch_pool=2, patch_depths=False, depth_summary=True, depth_scan=15, history=2)
        expert = Expert()
        env = SimEnv(7, 3, expert=expert)
        obs, state, rows, live = env.reset(), None, [], []
        for step in range(12):  # record a short expert episode with the live featurizer
            m, states = f.build_batch([obs], [state])
            state = states[0]
            live.append(dict(zip(f.columns, m[0])))
            x, y, yaw = env.pose()
            cmd = expert.act(env.state())
            obs, info = env.step(cmd)
            rows.append({"episode": 1, "scene_id": 7, "seed": 3, "step": step, "x": x, "y": y,
                         "executed_stride": info["stride"], "executed_turn": info["turn"],
                         "done_reason": info["done_reason"]})
            if info["done"]:
                break
        env.close()
        replay_env = SimEnv(7, 3, expert=Expert())
        frame, deviation, _ = replay_episode(replay_env, pd.DataFrame(rows), 15, 2, 10.0)
        replay_env.close()
        self.assertLess(deviation, 1e-6)
        self.assertEqual(len(new_columns(15, 2)), 4 + 15 + 8)
        for name in ["dc_ahead", "dc_near", "ds_0", "ds_7", "ds_14", "h1_dc_ahead", "h2_dc_left"]:
            np.testing.assert_allclose(frame[name].to_numpy(), [r[name] for r in live], atol=1e-4, err_msg=name)

    @unittest.skipUnless(HAVE_MUJOCO_GL, "needs MuJoCo rendering")
    def test_lidar_in_the_sim_matches_the_furniture(self):
        from nav.dagger.env import SimEnv
        env = SimEnv(100012, 99)
        obs = env.reset()
        self.assertEqual(obs["lidar"].shape, (36,))
        _, _, yaw = obs["robot_pose"]
        x, y, _ = env.sim.data.geom("lidar_geom").xpos  # beams start at the lidar, 2.3 cm behind the centre
        for k in (0, 9, 18, 27):  # check four beams against the true footprints in the lidar plane
            a = yaw + math.radians(10 * k)
            truth = 10.0
            for t in np.arange(0.05, 10.0, 0.01):
                px, py = x + t * math.cos(a), y + t * math.sin(a)
                if abs(px) > 3.5355 or abs(py) > 3.5355 or any(_inside(fp, px, py) for fp in env.footprints()):
                    truth = t
                    break
            self.assertLess(abs(float(obs["lidar"][k]) - truth), 0.06, "beam {}".format(k))
        env.close()

    def test_repool_matches_live_pooling(self):
        import pandas as pd
        from nav.dagger.featurize import columns, obstacle_summary, pool_grid
        from nav.dagger.repool import pool_arrays, repool_frame
        rng = np.random.default_rng(0)
        classes = rng.integers(0, 4, (3, 26, 46)).astype(np.int8)
        depths = rng.uniform(0.2, 10.0, (3, 26, 46)).astype(np.float32)
        batch_c, batch_d = pool_arrays(classes, depths, 2)
        for k in range(3):
            live_c, live_d = pool_grid(classes[k], depths[k], 2)
            np.testing.assert_array_equal(batch_c[k], live_c)
            np.testing.assert_allclose(batch_d[k], live_d, rtol=1e-5)
        feature, cat = columns(1)
        table = pd.DataFrame(np.concatenate([classes.reshape(3, -1), depths.reshape(3, -1),
                                             np.zeros((3, 8), np.float32)], axis=1), columns=feature)
        table["episode"] = [0, 0, 1]
        pooled = repool_frame(table, 2)
        self.assertEqual(list(pooled.columns), columns(2)[0] + ["episode"])
        self.assertEqual(len(columns(2)[0]), 299 * 2 + 8)
        expected = obstacle_summary(batch_c[1], batch_d[1])
        self.assertAlmostEqual(float(pooled["free_ahead"].iloc[1]), expected["free_ahead"], places=5)


class FakeEnv:
    """Walks one step per call; collides on step `collide_at`, reaches the goal on step `goal_at`."""

    def __init__(self, goal_at=5, collide_at=None):
        self.goal_at, self.collide_at = goal_at, collide_at
        self.goal, self.start = (1.0, 0.0), (0.0, 0.0, 0.0)

    def reset(self, scene_id, seed):
        self.scene_id, self.steps, self.path_length, self.executed = scene_id, 0, 0.0, []
        return self._obs()

    def _obs(self):
        return {"robot_pose": (0.1 * self.steps, 0.0, 0.0), "goal_point": self.goal}

    def state(self):
        return {"pose": self._obs()["robot_pose"]}

    def pose(self):
        return self._obs()["robot_pose"]

    def step(self, cmd):
        self.executed.append(tuple(cmd))
        self.steps += 1
        collision = self.collide_at is not None and self.steps >= self.collide_at
        reason = "collision" if collision else "goal" if self.steps >= self.goal_at else ""
        return (None if reason else self._obs()), {"collision": collision, "done": bool(reason),
                                                   "done_reason": reason, "stride": cmd[0], "turn": cmd[1]}


class FakeFeaturizer:
    columns, categorical = ["a", "b"], ["a"]

    def build_batch(self, observations, prev_states):
        return np.zeros((len(observations), 2), np.float32), prev_states


class FakeExpert:
    nopath = False

    def act(self, state):
        return (0.3, 0.1)


class FakePolicy:
    def __init__(self):
        self.calls = []

    def act_batch(self, rows):
        self.calls.append(len(rows))
        return np.tile([0.05, -0.2], (len(rows), 1))


@unittest.skipUnless(HAVE_ML, "needs the ml packages")
class RolloutTest(unittest.TestCase):
    def run_round(self, beta, envs, episodes=6):
        from nav.dagger.rollout import run_episodes, to_frame
        plan = [{"episode": i, "scene_id": i, "seed": 0} for i in range(episodes)]
        policy = FakePolicy()
        frames = []
        counts, steps = run_episodes(envs, FakeFeaturizer(), FakeExpert(), policy, beta, plan, 1,
                                     np.random.default_rng(0),
                                     on_chunk=lambda r: frames.append(to_frame(*r.take()[:2], ["a", "b"], ["a"])))
        table = __import__("pandas").concat([f for f in frames if len(f)], ignore_index=True)
        return counts, steps, table, policy

    def test_beta_one_runs_the_expert(self):
        counts, steps, table, policy = self.run_round(1.0, [FakeEnv() for _ in range(3)])
        self.assertEqual(policy.calls, [])  # no student needed
        self.assertTrue((table["executed_stride"] == 0.3).all())
        self.assertEqual(counts["goal"], 6)  # the quota of 6 episodes on 3 robots
        self.assertEqual(table["episode"].nunique(), 6)

    def test_beta_zero_runs_the_student_but_labels_with_the_expert(self):
        counts, steps, table, policy = self.run_round(0.0, [FakeEnv() for _ in range(3)])
        self.assertTrue(np.allclose(table["executed_stride"], 0.05))
        self.assertTrue(np.allclose(table["executed_turn"], -0.2))
        self.assertTrue((table["y_stride"] == 0.3).all() and (table["y_turn"] == 0.1).all())
        self.assertTrue(np.allclose(table["student_stride"], 0.05))
        self.assertEqual(sum(policy.calls), steps)  # every robot-step predicted ...
        self.assertEqual(len(policy.calls), steps // 3)  # ... in one call per lock-step

    def test_mixing_and_collisions(self):
        counts, steps, table, policy = self.run_round(0.5, [FakeEnv(goal_at=50, collide_at=3) for _ in range(2)],
                                                      episodes=40)
        self.assertEqual(counts["collision"], 40)
        last = table.groupby("episode").tail(1)
        self.assertTrue(last["collision"].all())  # the collision row is kept, with its expert label
        self.assertTrue((last["done_reason"] == "collision").all())
        self.assertTrue((table.groupby("episode").size() == 3).all())  # and the episode ended there
        share = table["used_expert"].mean()
        self.assertGreater(share, 0.35)
        self.assertLess(share, 0.65)


@unittest.skipUnless(HAVE_ML, "needs the ml packages")
class StudentTest(unittest.TestCase):
    def test_act_batch_clips_and_context_cap(self):
        import pandas as pd

        from nav.dagger.policy import TabPFNPolicy, choose_context

        class Wild:
            def __init__(self, value):
                self.value = value

            def predict(self, X):
                return np.full(len(X), self.value)

        policy = TabPFNPolicy(["a", "b"], ["a"])
        policy.models = {"y_stride": Wild(5.0), "y_turn": Wild(-5.0)}
        out = policy.act_batch(pd.DataFrame({"a": [0, 1, 2], "b": [0.1, 0.2, 0.3]}))
        self.assertEqual(out.shape, (3, 2))
        self.assertTrue(np.allclose(out[:, 0], 0.3) and np.allclose(out[:, 1], -0.25))
        rng = np.random.default_rng(0)
        table = pd.DataFrame({"y_stride": rng.uniform(0, 0.3, 500), "y_turn": rng.uniform(-0.25, 0.25, 500)})
        self.assertEqual(len(choose_context(table, None)), 500)
        capped = choose_context(table, 120)
        self.assertEqual(len(capped), 120)
        self.assertGreater(capped["y_turn"].std(), 0.1)  # still spread over the turn range


def _job(value):
    """For WorkerPoolTest: value 0 kills its own process, as the out-of-memory guard would; -1 sleeps."""
    import os
    import time
    if value == 0:
        os._exit(9)
    if value == -1:
        time.sleep(60)
    return value * 2


@unittest.skipUnless(HAVE_ML, "needs the ml packages")
class WorkerPoolTest(unittest.TestCase):
    def test_a_killed_process_raises_instead_of_hanging(self):
        from concurrent.futures.process import BrokenProcessPool

        from nav.dagger.worker import run_jobs
        self.assertEqual(run_jobs(_job, [1, 2, 3]), [2, 4, 6])
        with self.assertRaises(BrokenProcessPool):
            run_jobs(_job, [1, 0, 3])

    def test_memory_guard_stops_the_workers(self):
        import time

        from nav.dagger.worker import MemoryLow, run_jobs
        start = time.time()
        with self.assertRaises(MemoryLow):  # a reserve larger than any machine: the guard must trip
            run_jobs(_job, [-1, -1], log=lambda m: None, reserve=1e6, every=0.5)
        self.assertLess(time.time() - start, 20)  # stopped, not waiting for the 60 s jobs

    def test_process_count_fits_memory(self):
        from nav.dagger.guard import available_gb, fit_procs, reserve_gb
        self.assertEqual(fit_procs(32, available=40, reserve=8, per_proc=2.5), 12)
        self.assertEqual(fit_procs(8, available=40, reserve=8, per_proc=2.5), 8)
        self.assertEqual(fit_procs(8, available=5, reserve=8, per_proc=2.5), 1)  # never 0
        self.assertGreater(available_gb(), 0)
        self.assertGreaterEqual(reserve_gb(total=121), 12.0)  # 10% of a large machine


@unittest.skipUnless(HAVE_ML, "needs the ml packages")
class ArenaStepTest(unittest.TestCase):
    def test_one_real_step(self):
        from nav.dagger.env import SimEnv
        from nav.dagger.expert import Expert
        expert = Expert()
        try:
            env = SimEnv(simple=True, expert=expert)
            obs = env.reset(3, 0)
        except Exception as error:  # no OpenGL on this machine (same rule as the camera test)
            self.skipTest("cannot render: {}".format(error))
        self.assertEqual(obs["rgb"].shape, (360, 640, 3))
        self.assertEqual(obs["depth"].shape, (360, 640))
        self.assertGreater(len(env.footprints()), 8)
        obs, info = env.step(expert.act(env.state()))
        self.assertEqual(set(info) >= {"collision", "done", "done_reason"}, True)
        self.assertFalse(info["collision"])
        self.assertEqual(env.steps, 1)
        env.close()


if __name__ == "__main__":
    unittest.main()

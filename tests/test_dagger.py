"""Tests for nav/dagger: scenes, expert, features, rollout mixing, student wrapper, one real arena step.

No GPU, no TabPFN key and no downloads: TabPFN and DINOv2 are replaced by fakes.
"""

import math
import sys
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

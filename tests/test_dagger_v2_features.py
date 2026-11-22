"""Tests for the DAgger v2 feature variants (nav/dagger/featurize.py: derive, drop_constant, derive_live).

Synthetic grids only: no GPU, no TabPFN key, no downloads.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pandas as pd
    import yaml  # noqa: F401
    HAVE_ML = True
except ImportError:
    HAVE_ML = False

FLOOR, WALL, OBSTACLE, TARGET = 0, 1, 2, 3


def make_base(n=1, goal=(3.0, 0.0, 1.0), episode=None, pose=None):
    """n base rows: all floor, depth 10, goal straight ahead."""
    from nav.dagger.featurize import COLS, ROWS, base_columns
    pc, pdn = base_columns()
    frame = pd.DataFrame(np.zeros((n, ROWS * COLS), np.int8), columns=pc)
    frame = pd.concat([frame, pd.DataFrame(np.full((n, ROWS * COLS), 10.0, np.float32), columns=pdn)], axis=1)
    frame["goal_dist"], frame["goal_sin"], frame["goal_cos"] = goal
    frame["episode"] = 0 if episode is None else episode
    frame["step"] = np.arange(n)
    frame["x"], frame["y"], frame["yaw"] = (0.0, 0.0, 0.0) if pose is None else pose
    return frame


def set_cell(frame, row, r, c, cls=None, depth=None):
    if cls is not None:
        frame.loc[row, "pc_{}_{}".format(r, c)] = cls
    if depth is not None:
        frame.loc[row, "pd_{}_{}".format(r, c)] = depth


@unittest.skipUnless(HAVE_ML, "needs the ml packages: uv run --group ml --group vision")
class FeatureVariantsTest(unittest.TestCase):
    def test_default_reproduces_feature_columns(self):
        from nav.dagger.featurize import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, derive
        matrix, names, cat = derive(make_base(2))
        self.assertEqual(names, FEATURE_COLUMNS)
        self.assertEqual(cat, CATEGORICAL_COLUMNS)
        self.assertEqual(matrix.shape, (2, len(FEATURE_COLUMNS)))
        self.assertEqual(matrix.dtype, np.float32)

    def test_default_matches_per_row_build(self):
        """Same numbers as the existing per-row path (pool_grid + obstacle_summary), pooled or not."""
        from nav.dagger.featurize import (COLS, ROWS, columns, derive, obstacle_summary, pool_grid, SUMMARY)
        rng = np.random.default_rng(0)
        base = make_base(3)
        for pool in (1, 2):
            cls = rng.integers(0, 4, (3, ROWS, COLS)).astype(np.int8)
            dep = rng.uniform(0.5, 10, (3, ROWS, COLS)).astype(np.float32)
            base.loc[:, [c for c in base.columns if c.startswith("pc_")]] = cls.reshape(3, -1)
            base.loc[:, [c for c in base.columns if c.startswith("pd_")]] = dep.reshape(3, -1)
            matrix, names, _ = derive(base, {"pool": pool})
            self.assertEqual(names, columns(pool)[0])
            for i in range(3):
                c, d = pool_grid(cls[i], dep[i], pool)
                s = obstacle_summary(c, d)
                row = np.asarray([s[k] for k in SUMMARY], np.float32)
                np.testing.assert_allclose(matrix[i, -5:], row, rtol=1e-5)
                np.testing.assert_allclose(matrix[i, :c.size], c.reshape(-1))

    def test_any_solid_keeps_single_obstacle(self):
        from nav.dagger.featurize import derive
        base = make_base()
        set_cell(base, 0, 4, 6, OBSTACLE, 2.0)  # one patch in the 2x2 block (rows 4-5, cols 6-7)
        maj, names, _ = derive(base, {"pool": 2, "pool_rule": "majority"})
        anyp, _, _ = derive(base, {"pool": 2, "pool_rule": "any_solid"})
        i = names.index("pc_2_3")
        self.assertEqual(maj[0, i], FLOOR)
        self.assertEqual(anyp[0, i], OBSTACLE)
        # depth is the block mean
        j = names.index("pd_2_3")
        self.assertAlmostEqual(float(anyp[0, j]), (2.0 + 30.0) / 4, places=5)

    def test_any_solid_priority_and_leg(self):
        from nav.dagger.featurize import LEG, derive
        base = make_base()
        set_cell(base, 0, 0, 0, WALL)
        set_cell(base, 0, 0, 1, OBSTACLE)
        set_cell(base, 0, 1, 0, TARGET)
        set_cell(base, 0, 2, 0, LEG)  # a block with only legs and floor is floor
        set_cell(base, 0, 2, 1, LEG)
        set_cell(base, 0, 3, 0, LEG)
        set_cell(base, 0, 3, 1, LEG)
        matrix, names, _ = derive(base, {"pool": 2, "pool_rule": "any_solid"})
        self.assertEqual(matrix[0, names.index("pc_0_0")], TARGET)
        self.assertEqual(matrix[0, names.index("pc_1_0")], FLOOR)

    def test_pooled_shapes(self):
        from nav.dagger.featurize import derive
        for rule in ("majority", "any_solid"):
            matrix, names, cat = derive(make_base(2), {"pool": 2, "pool_rule": rule})
            self.assertEqual(sum(n.startswith("pc_") for n in names), 13 * 23)
            self.assertEqual(sum(n.startswith("pd_") for n in names), 13 * 23)
            self.assertEqual(len(cat), 13 * 23)
            self.assertEqual(matrix.shape[0], 2)

    def test_seven_sectors_and_goal_free(self):
        from nav.dagger.featurize import COLS, HFOV, column_bearings, derive, sector_edges
        base = make_base()
        edges = sector_edges(COLS)
        self.assertEqual(list(edges), [0, 7, 13, 20, 26, 33, 39, 46])
        for s, depth in ((0, 2.0), (1, 3.0), (2, 4.0)):
            for c in range(edges[s], edges[s + 1]):
                set_cell(base, 0, 10, c, OBSTACLE, depth)
        set_cell(base, 0, 12, 24, WALL, 1.5)  # in sector 3 (ahead)
        matrix, names, cat = derive(base, {"summary": "7"})
        v = dict(zip(names, matrix[0]))
        for s, want in enumerate([2.0, 3.0, 4.0, 1.5, 10.0, 10.0, 10.0]):
            self.assertAlmostEqual(v["free_s{}".format(s)], want, places=5)
        self.assertAlmostEqual(v["near_dist"], 1.5, places=5)
        self.assertAlmostEqual(v["near_bearing"], float(column_bearings(COLS)[24]), places=5)
        self.assertAlmostEqual(v["widest_bearing"], float(np.mean(column_bearings(COLS)[26:33])), places=5)
        self.assertLess(v["widest_bearing"], 0)  # sector 4 is right of centre
        self.assertAlmostEqual(v["goal_free"], 1.5, places=5)  # goal dead ahead: sector 3
        for k in ("free_left", "free_ahead", "free_right"):
            self.assertNotIn(k, names)
        # goal to the left, inside the view, in sector 1
        left = make_base(goal=(3.0, math.sin(0.3), math.cos(0.3)))
        left.loc[0, [c for c in left.columns if c.startswith("pd_")]] = 10.0
        for c in range(edges[1], edges[2]):
            set_cell(left, 0, 5, c, OBSTACLE, 3.0)
        m2, n2, _ = derive(left, {"summary": "7"})
        self.assertAlmostEqual(dict(zip(n2, m2[0]))["goal_free"], 3.0, places=5)
        # goal outside the camera's horizontal view: the cap
        outside = make_base(goal=(3.0, math.sin(HFOV), math.cos(HFOV)))
        set_cell(outside, 0, 5, 23, OBSTACLE, 1.0)
        m3, n3, _ = derive(outside, {"summary": "7", "cap": 10.0})
        self.assertEqual(dict(zip(n3, m3[0]))["goal_free"], 10.0)

    def test_history_same_episode_and_earlier_only(self):
        from nav.dagger.featurize import derive
        base = pd.concat([make_base(3, episode=0, pose=(0.0, 0.0, 0.0)),
                          make_base(2, episode=1, pose=(5.0, 5.0, 0.0))], ignore_index=True)
        base["step"] = [0, 1, 2, 0, 1]
        base["x"] = [0.0, 0.5, 1.0, 5.0, 5.0]
        base["y"] = [0.0, 0.0, 0.2, 5.0, 5.0]
        base["yaw"] = [0.0, 0.0, math.pi / 2, 0.0, 0.0]
        for r, d in enumerate([2.0, 3.0, 4.0, 9.0, 9.0]):  # distinct free distance per row, in sector 3
            set_cell(base, r, 10, 23, OBSTACLE, d)
        base["goal_sin"] = [0.0, 0.1, 0.2, 0.3, 0.4]
        matrix, names, _ = derive(base, {"summary": "7", "history": 2, "patches": False})
        v = {n: matrix[:, i] for i, n in enumerate(names)}
        # row 2 (episode 0, step 2): one step back is row 1, two steps back is row 0
        self.assertAlmostEqual(v["h1_free_s3"][2], 3.0, places=5)
        self.assertAlmostEqual(v["h2_free_s3"][2], 2.0, places=5)
        self.assertAlmostEqual(v["h1_goal_sin"][2], 0.1, places=5)
        self.assertAlmostEqual(v["odo_fwd_1"][2], 0.5, places=5)  # from (0.5, 0, yaw 0) to (1.0, 0.2)
        self.assertAlmostEqual(v["odo_left_1"][2], 0.2, places=5)
        self.assertAlmostEqual(v["odo_dyaw_1"][2], math.pi / 2, places=5)
        # first step of episode 0 repeats itself; first step of episode 1 does not see episode 0
        self.assertAlmostEqual(v["h1_free_s3"][0], 2.0, places=5)
        self.assertAlmostEqual(v["h1_free_s3"][3], 9.0, places=5)
        self.assertAlmostEqual(v["h2_free_s3"][3], 9.0, places=5)
        self.assertEqual(v["odo_fwd_1"][3], 0.0)
        self.assertAlmostEqual(v["h1_goal_sin"][4], 0.3, places=5)
        # a later row never leaks into an earlier one: change row 2, rows 0 and 1 stay the same
        base2 = base.copy()
        set_cell(base2, 2, 10, 23, OBSTACLE, 0.5)
        m2, _, _ = derive(base2, {"summary": "7", "history": 2, "patches": False})
        np.testing.assert_array_equal(m2[:2], matrix[:2])
        np.testing.assert_array_equal(m2[3:], matrix[3:])

    def test_history_unsorted_input(self):
        from nav.dagger.featurize import derive
        base = make_base(3)
        base["x"] = [0.0, 1.0, 2.0]
        for r, d in enumerate([2.0, 3.0, 4.0]):
            set_cell(base, r, 10, 23, OBSTACLE, d)
        want, names, _ = derive(base, {"history": 1, "patches": False, "summary": "7"})
        shuffled = base.iloc[[2, 0, 1]].reset_index(drop=True)
        got, _, _ = derive(shuffled, {"history": 1, "patches": False, "summary": "7"})
        np.testing.assert_allclose(got, want[[2, 0, 1]], rtol=1e-6)

    def test_summary_only_and_categorical_flag(self):
        from nav.dagger.featurize import derive
        matrix, names, cat = derive(make_base(), {"patches": False, "summary": "7"})
        self.assertFalse([n for n in names if n.startswith(("pc_", "pd_"))])
        self.assertEqual(cat, [])
        self.assertEqual(names[:3], ["goal_dist", "goal_sin", "goal_cos"])
        _, names2, cat2 = derive(make_base(), {"categorical": False})
        self.assertEqual(cat2, [])
        self.assertTrue(any(n.startswith("pc_") for n in names2))

    def test_drop_constant(self):
        from nav.dagger.featurize import drop_constant
        matrix = np.array([[1, 5, 0], [1, 6, 0], [1, 7, 1], [2, 8, 0]], np.float32)
        mask = drop_constant(matrix, ["a", "b", "c"], 0.7)
        # a: 3/4 = 0.75 > 0.7 dropped; b: 0.25 kept; c: 0.75 dropped
        self.assertEqual(mask.tolist(), [False, True, False])
        self.assertEqual(drop_constant(matrix, ["a", "b", "c"], 0.99).tolist(), [True, True, True])
        self.assertEqual(drop_constant(matrix, ["a", "b", "c"], 0.74).tolist(), [False, True, False])

    def test_derive_live_matches_fit_time(self):
        """Feeding rows one by one through derive_live gives the same features as derive on the whole episode."""
        from nav.dagger.featurize import derive, derive_live
        variant = {"summary": "7", "history": 2, "patches": False}
        base = make_base(4)
        base["x"] = [0.0, 0.3, 0.7, 1.2]
        base["yaw"] = [0.0, 0.1, 0.2, 0.3]
        for r, d in enumerate([2.0, 3.0, 4.0, 5.0]):
            set_cell(base, r, 10, 23, OBSTACLE, d)
        want, names, _ = derive(base, variant)
        state = [None]
        for r in range(4):
            got, names2, _, state = derive_live(base.iloc[[r]].reset_index(drop=True), state, variant)
            self.assertEqual(names2, names)
            np.testing.assert_allclose(got[0], want[r], rtol=1e-5, atol=1e-6)
            self.assertLessEqual(len(state[0]), 2)


if __name__ == "__main__":
    unittest.main()

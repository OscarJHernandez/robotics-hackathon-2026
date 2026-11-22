"""v2 feature pipeline (Features), the rule baseline and the data-set plans. No simulator, no TabPFN."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pandas as pd
    HAVE_ML = True
except ImportError:
    HAVE_ML = False


def base_rows(n=6, episodes=2, rays=36, seed=0):
    """Stored base rows: floor everywhere at 4 m, goal ahead, lidar at 5 m."""
    from nav.dagger.featurize import base_columns, lidar_columns
    from scene import FLOOR
    rng = np.random.default_rng(seed)
    pc, pdn = base_columns()
    frame = pd.DataFrame(np.full((n, len(pc)), FLOOR, np.int8), columns=pc)
    frame = pd.concat([frame, pd.DataFrame(np.full((n, len(pdn)), 4.0, np.float32), columns=pdn)], axis=1)
    frame["goal_dist"], frame["goal_sin"], frame["goal_cos"] = 2.0, 0.0, 1.0
    for k in ("near_dist", "near_bearing", "free_left", "free_ahead", "free_right"):
        frame[k] = 4.0
    lid = pd.DataFrame(np.full((n, rays), 5.0, np.float32), columns=lidar_columns(rays))
    frame = pd.concat([frame, lid], axis=1)
    frame["episode"] = np.repeat(np.arange(episodes), n // episodes)
    frame["step"] = np.tile(np.arange(n // episodes), episodes)
    frame["x"], frame["y"], frame["yaw"] = rng.normal(size=n), rng.normal(size=n), 0.0
    frame["look_bearing"] = 0.3
    return frame


@unittest.skipUnless(HAVE_ML, "needs the ml packages")
class FeaturesTest(unittest.TestCase):
    def test_unknown_key_and_missing_lidar_fail(self):
        from nav.dagger.featurize import Features
        with self.assertRaises(ValueError):
            Features({"colour": 1})
        rows = base_rows().drop(columns=[c for c in base_rows().columns if c.startswith("lr_")])
        with self.assertRaises(ValueError):
            Features({"patches": False, "lidar": True}).fit_transform(rows)

    def test_scan_finds_a_thin_obstacle_in_the_right_sector(self):
        from nav.dagger.featurize import BAND_ROWS, Features
        from scene import OBSTACLE
        rows = base_rows(n=2, episodes=1)
        col = 40  # far right of the picture: the last of 15 sectors
        for r in range(BAND_ROWS[0], BAND_ROWS[1]):
            rows["pc_{}_{}".format(r, col)] = OBSTACLE
            rows["pd_{}_{}".format(r, col)] = 0.7
        feats = Features({"patches": False, "summary": "none", "scan": 15})
        X = feats.fit_transform(rows)
        scan = X[0, [feats.columns.index("scan_{}".format(i)) for i in range(15)]]
        self.assertAlmostEqual(float(scan[13]), 0.7, places=5)  # 40 is in sector 13 of 15 (edges 39.8 .. 42.9)
        self.assertTrue(np.all(np.delete(scan, 13) == 10.0))  # floor only elsewhere: the cap

    def test_columns_match_at_fit_transform_and_live(self):
        from nav.dagger.featurize import Features
        rows = base_rows(n=8, episodes=2)
        band = np.random.default_rng(1).normal(size=(8, 46, 384)).astype(np.float16)
        cfg = {"patches": False, "summary": "none", "band_depth": 15, "band_pca": 4, "lidar": True}
        feats = Features(cfg)
        X = feats.fit_transform(rows, band)
        self.assertEqual(X.shape, (8, 3 + 15 + 15 * 4 + 36))
        self.assertEqual(feats.transform(rows, band).shape, X.shape)
        np.testing.assert_allclose(feats.transform(rows, band), X, rtol=1e-5, atol=1e-5)
        live, prev = feats.live(rows.iloc[:2].reset_index(drop=True), band[:2], [None, None])
        np.testing.assert_allclose(live, X[:2], rtol=1e-5, atol=1e-5)
        self.assertEqual(len(prev), 2)

    def test_pca_is_fitted_on_the_context_only(self):
        from nav.dagger.featurize import Features
        rows = base_rows(n=8, episodes=2)
        rng = np.random.default_rng(2)
        band_ctx = rng.normal(size=(8, 46, 16)).astype(np.float16)
        feats = Features({"patches": False, "summary": "none", "band_pca": 2})
        feats.fit_transform(rows, band_ctx)
        pca = [a.copy() for a in feats.pca[:3]]
        feats.transform(rows, rng.normal(size=(8, 46, 16)).astype(np.float16) * 50)  # held-out rows: never refit
        for before, after in zip(pca, feats.pca[:3]):
            np.testing.assert_array_equal(before, after)

    def test_drop_constant_keeps_goal_and_same_columns(self):
        from nav.dagger.featurize import Features
        rows = base_rows(n=6, episodes=2)
        feats = Features({"patches": True, "pool": 2, "drop_constant": 0.95})
        X = feats.fit_transform(rows)
        self.assertIn("goal_dist", feats.columns)  # constant here, but never dropped
        self.assertLess(len(feats.columns), 13 * 23 * 2 + 8)
        self.assertEqual(feats.transform(rows).shape, X.shape)

    def test_lookahead_is_only_there_when_asked(self):
        from nav.dagger.featurize import Features
        rows = base_rows()
        self.assertNotIn("look_sin", Features({"patches": False}).fit_transform(rows) is not None
                         and Features({"patches": False}).columns or [])
        feats = Features({"patches": False, "lookahead": True})
        X = feats.fit_transform(rows)
        self.assertAlmostEqual(float(X[0, feats.columns.index("look_sin")]), math.sin(0.3), places=5)


@unittest.skipUnless(HAVE_ML, "needs the ml packages")
class RuleAndPlansTest(unittest.TestCase):
    def test_rule_stops_when_the_goal_is_off_to_the_side(self):
        from nav.dagger.policy import RulePolicy, make_student
        cols = ["goal_dist", "goal_sin", "goal_cos"]
        rule = make_student({"kind": "rule"}, cols, [])
        self.assertIsInstance(rule, RulePolicy)
        X = np.array([[2.0, 0.0, 1.0], [2.0, math.sin(2.0), math.cos(2.0)], [2.0, math.sin(-1.5), math.cos(-1.5)]])
        out = rule.act_batch(X)
        self.assertGreater(out[0, 0], 0.25)  # goal ahead: walk
        self.assertEqual(out[1, 0], 0.0)  # goal behind-left: stop ...
        self.assertGreater(out[1, 1], 0.2)  # ... and turn left
        self.assertEqual(out[2, 0], 0.0)
        self.assertLess(out[2, 1], -0.2)

    def test_sets_are_disjoint(self):
        from nav.dagger.config import SETS, set_plan
        scenes = {name: {e["scene_id"] for e in set_plan(name)} for name in SETS}
        training = scenes["round0"] | scenes["perturb"]
        for name in ("offline_expert", "screen", "confirm", "final"):
            self.assertFalse(training & scenes[name], name)
        for a in ("offline_expert", "screen", "confirm", "final"):
            for b in ("offline_expert", "screen", "confirm", "final"):
                if a < b:
                    self.assertFalse(scenes[a] & scenes[b], (a, b))
        ids = [e["episode"] for name in SETS for e in set_plan(name)]
        self.assertEqual(len(ids), len(set(ids)))  # episode ids never repeat across sets
        self.assertEqual(len(set_plan("screen", 100, 100)), 100)

    def test_variant_files_load(self):
        from nav.dagger.experiment import load_variant
        folder = Path(__file__).resolve().parents[1] / "nav" / "dagger" / "variants"
        files = sorted(folder.glob("*.yaml"))
        self.assertGreater(len(files), 5)
        for path in files:
            variant = load_variant(path)
            self.assertEqual(variant["id"], path.stem)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_ML, "needs the ml packages")
class DaggerRoundSetsTest(unittest.TestCase):
    def test_dagger_rounds_are_disjoint_from_everything(self):
        from nav.dagger.config import SETS, set_plan
        fixed = {e["scene_id"] for name in SETS for e in set_plan(name)}
        fixed_ids = {e["episode"] for name in SETS for e in set_plan(name)}
        rounds = [set_plan("dagger_r{}".format(k)) for k in (1, 2, 3, 60)]
        for plan in rounds:
            self.assertFalse({e["scene_id"] for e in plan} & fixed)
            self.assertFalse({e["episode"] for e in plan} & fixed_ids)
        self.assertFalse({e["scene_id"] for e in rounds[0]} & {e["scene_id"] for e in rounds[1]})
        with self.assertRaises(KeyError):
            set_plan("dagger_r0")


@unittest.skipUnless(HAVE_ML, "needs the ml packages")
class LidarQuotaTest(unittest.TestCase):
    def test_rare_near_obstacle_stops_are_all_kept(self):
        from nav.dagger.policy import choose_context
        rng = np.random.default_rng(0)
        n = 2000
        table = pd.DataFrame({"y_stride": np.full(n, 0.3), "y_turn": rng.uniform(-0.1, 0.1, n),
                              "free_ahead": np.full(n, 3.0), "used_expert": np.ones(n, bool)})
        for i in range(36):
            table["lr_{}".format(i)] = 2.0
        rare = np.arange(20)  # 20 rows: obstacle 0.25 m beside, expert stops and turns
        table.loc[rare, "lr_9"] = 0.22
        table.loc[rare, "y_stride"] = 0.0
        table.loc[rare, "y_turn"] = 0.25
        picked = choose_context(table, 300, seed=0, quotas="mode_x_lidar")
        self.assertEqual(len(picked), 300)
        self.assertTrue(set(rare) <= set(picked.index))  # every rare row survives the cap
        uniform = choose_context(table, 300, seed=0, quotas="uniform")
        self.assertLess(len(set(rare) & set(uniform.index)), 10)


@unittest.skipUnless(HAVE_ML, "needs the ml packages")
class LidarShieldTest(unittest.TestCase):
    def test_stops_and_turns_away_only_when_close_in_front(self):
        from nav.dagger.policy import LidarShield

        class Walk:
            def act_batch(self, X):
                return np.tile([0.3, 0.05], (len(X), 1))

        cols = ["goal_dist"] + ["lr_{}".format(i) for i in range(36)]
        X = np.full((4, 37), 2.0)
        X[1, 1 + 6] = 0.2     # 60 degrees left, close: stop, turn right
        X[2, 1 + 30] = 0.2    # 300 = -60 degrees (right), close: stop, turn left
        X[3, 1 + 18] = 0.2    # straight behind: ignored
        out = LidarShield(Walk(), cols, min_range=0.27, fov_deg=90).act_batch(X)
        np.testing.assert_allclose(out[0], [0.3, 0.05])
        self.assertEqual(out[1, 0], 0.0)
        self.assertLess(out[1, 1], 0)
        self.assertEqual(out[2, 0], 0.0)
        self.assertGreater(out[2, 1], 0)
        np.testing.assert_allclose(out[3], [0.3, 0.05])

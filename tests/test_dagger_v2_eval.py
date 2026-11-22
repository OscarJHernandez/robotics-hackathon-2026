"""Tests for the DAgger v2 evaluation harness: statistics, plans, closed-loop summary, offline metrics."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pandas as pd
    import yaml  # noqa: F401
    HAVE_ML = True
except ImportError:
    HAVE_ML = False


class StatsTest(unittest.TestCase):
    def test_wilson(self):
        from nav.dagger.stats import wilson
        low, high = wilson(45, 50)
        self.assertAlmostEqual(low, 0.7864, places=3)
        self.assertAlmostEqual(high, 0.9565, places=3)
        self.assertEqual(wilson(0, 0), (0.0, 1.0))
        self.assertEqual(wilson(50, 50)[1], 1.0)
        self.assertEqual(wilson(0, 50)[0], 0.0)

    def test_sign_test(self):
        from nav.dagger.stats import sign_test
        self.assertAlmostEqual(sign_test(8, 1), 2 * 10 / 512)  # 2 * P(X<=1), n=9
        self.assertAlmostEqual(sign_test(1, 8), 2 * 10 / 512)
        self.assertEqual(sign_test(0, 0), 1.0)
        self.assertEqual(sign_test(5, 5), 1.0)
        self.assertAlmostEqual(sign_test(10, 0), 2 / 1024)

    def test_paired(self):
        from nav.dagger.stats import paired
        a = {1: "goal", 2: "goal", 3: "collision", 4: "goal", 5: "timeout", 9: "goal"}
        b = {1: "goal", 2: "collision", 3: "goal", 4: "fell", 5: "timeout", 8: "goal"}
        out = paired(a, b)  # key 9 and 8 are not shared
        self.assertEqual((out["wins"], out["losses"], out["ties"]), (2, 1, 2))
        self.assertAlmostEqual(out["p"], 1.0)


@unittest.skipUnless(HAVE_ML, "needs the ml packages: uv run --group ml --group vision")
class EvaluateTest(unittest.TestCase):
    def test_plan_ids(self):
        from nav.dagger.config import HELDOUT_START
        from nav.dagger.evaluate import eval_plan
        s99, s98 = eval_plan("screen-50"), eval_plan("screen-50", 98)
        f99 = eval_plan("final-200", 99)
        self.assertEqual(len(s99), 50)
        self.assertEqual(len(f99), 200)
        self.assertEqual([e["scene_id"] for e in s99], list(range(HELDOUT_START, HELDOUT_START + 50)))
        self.assertEqual(s99[0]["episode"], 90_000_000)
        self.assertEqual(s98[0]["episode"], 89_000_000)
        self.assertTrue({e["seed"] for e in s99} == {99} and {e["seed"] for e in s98} == {98})
        ids = [e["episode"] for e in f99 + s98 + eval_plan("final-200", 98)[50:]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse({e["episode"] for e in f99} & {e["episode"] for e in eval_plan("final-200", 98)})

    def test_closed_loop_summary(self):
        from nav.dagger.evaluate import closed_loop_summary
        episodes = ([{"episode": i, "done_reason": "goal", "steps": 10 + i} for i in range(4)]
                    + [{"episode": 4, "done_reason": "collision", "steps": 5},
                       {"episode": 5, "done_reason": "timeout", "steps": 300}])
        s = closed_loop_summary(episodes)
        self.assertEqual(s["episodes"], 6)
        self.assertEqual(s["outcomes"]["goal"]["count"], 4)
        self.assertEqual(s["outcomes"]["fell"]["count"], 0)
        self.assertAlmostEqual(s["outcomes"]["goal"]["rate"], 0.6667, places=4)
        self.assertLess(s["outcomes"]["goal"]["ci_low"], 0.6667)
        self.assertGreater(s["outcomes"]["goal"]["ci_high"], 0.6667)
        self.assertEqual(s["mean_steps_to_goal"], 11.5)
        self.assertEqual(s["median_steps_to_goal"], 11.5)
        self.assertEqual(closed_loop_summary(pd.DataFrame(episodes))["episodes"], 6)
        self.assertEqual(closed_loop_summary(episodes + episodes[:2])["episodes"], 6)  # duplicates collapse
        self.assertIsNone(closed_loop_summary([])["mean_steps_to_goal"])

    def test_compare_rounds(self):
        from nav.dagger.evaluate import compare_rounds
        a = [{"scene_id": 1, "seed": 99, "done_reason": "goal"}, {"scene_id": 2, "seed": 99, "done_reason": "goal"}]
        b = [{"scene_id": 1, "seed": 99, "done_reason": "collision"}, {"scene_id": 2, "seed": 99, "done_reason": "goal"},
             {"scene_id": 3, "seed": 99, "done_reason": "goal"}]
        out = compare_rounds(a, b)
        self.assertEqual((out["wins"], out["losses"], out["ties"], out["matched"]), (1, 0, 1, 2))

    def test_offline_metrics(self):
        from nav.dagger.evaluate import offline_metrics
        df = pd.DataFrame({
            "free_ahead": [0.2, 0.4, 0.7, 1.5, 2.0, 0.3],
            "y_stride": [0.0, 0.0, 0.1, 0.2, 0.2, 0.15],
            "y_turn": [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            "pred_stride": [0.0, 0.1, 0.1, 0.1, 0.2, 0.0],
            "pred_turn": [0.5, 0.0, 0.0, 0.0, 0.2, 0.0]})
        m = offline_metrics(df)
        self.assertEqual(m["by_free_ahead"]["<0.5"]["n"], 3)
        self.assertEqual(m["by_free_ahead"]["0.5-1"]["n"], 1)
        self.assertEqual(m["by_free_ahead"][">1"]["n"], 2)
        self.assertAlmostEqual(m["by_free_ahead"][">1"]["mae_stride"], 0.05)
        self.assertAlmostEqual(m["by_free_ahead"][">1"]["mae_turn"], 0.1)
        self.assertEqual(m["by_mode"]["stop_and_turn"]["n"], 2)
        self.assertEqual(m["by_mode"]["walk"]["n"], 4)
        self.assertAlmostEqual(m["by_mode"]["stop_and_turn"]["mae_stride"], 0.05)
        s = m["stop_and_turn"]  # expert stops: rows 0,1. predicted stops (<0.05): rows 0,5
        self.assertEqual((s["expert_stops"], s["predicted_stops"]), (2, 2))
        self.assertAlmostEqual(s["recall"], 0.5)
        self.assertAlmostEqual(s["precision"], 0.5)
        empty = offline_metrics(df.iloc[:0])
        self.assertIsNone(empty["stop_and_turn"]["recall"])


if __name__ == "__main__":
    unittest.main()

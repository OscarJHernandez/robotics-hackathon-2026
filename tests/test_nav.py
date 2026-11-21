"""Checks for walking a→b: geometry, the walker, the database recorder and hindsight relabelling.

Needs the ml packages: uv run --group ml python -m unittest tests/test_nav.py -v (or `just test-nav`).
Each test uses its own temporary data folder, never data/nav.
"""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nav"))

try:
    import duckdb  # noqa: F401
    import pandas  # noqa: F401
    HAVE_ML = True
except ImportError:
    HAVE_ML = False


@unittest.skipUnless(HAVE_ML, "needs the ml packages: uv run --group ml ...")
class NavTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.environ["NAV_DATA_DIR"] = self.tmp.name
        self.addCleanup(os.environ.pop, "NAV_DATA_DIR", None)

    def test_relative_bearing_and_goal(self):
        from common import MAX_HORIZON, MAX_STEP, make_goal, relative
        dist, bearing = relative((0.0, 0.0, math.pi / 2), 0.0, 1.0)  # facing +y, b straight ahead
        self.assertAlmostEqual(dist, 1.0)
        self.assertAlmostEqual(bearing, 0.0)
        _, left = relative((0.0, 0.0, 0.0), 0.0, 1.0)
        self.assertAlmostEqual(left, math.pi / 2)  # b to the left is a positive bearing
        far = make_goal(5.0, 0.3)
        self.assertEqual(far["horizon"], MAX_HORIZON)
        self.assertAlmostEqual(far["goal_dist"], MAX_HORIZON * MAX_STEP)
        self.assertEqual(make_goal(0.05, 0.0)["horizon"], 1)
        # delta_x/delta_y: the same capped goal in the body frame (ahead, left)
        self.assertAlmostEqual(math.hypot(far["delta_x"], far["delta_y"]), MAX_HORIZON * MAX_STEP)
        ahead = make_goal(0.5, 0.0)
        self.assertAlmostEqual(ahead["delta_x"], 0.5)
        self.assertAlmostEqual(ahead["delta_y"], 0.0)
        self.assertGreater(make_goal(0.5, 0.4)["delta_y"], 0.0)  # b to the left
        self.assertLess(make_goal(0.5, -0.4)["delta_y"], 0.0)

    def test_feature_groups(self):
        from hindsight import feature_columns
        self.assertEqual(feature_columns(("goal",), None), ["goal_dist", "goal_sin", "goal_cos", "horizon"])
        self.assertEqual(feature_columns(("goal", "delta"), None)[-2:], ["delta_x", "delta_y"])
        with self.assertRaises(ValueError):
            feature_columns(("nope",), None)

    def test_walker_records_a_cycle(self):
        from common import JetHexaSim, Walker
        sim = JetHexaSim()
        self.addCleanup(sim.close)
        walker = Walker(sim, (0.0, 0.0, math.pi / 2), (0.0, 1.0))
        self.assertAlmostEqual(walker.pose()[2], math.pi / 2, places=3)
        row = walker.observe()
        self.assertAlmostEqual(row["target_dist"], 1.0, places=2)
        self.assertGreater(row["target_cos"], 0.99)  # b is ahead
        out = walker.run_cycle(0.3, 0.0)
        self.assertGreater(out["y_next"], 0.05)  # walked toward +y, its heading
        self.assertLess(out["dist_next"], 0.97)
        self.assertEqual(len(walker.steps), 51)

    def test_recorder_round_trip_and_dictionary(self):
        from common import JetHexaSim, Walker
        from db import Recorder, query
        sim = JetHexaSim()
        self.addCleanup(sim.close)
        recorder = Recorder("test", policy="random")
        walker = Walker(sim, (0.0, 0.0, 0.0), (1.0, 0.0))
        rows = []
        for _ in range(3):
            row = walker.observe()
            row.update(source="random", hold=0, image="")
            row.update(walker.run_cycle(0.25, 0.0))
            rows.append(row)
        recorder.add_episode(0, rows, walker.steps, walker.summary())
        recorder.close()
        cycles = query("SELECT * FROM cycles ORDER BY cycle")
        self.assertEqual(len(cycles), 3)
        self.assertEqual(cycles["cycle"].tolist(), [0, 1, 2])
        self.assertFalse(cycles.drop(columns=["image"]).isna().any().any())
        described = set(query("SELECT name FROM columns")["name"])
        self.assertEqual(set(cycles.columns) - described, set())
        self.assertEqual(int(query("SELECT count(*) AS n FROM steps")["n"][0]), 151)
        episode = query("SELECT * FROM episodes").iloc[0]
        self.assertEqual(int(episode["cycles"]), 3)
        self.assertTrue((recorder.folder / "cycles.csv").exists())
        self.assertTrue((recorder.folder / "run.json").exists())

    def test_hindsight_uses_only_constant_command_stretches(self):
        import pandas as pd
        from hindsight import hindsight_rows
        base = dict(run_id="r", episode=0, sin_yaw=0.0, cos_yaw=1.0, partial=False, fell=False)
        rows = [  # walks along +x: two cycles with command A, then one with command B
            dict(base, cycle=0, x=0.0, y=0.0, x_next=0.1, y_next=0.0, stride=0.3, turn=0.0),
            dict(base, cycle=1, x=0.1, y=0.0, x_next=0.2, y_next=0.0, stride=0.3, turn=0.0),
            dict(base, cycle=2, x=0.2, y=0.0, x_next=0.25, y_next=0.05, stride=0.1, turn=0.2),
        ]
        table = hindsight_rows(pd.DataFrame(rows))
        # cycle 0: K=1 and K=2 (same command); cycle 1: K=1 only (cycle 2 changes command); cycle 2: K=1
        self.assertEqual(sorted(zip(table["cycle"], table["horizon"])), [(0, 1.0), (0, 2.0), (1, 1.0), (2, 1.0)])
        two = table[(table["cycle"] == 0) & (table["horizon"] == 2.0)].iloc[0]
        self.assertAlmostEqual(two["goal_dist"], 0.2)
        self.assertAlmostEqual(two["goal_cos"], 1.0)
        self.assertAlmostEqual(two["delta_x"], 0.2)
        self.assertAlmostEqual(two["delta_y"], 0.0)
        self.assertEqual((two["stride"], two["turn"]), (0.3, 0.0))
        # the deltas are the relabelled end point in the start pose's frame: cycle 2 ends 0.05 ahead, 0.05 left
        last = table[table["cycle"] == 2].iloc[0]
        self.assertAlmostEqual(last["delta_x"], 0.05)
        self.assertAlmostEqual(last["delta_y"], 0.05)

    def test_planner_picks_the_command_that_heads_for_b(self):
        import numpy as np
        import pandas as pd
        from dynamics import FEATURES, DynamicsPlanner, motion_rows
        rng = np.random.default_rng(0)
        rows = []
        for i in range(120):  # a made-up robot: forward = stride / 2, yaw change = 2 * turn
            stride, turn = rng.uniform(0, 0.3), rng.uniform(-0.25, 0.25)
            yaw = rng.uniform(-math.pi, math.pi)
            fwd, dyaw = stride / 2, 2 * turn
            rows.append(dict(run_id="r", episode=0, cycle=i, partial=False, fell=False, x=0.0, y=0.0,
                             sin_yaw=math.sin(yaw), cos_yaw=math.cos(yaw), x_next=fwd * math.cos(yaw),
                             y_next=fwd * math.sin(yaw), yaw_next=yaw + dyaw, stride=stride, turn=turn,
                             prev_stride=0.0, prev_turn=0.0, fwd_1=0.0, left_1=0.0, dyaw_1=0.0))
        table = motion_rows(pd.DataFrame(rows))
        self.assertAlmostEqual(float(table["m_dyaw"].iloc[0]), 2 * rows[0]["turn"], places=6)
        planner = DynamicsPlanner(table[FEATURES + ["m_fwd", "m_left", "m_dyaw"]], n_estimators=1)
        state = dict(prev_stride=0.0, prev_turn=0.0, fwd_1=0.0, left_1=0.0, dyaw_1=0.0)
        ahead, left = planner.act([dict(state, target_dist=1.0, target_sin=0.0, target_cos=1.0),
                                   dict(state, target_dist=1.0, target_sin=1.0, target_cos=0.0)])
        self.assertGreater(ahead[0], 0.2)  # b straight ahead: walk
        self.assertLess(abs(ahead[1]), 0.1)  # and do not turn
        self.assertGreater(left[1], 0.1)  # b to the left: turn left


class FitOnlyRuleTest(unittest.TestCase):
    """AGENTS.md: TabPFN is used only through fit() and predict(); its weights are never trained."""

    def test_no_fine_tuning_code_in_nav(self):
        banned = ("Finetuned", "tabpfn.finetuning", "save_tabpfn_model", "optimizer", ".backward(",
                  "load_state_dict", "requires_grad", "learning_rate")
        nav = Path(__file__).resolve().parents[1] / "nav"
        for path in sorted(nav.glob("*.py")):
            text = path.read_text()
            for word in banned:
                self.assertNotIn(word, text, "{} uses '{}': fine-tuning is not allowed (AGENTS.md)".format(
                    path.name, word))

if __name__ == "__main__":
    unittest.main()

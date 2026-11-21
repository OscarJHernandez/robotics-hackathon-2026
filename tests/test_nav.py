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
        self.assertEqual((two["stride"], two["turn"]), (0.3, 0.0))

    def test_success_rows_keep_the_more_direct_half(self):
        import pandas as pd
        from hindsight import success_rows
        episodes = pd.DataFrame([dict(run_id="r", episode=i, reached=True, directness=d)
                                 for i, d in enumerate([0.5, 0.7, 0.9, 0.95])])
        cycles = pd.DataFrame([dict(run_id="r", episode=i, cycle=0, partial=False, reached=False)
                               for i in range(4)])
        kept = success_rows(cycles, episodes)
        self.assertEqual(sorted(set(kept["episode"])), [1, 2, 3])  # the median one and above
        counts = kept["episode"].value_counts()
        self.assertEqual((counts[1], counts[3]), (1, 3))


if __name__ == "__main__":
    unittest.main()

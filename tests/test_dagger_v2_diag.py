"""Tests for the DAgger v2 diagnostics (episode table, failure types, paired tests, offline report) and the result
records. Hand-made episodes with exact expected values; no simulator, no TabPFN."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False


def make_episode(ep, n, goal_dist, bearing_deg, student, expert, outcome, path, final_dist, start_dist,
                 x=0.0, y=0.0, lidar_last=None, scene=None, seed=9):
    """Rows and the episode dict of one hand-made episode. goal_dist, bearing_deg, student, expert: a number or a
    list of n values. lidar_last: {ray index: range} for the last row (all other rays 2.0 m)."""
    def seq(v):
        return list(v) if isinstance(v, (list, tuple)) else [v] * n

    gd, br, st, ex = seq(goal_dist), seq(bearing_deg), seq(student), seq(expert)
    rows = []
    for i in range(n):
        row = {"episode": ep, "step": i, "scene_id": scene or ep, "seed": seed, "x": x, "y": y, "yaw": 0.0,
               "goal_dist": gd[i], "goal_sin": math.sin(math.radians(br[i])),
               "goal_cos": math.cos(math.radians(br[i])), "student_stride": st[i], "student_turn": 0.0,
               "expert_stride": ex[i], "expert_turn": 0.0, "collision": False, "done_reason": "",
               "free_ahead": 2.0}
        for k in range(36):
            row["lr_{}".format(k)] = 2.0
        rows.append(row)
    rows[-1]["done_reason"] = outcome
    rows[-1]["collision"] = outcome == "collision"
    for k, v in (lidar_last or {}).items():
        rows[-1]["lr_{}".format(k)] = v
    episode = {"episode": ep, "scene_id": scene or ep, "seed": seed, "steps": n, "done_reason": outcome,
               "path_length": path, "start_x": 0.0, "start_y": 0.0, "start_yaw": 0.0, "goal_x": 3.0, "goal_y": 0.0,
               "final_dist": final_dist, "start_dist": start_dist}
    return rows, episode


def build():
    """Six episodes: 1 goal, 2 walked past, 3 never turned, 4 collision side, 5 collision front, 6 spinning."""
    specs = [
        make_episode(1, 20, [3.0 - 0.14 * i for i in range(20)], 0, 0.2, [0.2, 0.2, 0.2, 0.0, 0.0] + [0.2] * 15,
                     "goal", 3.0, 0.2, 3.0),
        make_episode(2, 25, [3.0, 2.4, 1.8, 1.0, 0.4] + [1.0] * 5 + [1.5] * 15, 0, 0.2, 0.2, "timeout", 6.0, 1.5,
                     3.0),
        make_episode(3, 30, 2.5, 180, 0.2, 0.0, "timeout", 5.0, 3.0, 2.5),
        make_episode(4, 10, 3.0, 0, 0.2, 0.2, "collision", 2.0, 1.0, 3.0, x=3.2, lidar_last={9: 0.2}),
        make_episode(5, 10, 4.0, 0, 0.2, 0.2, "collision", 2.0, 1.0, 4.5, lidar_last={0: 0.2}),
        make_episode(6, 40, 3.0, [90] * 10 + [20] * 30, 0.0, 0.0, "timeout", 0.3, 3.0, 3.0),
    ]
    rows = pd.DataFrame([r for rs, _ in specs for r in rs])
    return rows, [e for _, e in specs]


@unittest.skipUnless(HAVE_PANDAS, "needs pandas: uv run --group ml")
class EpisodeTableTest(unittest.TestCase):
    def setUp(self):
        from nav.dagger.evaluate import episode_table
        self.rows, self.episodes = build()
        expert = [{"episode": 1, "scene_id": 1, "seed": 9, "done_reason": "goal", "path_length": 2.5},
                  {"episode": 2, "scene_id": 2, "seed": 9, "done_reason": "collision", "path_length": 1.0}]
        self.table = episode_table(self.rows, self.episodes, expert).set_index("episode")

    def test_failure_labels(self):
        t = self.table
        self.assertEqual(list(t["failure"]), ["goal", "walked_past", "never_turned", "collision_side",
                                              "collision_front", "spinning"])
        self.assertEqual(list(t["goal"]), [True, False, False, False, False, False])

    def test_progress_and_ratio(self):
        t = self.table
        for ep, want in {1: 2.8 / 3.0, 2: 0.5, 3: -0.2, 4: 2.0 / 3.0, 5: 3.5 / 4.5, 6: 0.0}.items():
            self.assertAlmostEqual(t.loc[ep, "progress"], want, places=9)
        self.assertAlmostEqual(t.loc[1, "path_ratio"], 3.0 / 2.5)
        self.assertTrue(np.isnan(t.loc[2, "path_ratio"]))  # the expert did not reach the goal there
        self.assertTrue(np.isnan(t.loc[3, "path_ratio"]))  # no expert episode

    def test_start_geometry(self):
        t = self.table
        self.assertEqual(list(t["start_side"]), ["ahead", "ahead", "behind", "ahead", "ahead", "side"])
        self.assertAlmostEqual(t.loc[3, "start_bearing_deg"], 180.0)
        self.assertAlmostEqual(t.loc[6, "start_bearing_deg"], 90.0)
        self.assertEqual(list(t["dist_bin"]), ["3-4", "3-4", "2-3", "3-4", ">4", "3-4"])

    def test_divergence(self):
        t = self.table
        self.assertEqual(t.loc[1, "first_divergence"], 3)  # expert stops at steps 3 and 4, student walks
        self.assertEqual((t.loc[1, "stop_ignored"], t.loc[1, "wrong_stop"]), (2, 0))
        self.assertEqual(t.loc[3, "first_divergence"], 0)
        self.assertEqual((t.loc[3, "stop_ignored"], t.loc[3, "wrong_stop"]), (30, 0))
        self.assertTrue(np.isnan(t.loc[2, "first_divergence"]))
        self.assertEqual(t["stop_ignored"].sum(), 32)

    def test_wrong_stop(self):
        from nav.dagger.evaluate import episode_table
        rows, ep = make_episode(1, 10, 3.0, 0, [0.2] * 4 + [0.0] * 6, 0.2, "timeout", 3.0, 3.0, 3.0)
        t = episode_table(pd.DataFrame(rows), [ep])
        self.assertEqual((t.loc[0, "first_divergence"], t.loc[0, "wrong_stop"], t.loc[0, "stop_ignored"]),
                         (4, 6, 0))

    def test_min_goal_dist_and_lidar(self):
        t = self.table
        self.assertAlmostEqual(t.loc[1, "min_goal_dist"], 3.0 - 0.14 * 19)
        self.assertAlmostEqual(t.loc[2, "min_goal_dist"], 0.4)
        self.assertEqual((t.loc[4, "last_lidar_min"], t.loc[4, "last_lidar_bearing_deg"]), (0.2, 90.0))
        self.assertEqual(t.loc[5, "last_lidar_bearing_deg"], 0.0)
        self.assertEqual(t.loc[1, "last_free_ahead"], 2.0)

    def test_near_wall(self):
        self.assertEqual(list(self.table["near_wall"]), [False, False, False, True, False, False])

    def test_lidar_bearing_wraps_to_negative(self):
        from nav.dagger.evaluate import episode_table
        rows, ep = make_episode(1, 5, 3.0, 0, 0.2, 0.2, "collision", 1.0, 1.0, 3.0, lidar_last={35: 0.1})
        t = episode_table(pd.DataFrame(rows), [ep])
        self.assertEqual(t.loc[0, "last_lidar_bearing_deg"], -10.0)
        self.assertEqual(t.loc[0, "failure"], "collision_front")

    def test_no_lidar_no_student_columns(self):
        from nav.dagger.evaluate import episode_table
        rows, ep = make_episode(1, 5, 3.0, 0, 0.2, 0.2, "collision", 1.0, 1.0, 3.0)
        df = pd.DataFrame(rows).drop(columns=["student_stride", "student_turn"] + ["lr_{}".format(i) for i in range(36)])
        ep.pop("start_dist")  # computed from start and goal: 3.0
        t = episode_table(df, [ep])
        self.assertEqual(t.loc[0, "failure"], "collision")
        self.assertTrue(np.isnan(t.loc[0, "last_lidar_min"]))
        self.assertTrue(np.isnan(t.loc[0, "first_divergence"]))
        self.assertAlmostEqual(t.loc[0, "start_dist"], 3.0)


@unittest.skipUnless(HAVE_PANDAS, "needs pandas: uv run --group ml")
class SummaryCompareTest(unittest.TestCase):
    def test_summarize(self):
        from nav.dagger.evaluate import episode_table, summarize
        rows, episodes = build()
        s = summarize(episode_table(rows, episodes))
        self.assertEqual(s["n"], 6)
        self.assertEqual((s["goal"]["count"], s["goal"]["rate"]), (1, round(1 / 6, 4)))
        self.assertAlmostEqual(s["goal"]["ci_low"], 0.0301, places=4)
        self.assertAlmostEqual(s["goal"]["ci_high"], 0.5635, places=4)
        self.assertEqual(s["collision"]["count"], 2)
        self.assertAlmostEqual(s["collision"]["ci_low"], 0.0968, places=4)
        self.assertAlmostEqual(s["collision"]["ci_high"], 0.7000, places=4)
        self.assertEqual(s["timeout"]["count"], 3)
        self.assertAlmostEqual(s["timeout"]["ci_low"], 0.1876, places=4)
        self.assertAlmostEqual(s["timeout"]["ci_high"], 0.8124, places=4)
        self.assertEqual((s["stuck"]["count"], s["fell"]["count"]), (0, 0))
        self.assertAlmostEqual(s["mean_progress"], 0.4463, places=4)
        self.assertAlmostEqual(s["median_progress"], 0.5833, places=4)
        self.assertEqual(s["mean_steps_goal"], 20.0)
        self.assertEqual(s["failures"], {"goal": 1, "walked_past": 1, "never_turned": 1, "collision_side": 1,
                                         "collision_front": 1, "spinning": 1})
        self.assertEqual(s["by_start_side"], {"ahead": {"n": 4, "goal_rate": 0.25},
                                              "side": {"n": 1, "goal_rate": 0.0},
                                              "behind": {"n": 1, "goal_rate": 0.0}})
        self.assertEqual(s["by_dist_bin"]["<2"], {"n": 0, "goal_rate": None})
        self.assertEqual(s["by_dist_bin"]["3-4"], {"n": 4, "goal_rate": 0.25})
        self.assertEqual((s["stop_ignored"], s["wrong_stop"]), (32, 0))
        self.assertEqual(s["collision_side_share"], 0.5)
        json.dumps(s)

    def test_compare(self):
        from nav.dagger.evaluate import compare
        a = pd.DataFrame({"scene_id": [1, 2, 3, 4, 5, 99], "seed": 9,
                          "goal": [True, True, False, True, True, True],
                          "outcome": ["goal", "goal", "timeout", "goal", "goal", "goal"],
                          "progress": [0.5, 0.25, 0.0, 0.375, 0.0625, 1.0]})
        b = pd.DataFrame({"scene_id": [1, 2, 3, 4, 5, 77], "seed": 9,
                          "goal": [False, False, True, False, True, True],
                          "outcome": ["collision", "collision", "goal", "timeout", "goal", "goal"],
                          "progress": [0.0, 0.0, 0.125, 0.0, 0.0, 1.0]})
        c = compare(a, b)
        self.assertEqual((c["pairs"], c["wins"], c["losses"], c["ties"]), (5, 3, 1, 1))
        self.assertAlmostEqual(c["sign_p"], 0.625)  # 2 * P(X <= 1), n = 4
        # differences .5, .25, -.125, .375, .0625: W- = 2 -> exact two-sided p = 2 * 3/32
        self.assertAlmostEqual(c["wilcoxon_p"], 0.1875)
        self.assertAlmostEqual(c["mean_progress_diff"], (0.5 + 0.25 - 0.125 + 0.375 + 0.0625) / 5)
        self.assertEqual((c["collisions_a"], c["collisions_b"]), (0, 2))

    def test_compare_identical_gives_p_one(self):
        from nav.dagger.evaluate import compare
        a = pd.DataFrame({"scene_id": [1, 2], "seed": 9, "goal": [True, False], "outcome": ["goal", "timeout"],
                          "progress": [1.0, 0.2]})
        c = compare(a, a.copy())
        self.assertEqual((c["wins"], c["losses"], c["ties"]), (0, 0, 2))
        self.assertEqual((c["sign_p"], c["wilcoxon_p"], c["mean_progress_diff"]), (1.0, 1.0, 0.0))

    def test_stats_helpers(self):
        from nav.dagger.stats import spearman, wilcoxon_p
        self.assertEqual(wilcoxon_p([0, 0.0, 0]), 1.0)
        self.assertEqual(wilcoxon_p([]), 1.0)
        self.assertAlmostEqual(wilcoxon_p([1, 2, 3, 4, 5]), 0.0625)
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))

    def test_rank_agreement(self):
        from nav.dagger.evaluate import rank_agreement
        out = rank_agreement({"a": 1.0, "b": 3.0, "c": 2.0, "x": 9.0}, {"a": 0.1, "b": 0.5, "c": 0.3, "y": 0.9})
        self.assertAlmostEqual(out["spearman"], 1.0)
        self.assertEqual((out["n"], out["top2_offline"], out["top2_closed"]), (3, ["b", "c"], ["b", "c"]))
        out = rank_agreement({"a": 1.0, "b": 3.0, "c": 2.0}, {"a": 0.9, "b": 0.1, "c": 0.5})
        self.assertAlmostEqual(out["spearman"], -1.0)
        self.assertEqual((out["top2_offline"], out["top2_closed"]), (["b", "c"], ["a", "c"]))


@unittest.skipUnless(HAVE_PANDAS, "needs pandas: uv run --group ml")
class OfflineReportTest(unittest.TestCase):
    def test_report(self):
        from nav.dagger.evaluate import offline_report
        df = pd.DataFrame({
            "y_stride": [0.3, 0.3, 0.0, 0.0, 0.01, 0.2],
            "y_turn": [0.0, 0.1, 0.25, -0.25, 0.2, 0.0],
            "free_ahead": [2.0, 0.8, 0.3, 0.4, 1.5, 0.5],
            "pred_stride": [0.25, 0.2, 0.0, 0.1, 0.0, 0.0],
            "pred_turn": [0.05, 0.0, 0.125, -0.25, -0.1, 0.0]})
        r = offline_report(df)
        self.assertEqual(r["n"], 6)
        self.assertAlmostEqual(r["all"]["mae_stride"], 0.46 / 6)
        self.assertAlmostEqual(r["all"]["mae_turn"], 0.575 / 6)
        b = r["by_free_ahead"]
        self.assertEqual((b["<0.5"]["n"], b["0.5-1"]["n"], b[">1"]["n"]), (2, 2, 2))
        self.assertAlmostEqual(b["<0.5"]["mae_stride"], 0.05)
        self.assertAlmostEqual(b["<0.5"]["mae_turn"], 0.0625)
        self.assertAlmostEqual(b["0.5-1"]["mae_stride"], 0.15)
        self.assertAlmostEqual(b["0.5-1"]["mae_turn"], 0.05)
        self.assertAlmostEqual(b[">1"]["mae_stride"], 0.03)
        self.assertAlmostEqual(b[">1"]["mae_turn"], 0.175)
        self.assertEqual((r["by_mode"]["walk"]["n"], r["by_mode"]["stop"]["n"]), (3, 3))
        self.assertAlmostEqual(r["by_mode"]["walk"]["mae_stride"], 0.35 / 3)
        self.assertAlmostEqual(r["by_mode"]["walk"]["mae_turn"], 0.05)
        self.assertAlmostEqual(r["by_mode"]["stop"]["mae_stride"], 0.11 / 3)
        self.assertAlmostEqual(r["by_mode"]["stop"]["mae_turn"], 0.425 / 3)
        s = r["stop"]
        self.assertEqual((s["expert_stops"], s["predicted_stops"], s["hits"]), (3, 3, 2))
        self.assertAlmostEqual(s["recall"], 2 / 3)
        self.assertAlmostEqual(s["precision"], 2 / 3)
        h = r["hard_turns"]
        self.assertEqual(h["n"], 3)
        self.assertAlmostEqual(h["mean_abs_pred_turn"], 0.475 / 3)
        self.assertAlmostEqual(h["mean_abs_expert_turn"], 0.7 / 3)
        self.assertAlmostEqual(h["ratio"], 0.475 / 0.7)
        self.assertAlmostEqual(h["same_sign_share"], 2 / 3)
        json.dumps(r)

    def test_custom_columns_and_no_hard_turns(self):
        from nav.dagger.evaluate import offline_report
        df = pd.DataFrame({"y_stride": [0.3], "y_turn": [0.0], "free_ahead": [2.0], "s": [0.3], "t": [0.0]})
        r = offline_report(df, pred_stride="s", pred_turn="t")
        self.assertEqual(r["hard_turns"]["n"], 0)
        self.assertIsNone(r["hard_turns"]["ratio"])
        self.assertIsNone(r["stop"]["recall"])
        self.assertEqual(r["all"]["mae_stride"], 0.0)


@unittest.skipUnless(HAVE_PANDAS, "needs pandas: uv run --group ml")
class JournalTest(unittest.TestCase):
    def test_write_read_update(self):
        from nav.dagger import journal
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "experiments" / "e1"
            r = journal.write_result(folder, id="e1", hypothesis="h", status="running")
            self.assertEqual(set(r), set(journal.RESULT_FIELDS))
            self.assertEqual(journal.read_result(folder), r)
            self.assertEqual(r["verdict"], "")
            journal.update_result(folder, status="done", verdict="promoted", metrics={"closed": {"goal_rate": 0.5}})
            got = journal.read_result(folder)
            self.assertEqual((got["id"], got["status"], got["verdict"]), ("e1", "done", "promoted"))
            self.assertEqual(got["metrics"]["closed"]["goal_rate"], 0.5)
            self.assertEqual([p.name for p in folder.iterdir()], ["result.json"])  # no temp file left behind

    def test_validation(self):
        from nav.dagger import journal
        with tempfile.TemporaryDirectory() as d:
            for bad in ({"colour": "red"}, {"status": "finished"}, {"verdict": "great"}, {"metrics": []}):
                with self.assertRaises(ValueError):
                    journal.write_result(d, id="x", **bad)
            self.assertFalse((Path(d) / "result.json").exists())
            journal.write_result(d, id="x")
            with self.assertRaises(ValueError):
                journal.update_result(d, status="nope")
            self.assertEqual(journal.read_result(d)["status"], "planned")  # failed update left it unchanged

    def test_atomic_write_failure_keeps_old_file(self):
        from nav.dagger import journal
        with tempfile.TemporaryDirectory() as d:
            journal.write_result(d, id="x", notes="old")
            with self.assertRaises(TypeError):
                journal.write_result(d, id="x", notes=object())  # not JSON-serialisable
            self.assertEqual(journal.read_result(d)["notes"], "old")
            self.assertEqual([p.name for p in Path(d).iterdir()], ["result.json"])

    def test_failures_and_incident(self):
        from nav.dagger import journal
        from nav.dagger.evaluate import episode_table
        rows, episodes = build()
        table = episode_table(rows, episodes)
        with tempfile.TemporaryDirectory() as d:
            items = journal.write_failures(d, table)
            self.assertEqual([i["episode"] for i in items], [2, 3, 4, 5, 6])
            on_disk = json.loads((Path(d) / "failures.json").read_text())
            self.assertEqual(on_disk[0], {"episode": 2, "scene_id": 2, "seed": 9, "failure": "walked_past",
                                          "steps": 25, "final_dist": 1.5})
            journal.append_incident(d, "worker died", "out of memory", "lower batch", commit="abc123")
            journal.append_incident(d, "second", "c", "f")
            text = (Path(d) / "incidents.md").read_text()
            self.assertEqual(text.count("## 20"), 2)
            self.assertIn("- What: worker died", text)
            self.assertIn("- Commit: abc123", text)

    def test_leaderboard_order(self):
        from nav.dagger import journal
        with tempfile.TemporaryDirectory() as d:
            for name, metrics in [("a", {"closed": {"goal_rate": 0.4}}), ("b", {"closed": {"goal_rate": 0.7}}),
                                  ("c", {}), ("d", {"closed": {"goal_rate": 0.55}}), ("e", {"closed": {}})]:
                journal.write_result(Path(d) / "experiments" / name, id=name, metrics=metrics)
            board = journal.leaderboard(d)
            self.assertEqual([r["id"] for r in board], ["b", "d", "a", "c", "e"])
            self.assertEqual([r["goal_rate"] for r in board], [0.7, 0.55, 0.4, None, None])
            self.assertEqual([r["id"] for r in json.loads((Path(d) / "leaderboard.json").read_text())],
                             ["b", "d", "a", "c", "e"])
            self.assertEqual(journal.leaderboard(Path(d) / "empty"), [])


if __name__ == "__main__":
    unittest.main()


class StudentOnlyCheckTest(unittest.TestCase):
    def test_only_runs_driven_entirely_by_the_student_pass(self):
        import pandas as pd

        from nav.dagger.report import student_only
        rows = pd.DataFrame({"used_expert": [False, False], "student_stride": [0.2, 0.35], "student_turn": [0.1, -0.3],
                             "executed_stride": [0.2, 0.3], "executed_turn": [0.1, -0.25]})
        self.assertTrue(student_only(rows))  # executed = the student's command (clipped to the gait's range)
        mixed = rows.copy()
        mixed.loc[1, "used_expert"] = True
        self.assertFalse(student_only(mixed))  # one step driven by the expert
        other = rows.copy()
        other.loc[0, "executed_turn"] = 0.0
        self.assertFalse(student_only(other))  # executed something else than the student's command

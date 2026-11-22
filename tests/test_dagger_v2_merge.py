"""DAgger v2 strict merge: parts, side files, episode summaries, plan."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nav.dagger.merge import MergeError, merge_dir, missing_episodes  # noqa: E402


def make_folder(root, parts, side=True, lengths=None):
    """parts: list of lists of episode ids; every episode has `lengths[e]` steps (default 3)."""
    root = Path(root)
    (root / "parts").mkdir(parents=True)
    (root / "episodes").mkdir()
    lengths = lengths or {}
    for i, episodes in enumerate(parts):
        rows = [(e, s) for e in episodes for s in range(lengths.get(e, 3))]
        pd.DataFrame({"episode": [r[0] for r in rows], "step": [r[1] for r in rows],
                      "x": [float(r[0] * 100 + r[1]) for r in rows]}).to_parquet(root / "parts" / "p{}.parquet".format(i))
        if side is True or (isinstance(side, (list, set)) and i in side):
            arr = np.zeros((len(rows), 2, 4), dtype=np.float16)
            for k, (e, s) in enumerate(rows):
                arr[k] = e * 100 + s
            np.save(root / "parts" / "p{}.hz.npy".format(i), arr)
        with open(root / "episodes" / "p{}.jsonl".format(i), "w") as f:
            for e in episodes:
                f.write(json.dumps({"episode": e, "done_reason": "goal" if e % 2 else "timeout"}) + "\n")
    return root


class MergeDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "round"

    def tearDown(self):
        self._tmp.cleanup()

    def test_clean_merge_sorted_and_side_aligned(self):
        make_folder(self.root, [[5, 1], [3, 0], [4, 2]])
        result = merge_dir(self.root, plan=[{"episode": e} for e in range(6)], log=lambda *_: None)
        rows = pd.read_parquet(self.root / "rows.parquet")
        self.assertEqual(len(rows), 18)
        self.assertEqual(list(zip(rows["episode"], rows["step"])), sorted(zip(rows["episode"], rows["step"])))
        side = np.load(self.root / "rows.hz.npy")
        self.assertEqual(side.shape, (18, 2, 4))
        self.assertEqual(side.dtype, np.float16)
        expected = (rows["episode"] * 100 + rows["step"]).to_numpy()
        self.assertTrue(np.array_equal(side[:, 0, 0].astype(np.int64), expected))
        self.assertTrue(np.array_equal(rows["x"].to_numpy(), expected.astype(float)))
        self.assertEqual((result["rows"], result["episodes"], result["parts"], result["has_side"]), (18, 6, 3, True))
        self.assertEqual(result["side_shape"], [18, 2, 4])
        self.assertEqual(result["outcomes"], {"goal": 3, "timeout": 3})
        self.assertEqual(len(result["rows_sha256"]), 64)
        self.assertEqual(json.loads((self.root / "merge.json").read_text()), result)
        self.assertEqual([e["episode"] for e in json.loads((self.root / "episodes.json").read_text())], list(range(6)))

    def test_chunked_copy_crosses_chunks(self):
        import nav.dagger.merge as merge
        old, merge.CHUNK_ROWS = merge.CHUNK_ROWS, 4
        try:
            make_folder(self.root, [[5, 1], [3, 0], [4, 2]])
            merge_dir(self.root, log=lambda *_: None)
        finally:
            merge.CHUNK_ROWS = old
        rows = pd.read_parquet(self.root / "rows.parquet")
        side = np.load(self.root / "rows.hz.npy")
        self.assertTrue(np.array_equal(side[:, 1, 3].astype(np.int64), (rows["episode"] * 100 + rows["step"]).to_numpy()))

    def test_no_side_files(self):
        make_folder(self.root, [[0, 1], [2]], side=False)
        result = merge_dir(self.root, log=lambda *_: None)
        self.assertFalse(result["has_side"])
        self.assertFalse((self.root / "rows.hz.npy").exists())

    def test_duplicate_row_fails(self):
        make_folder(self.root, [[0, 1], [1, 2]])
        with self.assertRaises(MergeError) as ctx:
            merge_dir(self.root, log=lambda *_: None)
        self.assertIn("duplicated", str(ctx.exception))

    def test_short_side_file_fails(self):
        make_folder(self.root, [[0, 1], [2]])
        path = self.root / "parts" / "p1.hz.npy"
        np.save(path, np.load(path)[:-1])
        with self.assertRaises(MergeError) as ctx:
            merge_dir(self.root, log=lambda *_: None)
        self.assertIn("p1", str(ctx.exception))

    def test_missing_planned_episode(self):
        make_folder(self.root, [[0, 1], [2]])
        plan = [{"episode": e} for e in range(4)]
        self.assertEqual(missing_episodes(self.root, plan), [{"episode": 3}])
        with self.assertRaises(MergeError) as ctx:
            merge_dir(self.root, plan=plan, log=lambda *_: None)
        self.assertIn("missing [3]", str(ctx.exception))

    def test_unexpected_episode_fails(self):
        make_folder(self.root, [[0, 1], [2]])
        with self.assertRaises(MergeError) as ctx:
            merge_dir(self.root, plan=[{"episode": 0}, {"episode": 1}], log=lambda *_: None)
        self.assertIn("unexpected [2]", str(ctx.exception))

    def test_gap_in_steps_fails(self):
        make_folder(self.root, [[0, 1]])
        path = self.root / "parts" / "p0.parquet"
        table = pd.read_parquet(path)
        keep = ~((table["episode"] == 1) & (table["step"] == 1))
        table[keep].to_parquet(path)
        np.save(self.root / "parts" / "p0.hz.npy", np.load(self.root / "parts" / "p0.hz.npy")[keep.to_numpy()])
        with self.assertRaises(MergeError) as ctx:
            merge_dir(self.root, log=lambda *_: None)
        self.assertIn("gaps", str(ctx.exception))

    def test_mixed_side_files_fail(self):
        make_folder(self.root, [[0], [1], [2]], side={0, 2})
        with self.assertRaises(MergeError) as ctx:
            merge_dir(self.root, log=lambda *_: None)
        self.assertIn("side files", str(ctx.exception))

    def test_rows_and_summaries_must_match(self):
        make_folder(self.root, [[0, 1]])
        (self.root / "episodes" / "extra.jsonl").write_text(json.dumps({"episode": 9, "done_reason": "goal"}) + "\n")
        with self.assertRaises(MergeError):
            merge_dir(self.root, log=lambda *_: None)

    def test_duplicate_summary_fails(self):
        make_folder(self.root, [[0, 1]])
        (self.root / "episodes" / "again.jsonl").write_text(json.dumps({"episode": 0, "done_reason": "goal"}) + "\n")
        with self.assertRaises(MergeError) as ctx:
            merge_dir(self.root, log=lambda *_: None)
        self.assertIn("duplicated episode", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

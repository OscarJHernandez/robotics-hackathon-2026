"""Checks the patch labels, the blocked-region margin and the probe. Does not download DINOv2."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision"))

from features import COLS, DIM, ROWS, patch_counts, patch_labels  # noqa: E402
from scene import FLOOR, IGNORE, OBSTACLE, TARGET, WALL  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def labels_for(counts_per_class):
    """Label of one patch with the given pixel counts (floor, wall, obstacle, target, ignore)."""
    counts = np.array(counts_per_class, dtype=np.uint8).reshape(1, 1, 5)
    return int(patch_labels(counts)[0, 0])


class PatchLabelTest(unittest.TestCase):
    def test_counts_cover_every_pixel_of_every_patch(self):
        classes = np.full((360, 640), FLOOR, dtype=np.uint8)
        classes[:175] = WALL
        classes[100:200, 300:340] = OBSTACLE
        counts = patch_counts(classes)
        self.assertEqual(counts.shape, (ROWS, COLS, 5))
        self.assertTrue((counts.sum(-1) == 196).all())
        labels = patch_labels(counts)
        self.assertEqual(int(labels[0, 0]), WALL)
        self.assertEqual(int(labels[-1, 0]), FLOOR)
        self.assertEqual(int(labels[10, 22]), OBSTACLE)
        self.assertEqual(int(labels[12, 0]), IGNORE)  # the row where wall meets floor is mixed

    def test_majority_needs_70_percent(self):
        self.assertEqual(labels_for([196, 0, 0, 0, 0]), FLOOR)
        self.assertEqual(labels_for([56, 0, 140, 0, 0]), OBSTACLE)  # 71 %
        self.assertEqual(labels_for([76, 0, 120, 0, 0]), IGNORE)  # 61 %: the couch-edge example

    def test_target_needs_only_30_percent(self):
        self.assertEqual(labels_for([130, 0, 0, 66, 0]), TARGET)  # 34 %
        self.assertEqual(labels_for([150, 0, 0, 46, 0]), FLOOR)  # 23 % target, 77 % floor
        self.assertEqual(labels_for([100, 50, 0, 46, 0]), IGNORE)

    def test_robot_legs_are_skipped(self):
        self.assertEqual(labels_for([40, 0, 0, 0, 156]), IGNORE)


@unittest.skipUnless(HAVE_TORCH, "needs torch: uv run --group vision ...")
class ProbeTest(unittest.TestCase):
    def test_margin_grows_the_blocked_region_by_one_patch(self):
        from probe import blocked_from_probs, grow
        mask = np.zeros((5, 7), dtype=bool)
        mask[2, 3] = True
        grown = grow(mask, 1)
        self.assertEqual(int(grown.sum()), 9)
        self.assertTrue(grown[1:4, 2:5].all())
        probs = np.zeros((5, 7, 4), dtype=np.float32)
        probs[..., FLOOR] = 1.0
        probs[2, 3] = (0.7, 0.1, 0.2, 0.0)  # floor is the top class, but wall + obstacle is over the bar
        self.assertEqual(int(blocked_from_probs(probs).sum()), 9)
        self.assertEqual(int(blocked_from_probs(probs, margin=0).sum()), 1)

    def test_probe_learns_separable_classes_and_reloads(self):
        import tempfile

        from probe import evaluate, load_probe, predict_probs, save_probe, split_episodes, train_probe
        rng = np.random.default_rng(0)
        centres = rng.normal(size=(4, DIM)).astype(np.float32)
        y = rng.integers(0, 4, size=4000)
        x = centres[y] + 0.5 * rng.normal(size=(4000, DIM)).astype(np.float32)
        for hidden in (0, 16):
            probe = train_probe(x, y, hidden=hidden, epochs=15, batch=256)
            self.assertGreater((predict_probs(probe, x).argmax(-1) == y).mean(), 0.95)
        with tempfile.TemporaryDirectory() as folder:
            save_probe(probe, Path(folder) / "probe.pt")
            again = load_probe(Path(folder) / "probe.pt")
        np.testing.assert_allclose(predict_probs(again, x[:50]), predict_probs(probe, x[:50]), atol=1e-6)

        train, test = split_episodes(np.repeat(np.arange(10), 4), 0.2, 0)
        self.assertEqual((len(train), len(test)), (8, 2))
        self.assertFalse(set(train) & set(test))

        probs = np.zeros((1, ROWS, COLS, 4), dtype=np.float32)
        probs[..., FLOOR] = 1.0
        labels = np.full((1, ROWS, COLS), FLOOR, dtype=np.uint8)
        labels[0, 5, 5] = OBSTACLE  # an obstacle the probe calls floor
        counts = np.zeros((1, ROWS, COLS, 5), dtype=np.uint8)
        counts[..., FLOOR] = 196
        metrics = evaluate(probs, labels, counts)
        self.assertEqual(metrics["blocked_vs_free"]["obstacle_left_free"], 1.0)
        self.assertEqual(metrics["blocked_vs_free"]["floor_blocked"], 0.0)


if __name__ == "__main__":
    unittest.main()

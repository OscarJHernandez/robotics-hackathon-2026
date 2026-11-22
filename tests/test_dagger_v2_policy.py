"""DAgger v2 student pieces, with fake models (no TabPFN download, no GPU)."""
import unittest

import numpy as np
import pandas as pd

from nav.dagger import policy as P

COLS = ["a", "b"]


class FakeClassifier:
    def __init__(self, classes, proba):
        self.classes_ = np.array(classes)
        self._p = np.array(proba, float)

    def predict_proba(self, X):
        return np.tile(self._p, (len(X), 1))


class FakeRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def rows(**kw):
    n = len(next(iter(kw.values())))
    return pd.DataFrame({"a": np.zeros(n), "b": np.zeros(n), **kw})


class Base:
    """Policy returning fixed commands."""
    columns = COLS

    def __init__(self, cmd):
        self.cmd = cmd

    def act_batch(self, r):
        return np.tile(self.cmd, (len(r), 1)).astype(float)


class ModeLabels(unittest.TestCase):
    def test_labels(self):
        got = P.mode_labels([0.2, 0.0, 0.01, 0.02, 0.0], [0.1, 0.2, -0.2, -0.2, 0.0])
        self.assertEqual(got.tolist(), [0, 1, 2, 0, 1])


class Gated(unittest.TestCase):
    def make(self, classes, proba):
        g = P.GatedPolicy(COLS, [])
        g.classifier = FakeClassifier(classes, proba)
        g.models = {"y_stride": FakeRegressor(0.2), "y_turn": FakeRegressor(0.05)}
        return g

    def test_missing_class(self):
        g = self.make([0, 1], [0.2, 0.8])  # class 2 absent
        out = g.act_batch(rows(a=[1.0, 2.0]))
        self.assertEqual(g.last_proba.shape, (2, 3))
        self.assertTrue((g.last_proba[:, 2] == 0).all())
        np.testing.assert_allclose(out, [[0, 0.25], [0, 0.25]])

    def test_walk_and_right(self):
        out = self.make([0, 1, 2], [0.7, 0.2, 0.1]).act_batch(rows(a=[1.0]))
        np.testing.assert_allclose(out, [[0.2, 0.05]])
        out = self.make([0, 2], [0.1, 0.9]).act_batch(rows(a=[1.0]))
        np.testing.assert_allclose(out, [[0, -0.25]])


class ShieldTests(unittest.TestCase):
    def test_close_turns_to_freer_side(self):
        s = P.Shield(Base((0.2, 0.0)), 0.35)
        r = rows(free_ahead=[0.2, 0.2, 2.0], free_left=[3.0, 1.0, 1.0], free_right=[1.0, 3.0, 1.0])
        out = s.act_batch(r)
        np.testing.assert_allclose(out, [[0, 0.25], [0, -0.25], [0.2, 0.0]])
        self.assertEqual(s.shielded, 2)

    def test_strong_turn_kept_and_free_s_columns(self):
        r = rows(free_ahead=[0.1], **{f"free_s{i}": [1.0 if i < 3 else 0.5] for i in range(7)})
        np.testing.assert_allclose(P.Shield(Base((0.2, -0.2))).act_batch(r), [[0, -0.2]])
        s = P.Shield(Base((0.2, 0.0)))
        np.testing.assert_allclose(s.act_batch(r), [[0, 0.25]])
        self.assertEqual(s.name, "student + shield")


def table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    stride = np.where(rng.random(n) < 0.3, 0.0, 0.25)
    return pd.DataFrame({"a": rng.random(n), "b": rng.random(n), "y_stride": stride,
                         "y_turn": rng.uniform(-0.25, 0.25, n), "free_ahead": rng.uniform(0, 2, n),
                         "used_expert": rng.random(n) < 0.8})


class Context(unittest.TestCase):
    def test_keeps_student_and_stop_rows(self):
        t = table()
        out = P.choose_context(t, max_rows=150, seed=1, quotas="mode_x_free")
        self.assertEqual(len(out), 150)
        self.assertEqual((~out["used_expert"]).sum(), (~t["used_expert"]).sum())
        # every non-empty group is represented
        self.assertEqual(set(P.context_summary(out)) - {"student_rows"}, set(P.context_summary(t)) - {"student_rows"})

    def test_all_rows_when_small(self):
        t = table(20)
        self.assertEqual(len(P.choose_context(t, max_rows=50, quotas="mode_x_free")), 20)

    def test_summary_and_default(self):
        t = table()
        s = P.context_summary(t)
        self.assertEqual(sum(v for k, v in s.items() if k != "student_rows"), len(t))
        self.assertEqual(s["student_rows"], int((~t["used_expert"]).sum()))
        self.assertEqual(len(P.choose_context(t, max_rows=60)), 60)  # turn_stride still works
        with self.assertRaises(ValueError):
            P.choose_context(t, max_rows=60, quotas="nope")


class Knn(unittest.TestCase):
    def test_fit_predict(self):
        t = table(100)
        t["y_stride"] = t["a"] * 0.3
        t["y_turn"] = (t["b"] - 0.5) * 0.5
        k = P.KNNPolicy(COLS, k=3)
        k.fit(t)
        out = k.act_batch(t[COLS].iloc[:5])
        self.assertEqual(out.shape, (5, 2))
        self.assertTrue((out[:, 0] >= 0).all() and (np.abs(out[:, 1]) <= 0.25).all())
        np.testing.assert_allclose(out[:, 0], t["y_stride"].iloc[:5], atol=0.06)

    def test_summary_columns(self):
        self.assertEqual(P.summary_columns(["x", "pc_0", "pd_1", "free_ahead"]), ["x", "free_ahead"])


class Factory(unittest.TestCase):
    cfg = {"device": "cpu", "n_estimators": 1, "model_version": "v3.5"}

    def test_kinds(self):
        self.assertIsInstance(P.make_student({**self.cfg, "kind": "tabpfn"}, COLS, [], 0), P.TabPFNPolicy)
        self.assertIsInstance(P.make_student({**self.cfg, "kind": "gated"}, COLS, [], 0), P.GatedPolicy)
        self.assertIsInstance(P.make_student({"kind": "knn"}, COLS, [], 0), P.KNNPolicy)
        self.assertIsInstance(P.make_student(self.cfg, COLS, [], 0), P.TabPFNPolicy)
        s = P.make_student({**self.cfg, "kind": "gated", "shield": {"free_ahead_min": 0.5}}, COLS, [], 0)
        self.assertIsInstance(s, P.Shield)
        self.assertEqual(s.free_ahead_min, 0.5)
        self.assertIsInstance(s.policy, P.GatedPolicy)
        with self.assertRaises(ValueError):
            P.make_student({"kind": "x"}, COLS, [], 0)


if __name__ == "__main__":
    unittest.main()

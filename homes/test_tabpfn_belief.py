"""Does TabPFN's belief make sense, and does the Bayesian search on top of it?

    uv run --group ml python -m unittest homes/test_tabpfn_belief.py -v      # about a minute: one fit()

These check behaviour, not exact numbers: what anyone would expect of "where is she?" given the inputs. They need
TabPFN (the ml group) and skip without it. The examples come from data/tabpfn_context.npz (made if missing).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

HOMES = Path(__file__).resolve().parent
sys.path.insert(0, str(HOMES))

import flats as F  # noqa: E402
import search as S  # noqa: E402

try:
    import tabpfn  # noqa: F401
    import tabpfn_belief as T
    HAVE_TABPFN = True
except ImportError:
    HAVE_TABPFN = False

HABITS = {"usual": {"0": ["kitchen", "living_room", "living_room"], "1": ["bedroom", "living_room", "living_room"]},
          "outings": [(3, 1)]}  # Thursday afternoons out
WED_MORNING, WED_AFTERNOON, THU_AFTERNOON = (10.0, 2), (15.0, 2), (15.0, 3)


@unittest.skipUnless(HAVE_TABPFN, "needs TabPFN: uv run --group ml ...")
class TabPFNBeliefTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = T.TabPFNBelief()

    def belief(self, when, journal=(), last=None, habits=HABITS):
        return self.model.belief(when[0], when[1], habits, list(journal), last)

    def test_a_probability_for_every_class_that_sums_to_one(self):
        b = self.belief(WED_AFTERNOON)
        self.assertEqual(list(b), list(F.CLASSES))
        self.assertAlmostEqual(sum(b.values()), 1.0, places=5)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in b.values()))
        self.assertIsNone(self.belief((3.0, 2)))  # night: no examples, the caller uses the stated default

    def test_heard_a_moment_ago_she_is_most_likely_still_there(self):
        for zone in F.ZONE_NAMES:
            b = self.belief(WED_AFTERNOON, last=(zone, 10.0))
            self.assertEqual(max(b, key=b.get), zone, "heard in the {} 10 min ago: {}".format(zone, b))
            self.assertGreater(b[zone], 0.5)

    def test_a_record_counts_less_the_older_it_is(self):
        fresh = self.belief(WED_AFTERNOON, last=("bedroom", 10.0))["bedroom"]
        old = self.belief(WED_AFTERNOON, last=("bedroom", 300.0))["bedroom"]  # this morning: another part of the day
        self.assertGreater(fresh, old + 0.3)

    def test_her_stated_habit_leads_when_nothing_recent_is_known(self):
        b = self.belief(WED_MORNING, last=("bedroom", 600.0))  # last heard yesterday evening
        home = {z: b[z] for z in F.ZONE_NAMES}
        self.assertEqual(max(home, key=home.get), "kitchen", b)  # weekday mornings: kitchen

    def test_a_stated_outing_makes_out_likely(self):
        out = self.belief(THU_AFTERNOON, last=("kitchen", 300.0))["out"]
        normal = self.belief(WED_AFTERNOON, last=("kitchen", 300.0))["out"]
        self.assertGreater(out, 0.5)
        self.assertGreater(out, normal + 0.3)

    def test_her_journal_moves_the_belief(self):
        # Two weeks of Wednesday ... Friday afternoons in the dining area, against a form that says living room.
        journal = [(wd, 1, "dining_area", day) for day, wd in enumerate([0, 1, 2, 3, 4] * 2)]
        without = self.belief(WED_AFTERNOON, last=("kitchen", 300.0))
        with_journal = self.belief(WED_AFTERNOON, journal, last=("kitchen", 300.0))
        self.assertGreater(with_journal["dining_area"], without["dining_area"] + 0.1)
        self.assertLess(with_journal["living_room"], without["living_room"])

    def test_better_than_the_population_alone_on_people_it_never_saw(self):
        from generate import build

        out = build(residents=150, days=16, seed=7)  # other residents and flats than the examples
        X, y = T.frame_rows(out, "test")
        self.assertGreater(len(X), 300)
        P = self.model.predict(X)
        top = np.mean(np.array(F.CLASSES)[P.argmax(axis=1)] == y)
        population = S.population_shares(out["diary"])
        base = np.mean([F.CLASSES[int(np.argmax(population[int(r[2]), int(r[3])]))] == z for r, z in zip(X, y)])
        self.assertGreater(top, base + 0.15, "TabPFN {:.2f}, population {:.2f}".format(top, base))
        self.assertGreater(top, 0.6)


@unittest.skipUnless(HAVE_TABPFN, "needs TabPFN: uv run --group ml ...")
class SearchOnTabPFNTest(unittest.TestCase):
    """The Bayesian search, started from TabPFN's belief."""

    @classmethod
    def setUpClass(cls):
        cls.model = T.TabPFNBelief()
        cls.flat = F.random_flat(np.random.default_rng([0, 0]), 0)

    def test_the_search_goes_most_likely_first(self):
        b = self.model.belief(15.0, 2, HABITS, [], ("dining_area", 15.0))
        order, _ = S.plan(self.flat, b)
        self.assertEqual(order[0], "dining_area")
        p = [b[z] for z in order]
        self.assertTrue(all(p[i] >= p[i + 1] - 0.01 for i in range(len(p) - 1)), p)

    def test_an_empty_look_lowers_that_zone_and_raises_the_rest_in_proportion(self):
        b = self.model.belief(15.0, 2, HABITS, [], ("kitchen", 300.0))
        top = max(F.ZONE_NAMES, key=b.get)
        after = S.after_miss(b, top, miss_rate=0.2)
        self.assertLess(after[top], b[top])
        self.assertGreater(after[top], 0.0)  # the look can miss her: not ruled out
        others = [z for z in F.CLASSES if z != top and b[z] > 0.01]
        ratios = [after[z] / b[z] for z in others]
        self.assertTrue(all(r > 1.0 for r in ratios))
        self.assertAlmostEqual(max(ratios), min(ratios), places=6)  # Bayes: all others grow by the same factor
        self.assertAlmostEqual(sum(after.values()), 1.0, places=6)

    def test_the_loop_finds_her_wherever_she_is(self):
        b = self.model.belief(15.0, 2, HABITS, [], ("kitchen", 300.0))
        rng = np.random.default_rng(0)
        for zone in F.ZONE_NAMES:
            found = sum(S.run(self.flat, b, zone, rng, miss_rate=0.2)[-1]["seen"] for _ in range(20))
            self.assertGreaterEqual(found, 18, zone)

    def test_the_likelier_she_is_somewhere_the_sooner_it_is_checked(self):
        b = self.model.belief(15.0, 2, HABITS, [], ("kitchen", 300.0))
        order, _ = S.plan(self.flat, b)
        rng = np.random.default_rng(1)
        looks = {z: np.mean([len(S.run(self.flat, b, z, rng, miss_rate=0.0)) for _ in range(3)]) for z in order}
        self.assertEqual(looks[order[0]], 1.0)
        self.assertLessEqual(looks[order[0]], looks[order[-1]])

    def test_nobody_home_ends_the_search(self):
        b = self.model.belief(15.0, 3, HABITS, [], ("kitchen", 300.0))  # Thursday afternoon: a stated outing
        steps = S.run(self.flat, b, "out", np.random.default_rng(2), miss_rate=0.2)
        self.assertFalse(any(s["seen"] for s in steps))
        self.assertGreaterEqual(steps[-1]["belief_after"]["out"], S.STOP_OUT)
        self.assertLess(len(steps), S.MAX_LOOKS)
        # With an outing stated it takes fewer empty looks to be sure than on an ordinary afternoon.
        ordinary = self.model.belief(15.0, 2, HABITS, [], ("kitchen", 300.0))
        usual = S.run(self.flat, ordinary, "out", np.random.default_rng(2), miss_rate=0.2)
        self.assertLess(len(steps), len(usual))
        # Every empty look makes "out" more likely.
        outs = [b["out"]] + [s["belief_after"]["out"] for s in steps]
        self.assertTrue(all(x < y for x, y in zip(outs, outs[1:])), outs)


if __name__ == "__main__":
    unittest.main()

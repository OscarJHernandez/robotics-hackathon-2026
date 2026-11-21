"""Checks for the Knock data in homes/: flats, residents, the generated tables and the cold-start prior.

    uv run --project <team repo> --group ml python -m unittest homes/test_homes.py -v

The flat tests need only numpy. The table tests need pandas and skip without it. Nothing here needs TabPFN.
"""

import re
import sys
import unittest
from pathlib import Path

import numpy as np

HOMES = Path(__file__).resolve().parent
sys.path.insert(0, str(HOMES))

import flats as F  # noqa: E402
import residents as R  # noqa: E402

try:
    import pandas as pd  # noqa: F401
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False

# The app's six zone names (KnockZone in app/web/src/lib/api.ts, ZONES in app/knock.py).
APP_ZONES = {"bed", "bathroom", "kitchen", "couch", "armchair", "entrance"}
SPEC_COLUMNS = ["day", "resident", "source", "hour", "weekday", "last_heard_zone",
                "min_since_heard", "voice_bearing_deg", "voice_loudness_db", "zone"]


def make_flat(i):
    return F.random_flat(np.random.default_rng([0, i]), i)


class FlatsTest(unittest.TestCase):
    def test_zone_counts_names_and_geometry(self):
        counts = set()
        for i in range(300):
            flat = make_flat(i)
            names = F.zone_names(flat)
            counts.add(len(names))
            self.assertTrue(3 <= len(names) <= 5)
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue({"bed", "kitchen", "bathroom"} <= set(names))
            self.assertTrue(set(names) <= APP_ZONES)
            w, d = flat["width"] / 2, flat["depth"] / 2
            rects = []
            for z in flat["zones"]:
                r = (z["x"] - z["size_x"] / 2, z["y"] - z["size_y"] / 2, z["x"] + z["size_x"] / 2, z["y"] + z["size_y"] / 2)
                tol = 2e-3  # the flat stores centre and size rounded to 1 mm, so a rectangle can differ by under 1 mm
                self.assertTrue(-w - tol <= r[0] and r[2] <= w + tol and -d - tol <= r[1] and r[3] <= d + tol)
                for other in rects:
                    self.assertFalse(F._overlap(r, other), "furniture overlaps in flat {}".format(i))
                rects.append(r)
                self.assertTrue(-w + F.CLEARANCE - 1e-6 <= z["view_x"] <= w - F.CLEARANCE + 1e-6)
                self.assertTrue(-d + F.CLEARANCE - 1e-6 <= z["view_y"] <= d - F.CLEARANCE + 1e-6)
                for o in flat["zones"]:
                    if o["solid"]:
                        orect = (o["x"] - o["size_x"] / 2, o["y"] - o["size_y"] / 2, o["x"] + o["size_x"] / 2,
                                 o["y"] + o["size_y"] / 2)
                        self.assertGreaterEqual(F._point_rect_dist(z["view_x"], z["view_y"], orect), F.CLEARANCE - 1e-6)
        self.assertEqual(counts, {3, 4, 5})

    def test_same_seed_same_flat(self):
        self.assertEqual(make_flat(7), make_flat(7))
        self.assertNotEqual(make_flat(7), make_flat(8))

    def test_to_app_zones_has_the_shape_of_app_knock_zones(self):
        zones = F.to_app_zones(make_flat(3))
        for name, (label, (x, y), heading, description) in zones.items():
            self.assertIn(name, APP_ZONES)
            self.assertIsInstance(label, str)
            self.assertIsInstance(description, str)
            self.assertTrue(-180.0 <= heading <= 180.0)
            self.assertTrue(isinstance(x, float) and isinstance(y, float))


class ResidentsTest(unittest.TestCase):
    def test_routine_tables_use_only_zones_of_the_flat(self):
        for i in range(100):
            flat = make_flat(i)
            p = R.sample_persona(np.random.default_rng([1, i]), i, flat)
            names = set(F.zone_names(flat))
            for wk in ("0", "1"):
                self.assertEqual(len(p["tables"][wk]), 3)
                for table in p["tables"][wk]:
                    self.assertTrue(set(table) <= names)
                    self.assertAlmostEqual(sum(table.values()), 1.0)
            self.assertEqual(len(p["out_p"]), 7)
            allowed = {F.CLASSES.index(n) for n in names} | {R.OUT}
            for day in range(7):
                codes = R.draw_day(p, day, np.random.default_rng([2, i, day]))
                self.assertEqual(len(codes), 3)
                self.assertTrue(set(codes) <= allowed)
            probs = R.zone_probs(p, 2, 1)
            self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_slots(self):
        self.assertEqual([R.slot_of(m) for m in (419, 420, 719, 720, 1079, 1080, 1349, 1350)],
                         [None, 0, 0, 1, 1, 2, 2, None])

    def test_no_fine_tuning_code_in_homes(self):
        banned = re.compile(r"finetun|fine_tun|optimizer|learning_rate|\.backward\(|loss\.backward", re.I)
        for path in HOMES.glob("*.py"):
            if path.name == "test_homes.py":
                continue
            self.assertIsNone(banned.search(path.read_text()), "fine-tuning words in {}".format(path.name))


@unittest.skipUnless(HAVE_PANDAS, "needs pandas: uv run --group ml ...")
class GenerateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import generate
        cls.generate = generate
        cls.diary, cls.checkins = generate.generate(residents=10, days=7, seed=0)
        cls.flats = {f["flat_id"]: f for f in generate.build(residents=10, days=7, seed=0)["flats"]}

    def test_same_seed_gives_the_same_frames(self):
        d2, c2 = self.generate.generate(residents=10, days=7, seed=0)
        pd.testing.assert_frame_equal(self.diary, d2)
        pd.testing.assert_frame_equal(self.checkins, c2)

    def test_resident_does_not_change_when_the_number_of_residents_does(self):
        d2, _ = self.generate.generate(residents=4, days=7, seed=0)
        a = self.diary[self.diary["resident_id"] <= 4].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, d2)

    def test_three_records_a_day_one_per_slot_and_no_pose(self):
        self.assertEqual(list(self.diary.columns[:len(SPEC_COLUMNS)]), SPEC_COLUMNS)
        self.assertEqual(list(self.checkins.columns[:len(SPEC_COLUMNS)]), SPEC_COLUMNS)
        for col in ("pose", "fell", "min_since_carer_left"):
            self.assertNotIn(col, self.diary.columns)
            self.assertNotIn(col, self.checkins.columns)
        self.assertTrue(self.diary["day"].between(0, 4).all())
        self.assertTrue(self.checkins["day"].between(5, 6).all())
        per_day = self.diary.groupby(["resident_id", "day"]).size()
        self.assertTrue((per_day == 3).all())
        self.assertEqual(len(self.diary), 10 * 5 * 3)
        for slot, (a, b) in enumerate(R.SLOT_WINDOWS):
            h = self.diary.loc[self.diary["slot"] == slot, "hour"]
            self.assertTrue(((h >= a / 60 - 1e-3) & (h <= b / 60 + 1e-3)).all())
        self.assertEqual(set(self.checkins["kind"]), {"checkin", "help"})

    def test_every_label_is_in_the_residents_flat_or_out(self):
        for df in (self.diary, self.checkins):
            for fid, g in df.groupby("flat_id"):
                allowed = set(F.zone_names(self.flats[fid])) | {"out"}
                self.assertTrue(set(g["zone"]) <= allowed)
                self.assertTrue(set(g["last_heard_zone"].dropna()) <= allowed)
        self.assertTrue(set(self.diary["zone"]) | set(self.checkins["zone"]) <= APP_ZONES | {"out"})

    def test_no_leakage_last_heard_is_the_previous_row(self):
        d = self.diary.sort_values(["resident_id", "day", "hour"])
        for _, g in d.groupby("resident_id"):
            self.assertTrue(pd.isna(g["last_heard_zone"].iloc[0]))
            self.assertEqual(g["last_heard_zone"].iloc[1:].tolist(), g["zone"].iloc[:-1].tolist())
            gap = (g["day"].diff() * 1440 + g["hour"].diff() * 60).iloc[1:]
            self.assertTrue(np.allclose(g["min_since_heard"].iloc[1:], gap, atol=0.3))  # hour and gap are rounded
            self.assertTrue((g["hist_n"][g["day"] == 0] == 0).all())  # nothing from before day 0

    def test_voice_only_on_heard_rows_and_on_help_calls(self):
        self.assertTrue(self.diary.loc[self.diary["source"] == "checkin", "voice_bearing_deg"].isna().all())
        self.assertTrue((self.diary.loc[self.diary["source"] == "checkin", "zone"] == "out").all())  # out: nobody heard
        self.assertTrue(self.diary.loc[self.diary["source"] == "heard", "voice_bearing_deg"].notna().all())
        self.assertTrue(self.checkins.loc[self.checkins["kind"] == "checkin", "voice_bearing_deg"].isna().all())
        self.assertTrue(self.checkins.loc[self.checkins["kind"] == "help", "voice_bearing_deg"].notna().all())
        self.assertTrue((self.checkins.loc[self.checkins["kind"] == "help", "zone"] != "out").all())

    def test_positions_only_when_home(self):
        home = self.checkins["zone"] != "out"
        self.assertTrue(self.checkins.loc[home, "x"].notna().all())
        self.assertTrue(self.checkins.loc[~home, "x"].isna().all())

    def test_splits_do_not_overlap_and_feature_frame_is_numeric(self):
        split = self.diary.groupby("resident_id")["resident_split"].nunique()
        self.assertTrue((split == 1).all())
        self.assertEqual(self.diary["resident_split"].isin(["train", "test"]).all(), True)
        X = self.generate.feature_frame(self.diary)
        self.assertTrue(all(str(t) in ("float32",) for t in X.dtypes))
        self.assertEqual(len(X), len(self.diary))


@unittest.skipUnless(HAVE_PANDAS, "needs pandas: uv run --group ml ...")
class PriorTest(unittest.TestCase):
    def test_cold_start_prior_sums_to_one_and_stays_in_the_flat(self):
        import generate
        import prior
        out = generate.build(residents=60, days=7, seed=1)
        sigs = {f["flat_id"]: F.signature(f) for f in out["flats"]}
        model = prior.ColdStartPrior.fit(out["diary"], sigs)
        for flat in out["flats"][:10]:
            for hour in (3.0, 7.2, 10.5, 14.0, 20.0, 23.0):
                for weekday in (1, 6):
                    zones, p_out = model.probs(flat, hour, weekday)
                    self.assertAlmostEqual(sum(zones.values()), 1.0)
                    self.assertTrue(set(zones) == set(F.zone_names(flat)))
                    self.assertTrue(0.0 <= p_out <= 1.0)
        zones, p_out = model.probs(out["flats"][0], 3.0, 1)  # night: the stated default
        self.assertEqual(p_out, 0.0)
        self.assertAlmostEqual(zones["bed"], 0.85)


if __name__ == "__main__":
    unittest.main()

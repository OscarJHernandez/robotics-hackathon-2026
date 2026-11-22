"""Where is she? The belief from TabPFN, with synthetic check-ins as its examples.

    uv run --group ml python homes/tabpfn_belief.py            # make homes/data/tabpfn_context.npz and try it

TabPFN is used only through `fit()` and `predict_proba()` (the rule in AGENTS.md): `fit()` stores the examples,
the pretrained weights never change. What makes it work is what goes into `fit()`:

- **Examples are check-ins, not diary records.** A check-in can come a few minutes after a record in the same
  part of the day, and then "where she was last heard" is almost always right. Diary records never show that
  (there is one per part of the day), which is why TabPFN lost to the "last heard" rule in homes/check.py.
- **The same inputs as the counting belief** (search.beliefs): the time, her stated habits, her own journal (as
  shares per zone for this part of the day), and where and how long ago she was last heard.

The examples come from synthetic residents (generate.build) with journals of 1, 2 and 4 weeks, so the model has
seen short and long journals. A resident's own journal is not among the examples; it reaches the model through
the features of the row to predict, so a pasted journal changes the belief without a new fit().
"""

import argparse
import warnings
from pathlib import Path

import numpy as np

from flats import CLASSES, ZONE_NAMES
from residents import slot_of, stated_habits

CONTEXT_FILE = Path(__file__).resolve().parent / "data" / "tabpfn_context.npz"
# TabPFN 3.5 when its API key is there (TABPFN_TOKEN, or TABPFN_TOKEN=... in the repo's gitignored .env, as for
# nav/dagger), else the v2 weights, which need no key.
VERSIONS = ("v3.5", "v2")
CONTEXT_ROWS = 2000
NEVER = 2880.0  # minutes since last heard when there is no record, and the cap
HIST_ALPHA = 0.1  # as generate.HIST_ALPHA
FEATURES = (["hour", "weekday", "is_weekend", "slot", "min_since_heard", "same_part"]
            + ["last_" + c for c in CLASSES] + ["hist_" + c for c in CLASSES] + ["hist_n"]
            + ["habit_" + z for z in ZONE_NAMES] + ["outing"])


def same_part(hour, min_since):
    """True when her newest record is from this same part of today."""
    slot = slot_of(int(hour * 60))
    return slot is not None and min_since < 16 * 60 and slot_of(int(hour * 60 - min_since)) == slot


def row(hour, weekday, last_zone, min_since, hist, hist_n, habit_zone, outing):
    """One row of FEATURES. last_zone: a class name or None; min_since: minutes since that record;
    hist: {class: share of her earlier-day records in this part of the day and weekday type}, hist_n: how many
    there are; habit_zone: her stated usual zone now; outing: True on a stated outing."""
    never = last_zone is None or min_since is None or not np.isfinite(min_since)
    since = NEVER if never else min(float(min_since), NEVER)
    out = [float(hour), float(weekday), float(weekday >= 5), float(slot_of(int(hour * 60)) or 0), since,
           float(not never and same_part(hour, since))]
    out += [float(last_zone == c) for c in CLASSES]
    out += [float(hist.get(c, 0.0)) for c in CLASSES] + [float(hist_n)]
    out += [float(habit_zone == z) for z in ZONE_NAMES] + [float(bool(outing))]
    return out


def history(journal, weekday, slot, before_day=None):
    """({class: smoothed share}, n) of her records in this part of the day and weekday type. journal: records
    (weekday, slot, zone) or (weekday, slot, zone, day); with `before_day` only records of earlier days count,
    as in the generator's hist_* columns."""
    counts = {c: 0.0 for c in CLASSES}
    for record in journal:
        if record[1] != slot or (record[0] >= 5) != (weekday >= 5):
            continue
        if before_day is not None and len(record) > 3 and record[3] >= before_day:
            continue
        counts[record[2]] += 1.0
    n = sum(counts.values())
    return {c: (v + HIST_ALPHA) / (n + HIST_ALPHA * len(CLASSES)) for c, v in counts.items()}, n


def frame_rows(out, split):
    """(X, y) for every `checkin` of the residents of a split in a generate.build() result."""
    checkins = out["checkins"][out["checkins"]["kind"] == "checkin"]
    habits = {p["resident_id"]: stated_habits(p) for p in out["personas"] if p["resident_split"] == split}
    X, y = [], []
    for r in checkins.itertuples():
        h = habits.get(r.resident_id)
        if h is None:
            continue
        wk, slot, weekday = "1" if r.weekday >= 5 else "0", int(r.slot), int(r.weekday)
        last = r.last_heard_zone if isinstance(r.last_heard_zone, str) else None
        hist = {c: getattr(r, "hist_" + c) for c in CLASSES}
        X.append(row(r.hour, weekday, last, r.min_since_heard, hist, r.hist_n, h["usual"][wk][slot],
                     (weekday, slot) in set(map(tuple, h["outings"]))))
        y.append(r.zone)
    return np.array(X, dtype=np.float32), np.array(y, dtype=object)


def make_context(rows=CONTEXT_ROWS, seed=1):
    """Synthetic check-ins of train residents with 1, 2 and 4 weeks of journal: (X, y), `rows` of them."""
    from generate import build

    parts = [frame_rows(build(residents=250, days=days, seed=seed + i), "train")
             for i, days in enumerate((9, 16, 30))]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    keep = np.random.default_rng(seed).permutation(len(X))[:rows]
    return X[keep], y[keep]


def load_context(path=CONTEXT_FILE, make=True):
    """The examples: from the file, or made (and saved) when it is missing."""
    path = Path(path)
    if path.exists():
        data = np.load(path, allow_pickle=False)
        if list(data["features"]) == FEATURES and list(data["classes"]) == list(CLASSES):
            return data["X"].astype(np.float32), data["y"].astype(object)
    if not make:
        raise FileNotFoundError(str(path))
    X, y = make_context()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y.astype(str), features=np.array(FEATURES), classes=np.array(CLASSES))
    return X, y


def pick_version(wanted=None):
    """The TabPFN version to use: `wanted`, else the newest one whose key is available. Loads the key into the
    environment (nav/dagger/envfile.py); the key is never printed."""
    import os
    import sys

    repo = str(Path(__file__).resolve().parents[1])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from nav.dagger.envfile import GATED, load_token

    os.environ.setdefault("TABPFN_NO_BROWSER", "1")  # never open a login page
    have_key = load_token() is not None
    for version in ([wanted] if wanted else VERSIONS):
        if version not in GATED or have_key:
            return version
    raise RuntimeError("TabPFN {} needs an API key (TABPFN_TOKEN or the repo's .env)".format(wanted))


class TabPFNBelief:
    """P(zone) and P(out) from TabPFN. `fit()` once on the examples; `predict(rows)` -> array [n, len(CLASSES)]."""

    name = "tabpfn"

    def __init__(self, X=None, y=None, seed=0, version=None):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        if X is None:
            X, y = load_context()
        self.rows = len(X)
        self.version = pick_version(version)
        self.model = TabPFNClassifier.create_default_for_version(
            ModelVersion(self.version), device="cpu", n_estimators=4, random_state=seed,
            fit_mode="fit_with_cache", ignore_pretraining_limits=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(np.asarray(X, np.float32), np.asarray(y))  # stores the examples; no weights change
        self._order = [list(self.model.classes_).index(c) if c in self.model.classes_ else None for c in CLASSES]

    def predict(self, rows):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = self.model.predict_proba(np.asarray(rows, np.float32))
        out = np.zeros((len(proba), len(CLASSES)))
        for j, i in enumerate(self._order):
            if i is not None:
                out[:, j] = proba[:, i]
        return out / out.sum(axis=1, keepdims=True)

    def belief(self, hour, weekday, habits, journal, last):
        """{class: p} at this hour and weekday. habits: residents.stated_habits(); journal: her records
        (weekday, slot, zone, day index or date) before now; last: (zone, minutes ago) or None."""
        slot = slot_of(int(hour * 60))
        if slot is None:
            return None  # night: no examples; the caller uses the stated default
        hist, n = history(journal, weekday, slot)
        usual = habits["usual"]["1" if weekday >= 5 else "0"][slot]
        outing = (weekday, slot) in set(map(tuple, habits["outings"]))
        features = row(hour, weekday, last[0] if last else None, last[1] if last else None, hist, n, usual, outing)
        return {c: float(p) for c, p in zip(CLASSES, self.predict([features])[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--remake", action="store_true", help="make the examples again even if the file exists")
    parser.add_argument("--version", default=None, help="TabPFN version (default: v3.5 with a key, else v2)")
    args = parser.parse_args()
    if args.remake and CONTEXT_FILE.exists():
        CONTEXT_FILE.unlink()
    X, y = load_context()
    names, counts = np.unique(y, return_counts=True)
    print("{} examples, {} features; classes: {}".format(len(X), X.shape[1], dict(zip(names, counts.tolist()))))
    print("in the same part of the day as the last record: {:.0f}%".format(100 * X[:, FEATURES.index("same_part")].mean()))
    model = TabPFNBelief(X, y, version=args.version)
    print("TabPFN {}".format(model.version))
    habits = {"usual": {"0": ["kitchen", "living_room", "living_room"], "1": ["bedroom"] * 3}, "outings": [(1, 1)]}
    for label, hour, weekday, last in (("Wed 15:00, last heard this morning", 15.0, 2, ("kitchen", 300.0)),
                                       ("Wed 15:00, heard in the kitchen 10 min ago", 15.0, 2, ("kitchen", 10.0)),
                                       ("Tue 14:00, a stated outing", 14.0, 1, ("kitchen", 300.0))):
        b = model.belief(hour, weekday, habits, [], last)
        print("{:<44} {}".format(label, {k: round(v, 2) for k, v in b.items()}))


if __name__ == "__main__":
    main()

"""Cold-start prior: where is she in this part of the day, before there is any diary for her?

    uv run --project <team repo> --group ml python homes/prior.py        # fits from data/diary.parquet -> data/cold_prior.json

P(zone | flat's zone set, weekday type, slot: morning / afternoon / evening), counted from the diary rows of TRAIN residents only
(so test residents are never in it), smoothed towards the population share. Residents with the same set of zones
(for example bed, bathroom, kitchen, couch) share one table, because "couch" means nothing in a flat that has none.

`probs(flat, hour, weekday)` returns `(zone_probs, p_out)`:
- `zone_probs`: {zone name: p} over the flat's zones, summing to 1, the same shape as PRIORS in app/knock.py;
- `p_out`: the separate probability that she is not home.

There are no diary rows between 22:30 and 07:00 (the robot only records in the day), so night uses a stated default:
bed 0.85, bathroom 0.15, nobody out (used by help calls at night). The saved file is small JSON; loading it needs only numpy.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from flats import CLASSES, signature, zone_names
from residents import slot_of

DATA = Path(__file__).resolve().parent / "data"
SLOTS = 3  # morning, afternoon, evening
ALPHA = 5.0  # pseudo-count that pulls a thin table towards the population share
NIGHT_DEFAULT = {"bed": 0.85, "bathroom": 0.15}


def _wk(weekday):
    return 1 if weekday >= 5 else 0


class ColdStartPrior:
    def __init__(self, counts, pooled, alpha=ALPHA):
        self.counts = counts  # {signature: array [2, 3, 7]}
        self.pooled = pooled  # array [2, 3, 7], every flat together
        self.alpha = alpha

    # --- fit / save / load ---

    @classmethod
    def fit(cls, diary, flat_zones, alpha=ALPHA):
        """`diary`: pandas frame with flat_id, is_weekend, hour, zone, resident_split. `flat_zones`: {flat_id: signature}."""
        train = diary[diary["resident_split"] == "train"]
        code = {c: i for i, c in enumerate(CLASSES)}
        slot = train["slot"].to_numpy().astype(int)
        wk = train["is_weekend"].to_numpy().astype(int)
        z = train["zone"].map(code).to_numpy()
        sig = train["flat_id"].map(flat_zones).to_numpy()
        counts = {}
        for s in sorted(set(flat_zones.values())):
            m = sig == s
            c = np.zeros((2, SLOTS, len(CLASSES)))
            np.add.at(c, (wk[m], slot[m], z[m]), 1.0)
            counts[s] = c
        pooled = sum(counts.values())
        return cls(counts, pooled, alpha)

    def save(self, path):
        data = {"classes": list(CLASSES), "alpha": self.alpha, "night_default": NIGHT_DEFAULT,
                "pooled": self.pooled.round(1).tolist(),
                "tables": {s: c.round(1).tolist() for s, c in self.counts.items()}}
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls({s: np.array(c) for s, c in data["tables"].items()}, np.array(data["pooled"]), data["alpha"])

    # --- the prior ---

    def probs(self, flat, hour, weekday):
        """(zone_probs, p_out) for this flat at this hour of day (0-24) and weekday (0 = Monday)."""
        names = zone_names(flat)
        wk, slot = _wk(weekday), slot_of(int(hour * 60))
        if slot is None:  # night: no diary rows, use the stated default
            zone_probs = {n: NIGHT_DEFAULT.get(n, 0.0) for n in names}
            total = sum(zone_probs.values())
            return {n: p / total for n, p in zone_probs.items()}, 0.0
        keep = [CLASSES.index(n) for n in names] + [CLASSES.index("out")]
        pooled = self.pooled[wk, slot]
        base = pooled[keep]
        base = base / base.sum()
        own = self.counts.get(signature(flat))
        c = own[wk, slot][keep] if own is not None else np.zeros(len(keep))
        p = (c + self.alpha * base) / (c.sum() + self.alpha)
        p_out = float(p[-1])
        z = p[:-1] / p[:-1].sum()
        return {n: float(z[i]) for i, n in enumerate(names)}, p_out


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=str(DATA))
    args = parser.parse_args()
    data = Path(args.data)
    diary = pd.read_parquet(data / "diary.parquet")
    flats = json.loads((data / "flats.json").read_text())
    prior = ColdStartPrior.fit(diary, {f["flat_id"]: signature(f) for f in flats})
    prior.save(data / "cold_prior.json")
    flat = next(f for f in flats if len(f["zones"]) == 4)
    for hour, wd in ((9.5, 1), (15.0, 2), (20.0, 4), (3.0, 1)):
        zone_probs, p_out = prior.probs(flat, hour, wd)
        print("flat {} hour {:>4} weekday {}: {}  out {:.2f}".format(
            flat["flat_id"], hour, wd, {k: round(v, 2) for k, v in zone_probs.items()}, p_out))
    print("wrote", data / "cold_prior.json")


if __name__ == "__main__":
    main()

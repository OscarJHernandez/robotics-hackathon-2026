"""One worked example: a held-out resident, her flat, routine table, journal and what TabPFN answers for one check-in.

    uv run --project <team repo> --group ml python homes/example.py [--resident 5] [--row 1] [--own-days 3]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import render  # noqa: E402
from check import cold_proba, fit_classifier, restrict  # noqa: E402
from flats import CLASSES, distance, signature, zone_names  # noqa: E402
from generate import VOICE_FEATURES, _history, build, feature_frame  # noqa: E402
from prior import ColdStartPrior  # noqa: E402
from residents import zone_probs  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
CODE = {c: i for i, c in enumerate(CLASSES)}
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resident", type=int, default=5)
    ap.add_argument("--row", type=int, default=0, help="which of her test check-ins to use")
    ap.add_argument("--own-days", type=int, default=3)
    args = ap.parse_args()
    rid, k = args.resident, args.own_days

    diary = pd.read_parquet(DATA / "diary.parquet")
    checkins = pd.read_parquet(DATA / "checkins.parquet")
    flats = {f["flat_id"]: f for f in json.loads((DATA / "flats.json").read_text())}
    flat = flats[rid]
    prior = ColdStartPrior.load(DATA / "cold_prior.json")
    p = build(residents=rid, days=7, seed=0, records_per_day=len(json.loads(pd.read_parquet(DATA / "residents.parquet").iloc[rid - 1]["tables"])["0"]))["personas"][rid - 1]  # same seeds: the persona of resident rid
    assert p["resident_split"] == "test", "pick a held-out resident (every 5th: 5, 10, 15 ...)"

    q = checkins[(checkins.resident_id == rid) & (checkins.kind == "checkin")].reset_index(drop=True)
    row = q.iloc[args.row]
    fig, ax = plt.subplots(figsize=(6, 4.8))
    render.draw(ax, flat)
    if not pd.isna(row["x"]):
        ax.plot(row["x"], row["y"], "*", ms=16, color="#d4a000", zorder=7)
    ax.set_xlabel(", ".join(zone_names(flat)) + "   |   star = where she really is", fontsize=8)
    fig.savefig(DATA / "example_map.png", dpi=130, bbox_inches="tight")

    print("=== THE FLAT ===")
    print("flat {}: {:.1f} x {:.1f} m, zones: {}   (map: {})".format(
        flat["flat_id"], flat["width"], flat["depth"], zone_names(flat), DATA / "example_map.png"))

    print("\n=== HER ROUTINE (hidden table the generator draws from; the model never sees it) ===")
    print("{} ({}), type '{}'. P(zone) per weekday and slot, 'out' = not home:".format(p["name"], p["resident"], p["archetype"]))
    for wd in (1, 5):  # a Tuesday and a Saturday
        for s, name in enumerate(range(len(p["tables"]["0"]))):
            probs = {z: round(v, 2) for z, v in sorted(zone_probs(p, wd, s).items(), key=lambda kv: -kv[1]) if v >= 0.02}
            print("  {} {:9s} {}".format(DAYS[wd], name, probs))
    print("  other weekdays differ only in 'out':", {DAYS[d]: [round(x, 2) for x in p["out_p"][d]] for d in range(7)
                                                    if any(x > 0.1 for x in p["out_p"][d])})

    print("\n=== HER JOURNAL (the 15 records the robot has: 3 a day, days 0-4) ===")
    j = diary[diary.resident_id == rid][["day", "weekday", "source", "hour", "zone", "true_zone"]].copy()
    j["weekday"] = j["weekday"].map(lambda w: DAYS[w])
    print(j.round(2).to_string(index=False))

    print("\n=== THE QUESTION (a test check-in: she doesn't answer, where is she?) ===")
    print("day {} ({}), {:02d}:{:02d} (slot {}), last heard in the {}, {:.0f} min earlier".format(
        row["day"], DAYS[row["weekday"]], int(row["hour"]), int(row["hour"] % 1 * 60), row["slot"] if row["slot"] < len(p["tables"]["0"]) else "night",
        row["last_heard_zone"], row["min_since_heard"]))
    print("TRUTH:", row["zone"])

    allowed = set(zone_names(flat)) | {"out"}
    train = diary[diary.resident_split == "train"]
    pool = train[train.flat_id.map({f: signature(g) for f, g in flats.items()}) == signature(flat)]
    rng = np.random.default_rng(rid)
    own = diary[(diary.resident_id == rid) & (diary.day < k)]
    outs = pool[pool.zone == "out"]
    n_out = min(100, len(outs))
    ctx_other = pd.concat([outs.iloc[rng.permutation(len(outs))[:n_out]],
                           pool[pool.zone != "out"].sample(1000 - len(own) - n_out, random_state=rid)])
    ctx = pd.concat([own, ctx_other])
    Xc, Xq = feature_frame(ctx), feature_frame(q.iloc[[args.row]])
    qq = q.iloc[[args.row]]
    share, n = _history(qq["day"].to_numpy(), qq["slot"].to_numpy(), qq["is_weekend"].to_numpy(), own["day"].to_numpy(),
                        own["slot"].to_numpy(), own["is_weekend"].to_numpy(), own["zone"].map(CODE).to_numpy())
    for i, c in enumerate(CLASSES):
        Xq["hist_" + c] = share[:, i].astype("float32")
    Xq["hist_n"] = n.astype("float32")
    drop = [c for c in Xc.columns if c in VOICE_FEATURES or c.startswith("vzg_")]
    Xc, Xq = Xc.drop(columns=drop), Xq.drop(columns=drop)
    Xc["own"] = np.r_[np.ones(len(own)), np.zeros(len(ctx_other))].astype("float32")
    Xq["own"] = np.float32(1.0)
    keep = [c for c in Xc.columns if Xc[c].nunique() > 1]
    model = fit_classifier(Xc[keep].to_numpy(), ctx["zone"].to_numpy())
    pt = restrict(model.predict_proba(Xq[keep].to_numpy(np.float32)), list(model.classes_), allowed)[0]
    pc = cold_proba(prior, flat, qq)[0]

    print("\n=== WHAT THE MODEL ANSWERS ===")
    print("context: {} rows = her first {} diary days ({} rows) + {} rows of other residents whose flat has the same zones".format(
        len(ctx), k, len(own), len(ctx_other)))
    print("{:10s} {:>8s} {:>11s}".format("zone", "TabPFN", "cold-start"))
    for name in sorted(allowed, key=lambda z: -pt[CODE[z]]):
        mark = "  <-- truth" if name == row["zone"] else ("  <-- last heard" if name == row["last_heard_zone"] else "")
        print("{:10s} {:8.1%} {:11.1%}{}".format(name, pt[CODE[name]], pc[CODE[name]], mark))
    print("\n'last heard' rule says: {}  |  TabPFN's top: {}  |  truth: {}".format(
        row["last_heard_zone"], CLASSES[int(pt.argmax())], row["zone"]))
    left, pos, order = set(zone_names(flat)), "dock", []
    probs = {z: pt[CODE[z]] for z in zone_names(flat)}
    while left:
        z = max(left, key=lambda n: probs[n] / max(distance(flat, pos, n), 0.5))
        order.append(z), left.remove(z)
        pos = z
    print("search order (probability / walking distance): " + " -> ".join(order))
    if row["zone"] in order:
        print("zones checked before finding her:", order.index(row["zone"]) + 1)


if __name__ == "__main__":
    main()

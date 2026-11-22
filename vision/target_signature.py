"""The target's DINOv2 signature and how well it finds the target: just vision-signature full60 [--name pillar]

The navigation student already computes the camera's horizon band every step: the DINOv2 features of patch rows
11-14 averaged per patch column, 46 columns of 384 numbers (`nav/dagger/featurize.py`, BAND_ROWS). The signature
is made from band columns that show the target, in the TRAINING layouts of a vision run: every column is first
centred on the mean background column (DINOv2 features share a large common part), then the target columns are
grouped into `--prototypes` clusters (k-means on the cosine); their means are the signature vectors. A band column
matches when its cosine similarity to the nearest vector is above the threshold, chosen on the training layouts
for `--precision`; precision and recall are then reported on the held-out layouts (the same 12 as the probe).
On full60: one plain mean found the pillar in no picture beyond 2 m; centring plus 3 prototypes in 76% at 2-3 m.

The full 384 numbers are compared, not the few PCA values the student gets: those keep too little to tell the
target from look-alikes.

Output: data/vision/<run>/signature-<name>.npz (vectors (k, 384), center (384,), threshold, background mean and
spread) and
signature-<name>.json (the numbers). `nav/dagger/target.py` loads the .npz. A signature for another target (a
person standing, a person lying) is made the same way from pictures labelled with it; several vectors in one
file are matched by their maximum.
"""

import argparse
import json
import math

import numpy as np

from scene import TARGET

BAND_ROWS = (11, 15)  # the same patch rows as nav/dagger/featurize.py
PIXELS = 4 * 196  # pixels of one band column (4 patch rows of 14 x 14)
DIST_BINS = (0.0, 1.0, 2.0, 3.0, 4.0, 10.0)


def band_of(features):
    """(n, 26, 46, 384) patch features -> (n, 46, 384) float32 horizon band."""
    return np.asarray(features[:, BAND_ROWS[0]:BAND_ROWS[1]], dtype=np.float32).mean(axis=1)


def target_share(counts):
    """(n, 26, 46, 5) pixel counts per patch -> (n, 46) share of target pixels in every band column."""
    return counts[:, BAND_ROWS[0]:BAND_ROWS[1], :, TARGET].astype(np.float32).sum(axis=1) / PIXELS


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def similarity(band, vectors, center=None):
    """Cosine similarity of every band column to the signature: (..., 46, 384) x (k, 384) -> (..., 46), max over k.

    `center` (384,) is subtracted from the columns first (the signature vectors are already centred)."""
    band = np.asarray(band, dtype=np.float32)
    if center is not None:
        band = band - center
    return (unit(band) @ unit(vectors).T).max(axis=-1)


def load_signature(path):
    data = np.load(path)
    return {"vectors": data["vectors"].astype(np.float32), "center": data["center"].astype(np.float32),
            "threshold": float(data["threshold"]), "bg_mean": float(data["bg_mean"]), "bg_std": float(data["bg_std"]),
            "name": str(data["name"])}


def prototypes(columns, k, seed=0, iterations=30):
    """k cluster means of (n, 384) columns, k-means on the cosine. k = 1 is the plain mean."""
    if k <= 1 or len(columns) <= k:
        return columns.mean(axis=0)[None]
    rng = np.random.default_rng(seed)
    vectors = columns[rng.choice(len(columns), k, replace=False)]
    for _ in range(iterations):
        nearest = (unit(columns) @ unit(vectors).T).argmax(axis=1)
        vectors = np.stack([columns[nearest == j].mean(axis=0) if (nearest == j).any() else vectors[j]
                            for j in range(k)])
    return vectors


def pick_threshold(sim, positive, negative, precision):
    """Lowest threshold whose precision (positive / (positive + negative) columns above it) reaches `precision`."""
    values = np.concatenate([sim[positive], sim[negative]])
    is_pos = np.concatenate([np.ones(positive.sum(), bool), np.zeros(negative.sum(), bool)])
    order = np.argsort(-values)
    tp = np.cumsum(is_pos[order])
    prec = tp / np.arange(1, len(order) + 1)
    ok = np.flatnonzero(prec >= precision)
    if len(ok) == 0:
        return float(values.max())
    # the longest prefix that still has the precision: the last index where it holds
    return float(values[order][ok[-1]])


def column_scores(sim, positive, negative, threshold):
    hit = sim >= threshold
    tp, fp = int((hit & positive).sum()), int((hit & negative).sum())
    fn = int((~hit & positive).sum())
    return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "true_pos": tp, "false_pos": fp,
            "false_neg": fn, "positive_columns": int(positive.sum()), "negative_columns": int(negative.sum())}


def picture_scores(sim, share, threshold, distance):
    """Per picture: is the best column above the threshold, and does it show the target (share > 0)?"""
    best = sim.argmax(axis=1)
    above = sim.max(axis=1) >= threshold
    has_target = (share > 0).any(axis=1)
    on_target = share[np.arange(len(best)), best] > 0
    out = {"pictures_with_target": int(has_target.sum()), "pictures_without": int((~has_target).sum()),
           "found": int((above & on_target & has_target).sum()),
           "wrong_place": int((above & ~on_target & has_target).sum()),
           "false_alarm": int((above & ~has_target).sum())}
    out["found_rate"] = out["found"] / max(out["pictures_with_target"], 1)
    out["false_alarm_rate"] = out["false_alarm"] / max(out["pictures_without"], 1)
    by = []
    for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
        sel = has_target & (distance >= lo) & (distance < hi)
        by.append({"from_m": lo, "to_m": hi, "pictures": int(sel.sum()),
                   "found_rate": float((above & on_target)[sel].mean()) if sel.any() else None})
    out["by_distance"] = by
    return out


def build(features, counts, frames, layouts, train_eps, test_eps, min_share=0.25, precision=0.97, k=3, seed=0):
    """Signature from the training layouts' target columns, threshold from the training layouts; held-out scores.

    Returns (vectors (k, 384), center (384,), report)."""
    episodes = np.array([f["episode"] for f in frames])
    band = band_of(features)
    share = target_share(counts)
    positive, negative = share >= min_share, share == 0
    train = np.isin(episodes, train_eps)
    test = np.isin(episodes, test_eps)
    center = band[train][negative[train]].mean(axis=0)
    vectors = unit(prototypes(band[train][positive[train]] - center, k, seed))
    sim = similarity(band, vectors, center)
    threshold = pick_threshold(sim[train], positive[train], negative[train], precision)
    bg = sim[train][negative[train]]
    distance = np.array([math.hypot(f["pose"][0] - layouts[f["episode"]]["target"][0],
                                    f["pose"][1] - layouts[f["episode"]]["target"][1]) for f in frames])
    report = {"min_share": min_share, "precision_goal": precision, "prototypes": int(len(vectors)),
              "threshold": threshold,
              "bg_mean": float(bg.mean()), "bg_std": float(bg.std()),
              "target_mean": float(sim[train][positive[train]].mean()),
              "train_columns": column_scores(sim[train], positive[train], negative[train], threshold),
              "test_columns": column_scores(sim[test], positive[test], negative[test], threshold),
              "test_pictures": picture_scores(sim[test], share[test], threshold, distance[test]),
              "train_episodes": [int(e) for e in train_eps], "test_episodes": [int(e) for e in test_eps]}
    return vectors, center, report


def main():
    from dataset import run_dir
    from probe import split_episodes

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", nargs="?", default="full60", help="run name in data/vision (default: full60)")
    parser.add_argument("--name", default="pillar", help="signature name: signature-<name>.npz")
    parser.add_argument("--min-share", type=float, default=0.25, help="target pixel share of a target column")
    parser.add_argument("--precision", type=float, default=0.97, help="column precision the threshold aims for")
    parser.add_argument("--prototypes", type=int, default=3, help="signature vectors (k-means clusters)")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run = run_dir(args.run)
    meta = json.loads((run / "meta.json").read_text())
    frames, layouts = meta["frames"], meta["layouts"]
    episodes = np.array([f["episode"] for f in frames])
    train_eps, test_eps = split_episodes(episodes, args.test_fraction, args.seed)
    features = np.load(run / "features.npy", mmap_mode="r")
    counts = np.load(run / "patch_counts.npy")
    vectors, center, report = build(features, counts, frames, layouts, train_eps, test_eps, args.min_share,
                                    args.precision, args.prototypes, args.seed)
    report.update(run=run.name, name=args.name)
    np.savez(run / "signature-{}.npz".format(args.name), vectors=vectors, center=center, threshold=report["threshold"],
             bg_mean=report["bg_mean"], bg_std=report["bg_std"], name=args.name, band_rows=np.array(BAND_ROWS))
    (run / "signature-{}.json".format(args.name)).write_text(json.dumps(report, indent=1))
    tc, tp = report["test_columns"], report["test_pictures"]
    print("signature {} from {} training layouts: threshold {:.3f} (background {:.3f} +- {:.3f}, target {:.3f})"
          .format(args.name, len(train_eps), report["threshold"], report["bg_mean"], report["bg_std"],
                  report["target_mean"]))
    print("held-out columns ({} layouts): precision {:.1%}, recall {:.1%} ({} target, {} background columns)".format(
        len(test_eps), tc["precision"], tc["recall"], tc["positive_columns"], tc["negative_columns"]))
    print("held-out pictures: target found in {:.1%} of {}, wrong place {}, false alarm in {:.1%} of {}".format(
        tp["found_rate"], tp["pictures_with_target"], tp["wrong_place"], tp["false_alarm_rate"],
        tp["pictures_without"]))
    for b in tp["by_distance"]:
        if b["pictures"]:
            print("  {:.0f}-{:.0f} m: {:.1%} of {}".format(b["from_m"], b["to_m"], b["found_rate"], b["pictures"]))
    print("saved", run / "signature-{}.npz".format(args.name))


if __name__ == "__main__":
    main()

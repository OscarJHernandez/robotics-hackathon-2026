"""Train and test the patch classifier on DINOv2 features: just vision-probe latest [--hidden 256]

The probe is one linear layer (or one hidden layer with --hidden) from a patch's 384 features to four
classes: floor, wall, obstacle, target. Whole episodes (room layouts) are held out for the test.

For obstacle avoidance use `PatchClassifier`: `predict_patches(image)` gives the class grid and
`blocked_grid(image)` says which patches to stay away from.

Output in data/vision/<run>/: probe.pt (or probe-mlp256.pt), metrics*.json and overlays*/ pictures.
"""

import argparse
import json

import numpy as np

from features import COLS, DIM, ROWS, blocked_share, extract, load_dino
from scene import CLASSES, FLOOR, IGNORE, OBSTACLE, TARGET, WALL

MARGIN = 1  # patches: how far the blocked region is grown
THRESHOLD = 0.25  # blocked when P(wall) + P(obstacle) is above this; low on purpose
CAP = 150  # floor and wall patches kept per picture for training (obstacle and target: all)
EDGE_BUCKETS = ((0.1, 0.2), (0.2, 0.5), (0.5, 0.7))  # wall + obstacle share of a patch on an edge
COLORS = np.array([[70, 200, 90], [70, 110, 230], [240, 150, 30], [230, 40, 40]], dtype=np.float32)


def grow(mask, margin=MARGIN):
    """Grow the true region of a 2D mask by `margin` cells in every direction (also diagonally)."""
    out = mask.copy()
    for _ in range(margin):
        padded = np.pad(out, 1)
        out = np.zeros_like(out)
        for dy in range(3):
            for dx in range(3):
                out |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return out


def blocked_from_probs(probs, margin=MARGIN, threshold=THRESHOLD):
    """Which patches to avoid, from class probabilities (..., rows, cols, 4)."""
    raw = probs[..., WALL] + probs[..., OBSTACLE] > threshold
    if raw.ndim == 2:
        return grow(raw, margin)
    return np.stack([grow(r, margin) for r in raw])


def split_episodes(episodes, fraction, seed):
    """Hold out `fraction` of the episodes. Returns (train ids, test ids)."""
    unique = np.unique(episodes)
    order = np.random.default_rng(seed).permutation(unique)
    n_test = max(1, round(len(unique) * fraction))
    return np.sort(order[n_test:]), np.sort(order[:n_test])


def build_net(hidden):
    import torch.nn as nn
    if hidden:
        return nn.Sequential(nn.Linear(DIM, hidden), nn.ReLU(), nn.Linear(hidden, len(CLASSES)))
    return nn.Linear(DIM, len(CLASSES))


def train_probe(x, y, hidden=0, epochs=20, lr=1e-3, weight_decay=1e-4, batch=4096, seed=0, log=None):
    """Fit the probe on features x (n, 384) and labels y (n,). Returns {"net", "mean", "std", "hidden"}."""
    import torch
    torch.manual_seed(seed)
    x = torch.from_numpy(np.asarray(x, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y, dtype=np.int64))
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    x = (x - mean) / std
    counts = torch.bincount(y, minlength=len(CLASSES)).float().clamp_min(1)
    weights = counts.sum() / (len(CLASSES) * counts)  # rare classes count more
    net = build_net(hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        order = torch.randperm(len(x))
        total = 0.0
        for start in range(0, len(x), batch):
            idx = order[start:start + batch]
            loss = torch.nn.functional.cross_entropy(net(x[idx]), y[idx], weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        if log:
            log("epoch {:2d}  loss {:.4f}".format(epoch + 1, total / len(x)))
    net.eval()
    return {"net": net, "mean": mean, "std": std, "hidden": hidden}


def predict_probs(probe, features):
    """Class probabilities for features (..., 384): float32 array (..., 4)."""
    import torch
    flat = torch.from_numpy(np.asarray(features, dtype=np.float32).reshape(-1, DIM))
    out = []
    with torch.no_grad():
        for start in range(0, len(flat), 65536):
            chunk = (flat[start:start + 65536] - probe["mean"]) / probe["std"]
            out.append(torch.softmax(probe["net"](chunk), dim=-1))
    return torch.cat(out).numpy().reshape(features.shape[:-1] + (len(CLASSES),))


def save_probe(probe, path):
    import torch
    torch.save({"state": probe["net"].state_dict(), "mean": probe["mean"], "std": probe["std"],
                "hidden": probe["hidden"], "classes": list(CLASSES)}, path)


def load_probe(path):
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=True)
    net = build_net(saved["hidden"])
    net.load_state_dict(saved["state"])
    net.eval()
    return {"net": net, "mean": saved["mean"], "std": saved["std"], "hidden": saved["hidden"]}


class PatchClassifier:
    """Camera picture in, patch classes out. Loads DINOv2 and a trained probe."""

    def __init__(self, probe_path, device=None):
        self.probe = load_probe(probe_path)
        self.dino = load_dino(device)

    def predict_probs(self, image):
        """Class probabilities per patch (26, 46, 4) for one RGB uint8 picture (360, 640, 3)."""
        return predict_probs(self.probe, extract(self.dino, image[None])[0])

    def predict_patches(self, image):
        """Class per patch, (26, 46): 0 floor, 1 wall, 2 obstacle, 3 target."""
        return self.predict_probs(image).argmax(-1)

    def blocked_grid(self, image, margin=MARGIN, threshold=THRESHOLD):
        """True for every patch the robot should stay away from, (26, 46)."""
        return blocked_from_probs(self.predict_probs(image), margin, threshold)


def evaluate(probs, labels, counts, margin=MARGIN, threshold=THRESHOLD):
    """Scores on held-out pictures: probs (n, 26, 46, 4), labels and counts from features.py."""
    n = len(CLASSES)
    pred = probs.argmax(-1)
    keep = labels != IGNORE
    confusion = np.zeros((n, n), dtype=np.int64)  # rows: true class, columns: predicted
    np.add.at(confusion, (labels[keep], pred[keep]), 1)
    recall = confusion.diagonal() / np.maximum(confusion.sum(1), 1)
    precision = confusion.diagonal() / np.maximum(confusion.sum(0), 1)

    raw = probs[..., WALL] + probs[..., OBSTACLE] > threshold
    grown = np.stack([grow(r, margin) for r in raw])
    solid = (labels == WALL) | (labels == OBSTACLE)
    obstacle, floor = labels == OBSTACLE, labels == FLOOR

    def rate(mask, hit):
        return float(hit[mask].mean()) if mask.any() else None

    share = blocked_share(counts)
    mixed = counts[..., -1] < 0.1 * counts.sum(-1)  # leave out patches that show the robot's own legs
    edges = []
    for low, high in EDGE_BUCKETS:
        bucket = mixed & (share >= low) & (share < high)
        edges.append({"share": "{:.0%} to {:.0%}".format(low, high), "patches": int(bucket.sum()),
                      "blocked": rate(bucket, raw), "blocked_after_margin": rate(bucket, grown)})
    return {
        "classes": list(CLASSES),
        "patches": int(keep.sum()),
        "accuracy": float((pred[keep] == labels[keep]).mean()),
        "balanced_accuracy": float(recall.mean()),
        "recall": dict(zip(CLASSES, map(float, recall))),
        "precision": dict(zip(CLASSES, map(float, precision))),
        "confusion": confusion.tolist(),
        "blocked_vs_free": {
            "threshold": threshold, "margin": margin,
            # The dangerous error: something solid that the robot is told is free.
            "solid_left_free": rate(solid, ~raw),
            "solid_left_free_after_margin": rate(solid, ~grown),
            "obstacle_left_free": rate(obstacle, ~raw),
            "obstacle_left_free_after_margin": rate(obstacle, ~grown),
            "solid_predicted_floor": rate(solid, pred == FLOOR),
            # The cost of being careful: free floor that is marked blocked.
            "floor_blocked": rate(floor, raw),
            "floor_blocked_after_margin": rate(floor, grown),
        },
        "edges": edges,
    }


def save_overlays(run, frames, picks, probs, out):
    """Pictures with the predicted class painted over every patch, and blocked patches outlined."""
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    for index, p in zip(picks, probs):
        image = np.asarray(Image.open(run / frames[index]["image"]).convert("RGB"), dtype=np.float32)
        color = np.asarray(Image.fromarray(COLORS[p.argmax(-1)].astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]), Image.NEAREST), dtype=np.float32)
        blocked = np.asarray(Image.fromarray(blocked_from_probs(p).astype(np.uint8) * 255).resize(
            (image.shape[1], image.shape[0]), Image.NEAREST), dtype=np.float32)[..., None] / 255
        side = image * (1 - 0.6 * blocked)  # second panel: blocked patches darkened
        both = np.concatenate([0.5 * image + 0.5 * color, side], axis=0)
        Image.fromarray(both.astype(np.uint8)).save(out / frames[index]["image"].split("/")[-1])


def gather_training(features, labels, images, cap, rng):
    """Features and labels of the training patches; floor and wall are capped per picture."""
    xs, ys = [], []
    for i in images:
        flat = labels[i].reshape(-1)
        chosen = []
        for c in range(len(CLASSES)):
            idx = np.flatnonzero(flat == c)
            if c in (FLOOR, WALL) and len(idx) > cap:
                idx = rng.choice(idx, cap, replace=False)
            chosen.append(idx)
        idx = np.sort(np.concatenate(chosen))
        xs.append(np.asarray(features[i]).reshape(-1, DIM)[idx].astype(np.float32))
        ys.append(flat[idx])
    return np.concatenate(xs), np.concatenate(ys)


def report(metrics):
    lines = ["held-out patches: {}   accuracy {:.1%}   balanced accuracy {:.1%}".format(
        metrics["patches"], metrics["accuracy"], metrics["balanced_accuracy"])]
    lines.append("class      recall  precision")
    for name in CLASSES:
        lines.append("{:9s}  {:6.1%}  {:6.1%}".format(name, metrics["recall"][name], metrics["precision"][name]))
    lines.append("confusion (rows true, columns predicted: {}):".format(", ".join(CLASSES)))
    lines += ["  " + "  ".join("{:7d}".format(v) for v in row) for row in metrics["confusion"]]
    b = metrics["blocked_vs_free"]

    def pct(v):
        return "n/a" if v is None else "{:.2%}".format(v)

    lines.append("blocked vs free (threshold {}, margin {} patch):".format(b["threshold"], b["margin"]))
    lines.append("  wall or obstacle left free: {}  (after margin {})".format(
        pct(b["solid_left_free"]), pct(b["solid_left_free_after_margin"])))
    lines.append("  obstacle left free:         {}  (after margin {})".format(
        pct(b["obstacle_left_free"]), pct(b["obstacle_left_free_after_margin"])))
    lines.append("  floor marked blocked:       {}  (after margin {})".format(
        pct(b["floor_blocked"]), pct(b["floor_blocked_after_margin"])))
    lines.append("edge patches, by the share of their pixels that are wall or obstacle:")
    for e in metrics["edges"]:
        lines.append("  {:10s} {:6d} patches   blocked {}   after margin {}".format(
            e["share"], e["patches"], pct(e["blocked"]), pct(e["blocked_after_margin"])))
    return "\n".join(lines)


def main():
    from dataset import run_dir

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", nargs="?", default="latest", help="run name in data/vision (default: latest)")
    parser.add_argument("--hidden", type=int, default=0, help="hidden units; 0 = linear probe")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overlays", type=int, default=12, help="held-out pictures to save with predictions")
    args = parser.parse_args()
    run = run_dir(args.run)
    if not (run / "features.npy").exists():
        raise SystemExit("no features in {}: run `just vision-features` first".format(run))
    frames = json.loads((run / "meta.json").read_text())["frames"]
    episodes = np.array([f["episode"] for f in frames])
    features = np.load(run / "features.npy", mmap_mode="r")
    labels = np.load(run / "patch_labels.npy")
    counts = np.load(run / "patch_counts.npy")

    train_eps, test_eps = split_episodes(episodes, args.test_fraction, args.seed)
    train_images = np.flatnonzero(np.isin(episodes, train_eps))
    test_images = np.flatnonzero(np.isin(episodes, test_eps))
    rng = np.random.default_rng(args.seed)
    x, y = gather_training(features, labels, train_images, CAP, rng)
    print("training on {} patches from {} episodes ({}); testing on {} episodes".format(
        len(y), len(train_eps), ", ".join("{} {}".format(n, (y == i).sum()) for i, n in enumerate(CLASSES)),
        len(test_eps)))
    probe = train_probe(x, y, hidden=args.hidden, epochs=args.epochs, lr=args.lr, seed=args.seed, log=print)

    probs = predict_probs(probe, np.asarray(features[test_images], dtype=np.float32))
    metrics = evaluate(probs, labels[test_images], counts[test_images])
    metrics.update({"hidden": args.hidden, "epochs": args.epochs, "train_patches": int(len(y)),
                    "train_episodes": train_eps.tolist(), "test_episodes": test_eps.tolist()})
    tag = "-mlp{}".format(args.hidden) if args.hidden else ""
    save_probe(probe, run / "probe{}.pt".format(tag))
    (run / "metrics{}.json".format(tag)).write_text(json.dumps(metrics, indent=1))
    picks = rng.choice(len(test_images), min(args.overlays, len(test_images)), replace=False)
    save_overlays(run, frames, test_images[picks], probs[picks], run / "overlays{}".format(tag))
    print(report(metrics))
    print("saved", run / "probe{}.pt".format(tag))


if __name__ == "__main__":
    main()

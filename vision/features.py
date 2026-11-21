"""DINOv2 patch features and one label per patch: just vision-features latest

The frozen `dinov2_vits14_reg` model cuts a picture into 14 x 14 pixel patches and gives 384 numbers per
patch. The 640 x 360 camera picture is resized to 644 x 364 (multiples of 14), which is 26 rows x 46 columns =
1,196 patches. The label of a patch comes from the classes of its 196 pixels (see `patch_labels`).

Output in data/vision/<run>/: features.npy (pictures, 26, 46, 384) float16, patch_labels.npy and
patch_counts.npy (pixels per class in every patch).
"""

import argparse
import json

import numpy as np

from scene import CLASSES, FLOOR, IGNORE, OBSTACLE, TARGET, WALL

PATCH = 14
ROWS, COLS = 26, 46  # 364 / 14 and 644 / 14
DIM = 384
PIXELS = PATCH * PATCH
PURITY = 0.7  # floor, wall, obstacle: skip the patch when its majority class covers less than this
TARGET_PURITY = 0.3  # the thin target pillar: this share of target pixels is enough
MEAN = (0.485, 0.456, 0.406)  # ImageNet statistics, which DINOv2 expects
STD = (0.229, 0.224, 0.225)


def patch_counts(classes):
    """Pixels per class in every patch: uint8 (26, 46, 5); the last channel counts IGNORE pixels.

    `classes` is the per-pixel class map (360, 640). It is resized to 364 x 644 by nearest neighbour, so
    classes are never blended.
    """
    height, width = classes.shape
    rows = np.arange(ROWS * PATCH) * height // (ROWS * PATCH)
    cols = np.arange(COLS * PATCH) * width // (COLS * PATCH)
    big = classes[rows][:, cols].reshape(ROWS, PATCH, COLS, PATCH)
    counts = np.zeros((ROWS, COLS, len(CLASSES) + 1), dtype=np.uint8)
    for c in range(len(CLASSES)):
        counts[..., c] = (big == c).sum(axis=(1, 3))
    counts[..., -1] = PIXELS - counts[..., :-1].sum(axis=-1)
    return counts


def patch_labels(counts, purity=PURITY, target_purity=TARGET_PURITY):
    """One label per patch from `patch_counts`: a class, or IGNORE for patches left out of training.

    - target when at least `target_purity` of the pixels are target;
    - else the majority class, when it covers at least `purity` of the patch and is not "ignore";
    - else IGNORE (a mixed patch on an object's edge, or the robot's own legs).
    """
    share = counts.astype(np.float32) / PIXELS
    majority = share.argmax(axis=-1)
    labels = np.where((majority < len(CLASSES)) & (share.max(axis=-1) >= purity), majority, IGNORE)
    labels = np.where(share[..., TARGET] >= target_purity, TARGET, labels)
    return labels.astype(np.uint8)


def blocked_share(counts):
    """Share of every patch's pixels that are wall or obstacle."""
    return (counts[..., WALL].astype(np.float32) + counts[..., OBSTACLE]) / PIXELS


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_dino(device=None):
    """The frozen DINOv2 ViT-S/14 with registers. Downloads the code and weights (88 MB) on first use."""
    import warnings

    import torch
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="xFormers is not available")  # optional speed-up, not needed
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg", trust_repo=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device or pick_device())


def extract(model, images):
    """Patch features of RGB uint8 pictures (n, h, w, 3): float32 array (n, 26, 46, 384)."""
    import torch
    device = next(model.parameters()).device
    x = torch.from_numpy(np.array(images)).to(device).permute(0, 3, 1, 2).float() / 255.0
    x = torch.nn.functional.interpolate(x, size=(ROWS * PATCH, COLS * PATCH), mode="bilinear", align_corners=False)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        tokens = model.forward_features((x - mean) / std)["x_norm_patchtokens"]
    return tokens.reshape(len(images), ROWS, COLS, DIM).float().cpu().numpy()


def main():
    from PIL import Image

    from dataset import run_dir

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", nargs="?", default="latest", help="run name in data/vision (default: latest)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--purity", type=float, default=PURITY)
    parser.add_argument("--target-purity", type=float, default=TARGET_PURITY)
    args = parser.parse_args()
    run = run_dir(args.run)
    meta = json.loads((run / "meta.json").read_text())
    frames = meta["frames"]
    classes = np.load(run / "labels.npz")["classes"]

    counts = np.stack([patch_counts(c) for c in classes])
    labels = patch_labels(counts, args.purity, args.target_purity)
    np.save(run / "patch_counts.npy", counts)
    np.save(run / "patch_labels.npy", labels)
    total = labels.size
    print("patches: " + ", ".join("{} {:.1%}".format(name, (labels == i).sum() / total)
                                  for i, name in enumerate(CLASSES))
          + ", skipped {:.1%}".format((labels == IGNORE).sum() / total))

    model = load_dino()
    features = np.lib.format.open_memmap(run / "features.npy", mode="w+", dtype=np.float16,
                                         shape=(len(frames), ROWS, COLS, DIM))
    for start in range(0, len(frames), args.batch):
        batch = frames[start:start + args.batch]
        images = np.stack([np.asarray(Image.open(run / f["image"]).convert("RGB")) for f in batch])
        features[start:start + len(batch)] = extract(model, images)
        if (start // args.batch) % 10 == 0:
            print("features {}/{}".format(start + len(batch), len(frames)))
    features.flush()
    (run / "features.json").write_text(json.dumps({"model": "dinov2_vits14_reg", "purity": args.purity,
                                                   "target_purity": args.target_purity}))
    print("saved", run / "features.npy")


if __name__ == "__main__":
    main()

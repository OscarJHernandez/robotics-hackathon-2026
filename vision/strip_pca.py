"""A 3-value PCA of DINOv2 features on the horizon strip: python vision/strip_pca.py full60

The level robot camera sees the lidar's plane on the middle rows of its picture. Only a band of BAND patch
rows around that horizon goes through DINOv2 (features.extract(band=...)); its middle 2 rows, averaged in
blocks of 2 x 2 patches, give 23 cells across the 62 degree view. The PCA is fitted on those cells of every
picture of a vision run and saved as data/vision/<run>/pca3-strip.npz: mean (384,), components (3, 384),
explained variance ratio (3,), band, cells. nav/dagger/featurize.py (perception: strip) uses the same steps.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from features import COLS, DIM, extract, load_dino

BAND = 6  # patch rows through DINOv2: 84 of the 364 pixel rows, centred on the horizon
CELLS = COLS // 2  # 23 cells of 2 x 2 patches across the view
COMPONENTS = 3
DATA = Path(__file__).resolve().parent.parent / "data" / "vision"


def strip_cells(tokens):
    """Band features (n, BAND, 46, 384) -> horizon cells (n, 23, 384): the band's middle 2 rows, 2 x 2 means."""
    mid = tokens.shape[1] // 2
    rows = tokens[:, mid - 1:mid + 1]  # the 2 rows on the horizon
    n = rows.shape[0]
    return rows.reshape(n, 2, CELLS, 2, DIM).mean(axis=(1, 3))


def project(cells, pca):
    """Horizon cells (n, 23, 384) -> PCA values (n, 23, 3)."""
    return (cells - pca["mean"]) @ pca["components"].T


def load_pca(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", default="full60", help="vision run in data/vision with images/")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    folder = DATA / args.run
    paths = sorted((folder / "images").glob("*.png"))
    if not paths:
        raise SystemExit("no pictures in {}".format(folder / "images"))
    model = load_dino()
    cells = []
    for i in range(0, len(paths), args.batch):
        images = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths[i:i + args.batch]])
        cells.append(strip_cells(extract(model, images, band=BAND)))
        print("{} of {} pictures".format(min(i + args.batch, len(paths)), len(paths)), flush=True)
    x = np.concatenate(cells).reshape(-1, DIM).astype(np.float64)
    mean = x.mean(axis=0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    explained = (s ** 2) / (s ** 2).sum()
    out = folder / "pca3-strip.npz"
    np.savez(out, mean=mean.astype(np.float32), components=vt[:COMPONENTS].astype(np.float32),
             explained=explained[:COMPONENTS].astype(np.float32), band=BAND, cells=CELLS)
    print(json.dumps({"pictures": len(paths), "cells": int(x.shape[0]), "explained": explained[:COMPONENTS].round(4).tolist(),
                      "explained_total": round(float(explained[:COMPONENTS].sum()), 4), "saved": str(out)}))


if __name__ == "__main__":
    main()

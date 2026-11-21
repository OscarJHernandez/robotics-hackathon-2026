"""Re-pool a finished DAgger run's rows to a coarser patch grid, without the simulator or DINOv2.

    uv run --group ml --group vision python -m nav.dagger.repool --run dagger1 --out dagger1-pool2 --pool 2

Every round_<R>.parquet in data/nav/dagger/<run>/ is rewritten into data/nav/dagger/<out>/ with the patch
columns of featurize.columns(pool): per block of pool x pool patches the majority class and the mean depth
(as featurize.pool_grid does live), and the obstacle summary recomputed from the pooled grid (as the live
featurizer does). Goal columns and bookkeeping columns are copied unchanged, so the rows equal what a run
with `patch_pool: <pool>` would have written. The round_<R>.json, rounds.json and timings.json files and the
plans/ folder are copied as they are. The source run must have been made with patch_pool 1.
"""

import argparse
import shutil

import numpy as np
import pandas as pd

from .config import run_dir, runs_dir
from .featurize import COLS, LEG, ROWS, SUMMARY, columns, grid_shape, obstacle_summary


def pool_arrays(classes, depths, pool):
    """Batched pool_grid: classes (n, 26, 46) int, depths (n, 26, 46) -> (n, rows, cols) majority class, mean depth."""
    if pool == 1:
        return classes, depths
    rows, cols = grid_shape(pool)
    if (rows * pool, cols * pool) != (ROWS, COLS):
        raise SystemExit("pool {} does not divide the {} x {} grid".format(pool, ROWS, COLS))
    n = len(classes)
    blocks_c = classes.reshape(n, rows, pool, cols, pool)
    counts = np.stack([(blocks_c == k).sum(axis=(2, 4)) for k in range(LEG + 1)], axis=-1)
    pooled_c = counts.argmax(-1).astype(np.int8)  # ties go to the lower class, like np.bincount(...).argmax()
    pooled_d = depths.reshape(n, rows, pool, cols, pool).mean(axis=(2, 4), dtype=np.float64).astype(np.float32)
    return pooled_c, pooled_d


def repool_frame(table, pool, depth_cap=10.0):
    """A round's rows with patch_pool 1 columns -> the same rows with patch_pool `pool` columns."""
    src_feature, src_classes = columns(1)
    src_depths = src_feature[len(src_classes):len(src_classes) * 2]
    missing = [c for c in src_feature[:-len(SUMMARY)] if c not in table]
    if missing:
        raise SystemExit("not a patch_pool 1 table: {} columns missing, e.g. {}".format(len(missing), missing[0]))
    n = len(table)
    classes = table[src_classes].to_numpy().astype(np.int8).reshape(n, ROWS, COLS)
    depths = table[src_depths].to_numpy().astype(np.float32).reshape(n, ROWS, COLS)
    pooled_c, pooled_d = pool_arrays(classes, depths, pool)

    feature, cat = columns(pool)
    new_c, new_d = cat, feature[len(cat):len(cat) * 2]
    out = pd.DataFrame(pooled_c.reshape(n, -1), columns=new_c, index=table.index).astype(np.int8)
    out = pd.concat([out, pd.DataFrame(pooled_d.reshape(n, -1), columns=new_d, index=table.index)], axis=1)
    summaries = [obstacle_summary(c, d, depth_cap) for c, d in zip(pooled_c, pooled_d)]
    for key in SUMMARY:
        out[key] = np.asarray([s[key] for s in summaries], np.float32)
    rest = [c for c in table.columns if c not in set(src_classes) | set(src_depths) | set(SUMMARY)]
    out = pd.concat([out, table[rest]], axis=1)
    order = feature + [c for c in out.columns if c not in set(feature)]  # featurize's column order, then the rest
    return out[[c for c in order if c in out.columns]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="source run in data/nav/dagger/ (made with patch_pool 1)")
    parser.add_argument("--out", required=True, help="new run name for the pooled rows")
    parser.add_argument("--pool", type=int, default=2)
    parser.add_argument("--depth-cap", type=float, default=10.0, help="as features.depth_cap in dagger.yaml")
    args = parser.parse_args()
    src, dst = runs_dir() / args.run, runs_dir() / args.out
    if not any(src.glob("round_*.parquet")):
        raise SystemExit("no round_*.parquet in {}".format(src))
    if dst.exists():
        raise SystemExit("{} already exists; pick another --out".format(dst))
    dst = run_dir(args.out)
    for path in sorted(src.glob("round_*.parquet")):
        table = pd.read_parquet(path)
        pooled = repool_frame(table, args.pool, args.depth_cap)
        pooled.to_parquet(dst / path.name, index=False)
        print("{}: {} rows, {} -> {} columns".format(path.name, len(table), table.shape[1], pooled.shape[1]))
    for name in ["rounds.json", "timings.json"] + [p.name for p in src.glob("round_*.json")]:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    if (src / "plans").is_dir():
        shutil.copytree(src / "plans", dst / "plans")
    (dst / "SOURCE.txt").write_text("Rows of run {} re-pooled with nav/dagger/repool.py --pool {} "
                                    "(patch_pool {}).\n".format(args.run, args.pool, args.pool))
    print("written to {}".format(dst))


if __name__ == "__main__":
    main()

"""Export a student's context rows once, so it can be fitted elsewhere (the app, the TabPFN client) without the raw data.

  python -m nav.dagger.export --variant nav/dagger/variants/v29-lidarplan.yaml --out <folder>

Reads the variant's context rows exactly as `experiment.build_student` does (load_rows, choose_context,
Features.fit_transform), but does NOT fit anything. Writes into <folder>:

  features.parquet  the student's input columns (in fit order) + y_stride, y_turn, float32
  pca.npz           the band PCA learnt by Features.fit_transform: mean, scale, comps, var_ratios
  columns.json      {"columns", "all", "categorical", "features" (the variant's feature config)}
  meta.json         rows, variant, sources, git sha, host, time
  sample.parquet + sample_band.npy
                    a few stored base rows and their band arrays, so a test can check that the drive-time path
                    (`Features.live` with the saved PCA) gives the exported rows back

`load_export(folder)` returns (Features ready for `live`/`transform`, the feature table, meta).
Absolute imports, so the file also runs as a plain script from outside the repo (with the repo on PYTHONPATH).
No TabPFN here: these are transforms of the inputs (AGENTS.md: fit()/predict() only, never fine-tuning).
"""

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path

SAMPLE_ROWS = 12


def _git_sha(folder):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=folder, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def export(variant, out, sample_rows=SAMPLE_ROWS, seed=0):
    import numpy as np
    import pandas as pd

    from nav.dagger import REPO
    from nav.dagger.experiment import load_rows
    from nav.dagger.featurize import Features
    from nav.dagger.policy import choose_context

    ctx_cfg = variant["context"]
    features = Features(variant.get("features"))
    tables, bands = [], []
    for name in ctx_cfg["sources"]:
        rows, band = load_rows(name, features.needs_band)
        rows["_source"], rows["_row"] = name, np.arange(len(rows))
        tables.append(rows)
        bands.append(band)
    table = pd.concat(tables, ignore_index=True)
    table["_all"] = np.arange(len(table))
    context = choose_context(table, ctx_cfg.get("max_rows"), ctx_cfg.get("seed", 0),
                             quotas=ctx_cfg.get("quotas", "uniform"))
    band = None
    if features.needs_band:
        offsets = np.cumsum([0] + [len(t) for t in tables])
        src = np.searchsorted(offsets, context["_all"].to_numpy(), side="right") - 1
        band = np.stack([bands[s][r] for s, r in zip(src, context["_row"].to_numpy())])
    X = features.fit_transform(context, band)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(X.astype(np.float32), columns=features.columns)
    frame["y_stride"] = context["y_stride"].to_numpy(np.float32)
    frame["y_turn"] = context["y_turn"].to_numpy(np.float32)
    frame.to_parquet(out / "features.parquet", index=False)
    if features.pca is not None:
        mean, scale, comps, ratios = features.pca
        np.savez(out / "pca.npz", mean=mean, scale=scale, comps=comps, var_ratios=np.asarray(ratios, np.float64))
    (out / "columns.json").write_text(json.dumps({
        "columns": features.columns, "all": features._all, "categorical": features.categorical,
        "keep": [bool(k) for k in features.keep], "features": features.cfg}, indent=1))

    # A few base rows (the stored columns the live path reads) + their band arrays, for the round-trip test.
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(context), min(sample_rows, len(context)), replace=False))
    base_cols = [c for c in context.columns if not c.startswith("_")]
    sample = context.iloc[pick][base_cols].reset_index(drop=True)
    sample["_export_row"] = pick
    sample.to_parquet(out / "sample.parquet", index=False)
    if band is not None:
        np.save(out / "sample_band.npy", np.asarray(band[pick]))

    meta = {"rows": int(len(frame)), "columns": len(features.columns), "variant": variant["id"],
            "sources": {t["_source"].iloc[0]: int(len(t)) for t in tables}, "context": ctx_cfg,
            "git_sha": _git_sha(REPO), "host": socket.gethostname(), "repo": str(REPO),
            "exported": time.strftime("%Y-%m-%dT%H:%M:%S"), "describe": features.describe(),
            "sample_rows": [int(p) for p in pick]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def load_export(folder):
    """(features, table, meta): a `Features` set up as at fit time (columns, mask, PCA), ready for `live`."""
    import numpy as np
    import pandas as pd

    from nav.dagger.featurize import Features
    folder = Path(folder)
    cols = json.loads((folder / "columns.json").read_text())
    meta = json.loads((folder / "meta.json").read_text())
    features = Features(cols["features"])
    features.columns = list(cols["columns"])
    features.categorical = list(cols["categorical"])
    features._all = list(cols["all"])
    features.keep = np.asarray(cols["keep"], bool)
    if (folder / "pca.npz").exists():
        p = np.load(folder / "pca.npz")
        features.pca = (p["mean"].astype(np.float32), p["scale"].astype(np.float32), p["comps"].astype(np.float32),
                        p["var_ratios"].tolist())
    table = pd.read_parquet(folder / "features.parquet")
    if list(table.columns[:len(features.columns)]) != features.columns:
        raise ValueError("{}: features.parquet columns do not match columns.json".format(folder))
    return features, table, meta


def main():
    from nav.dagger.experiment import load_variant
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--variant", default=str(Path(__file__).resolve().parent / "variants" / "v29-lidarplan.yaml"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    started = time.time()
    meta = export(load_variant(args.variant), args.out)
    print("exported {} rows x {} columns of {} to {} in {:.0f} s".format(meta["rows"], meta["columns"], meta["variant"],
                                                                       args.out, time.time() - started), flush=True)


if __name__ == "__main__":
    main()

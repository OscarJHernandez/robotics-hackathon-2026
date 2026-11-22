"""Smoke test: can TabPFN learn from this data? A small check, not the full evaluation.

    uv run --project <team repo> --group ml python homes/check.py                       # 10 test residents
    uv run --project <team repo> --group ml python homes/check.py --residents 4 --own-days 3

For each held-out resident (never in the cold-start prior or the context), TabPFN's built-in fit() takes a small
context and predict_proba() answers her test check-ins; nothing is trained and no weights change (rule in AGENTS.md).
The context holds `--context-rows` rows: the resident's own first k days (own = 1) plus rows of TRAIN residents whose
flat has the same set of zones (own = 0), including up to 100 of their "out" rows (rare, so kept on purpose).
Her own-history columns (hist_*) are recomputed from those k days only, so k really is "days of her diary".

Compared with, using the same k own days: "last heard" (last_heard_zone), "usual place in this slot" (her own rows
in the context, hour and weekday type) and the cold-start prior. Two models, as in docs/knock-routines.md: without
the voice columns for check-ins, and with them (heard rows only) for help calls. Prints top-1, log loss and the
seconds per fit.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from flats import CLASSES, signature, zone_names
from generate import HIST, VOICE_FEATURES, _history, feature_frame
from prior import ColdStartPrior

DATA = Path(__file__).resolve().parent / "data"
MODEL_VERSION = "v2"  # TabPFN v2 weights, as in nav/policy.py
CODE = {c: i for i, c in enumerate(CLASSES)}
EPS = 1e-3


def fit_classifier(X, y, seed=0):
    """A pretrained TabPFN classifier with (X, y) as its context: TabPFN's own fit(), nothing else."""
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion(MODEL_VERSION), device="cpu", n_estimators=4, random_state=seed,
        fit_mode="fit_with_cache", ignore_pretraining_limits=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(np.asarray(X, np.float32), np.asarray(y))
    return model


def restrict(proba, classes, allowed):
    """Keep the classes this flat can answer (its zones and 'out') and renormalize; returns an array over CLASSES."""
    out = np.zeros((len(proba), len(CLASSES)))
    for j, c in enumerate(classes):
        if c in allowed:
            out[:, CODE[c]] = proba[:, j]
    s = out.sum(axis=1, keepdims=True)
    return out / np.where(s > 0, s, 1.0)


def score(proba, truth):
    """(top-1, log loss) of probabilities over CLASSES against true class names."""
    t = np.array([CODE[z] for z in truth])
    top1 = float((proba.argmax(axis=1) == t).mean())
    loss = float(-np.log(np.clip(proba[np.arange(len(t)), t], EPS, 1.0)).mean())
    return top1, loss


def cold_proba(prior, flat, q):
    out = np.zeros((len(q), len(CLASSES)))
    for i, (hour, wd) in enumerate(zip(q["hour"].to_numpy(), q["weekday"].to_numpy())):
        zones, p_out = prior.probs(flat, float(hour), int(wd))
        for name, p in zones.items():
            out[i, CODE[name]] = p * (1 - p_out)
        out[i, CODE["out"]] = p_out
    return out


def usual_pred(own, q, fallback):
    """The most frequent logged zone in her own rows for this hour and weekday type (else her overall mode)."""
    if own.empty:
        return list(fallback)
    key = own.groupby(["slot", "is_weekend"])["zone"].agg(lambda s: s.value_counts().index[0])
    overall = own["zone"].value_counts().index[0]
    return [key.get((int(sl), int(w)), overall) for sl, w in zip(q["slot"], q["is_weekend"])]


def one_resident(rid, k, args, tables):
    diary, checkins, train, flats, prior, sigs = tables
    flat = flats[rid]
    sig = signature(flat)
    allowed = set(zone_names(flat)) | {"out"}
    own = diary[(diary["resident_id"] == rid) & (diary["day"] < k)]  # her first k diary days (3 rows each)
    pool = train[train["flat_id"].map(sigs) == sig]
    rng = np.random.default_rng(rid)
    out_rows = pool[pool["zone"] == "out"]
    out_rows = out_rows.iloc[rng.permutation(len(out_rows))[:100]]
    rest = pool[pool["zone"] != "out"]
    n_rest = max(0, args.context_rows - len(own) - len(out_rows))
    rest = rest.iloc[rng.permutation(len(rest))[:n_rest]]
    results = {}
    for kind, variant in (("checkin", "no_voice"), ("help", "voice")):
        q = checkins[(checkins["resident_id"] == rid) & (checkins["kind"] == kind)].reset_index(drop=True)
        ctx_own, ctx_other = own, pd.concat([out_rows, rest])
        if variant == "voice":  # only heard rows have a voice
            ctx_own, ctx_other = ctx_own[ctx_own["source"] == "heard"], ctx_other[ctx_other["source"] == "heard"]
        ctx = pd.concat([ctx_own, ctx_other])
        Xc, Xq = feature_frame(ctx), feature_frame(q)
        # Her own-history columns from only her first k days (the context rows keep theirs).
        if k > 0:
            codes = own["zone"].map(CODE).to_numpy()
            share, n = _history(q["day"].to_numpy(), q["slot"].to_numpy(), q["is_weekend"].to_numpy(),
                                own["day"].to_numpy(), own["slot"].to_numpy(), own["is_weekend"].to_numpy(), codes)
            for i, c in enumerate(CLASSES):
                Xq["hist_" + c] = share[:, i].astype("float32")
            Xq["hist_n"] = n.astype("float32")
        else:
            Xc, Xq = Xc.drop(columns=HIST + ["hist_n"]), Xq.drop(columns=HIST + ["hist_n"])
        if variant == "no_voice":
            drop = [c for c in Xc.columns if c in VOICE_FEATURES or c.startswith("vzg_")]
            Xc, Xq = Xc.drop(columns=drop), Xq.drop(columns=drop)
        if k > 0:
            Xc["own"] = np.r_[np.ones(len(ctx_own)), np.zeros(len(ctx_other))].astype("float32")
            Xq["own"] = np.float32(1.0)
        keep = [c for c in Xc.columns if Xc[c].nunique(dropna=False) > 1]  # constant columns teach nothing
        start = time.time()
        model = fit_classifier(Xc[keep].to_numpy(), ctx["zone"].to_numpy())
        fit_s = time.time() - start
        proba = restrict(model.predict_proba(Xq[keep].to_numpy(np.float32)), list(model.classes_), allowed)
        truth = q["zone"].tolist()
        cold = cold_proba(prior, flat, q)
        usual = usual_pred(own, q, [CLASSES[i] for i in cold.argmax(axis=1)])  # no own rows: the cold-start answer
        row = {"tabpfn": score(proba, truth), "cold_start": score(cold, truth),
               "last_heard": (float((q["last_heard_zone"] == q["zone"]).mean()), None),
               "usual_place": (float(np.mean([a == b for a, b in zip(usual, truth)])), None),
               "fit_seconds": fit_s, "context_rows": int(len(ctx)), "features": len(keep), "queries": len(q)}
        results[kind] = row
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--residents", type=int, default=10)
    parser.add_argument("--own-days", type=int, nargs="+", default=[0, 3])  # of the 5 diary days
    parser.add_argument("--context-rows", type=int, default=1000)
    args = parser.parse_args()
    data = Path(args.data)
    diary = pd.read_parquet(data / "diary.parquet")
    checkins = pd.read_parquet(data / "checkins.parquet")
    flats = {f["flat_id"]: f for f in json.loads((data / "flats.json").read_text())}
    sigs = {fid: signature(f) for fid, f in flats.items()}
    prior = ColdStartPrior.load(data / "cold_prior.json")
    train = diary[diary["resident_split"] == "train"]
    tables = (diary, checkins, train, flats, prior, sigs)
    test_ids = sorted(diary.loc[diary["resident_split"] == "test", "resident_id"].unique())[: args.residents]

    summary = {}
    for k in args.own_days:
        agg = {}
        for rid in test_ids:
            for kind, row in one_resident(int(rid), k, args, tables).items():
                for name in ("tabpfn", "cold_start", "last_heard", "usual_place"):
                    agg.setdefault((kind, name), []).append(row[name])
                agg.setdefault((kind, "fit_s"), []).append(row["fit_seconds"])
                agg.setdefault((kind, "features"), []).append(row["features"])
        for kind in ("checkin", "help"):
            line = {}
            for name in ("tabpfn", "cold_start", "last_heard", "usual_place"):
                vals = agg[(kind, name)]
                line[name] = {"top1": round(float(np.mean([v[0] for v in vals])), 3),
                              "log_loss": None if vals[0][1] is None else round(float(np.mean([v[1] for v in vals])), 3)}
            line["fit_seconds_mean"] = round(float(np.mean(agg[(kind, "fit_s")])), 1)
            line["features"] = int(np.mean(agg[(kind, "features")]))
            summary["{}_own_days_{}".format(kind, k)] = line
            print("{:8s} own days {:2d}: TabPFN top-1 {:.3f} (log loss {:.3f}) | cold-start {:.3f} ({:.3f}) | "
                  "last heard {:.3f} | usual place {:.3f} | fit {:.1f}s, {} features".format(
                      kind, k, line["tabpfn"]["top1"], line["tabpfn"]["log_loss"], line["cold_start"]["top1"],
                      line["cold_start"]["log_loss"], line["last_heard"]["top1"], line["usual_place"]["top1"],
                      line["fit_seconds_mean"], line["features"]))
    (data / "check_results.json").write_text(json.dumps(
        {"residents": len(test_ids), "context_rows": args.context_rows, "results": summary}, indent=1))
    print("wrote", data / "check_results.json")


if __name__ == "__main__":
    main()

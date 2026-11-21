"""Generate the Knock training data: random flats, random residents, their diaries and test check-ins.

    uv run --project <team repo> --group ml python homes/generate.py                  # 8500 residents -> data/
    uv run --project <team repo> --group ml python homes/generate.py --residents 50   # a quick small run

Each resident lives in her own flat (the MuJoCo arena with five named zones, flats.py) and has a weekly routine table (residents.py). The data covers 7 days with
exactly 3 diary records a day: one at a random time in the morning (07:00-12:00), afternoon (12:00-18:00) and evening
(18:00-22:30) slot. A record is what the robot learns at that moment: if she is home it hears her and logs a zone
(from the direction and loudness of the sound; 10% land in a neighbouring zone), if she is out it finds nobody and logs
`out`. Days 0-4 are the diary, days 5-6 the test check-ins. The same seed gives the same files, and resident i does not
change when the number of residents does. Only past information goes into a row.
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from flats import CLASSES, ZONE_NAMES, bearing_from_dock, random_flat, zone_names
from residents import CODE, OUT, draw_day, sample_persona, slot_of, windows

DATA = Path(__file__).resolve().parent / "data"
HEARD_MISPLACED = 0.1  # share of heard records logged in a neighbouring zone
HIST_ALPHA = 0.1
# the `slot` value of a help call at night is n (the number of slots)
CLASS_ARR = np.array(CLASSES, dtype=object)
HIST = ["hist_" + c for c in CLASSES]
SPEC_COLUMNS = ["day", "resident", "source", "hour", "weekday", "last_heard_zone", "min_since_heard",
                "voice_bearing_deg", "voice_loudness_db", "zone"]
EXTRA_COLUMNS = ["slot", "is_weekend", "voice_zone_guess", "voice_dist_m"] + HIST + ["hist_n"]
ID_COLUMNS = ["resident_id", "resident_name", "resident_split", "flat_id", "archetype"]
NUMERIC_FEATURES = ["hour", "weekday", "is_weekend", "slot", "min_since_heard", "voice_bearing_deg",
                    "voice_loudness_db", "voice_dist_m"] + HIST + ["hist_n"]
VOICE_FEATURES = ["voice_bearing_deg", "voice_loudness_db", "voice_dist_m"]
DIARY_COLUMNS = SPEC_COLUMNS + EXTRA_COLUMNS + ID_COLUMNS + ["true_zone"]
CHECKIN_COLUMNS = SPEC_COLUMNS + EXTRA_COLUMNS + ID_COLUMNS + ["kind", "x", "y"]


# --- hearing ---

def _person_xy(flat, name, rng):
    """Where she is in a zone: near one of its spots (the couch or the armchair in the living room)."""
    z = next(z for z in flat["zones"] if z["name"] == name)
    spot = z["spots"][int(rng.integers(len(z["spots"])))]
    w, d = flat["width"] / 2 - 0.05, flat["depth"] / 2 - 0.05
    return (float(np.clip(spot["person_x"] + rng.normal(0, 0.1), -w, w)),
            float(np.clip(spot["person_y"] + rng.normal(0, 0.1), -d, d)))


def _voice(flat, x, y, rng):
    """(bearing in degrees from the dock, loudness in dB): what the microphones report about a sound at (x, y)."""
    dist = max(math.hypot(x - flat["dock"]["x"], y - flat["dock"]["y"]), 0.3)
    bearing = (bearing_from_dock(flat, x, y) + rng.normal(0, 25)) % 360.0
    return bearing, 65 - 20 * math.log10(dist) + rng.normal(0, 3)


def _zone_guess(flat, bearings):
    """The flat zone closest in direction to each measured bearing (a map-independent feature)."""
    names = zone_names(flat)
    ref = np.array([bearing_from_dock(flat, z["person_x"], z["person_y"]) for z in flat["zones"]])
    out = np.empty(len(bearings), object)
    for i, b in enumerate(bearings):
        out[i] = None if np.isnan(b) else names[int(np.argmin(np.abs((ref - b + 180.0) % 360.0 - 180.0)))]
    return out


# --- features ---

def _history(q_day, q_slot, q_wk, s_day, s_slot, s_wk, s_zone):
    """Own history: smoothed share of her earlier-day records in the same slot and weekday type, per class."""
    k = len(CLASSES)
    if len(s_day) == 0:
        return np.full((len(q_day), k), 1.0 / k), np.zeros(len(q_day))
    onehot = np.eye(k, dtype=np.float32)[s_zone]
    mask = (s_day[None, :] < q_day[:, None]) & (s_wk[None, :] == q_wk[:, None]) & (s_slot[None, :] == q_slot[:, None])
    counts = mask.astype(np.float32) @ onehot
    n = counts.sum(axis=1)
    return (counts + HIST_ALPHA) / (n[:, None] + HIST_ALPHA * k), n


def _frame(flat, p, a, share, hist_n, last_zone, min_since, target):
    n = len(target)
    f = {
        "day": a["day"], "resident": np.full(n, p["resident"], object), "source": a["source"],
        "hour": np.round(a["hour"], 3), "weekday": a["weekday"].astype(np.int8),
        "last_heard_zone": np.where(last_zone >= 0, CLASS_ARR[np.maximum(last_zone, 0)], None),
        "min_since_heard": np.round(min_since, 1),
        "voice_bearing_deg": np.round(a["vb"], 1), "voice_loudness_db": np.round(a["vl"], 1),
        "zone": CLASS_ARR[target],
        "slot": a["slot"].astype(np.int8), "is_weekend": (a["weekday"] >= 5).astype(np.int8),
        "voice_zone_guess": _zone_guess(flat, a["vb"]), "voice_dist_m": np.round(10 ** ((65 - a["vl"]) / 20), 2),
    }
    for i, c in enumerate(CLASSES):
        f["hist_" + c] = np.round(share[:, i], 4).astype(np.float32)
    f["hist_n"] = hist_n.astype(np.float32)
    f["resident_id"] = np.full(n, p["resident_id"], np.int32)
    f["resident_name"] = np.full(n, p["name"], object)
    f["resident_split"] = np.full(n, p["resident_split"], object)
    f["flat_id"] = np.full(n, p["flat_id"], np.int32)
    f["archetype"] = np.full(n, p["archetype"], object)
    return pd.DataFrame(f)


# --- one resident ---

def build_resident(task):
    """Flat, persona, diary rows (days before the test days) and test check-ins of one resident."""
    idx, seed_seq, days, test_days, n = task
    s_flat, s_persona, s_sim, s_query = seed_seq.spawn(4)
    flat = random_flat(np.random.default_rng(s_flat), idx + 1)
    p = sample_persona(np.random.default_rng(s_persona), idx, flat, n)
    rng = np.random.default_rng(s_sim)
    first_test = days - test_days
    neighbours = {z["name"]: z["neighbours"] for z in flat["zones"]}

    keys = ("abs", "day", "slot", "hour", "weekday", "true", "logged", "source", "vb", "vl")
    rows = {k: [] for k in keys}
    day_codes = []
    for day in range(days):
        codes = draw_day(p, day, rng)
        day_codes.append(codes)
        weekday = (p["start_weekday"] + day) % 7
        for slot, (a, b) in enumerate(windows(n)):
            t = float(rng.uniform(a, b))
            true = codes[slot]
            logged, vb, vl, source = true, np.nan, np.nan, "checkin"  # out: the robot finds nobody
            if true != OUT:
                source = "heard"
                name = CLASSES[true]
                if rng.random() < HEARD_MISPLACED:
                    logged = CODE[neighbours[name][int(rng.integers(len(neighbours[name])))]]
                x, y = _person_xy(flat, name, rng)
                vb, vl = _voice(flat, x, y, rng)
            for key, val in zip(keys, (day * 1440 + t, day, slot, t / 60.0, weekday, true, logged, source, vb, vl)):
                rows[key].append(val)
    s = {k: np.array(v) for k, v in rows.items()}
    s["source"] = s["source"].astype(object)
    s_wk = (s["weekday"] >= 5).astype(int)

    def previous(abs_q):
        i = np.searchsorted(s["abs"], abs_q, side="left") - 1
        last = np.where(i >= 0, s["logged"][np.maximum(i, 0)], -1)
        return last, np.where(i >= 0, abs_q - s["abs"][np.maximum(i, 0)], np.nan)

    share, hist_n = _history(s["day"], s["slot"], s_wk, s["day"], s["slot"], s_wk, s["logged"])
    last, since = previous(s["abs"])
    keep = s["day"] < first_test
    diary = _frame(flat, p, {k: s[k][keep] for k in ("day", "source", "hour", "weekday", "vb", "vl", "slot")},
                   share[keep], hist_n[keep], last[keep], since[keep], s["logged"][keep])
    diary["true_zone"] = CLASS_ARR[s["true"][keep]]

    # Test check-ins: 20 "checkin" (nobody answers) and 5 "help" (she calls; voice filled) per test day.
    rq = np.random.default_rng(s_query)
    q = {k: [] for k in ("abs", "day", "slot", "hour", "weekday", "true", "kind", "x", "y", "vb", "vl")}
    for day in range(first_test, days):
        weekday = (p["start_weekday"] + day) % 7
        for kind in ["checkin"] * 20 + ["help"] * 5:
            vb = vl = np.nan
            if kind == "checkin":
                t = float(rq.uniform(420, 1350))
                slot = slot_of(int(t), n)
                true = day_codes[day][slot]
            else:
                for _ in range(20):
                    t = float(rq.uniform(0, 1440))
                    slot = slot_of(int(t), n)
                    true = CODE["bedroom"] if slot is None else day_codes[day][slot]  # night: asleep at home
                    if true != OUT:
                        break
                else:
                    continue
            x = y = np.nan
            if true != OUT:
                x, y = _person_xy(flat, CLASSES[true], rq)
            if kind == "help":
                vb, vl = _voice(flat, x, y, rq)
            for key, val in (("abs", day * 1440 + t), ("day", day), ("slot", n if slot is None else slot),
                             ("hour", t / 60.0), ("weekday", weekday), ("true", true), ("kind", kind), ("x", x),
                             ("y", y), ("vb", vb), ("vl", vl)):
                q[key].append(val)
    qa = {k: np.array(v) for k, v in q.items()}
    share_q, hist_nq = _history(qa["day"], qa["slot"], (qa["weekday"] >= 5).astype(int), s["day"], s["slot"], s_wk,
                                s["logged"])
    last_q, since_q = previous(qa["abs"])
    qa["source"] = qa["kind"].astype(object)
    checkins = _frame(flat, p, qa, share_q, hist_nq, last_q, since_q, qa["true"].astype(int))
    checkins["kind"] = qa["kind"].astype(object)
    checkins["x"] = np.round(qa["x"].astype(float), 3)
    checkins["y"] = np.round(qa["y"].astype(float), 3)
    return {"flat": flat, "persona": p, "diary": diary, "checkins": checkins}


# --- the whole population ---

def build(residents=6, days=7, seed=0, test_days=2, workers=1, records_per_day=3):
    seeds = np.random.SeedSequence(seed).spawn(residents)
    tasks = [(i, seeds[i], days, test_days, records_per_day) for i in range(residents)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(build_resident, tasks, chunksize=32))
    else:
        results = [build_resident(t) for t in tasks]
    return {"flats": [r["flat"] for r in results], "personas": [r["persona"] for r in results],
            "diary": pd.concat([r["diary"] for r in results], ignore_index=True),
            "checkins": pd.concat([r["checkins"] for r in results], ignore_index=True)}


def generate(residents=6, days=7, seed=0, test_days=2, workers=1, records_per_day=3):
    """Returns (diary_rows, test_checkins). diary_rows are the days before the test days, 3 a day per resident."""
    out = build(residents, days, seed, test_days, workers, records_per_day)
    return out["diary"][DIARY_COLUMNS], out["checkins"][CHECKIN_COLUMNS]


def feature_frame(df):
    """The all-numeric table TabPFN takes: numeric features plus one-hot last_heard_zone and voice_zone_guess."""
    X = df[NUMERIC_FEATURES].astype("float32").copy()
    for c in CLASSES:
        X["lhz_" + c] = (df["last_heard_zone"] == c).astype("float32")
    for c in ZONE_NAMES:
        X["vzg_" + c] = (df["voice_zone_guess"] == c).astype("float32")
    return X


# --- files ---

def residents_table(personas, flats):
    rows = []
    for p, f in zip(personas, flats):
        row = {k: v for k, v in p.items() if k not in ("tables", "out_p")}
        row.update({"n_zones": len(f["zones"]), "zones": ",".join(zone_names(f)), "room_w": f["width"],
                    "room_d": f["depth"], "room_area_m2": f["area_m2"],
                    "tables": json.dumps(p["tables"]), "out_p": json.dumps(p["out_p"])})
        rows.append(row)
    return pd.DataFrame(rows)


def zones_table(flats):
    rows = []
    for f in flats:
        for z in f["zones"]:
            rows.append(dict({k: v for k, v in z.items() if k not in ("neighbours", "furniture", "spots")},
                             flat_id=f["flat_id"], furniture=",".join(z["furniture"]), n_spots=len(z["spots"]),
                             neighbours=",".join(z["neighbours"]), room_w=f["width"], room_d=f["depth"]))
    return pd.DataFrame(rows)


def baselines(diary, checkins):
    """'Last heard' and 'usual place in this slot' hits on the test check-ins (usual place: her own diary rows)."""
    usual = (diary.groupby(["resident_id", "slot", "is_weekend"])["zone"]
             .agg(lambda s: s.value_counts().index[0]).rename("usual"))
    overall = diary.groupby("resident_id")["zone"].agg(lambda s: s.value_counts().index[0]).rename("overall")
    c = checkins.join(usual, on=["resident_id", "slot", "is_weekend"]).join(overall, on="resident_id")
    c["usual"] = c["usual"].fillna(c["overall"])
    c["last_hit"] = (c["last_heard_zone"] == c["zone"]).astype(float)
    c["usual_hit"] = (c["usual"] == c["zone"]).astype(float)
    c["moved"] = (c["last_heard_zone"] != c["zone"]).astype(float)
    return c


def summarize(diary, checkins, personas, seconds):
    train = diary[diary["resident_split"] == "train"]
    c = baselines(diary, checkins[checkins["kind"] == "checkin"])

    def block(g):
        return {"rows": int(len(g)), "zone_differs_from_last_heard": round(float(g["moved"].mean()), 3),
                "last_heard_top1": round(float(g["last_hit"].mean()), 3),
                "usual_place_top1": round(float(g["usual_hit"].mean()), 3),
                "share_out": round(float((g["zone"] == "out").mean()), 3)}

    return {
        "seconds": round(seconds, 1), "residents": len(personas),
        "train_residents": int(sum(p["resident_split"] == "train" for p in personas)),
        "diary_rows": int(len(diary)), "train_rows": int(len(train)), "diary_rows_per_resident": round(len(diary) / len(personas), 1),
        "checkins": {k: int(v) for k, v in checkins["kind"].value_counts().items()},
        "class_share_train": {k: round(float(v), 4) for k, v in train["zone"].value_counts(normalize=True).items()},
        "rows_per_slot_train": {int(k): int(v) for k, v in train["slot"].value_counts().sort_index().items()},
        "test_checkins_all_residents": block(c),
        "by_archetype": {k: block(g) for k, g in c.groupby("archetype")},
        "by_slot": {int(k): block(g) for k, g in c.groupby("slot")},
    }


def write(out, data_dir, csv=False, started=None):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    diary, checkins, flats = out["diary"][DIARY_COLUMNS], out["checkins"][CHECKIN_COLUMNS], out["flats"]
    (data_dir / "flats.json").write_text(json.dumps(flats))
    zones_table(flats).to_parquet(data_dir / "zones.parquet", index=False)
    residents_table(out["personas"], flats).to_parquet(data_dir / "residents.parquet", index=False)
    diary.to_parquet(data_dir / "diary.parquet", index=False)
    checkins.to_parquet(data_dir / "checkins.parquet", index=False)
    train = diary[diary["resident_split"] == "train"].reset_index(drop=True)
    table = pd.concat([train[["resident_id", "day", "source"]], feature_frame(train), train[["zone"]]], axis=1)
    table.to_parquet(data_dir / "train.parquet", index=False)
    (data_dir / "train_columns.json").write_text(json.dumps(
        {"features": list(feature_frame(train.head(1)).columns), "target": "zone", "classes": list(CLASSES),
         "voice_features": VOICE_FEATURES + ["vzg_" + c for c in ZONE_NAMES]}, indent=1))
    summary = summarize(diary, checkins, out["personas"], time.time() - started if started else 0.0)
    summary["zones_per_flat"] = {int(k): int(v) for k, v in
                                 pd.Series([len(f["zones"]) for f in flats]).value_counts().sort_index().items()}
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    if csv:
        for name, df in (("diary", diary), ("checkins", checkins), ("train", table)):
            df.to_csv(data_dir / (name + ".csv"), index=False)
        residents_table(out["personas"], flats).to_csv(data_dir / "residents.csv", index=False)
        zones_table(flats).to_csv(data_dir / "zones.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--residents", type=int, default=None,
                        help="default: enough for about 102,000 training rows (8500 at 3 records a day)")
    parser.add_argument("--records-per-day", type=int, default=3,
                        help="records a day, one per slot at a random time (3: morning, afternoon, evening; more: finer slots)")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--out", default=str(DATA))
    parser.add_argument("--csv", action="store_true", help="also write CSV copies")
    args = parser.parse_args()
    started = time.time()
    if args.residents is None:
        args.residents = int(np.ceil(102000 / (0.8 * (args.days - args.test_days) * args.records_per_day)))
    out = build(args.residents, args.days, args.seed, args.test_days, args.workers, args.records_per_day)
    summary = write(out, args.out, csv=args.csv, started=started)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("by_archetype", "by_slot")}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

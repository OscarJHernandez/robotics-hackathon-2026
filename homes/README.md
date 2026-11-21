# homes: synthetic training data for Knock (v2, simple)

Random one-room flats (3–5 named zones), random residents with a small weekly routine, a diary of **3 records a day for 7
days**, test check-ins, and a cold-start zone prior. Everything is simulated and drawn from a seed: the same seed gives the
same files. v1 (a minute-by-minute day with trips, naps, carer, falls and `pose`) is kept in `data_v1/` and `v1_code/`.

Default run (`--residents 8500`, 7 days, seed 0, ~35 s): **102,000 training rows** (6,800 train residents × 15), 127,500
diary rows in all, 425,000 test check-ins (340,000 `checkin` + 85,000 `help`), 8,500 flats.

## Committed data

`homes/data/` holds the parquet tables, `cold_prior.json`, `summary.json` and `maps/sheet.png`; the CSV copies and the
per-flat plans are not committed (regenerate with `--csv` / `render.py`). The flats are `flats.json.gz`: run
`gunzip -k homes/data/flats.json.gz` before `render.py`, `prior.py`, `check.py` or `example.py`, which read `data/flats.json`.
Everything regenerates in about a minute from `generate.py` (same seed, same files).

## Commands

Run from this folder. `PROJECT` is the team repo that holds the Python environment (`uv sync --group ml` there).

```bash
PROJECT=.   # the repo root (needs `uv sync --group ml`)
uv run --project $PROJECT --group ml python generate.py --residents 8500 --csv   # data/*.parquet (+ .csv)
uv run --project $PROJECT --group ml python render.py                             # data/maps/sheet.png + 24 floor plans
uv run --project $PROJECT --group ml python prior.py                              # data/cold_prior.json
uv run --project $PROJECT --group ml python check.py --residents 10               # TabPFN smoke test, ~6 min on CPU
uv run --project $PROJECT --group ml python example.py --row 30                   # one worked example, explained
uv run --project $PROJECT --group ml python -m unittest test_homes.py -v          # 15 tests
```

## The person: a routine table

A day has 3 slots: **morning 07:00–12:00, afternoon 12:00–18:00, evening 18:00–22:30**. Each resident has, for weekdays and
for weekends, a probability over the zones of her flat for each slot, plus a probability of being out. That table *is* her
routine and her deviation at once: if she is on the couch in 56% of her evenings, the table says 0.56. Each day she draws
one zone per slot. At night she is in bed (used only by help calls).

Residents come in 8 flavours (`standard, early_riser, night_owl, kitchen_person, day_care, restless, frail,
armchair_reader`) that set the base tables; each resident's table is a jittered copy, and she may have shop afternoons,
day-care mornings or a weekend outing on chosen weekdays (those only change her `out` probability). If a flat has no couch or
armchair, daytime sitting moves to the kitchen table. The flavour is stored as `archetype` for analysis, never as a feature.

**The model is never told this table or that there is a routine.** It sees only records (hour, weekday, slot, where she was
last heard, her own earlier records in that slot) and has to find the pattern, in context, from them.

## The flats

Each resident lives in her own room, 2.6–4.5 m × 2.0–3.5 m, with a door on a random wall and the dock just inside it. **Bed,
kitchen and bathroom are in every flat**, plus 0–2 of couch, armchair and entrance (3, 4 or 5 zones in about 25% / 40% / 35%
of flats). Zones are identified by **name** (the app's six: `bed, bathroom, kitchen, couch, armchair, entrance`); `out` means not
home, so there are 7 classes. Furniture stands against the walls with a free aisle; each zone has a viewing pose in the aisle
(heading in degrees, 0 = +x, counter-clockwise). `flats.to_app_zones(flat)` returns the shape of `ZONES` in `app/knock.py`.

## How the robot "hears"

Each record is what the robot learns at one moment:

- **She is home → the robot hears her** (her voice, the kettle, the TV) and places the sound in a zone from the **direction**
  its microphones report and the **loudness**. Placing is imprecise: **10% of records are logged in a neighbouring zone**
  (`zone` ≠ `true_zone`), the direction has 25° noise and the loudness 3 dB. `source = heard`, voice columns filled.
- **She is out → there is nothing to hear.** The robot's check finds nobody: `zone = out`, `source = checkin`, voice empty.

In this data the hearing is simulated from her position, not real audio. On the real JetHexa the microphone board gives a
direction when it hears the wake word; a kettle or TV switching on would be another cue. Direction and loudness are
turned into a zone the same way: closest zone in direction from the dock.

## Rows

**Diary** (`diary.parquet`, days 0–4): exactly 3 records a day, one at a random time in each slot, so 15 per resident.

| Columns | Meaning |
|---|---|
| `day, resident, source, hour, weekday, last_heard_zone, min_since_heard, voice_bearing_deg, voice_loudness_db, zone` | The spec's columns with the spec's names. `zone` is the target; `day`, `resident`, `source` are not features |
| `slot, is_weekend, voice_zone_guess, voice_dist_m` | Slot 0/1/2 (3 = night, help calls only); the zone closest in direction to the voice; distance from the loudness |
| `n_zones, has_couch, has_armchair, has_entrance, room_area_m2` | About the flat |
| `hist_bed … hist_out`, `hist_n` | Her own history: the smoothed share of her earlier records **in the same slot and weekday type** per class, and how many there are |
| `resident_id, resident_name, resident_split, flat_id, archetype, true_zone` | IDs, never features. `true_zone` differs from `zone` on mis-heard records |

Only past information goes into a record: `last_heard_zone` and `min_since_heard` come from her previous record, `hist_*` from
earlier days only (0 on day 0). Removed in v2: `min_since_carer_left`, `pose`, `fell`.

**Test check-ins** (`checkins.parquet`, days 5–6, every resident): the same columns plus `kind, x, y`. 20 `checkin` per day
(random time 07:00–22:30, nobody answers) and 5 `help` per day (any hour, only while she is home, voice filled). `x, y` is where she
is in the flat (empty when out). Her state in a slot is the same zone as her diary record of that slot and day.

**Split.** `resident_split` is `test` for every fifth resident, fixed. Fit and build priors from `train` only; judge on `test`
residents (people and flats the model has never seen).

## The cold-start prior

`prior.ColdStartPrior`: P(zone | the flat's set of zones, weekday type, slot), counted from the train residents' diary and
smoothed towards the population share. `probs(flat, hour, weekday)` returns `(zone_probs, p_out)`: `zone_probs` is
`{zone name: p}` over the flat's zones summing to 1 (the shape of `PRIORS` in `app/knock.py`), `p_out` the chance she is not home.
Night (no records) uses a stated default: bed 0.85, bathroom 0.15. Use it when there is no diary; once records exist, TabPFN's
`predict_proba` is the better prior.

## Fitting TabPFN on it

Only `fit()` and `predict()` (the repo's rule: no fine-tuning). `train.parquet` is all numeric (34 features, target `zone`):

```python
import pandas as pd
from check import fit_classifier
train = pd.read_parquet("data/train.parquet").sample(1000, random_state=0)   # a context of 1,000 rows
X = train.drop(columns=["resident_id", "day", "source", "zone"])
model = fit_classifier(X.to_numpy(), train["zone"])                          # TabPFN's own fit(): stores the context
model.predict_proba(X.head(5).to_numpy())                                    # P(zone) for 5 rows, in model.classes_ order
```

A context of a few hundred to a few thousand rows works best, chosen per query: rows from residents whose flat has the same
zones, her own rows, and the `out` rows. Drop the voice columns (`train_columns.json` → `voice_features`) for check-ins, keep them
for `help` calls with `heard` rows as context.

## Variant: more records a day (`data_r12/`)

`generate.py --records-per-day 12` cuts the day into 12 slots (about one record every 80 minutes), each with its own zone draw
around the morning, afternoon or evening mix, so she moves around inside the day. `data_r12/` is that run: 2,125 residents × 60
records, **102,000 training rows** (the same size as the 3-a-day set), 15% `out`. The default (3 a day) data is unchanged.
Smoke test (10 held-out residents, 1,000-row context, `data_r12/check_results.json`), top-1:

| | TabPFN (0 / 1 / 3 own days) | last heard | cold-start prior | usual place |
|---|---|---|---|---|
| Check-ins | 0.485 / 0.480 / 0.475 | **0.665** | 0.453 | 0.453 / 0.392 / 0.402 |
| Help calls | 0.570 / 0.590 / 0.590 | 0.560 | **0.700** | 0.700 / 0.400 / 0.380 |

More records per day did **not** make TabPFN beat the simple rules in this test: "last heard" is stronger when records are closer
together, and TabPFN is below the cold-start prior on help calls. TabPFN's log loss is better than the cold-start prior's on
check-ins (1.22 against 1.40) but worse on help calls (1.00 against 0.86). Possible causes to check before drawing conclusions:
a 1,000-row context drawn from other residents, a context that does not include her own rows beyond 3 days, and a routine
where each slot is an independent draw (there is no persistence from one slot to the next, so "last heard" tells TabPFN less
than it would for a real person).

## Limits (read before judging a model)

- **With 3 records a day the model learns little about one person.** After 3 diary days she has 9 records, too few to see
  weekday-specific habits (a Thursday day-care morning, a Monday shop afternoon). Expect the model to be mostly the population prior
  plus the slot and where she was last heard. `check.py` and `example.py` show this honestly.
- **Smoke-test result (`data/check_results.json`, 10 held-out residents, 1,000-row context): TabPFN does not beat the simple
  rules on check-ins.** Top-1 on check-ins: TabPFN 0.43 (0 own days) / 0.42 (3 own days), "last heard" 0.53, cold-start prior 0.39,
  usual place 0.39 / 0.43. On help calls (voice columns): TabPFN 0.70 / 0.71, cold-start 0.71, last heard 0.47. So with this data
  TabPFN is at best level with the population prior, and "last heard" is the one to beat on check-ins. v1 (many records a
  day) gave TabPFN a 7-point lead. More records per day, or more own days, are what would change it.
- **Test queries are on days 5–6 only** (two weekdays per resident, which differ by resident because the start weekday is random).
- **Freeze the numbers before comparing models.** Do not tune `residents.py` until TabPFN wins.
- Zone names, not `z1…z6`; no `desk` zone (the app has none); the spec's carer, habit change and Margarete numbers are not reproduced.
- `check.py` is a smoke test (a few residents, one context size), not the full evaluation or the search-cost test.

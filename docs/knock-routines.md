# Routine spec for Knock: where Margarete is

For Krishna. This is what `generate()` simulates for Knock: where an older woman living alone is through the
day in the small studio flat, the moments the robot knows where she is (the diary rows), and the test
check-ins the robot has to answer. It replaces the object routines in [routines.md](routines.md) for Knock;
that file stays for Recall. The plan is on the
[Knock Architecture page](https://claude.ai/artifact/LNmANYSUocKoTr41kJCnG5) and the story on the
[Knock User Journey page](https://claude.ai/artifact/DEh7egdji2AmNuHswoP3pR).

The idea in one line: **she follows a routine, but not exactly**, so where she is now depends on the time,
the weekday, where she was last heard and how long ago, and whether the carer has just left. "Last heard" goes
wrong once she has moved on, and "usual place at this hour" can't use how recently she was heard. TabPFN can
use both, and it needs only a few days of rows for a new person.

## Flat and zones

The same studio as in [routines.md](routines.md): 3 m × 2 m in the simulator's world frame, x = −1.5 … 1.5 m,
y = −1 … 1 m, split into a 2 × 3 grid of 1 m cells. Two cells change their meaning: the desk cell becomes the
bathroom, and the TV cell is the armchair next to the TV.

| Zone | Name | Furniture | Cell centre (x, y) |
|---|---|---|---|
| `z1` | bed | bed and nightstand | (−1.0, 0.5) |
| `z2` | bathroom | shower tray, toilet, sink | (0.0, 0.5) |
| `z3` | kitchen | kitchen table and chair | (1.0, 0.5) |
| `z4` | couch | couch | (−1.0, −0.5) |
| `z5` | armchair | armchair next to the low TV sideboard | (0.0, −0.5) |
| `z6` | entrance | bench, coat rack, front door | (1.0, −0.5) |
| `out` | not home | | |

The robot's dock is at (0.9, −0.2), by the entrance. The flat has no inner walls in the simulation; the
bathroom is a marked zone.

## People

**`r1`, Margarete**: 81, lives alone, early dementia, unsteady on her feet. Her day is below.

**The carer**: visits on weekdays 08:00–08:30 and 18:00–18:30 and is with her in the kitchen. The carer's
voice is never logged as hers.

**`r2` … `r6`**: five other residents in the same kind of flat, with the changes listed under
[Other residents](#other-residents). They exist for one reason: to test TabPFN on a person it has not seen.

## Margarete's day

Every start and end time gets random jitter with a standard deviation of 15 minutes. Where a row has a zone
mix, draw one zone for the whole block each day.

| Weekdays | Activity | Zone |
|---|---|---|
| 22:30–07:30 | sleep | `z1` bed |
| 07:30–08:00 | wash | `z2` bathroom |
| 08:00–08:30 | carer visit: medicines, breakfast ready | `z3` kitchen |
| 08:30–09:30 | breakfast, radio | `z3` kitchen |
| 09:30–11:30 | morning | couch 0.50, armchair 0.20, bed 0.15, kitchen 0.15 |
| 11:30–12:30 | lunch | `z3` kitchen |
| 12:30–14:00 | nap | bed 0.70, couch 0.30 (dozes off on the couch) |
| 14:00–16:00 | afternoon: knitting, phone calls | armchair 0.50, couch 0.30, bed 0.20 |
| 16:00–17:00 | walk to the shop, Mon, Wed and Fri | `out` with p = 0.8, otherwise armchair |
| 17:00–18:00 | rest | `z5` armchair |
| 18:00–18:30 | carer visit | `z3` kitchen |
| 18:30–19:30 | dinner | `z3` kitchen |
| 19:30–22:00 | TV | couch 0.70, armchair 0.20, bed 0.10 |
| 22:00–22:30 | get ready for bed | `z2` bathroom |

**Thursday**: day care 09:30–12:30, so she is `out` (p = 0.9; otherwise a normal morning).

**Weekends** (weekday 5 and 6): no carer. She wakes at 08:30 and has breakfast until 09:30. Saturday
10:00–11:30 she goes shopping with her neighbour (`out`, p = 0.6). Sunday 14:00–17:00 family visits and she
sits at the kitchen table (`z3`).

**Short trips**: from 07:00 to 22:00, at home and not napping, she makes a trip about every 50 minutes
(exponential gap): to the bathroom with p = 0.5, the kitchen with p = 0.35, the entrance with p = 0.15 (the
door, the post). Each trip lasts 5–15 minutes, then she goes back.

**Night**: 1 or 2 bathroom trips between 00:00 and 07:00, each 5–10 minutes.

**Restless evenings**: with p = 0.15 per evening, from 19:30 to 21:00 she moves to a random home zone every
10–20 minutes instead of sitting down. People with dementia are often restless in the evening, and it makes
some evenings hard to predict.

## How she gets into the diary (the rows)

The robot only knows where she is at certain moments. Each moment is one row. There are two sources:

1. **Heard**: from 07:00 to 22:30, while she is at home, the robot hears her and places the sound in a
   zone: she talks to it, talks on the phone, or the kettle or TV goes on. On average **1 per hour**, as a
   Poisson process. With p = 0.1 the zone is logged as a neighbouring zone (the 2 or 3 cells sharing an
   edge), because placing a sound is not exact.
2. **Check-in**: about once a day, at a random time from 07:00 to 22:30, someone in the family starts a
   check-in. The row records where the robot found her, or `out` if she wasn't home. Without these rows
   TabPFN would never see `out` as an answer.

Each row:

| Column | Type | Meaning |
|---|---|---|
| `day` | int | 0 … days − 1, for splitting; not a model feature |
| `resident` | str | `r1` … `r6`, for splitting; never a model feature |
| `source` | str | `heard` or `checkin`; not a model feature |
| `hour` | float | time of day, e.g. 10.58 |
| `weekday` | int | 0 = Monday … 6 = Sunday |
| `min_since_carer_left` | float | minutes since the carer's last visit ended today; empty (NaN) at weekends and before 08:30 |
| `last_heard_zone` | str | zone of this resident's previous row (`z1` … `z6`, or `out`) |
| `min_since_heard` | float | minutes since this resident's previous row |
| `voice_bearing_deg` | float | `heard` rows only: direction of the sound from the dock, see below; NaN otherwise |
| `voice_loudness_db` | float | `heard` rows only, see below; NaN otherwise |
| `zone` | str | **target**: `z1` … `z6` or `out` |

Only past information may go into a row: nothing from after its time.

The architecture page also lists `phone_unanswered` and `answered_robot`. Leave them out of the model:
in `heard` rows they are always the same value, so TabPFN can't learn anything from them. They stay in the
check-in log for the report.

### Voice direction and loudness

For the "she calls for help" path. Real JetHexa: its microphone board reports a direction when it hears the
wake word (see [LEARNINGS.md](LEARNINGS.md)), so a direction per call is realistic.

- `voice_bearing_deg`: the angle from the dock at (0.9, −0.2) to her position, in degrees, 0 = +x,
  counter-clockwise, plus Gaussian noise with a standard deviation of 25°. Bed, bathroom, couch and armchair
  all lie roughly to the left of the dock, so direction alone doesn't settle the zone.
- `voice_loudness_db`: 65 − 20 · log10(distance in metres), plus noise with a standard deviation of 3 dB,
  and 6 dB quieter when she is lying on the floor.

Fit the zone model twice from the same rows: with the voice columns for "help" calls, and without them for
check-ins, where nobody heard her.

## Test check-ins

From the held-out days, two kinds. Both assume she didn't answer, so the robot has to search.

| Kind | How many | When | Her state |
|---|---|---|---|
| `checkin` | 20 per resident per day | random times 07:00–22:30 | where the routine puts her. With p = 0.2 she **fell** 5–60 minutes earlier: she stays in the zone where she fell, lying on the floor |
| `help` | 5 per resident per day, only while she is at home | any time, night included | where she is. Lying on the floor with p = 0.5, otherwise sitting or standing. The voice columns are filled |

Each test check-in has the row columns (without the target) plus the truth:

| Column | Meaning |
|---|---|
| `kind` | `checkin` or `help` |
| `zone` | the true zone, or `out` |
| `pose` | `lying_floor`, `lying_bed`, `sitting` or `standing` |
| `x`, `y` | her position: inside the zone, on the half nearest the open aisle so the 15 cm camera can see her, with random jitter |
| `fell` | true if she fell before the check-in |

Poses by activity: sleep and nap in bed → `lying_bed`; couch, armchair and kitchen → `sitting`; bathroom
and entrance → `standing`; a fall → `lying_floor`. Oscar's `env.place_person(zone, pose)` turns these into
the body in the scene.

## Other residents

Each is Margarete's day with these changes. Keep everything else the same.

| Resident | Changes |
|---|---|
| `r2` early riser | every time 60 minutes earlier; no nap; morning mix armchair 0.6, couch 0.3, kitchen 0.1 |
| `r3` night owl | every time 75 minutes later; TV until 23:30; 2–3 bathroom trips a night |
| `r4` kitchen person | morning and afternoon mixes are kitchen 0.6, armchair 0.3, couch 0.1; couch only in the evening |
| `r5` day care | day care (`out`) every weekday 09:30–12:30; short trips every 35 minutes |
| `r6` restless | restless evenings with p = 0.4; nap mix bed 0.4, couch 0.6 |

## Output of `generate()`

```python
generate(residents=6, days=21, seed=0, habit_change_day=None, heard_per_hour=1.0)
    -> (diary_rows, test_checkins)
```

- `diary_rows`: every row from days 0 … 15, for all residents (pandas DataFrame with the columns above).
- `test_checkins`: the test check-ins from days 16 … 20.
- The same `seed` always gives the same data.

## The three tests

1. **Held-out days**: context is days 0 … 15 of all residents; score Margarete's days 16 … 20.
2. **A new resident**: for each resident in turn, context is the other five residents' rows plus only the
   first 2 or 3 days of this one; score this one's test days. Add a column `own` (1 for the resident's own
   rows, 0 for the others) when building the context, and set it to 1 in the query, so TabPFN can tell her
   rows from the rest. This is also Prior Labs' lesson "choose the rows you show".
3. **Learning curve**: the same as test 2 with 1, 3, 7 and 14 of the resident's own days. Plot search time
   against days for TabPFN and the three baselines. This is the main chart in the pitch.

Baselines: fixed order (nearest zone first), "last heard" (`last_heard_zone` first, then nearest), and
"usual place at this hour" (the most frequent zone for this resident, hour and weekday type in the context).

## Habit change (separate scenario)

`habit_change_day=d`: from day `d` on, after a new medicine, Margarete lies down in bed from 10:00 to 11:30
instead of her morning mix (09:30–10:00 stays as it was).

The test: 16 normal days plus 5 after the change (`days=21, habit_change_day=16`). Give the model the normal
days plus the first 0, 1, 2 and 3 days after the change as context, and score the morning check-ins on the
rest. The point to show: TabPFN moves its top zone for a 10:35 check-in to the bed after a day or two, while
the lookup table follows slowly.

## Targets and tuning

A quick simulation of Margarete's weekdays alone (not the real generator, so treat them as a guide) gave the
numbers in brackets. Run the real generator once with the default numbers and report these. Tune only the
zone mixes, the trip rate and `heard_per_hour`, within ±50% of the numbers above.

| What | Aim |
|---|---|
| Diary rows per resident in 16 days | about 250 (about 15 heard a day plus 1 check-in) |
| Share of test check-ins where `zone != last_heard_zone` | 50–65% (56%) |
| "Last heard" top-1 on test check-ins | 35–50% (42%) |
| "Usual place at this hour", top-1 when its table comes from a 400-day run (its best case) | 45–60% (51%) |
| Best possible top-1 from hour, `last_heard_zone` and the time since, on a 400-day run | at least 6 points above the line before (59%) |
| Share of weekday check-ins where she is out | 5–12% (6%) |

With 0.6 heard events an hour, knowing where she was last heard barely helped (55% against 51%). Keep it at
about 1 an hour.

**Freeze the numbers before anyone runs TabPFN against the baselines.** Then leave them alone, even if
TabPFN loses. Tuning the data so that our model wins is the one thing judges must never be able to accuse us
of.

## Demo questions this must support

Check these on a long run (400 days) by counting the true zones, not with a model. The brackets are the quick
simulation.

| # | Context | Most likely zone |
|---|---|---|
| D1 | Margarete, Tuesday 10:35, last heard in the kitchen 85 minutes ago, carer left 08:30 | `z4` couch at the top (about 42%), then kitchen, armchair and bed, bathroom about 10% |
| D2 | the same, but heard in the kitchen 5 minutes ago | `z3` kitchen (about 85%) |
| D3 | Margarete, Thursday 10:35 | `out`, at day care (about 90%) |
| D4 | D1 after the habit change | `z1` bed (about 80%) |
| D5 | a help call at 02:30, voice from about 140°, loud | `z2` bathroom |

D1 and D2 differ only in how long ago she was heard. That is the comparison to show on stage: "usual place at
this hour" gives the same answer for both, TabPFN does not.

The journey page's numbers for D1 (couch 36%, bathroom 30%) were an illustration. Use the generator's numbers
in the demo.

## Fit with the existing code

- **Navigation:** `nav/` walks the robot on an empty floor with no obstacle avoidance
  ([navigation.md](navigation.md)). Keep the furniture against the walls and each zone's viewing pose in the
  open aisle (y between −0.3 and 0.3 m), facing the zone.
- **TabPFN:** the same TabPFN v2 install as `nav/` (`uv sync --group ml`), and only `fit()` and `predict()`,
  as [AGENTS.md](../AGENTS.md) requires. About 1,500 rows for 6 residents is well within what TabPFN handles.

## Not in scope

- Several people in the flat at once, apart from the carer, who is never logged.
- Real audio: the heard events, the direction and the loudness are drawn from the rules above.
- How she fell or whether she is hurt. A fall only freezes where she is and sets the pose.

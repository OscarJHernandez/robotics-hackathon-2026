# Routines spec for the data generator

For Krishna. This is what `generate()` simulates: one older resident and a visiting carer in a small studio
flat, three objects that move as they go about their day, and a log of "sightings" (someone looks for an
object and finds it). Each sighting becomes one TabPFN row. The plan is on the
[Recall Architecture page](https://claude.ai/artifact/FUvvPP3s2qNpgRsN4XSPeU).

Locations come at two levels: 6 **zones** (furniture areas) and 3 named **spots** on the floor around the
furniture in each zone (for example in front of the couch, under the couch, by the armrest). The **global**
TabPFN answers "found or not found" for each zone. The **local** TabPFN works at the level of one piece of
furniture: which spot around it. Each sighting carries both labels.

The idea in one line: **objects move with people**, so where an object is depends on who used it last,
where they came from and how long ago. A "last seen" rule gets that wrong often, and a lookup table can't
use the continuous features. TabPFN should do better.

## Flat and zones

Studio of 3 m × 2 m, split into a 2 × 3 grid of 1 m cells. Coordinates are the simulator's world frame:
the robot starts at the origin facing +x, and +y is to its left. The flat spans x = −1.5 … 1.5 m and
y = −1 … 1 m, so the robot starts in the middle and every zone is one cell away. The MuJoCo floor is 6 × 6 m,
so this fits with room to spare. Oscar uses the same layout for the MuJoCo scene.

| Zone | Name | Furniture | Home spot of | Cell centre (x, y) |
|---|---|---|---|---|
| `z1` | bed | bed and nightstand | | (−1.0, 0.5) |
| `z2` | desk | desk and chair | phone (charger) | (0.0, 0.5) |
| `z3` | kitchen | kitchen table | cup | (1.0, 0.5) |
| `z4` | couch | couch | | (−1.0, −0.5) |
| `z5` | tv | low TV sideboard | remote | (0.0, −0.5) |
| `z6` | entrance | bench by the door | | (1.0, −0.5) |

Objects: `remote`, `phone`, `cup`. All start the first day at their home spot. In the simulator they can be
mocap bodies without collisions, like the `target` pillar in Oscar's pull request #3, so `env.place()` only
sets `mocap_pos`. Make them 5–8 cm tall and brightly coloured so the camera, about 15 cm above the floor,
sees them from a metre away.

## Spots inside each zone

**All objects lie on the floor.** The robot's camera is about 15 cm above the floor and cannot see table
tops, so the sim leaves them out. This is also the pitch: the robot looks low and underneath, where people
can't easily look. (A real product would add a higher camera for table tops.)

Every zone has 3 floor spots around its piece of furniture, always labelled `a`, `b` and `c`. Spot `a` is
the open floor where things usually get put down (and the home spot where the zone has one), spot `b` is
under the furniture (half hidden, so harder to detect), and spot `c` is the side or corner. Each spot is a
place the robot can see from one viewing pose.

| Zone | `a` open floor | `b` under | `c` side |
|---|---|---|---|
| `z1` bed | beside the nightstand | under the bed | at the foot of the bed |
| `z2` desk | by the desk chair (home of phone) | under the desk | by the wastebasket |
| `z3` kitchen | by the kitchen table (home of cup) | under the table | by the far chair |
| `z4` couch | in front of the couch | under the couch | by the armrest |
| `z5` tv | in front of the TV sideboard (home of remote) | under the sideboard | beside the TV stand |
| `z6` entrance | on the doormat | under the bench | by the shoe rack |

**Where a put-down lands inside a zone**: when someone puts an object down in a zone, the spot is drawn from
that zone's put-down mix:

| Zone | `a` open floor | `b` under | `c` side |
|---|---|---|---|
| `z1` bed | 0.6 | 0.05 | 0.35 |
| `z2` desk | 0.6 | 0.1 | 0.3 |
| `z3` kitchen | 0.7 | 0.1 | 0.2 |
| `z4` couch | 0.6 | 0.05 | 0.35 |
| `z5` tv | 0.7 | 0.05 | 0.25 |
| `z6` entrance | 0.7 | 0.05 | 0.25 |

Most objects end up under furniture through the misplacement and "kicked under" rules below, not through
being put there.

## People

**`resident_a`**: 81, lives alone, early-stage dementia. Puts things down wherever she is and forgets them.

**`resident_b`**: the carer. Visits on weekdays 08:00–08:45 and 18:00–18:45. Tidies objects back to their
home spot.

## Resident's day

Start times get ±20 min of random jitter. Each activity happens in one zone.

| Weekdays | Activity | Zone |
|---|---|---|
| 07:30–08:00 | wake up | `z1` bed |
| 08:00–09:00 | breakfast | `z3` kitchen |
| 09:00–12:00 | newspaper, phone calls | `z2` desk |
| 12:00–13:00 | lunch | `z3` kitchen |
| 13:00–14:00 | nap | `z1` bed |
| 14:00–17:00 | TV | `z4` couch |
| 17:00–18:00 | walk outside | leaves and returns via `z6` entrance |
| 18:00–19:00 | dinner | `z3` kitchen |
| 19:00–22:00 | TV | `z4` couch |
| 22:00 | bed | `z1` bed |

Weekends (weekday 5 and 6): she wakes at 08:30, breakfast runs 08:30–09:30, there is no walk and no carer.
On Sunday 14:00–17:00 family visits and she sits at the kitchen table (`z3`) instead of the couch.

**Short trips**: during any activity longer than 1 hour, about every 45 minutes (exponential gap) she makes
a 5–10 minute trip: to the kitchen with p = 0.6, to bed with p = 0.2, to the entrance with p = 0.2. Then she
returns to the activity's zone. Short trips are what make "I just came back from the kitchen" matter.

## How objects move

Apply these each time the resident changes zone (activity change or short trip), from zone A to zone B:

1. **Carry.** Each object lying in zone A goes with her with probability `p_carry`:

   | Object | `p_carry` (activity change) | `p_carry` (short trip, there and back) |
   |---|---|---|
   | phone | 0.7 | 0.5 |
   | cup | 0.5 | 0.4 |
   | remote | 0.3 | 0.3 |

   A carried object is put down in zone B, at a spot drawn from zone B's put-down mix.
2. **Misplace.** Each put-down goes wrong with p = 0.08. Half of those land under the furniture (spot `b`)
   of the same zone, the other half under the furniture of a random neighbouring zone (the 2 or 3 cells
   sharing an edge). This is the "it rolled under the couch" noise.
3. **Fetch the remote.** At the start of a TV activity, if the remote is anywhere else, she fetches it to
   the couch (`z4`, `a`) with p = 0.8.
4. **Kicked under the couch.** An object in front of the couch (`z4`, `a`) while she sits there gets pushed
   under the couch (`z4`, `b`) with p = 0.3 per hour. This makes `hours_since_seen` matter at the spot
   level.
5. **Walk.** She takes the phone on the walk with p = 0.9. When she comes back she drops it on the doormat
   (`z6`, `a`) with p = 0.6; otherwise she keeps it and it moves on with her.
6. **Carer tidying.** At each carer visit, every object not at its home spot is put back there with
   p = 0.7. Set `last_user = resident_b` for the objects the carer moved. The carer finds objects under
   furniture (spot `b`) only with p = 0.4, so those tend to stay lost.

Every move updates the object's zone, spot, the time it was moved and who moved it (`last_user`).

## Sightings (the rows)

Sightings happen at random times while she is awake: on average 4 per object per day, spaced as a Poisson
process. A sighting does not move the object. Each sighting writes one row:

| Column | Type | Meaning |
|---|---|---|
| `day` | int | 0 … days − 1, used for splitting; not a model feature |
| `object` | str | `remote`, `phone`, `cup` |
| `hour` | float | time of day, e.g. 21.5 |
| `weekday` | int | 0 = Monday … 6 = Sunday |
| `last_user` | str | who moved the object last: `resident_a` or `resident_b` |
| `user_came_from` | str | name of the zone the resident was in before her current one |
| `hours_since_seen` | float | hours since this object's previous sighting (24.0 for its first) |
| `last_zone` | str | zone at this object's previous sighting (its home zone for the first) |
| `last_spot` | str | spot (`a`, `b`, `c`) at this object's previous sighting |
| `days_ago` | int | `days − 1 − day`; recent rows can count for more |
| `zone` | str | **global target**: the zone the object is really in now |
| `spot` | str | **local target**: the spot inside that zone, `a`, `b` or `c` |

`last_zone` is what the logs know, not the truth. If the object has moved since its last sighting,
`last_zone` is wrong. That gap is where the "last seen" baseline loses and TabPFN can win.

Only past information may go into a row. Nothing from after the sighting.

## The two models

### Global: found or not found in each zone

A binary TabPFN classifier. Each sighting becomes **6 rows, one per zone**, labelled `found = 1` for the
true zone and `found = 0` for the other five. Each row has the sighting's columns plus features of the zone
being asked about:

| Zone feature | Meaning |
|---|---|
| `cand_zone` | the zone this row asks about |
| `cand_x`, `cand_y` | that zone's centre in metres |
| `is_last_zone` | 1 if `cand_zone == last_zone` |
| `is_came_from` | 1 if `cand_zone` is where the resident just came from |
| `is_home` | 1 if `cand_zone` holds this object's home spot |
| `dist_from_last` | metres from `last_zone` to `cand_zone` |

To predict, score all 6 zones and divide each P(found) by their sum, so the zone probabilities add up to 1.
About 160 sightings give about 960 rows, well within what TabPFN handles.

This framing has three advantages over one 6-class model. The zone features let it generalise across zones
(for example "it's usually near where she just was"). It scales to any number of zones. And it can later
take the robot's own search results as rows: "checked the couch, not found". Only add those once detection
is reliable, because a miss can be a detection failure.

### Local: which spot around the furniture

**One** TabPFN classifier shared by all zones, with target `spot` (3 classes: `a`, `b`, `c`) and the zone as
an extra feature. A shared model sees every sighting as context, while six per-zone models would each see
only about 30 rows in 14 days. To predict, set the zone to the one the robot is searching.

### Combined, and baselines

- P(zone, spot) = P(zone) × P(spot | zone). The planner searches spot by spot.
- Zone baselines: "last seen zone", and the lookup table's most frequent zone (object × time of day × last
  user).
- Spot baselines: "last seen spot" (if `last_zone` is the zone being searched), and the lookup table's most
  frequent spot per zone and object.

## Output of `generate()`

```python
generate(days=14, seed=0, habit_change_day=None, sightings_per_object_per_day=4)
    -> (history_rows, test_episodes)
```

- `history_rows`: every sighting from days 0 … days − 4 (pandas DataFrame with the columns above).
- `test_episodes`: every sighting from the last 3 days. Same columns. Oscar's `env.place(object, zone,
  spot)` turns the zone and spot into a position in the scene, with a little random jitter.
- The same `seed` must always give the same data.

## Habit change (separate scenario)

`habit_change_day=d`: from day `d` on, the family has moved the TV into the bedroom. Evening TV (19:00–22:00)
happens in `z1` bed instead of `z4` couch, and the remote is fetched to the bed. Afternoon TV stays on the
couch.

The test for this scenario: generate 14 normal days plus 3 days after the change
(`days=17, habit_change_day=14`). Give the model the 14 normal days plus the first 0, 3 and 6 sightings
after the change as context, and score it on the rest. The point to show: TabPFN's top zone for evening
remote questions moves to the bed after a few new sightings, while the lookup table is slow to follow.

## Targets and tuning

Run once with the default numbers and report these. Tune only `p_carry`, the misplacement rate and the
tidying probability, within ±50% of the numbers above.

| What | Aim |
|---|---|
| Rows per 14 days | about 160 (3 objects × 4 a day × 11 history days, plus 36 test) |
| Share of sightings where `zone != last_zone` | 40–55% (otherwise "last seen" is too strong or too weak) |
| Best possible top-1 (the most common true zone for each exact context, computed on a 200-day run) | 60–75% |
| Share of sightings where `spot != a` | 30–45% (otherwise the local model has nothing to learn) |

**Freeze the numbers before anyone runs TabPFN against the baselines.** Then leave them alone, even if
TabPFN loses. Tuning the data to make our model win is the one thing judges must never be able to accuse us
of.

## Demo questions this must support

Check these on a long run (200 days) by counting the true zones, not with a model:

| # | Question context | Most likely zone |
|---|---|---|
| D1 | remote, weekday 21:00, `last_user` resident_a, came from entrance, 3 h since seen | `z4` couch (about 45–55%), then `z5` tv |
| D2 | same, but came from kitchen and 0.3 h since seen | `z3` kitchen moves to the top or close behind the couch |
| D3 | remote, weekday 21:00, after the habit change | `z1` bed |
| D4 | remote in `z4` couch, `last_spot` a, 3 h or more since seen | spot `b`, under the couch, at the top or close behind spot `a` |

If D2 doesn't hold, raise the remote's short-trip `p_carry` (within the tuning range) before the freeze.

## Fit with the existing code

- **Navigation:** `nav/` walks the robot from a to b with TabPFN on an empty floor, with no obstacle
  avoidance and up to about 1 m per 8 gait cycles ([navigation.md](navigation.md)). So put the furniture
  against the walls and every spot's viewing pose in the open aisle (y between −0.3 and 0.3 m), facing the
  furniture. Then a walk between two viewing poses never crosses furniture.
- **TabPFN package:** use the same TabPFN v2 install as `nav/` (`uv sync --group ml`). Its weights are not
  licence-gated and run locally.

## Not in scope

- More objects, more people, or keys and glasses (not detectable in the demo).
- Rooms other than these 6 zones. A bathroom trip simply isn't modelled.
- Realistic timing of when people search. Sightings are random times while she is awake.

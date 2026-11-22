# Walk the simulated JetHexa from a to b with TabPFN

The simulated robot walks from a start point a to a target point b (the red pillar) on a flat, empty floor.
[TabPFN](https://github.com/PriorLabs/TabPFN), a pretrained model for tables, steers it. **TabPFN is
used only through its built-in `fit()` and `predict()`**: `fit()` is given recorded examples and the
pretrained weights are never changed (rule in [AGENTS.md](../AGENTS.md)). Every run, episode and gait cycle
is stored in one database you can query with SQL.

No obstacles yet: no lidar and no path planning around things.

## Run it

```bash
uv sync --group ml
```

```bash
just collect --episodes 200 --images
```

```bash
just controller --name default
```

```bash
just nav --controller default --target 1.0 0.6 --yaw 3.0
```

```bash
just dashboard
```

| Command | What it does |
|---|---|
| `just collect` | Random walking; one recorded row per gait cycle. `--images` also saves the robot camera's picture of every cycle |
| `just controller --name N` | Builds controller N: picks recorded examples, calls TabPFN's `fit()` on them, checks its predictions on held-out examples and walks the 20 test tasks next to the baseline. About 1 minute |
| `just nav-eval --controller N` | Walks the 20 test tasks again with controller N and the baseline |
| `just nav --controller N --target X Y` | One a→b walk with a video (`demo.gif`) and a path plot (`path.png`) |
| `just dashboard` | Live dashboard at http://127.0.0.1:8770, rebuilt from the database every 30 s |
| `just dashboard-file` | The same dashboard as one file, `data/nav/dashboard.html`, that works offline |
| `just query "SQL"` | Runs SQL on the database and prints the result |
| `just test-nav` | Tests for the navigation code |

The TabPFN v2 weights are used because they are not licence-gated, so no login is needed. They download
once (about 30 MB) into `~/Library/Caches/tabpfn`.

## How it works

### What the controller decides

The legs follow the tripod gait in `sim/jethexa_sim.py`. Once per gait cycle (1 s) the controller picks:

- `stride`: how far the legs swing, 0 to 0.3 rad (0.3 is about 0.13 m per cycle)
- `turn`: the difference between left and right swing, -0.25 to 0.25 rad (positive turns left)

### In plain words: how the robot walks straight to b

1. **TabPFN learns how the robot moves.** From the recorded random walks it learns "if the legs do this for
   one second, the body ends up here". It never sees the target. Training is only `fit()` on 300 examples.
2. **Every second, the planner tries 45 possible commands** (5 stride lengths × 9 turn amounts) and asks
   TabPFN where each would leave the robot.
3. **Each predicted spot gets a score** from where b would then be, seen from the robot:
   - `delta_x`: how far b is straight ahead,
   - `delta_y`: how far b is off to the side (counted double, because sideways error is what makes a path curve),
   - plus a penalty for not facing b, so that with b behind it the robot turns around first.
4. **The command with the lowest score is sent to the legs.** Then it repeats.

The result: the robot turns on the spot until it faces b, then walks almost straight to it. Directness rose
from 0.89 to 0.96 on the 20 test tasks, as straight as the hand-written baseline.

Why each part is needed (measured, see the tables below): without the sideways term the path curves (0.89);
without the facing penalty the robot cannot turn around and misses every target behind it (10 of 20 reached).

### Two kinds of controller

Both get their examples from the random walks in the database, and both only call `fit()` and `predict()`.

| Kind | TabPFN predicts | How the command is chosen | Examples given to `fit()` |
|---|---|---|---|
| `planner` (default) | the body's motion over the next cycle (forward, sideways, turn) from the last motion and a candidate command | 45 candidate commands are scored (`--score`, default `hybrid`); the one whose predicted end pose leaves b least off to the side and most nearly ahead wins | 300 recorded cycles |
| `goal` | the command, from b's distance and bearing plus b's offset ahead and to the left | used directly | 150 hindsight examples |

**Hindsight examples** (`--kind goal`): in random walks a command is held for 1 to 8 cycles. Wherever the
robot ended up is treated as if it had been the target: "from here, to reach that point in K cycles, use
this command" (`nav/hindsight.py`).

**Goal inputs** (`--groups`, default `goal,delta`):

| Group | Columns | Meaning |
|---|---|---|
| `goal` | `goal_dist`, `goal_sin`, `goal_cos`, `horizon` | b's distance and bearing, and the cycles to get there |
| `delta` | `delta_x`, `delta_y` | b in the robot's body frame: metres ahead and metres to the left |

Both are capped at what 8 cycles can reach (1.04 m), pulled in along the bearing, because the hindsight
examples never go farther. In hindsight examples the deltas are worked out from the relabelled end point,
not from the episode's `target_x`/`target_y`.

### Results (2026-11-21, 20 held-out tasks, 6,000 recorded cycles)

| Controller | Examples in `fit()` | Reached b | Directness |
|---|---|---|---|
| `planner`, score `hybrid` (default) | 300 | 20 of 20 | **0.96** |
| `planner`, score `polar` | 300 | 20 of 20 | 0.89 |
| `planner`, score `polar` | 2,000 | 20 of 20 | 0.89 (9 times slower) |
| `goal`, groups `goal,delta` (default) | 150 | 20 of 20 | 0.84 |
| `goal`, groups `goal` | 150 | 20 of 20 | 0.77 |
| `goal`, groups `delta` | 150 | 20 of 20 | 0.73 |
| `goal`, groups `goal` | 3,000 | 20 of 20 | 0.72 |
| hand-written baseline | n/a | 20 of 20 | 0.96 |

Directness is straight-line distance / path length (1.0 = a straight line). The planner walks straighter
because it predicts what a command does instead of copying the curved arcs of random walking.

Adding `delta_x`/`delta_y` to the goal controller made 10 of the 20 tasks straighter and none worse by more
than 0.01. The gain is on tasks where b starts behind or far to the side: the controller now turns on the spot
before walking instead of walking a wide loop (task 7: 0.46 to 0.79, 24 to 18 cycles). Tasks with b roughly
ahead did not change; the path still bends a little after the turn.

**Planner score** (`--score`, kind `planner` only): the planner's TabPFN model never sees b (it predicts
motion), so b enters only where candidate commands are scored. `hybrid` is the default. `polar` (the old
default) scores distance to b plus a small cost for not facing it. `delta` scores b's offset from the predicted end pose, ahead and sideways (a metre
sideways counts double), with no bearing term. `hybrid` is `delta` plus a cost for not facing b, scaled by
the distance to b (`HYBRID_FACING` in `nav/dynamics.py`). Results with 300 examples:

| Score | Reached b | Directness | Mean cycles |
|---|---|---|---|
| `hybrid` (default) | 20 of 20 | **0.96** | 9.8 |
| `polar` | 20 of 20 | 0.89 | 9.1 |
| `delta` | 10 of 20 | 0.48 | n/a |
| hand-written baseline | 20 of 20 | 0.96 | 9.8 |

`hybrid` turns on the spot until it faces b, then walks almost straight: 15 of 20 tasks straighter than
`polar`, none worse, at the cost of 0.7 cycles on average for the turn. `delta` alone reaches every target
within 72 degrees of straight ahead and none more than 90 degrees away: with b behind, a turn on the spot
does not bring b closer, so the one-cycle planner never starts turning. The facing cost has to scale with
distance and be large (1.5, not `polar`'s fixed 0.08 m per radian), or turning toward b loses to the
growing sideways offset and `hybrid` stalls the same way.

## DAgger in the arena (`nav/dagger/`)

The controllers above walk an empty floor. `nav/dagger/` drives the robot to the red target pillar in the
50 m² arena with furniture (`vision/scene.py`), from the camera, and improves the student with DAgger:

1. **Round 0**: the expert drives 1000 episodes, one per random layout (scene ids 0 to 999). Every step is a row.
2. **Rounds 1 to 4**: TabPFN's `fit()` on every row so far, then 100 episodes where the executed command is
   the expert's with probability beta (0.5, 0.25, 0, 0), else the student's. The label is always the
   expert's command, so rows from states the student drove into teach it how to get back.
3. **Held-out test**: the last student drives alone in 200 layouts it never saw (scene ids from 100000).

| Piece | File | What it is |
|---|---|---|
| Environment | `env.py` | `SimEnv(scene_id, seed)`: `reset()`, `step((stride, turn))` = one 1 s gait cycle; ends on goal (0.25 m), collision (any robot contact with a wall or furniture), fall or 300 steps |
| Expert | `expert.py` | Shortest path over the true furniture footprints (Dijkstra from the goal, so it labels any state), 0.32 m safety margin, pure pursuit 0.4 m ahead |
| Features | `featurize.py` | Per patch of the 26 x 46 DINOv2 grid: class from the frozen DINOv2 + patch probe (`vision/probe.py`) and mean depth; nearest obstacle and free distance left, ahead, right; goal distance and bearing. 2,400 columns. The previous command is off by default (`use_prev_command`), because in expert data it almost equals the label |
| Student | `policy.py` | Two `TabPFNRegressor`s (stride, turn), TabPFN 3.5, `fit()` once per round, one `predict()` per lock-step for all robots |
| Rollout | `rollout.py` | 16 robots in lock-step per process |
| Machines | `worker.py`, `merge.py`, `loop.py`, `bench.py` | A round's episodes split between machines; each runs several processes and writes Parquet parts, merged and checked on the laptop |

Settings are in `nav/dagger/dagger.yaml`. TabPFN 3.5 needs an API key from https://ux.priorlabs.ai with
the 3.5 licence accepted once: set `TABPFN_TOKEN`, or put `TABPFN_TOKEN=...` in the repo's `.env`
(gitignored, never commit it). Inference stays on the machine; only the key is checked online.

```bash
just dagger local --run try1 --smoke          # every round on this machine, tiny: about 3 minutes
just dagger local --run try1 --procs 4        # the full schedule on one machine
just dagger-bench --procs 1,4,8 --seconds 60  # speed of this machine, and the projected time for round 0
```

On several machines: `just dagger plan --run R --round 0 --weights laptop=1,vendetta=2,...` writes one
plan file per machine; each runs `just dagger-worker --run R --round 0 --plan <its file> --procs N`; copy
`data/nav/dagger/R/round_0/` and `logs/` back, then `just dagger-merge --run R --round 0 --timings` and
`just dagger context --run R --round 1` for the next round's `fit()`.

Output in `data/nav/dagger/<run>/`: `round_<N>.parquet` (every row: features, the student's, expert's and
executed commands, labels, pose, done reason), `round_<N>.json` (outcome rates), `rounds.json`,
`timings.json`, and `logs/<host>-r<round>-p<process>.log` and `.jsonl`: start (machine, versions, config),
a heartbeat every 30 s (episodes done, steps per second, elapsed, ETA, finish time, outcomes, time per
phase) and one line per episode. Linux needs headless OpenGL: `MUJOCO_GL=egl` is set automatically.

## DAgger v2 in the arena: the best model (2026-11-22)

The full record (every variant, what worked and why, why the search stopped, GIFs of the student driving alone) is
[dagger-report.md](dagger-report.md).

A research loop (vendetta and metacon, coordinated by Claude) tested 26 student variants after dagger1, each driving alone
(beta 0) on held-out arena layouts, compared in pairs on the same layouts. Result:

**Best model, `v29-lidarplan`**: TabPFN 3.5 (`fit()`/`predict()` only, 4 estimators) on all 31,586 round 0 expert rows,
116 input columns:

- goal distance and bearing (3);
- the **simulated 360 degree lidar**: 36 rays, 10 degree steps (`JetHexaSim.lidar`; figures are assumptions);
- the camera's horizon band: nearest depth and 4 DINOv2 PCA components in each of 15 sectors;
- **the robot's own plan**: lidar hits accumulated over the episode become an occupancy map, a shortest path to the
  goal is planned on it (the expert's planner, on what the lidar has seen), and the bearing of the point 0.4 m along
  that path is the feature (`nav/dagger/lidarmap.py`). Only the lidar, the odometry pose and the goal are used.

| robust check (student alone, 300 steps) | goal (95% CI) | collisions |
|---|---|---|
| 200 fresh layouts, start seed 99 | 98.0% (95.0-99.2) | 2.0% |
| same layouts, start seed 98 | 97.5% (94.3-98.9) | 2.5% |
| the expert (anchor) | 99.5% / 100% | 0.5% / 0% |

How the directions compared (200 screening layouts unless noted):

| step | goal | collisions | what it showed |
|---|---|---|---|
| camera patch grid (606 columns, the dagger1 setup) | 64% | 36% | the dagger1 failure: walks on where the expert stops and turns |
| compact camera inputs (summary, horizon scan, horizon DINO PCA) | 63-70% (100) | 30-36% | fewer columns beat the grid, but the camera cannot see beside the robot |
| **+ 360 degree lidar** | 90% | 9% | the big step: most crashes were into furniture beside the robot |
| + all 31k rows instead of 8k | 94% | 6% | more real rows help; reshuffling, balancing, memory, gating, 8 estimators did not |
| DAgger rounds 1-4, heading perturbations | 89-96% (100) | 4-10% | no gain on top of all rows |
| PRIVILEGED ceiling: the expert's look-ahead bearing | 100% (100) | 0% | the missing piece is where the path goes, not sensing |
| **+ the robot's own plan on its lidar map** | **98%** | **2%** | most of the ceiling, without privileged information |
| v31: v29 with every row collected (81k, incl. DAgger rounds) | 92% (100) | 7% | more mixed data made it worse |

Commands (all on branch `feat/dagger-v2-fixpass`): `python -m nav.dagger.experiment collect|dagger|run`,
`python -m nav.dagger.stage run|analyse`, `python -m nav.dagger.lidarmap add --name <set>`, variants in
`nav/dagger/variants/*.yaml`, data and results in `data/nav/dagger/v2/` (STATUS.md, incidents.md, experiments/).

## Finding the target itself: DINOv2 signature, range, search (test only, 2026-11-22)

Above, the student is always handed the true target. `nav/dagger/target.py` replaces that with what a real robot
would have: a few candidate locations and its own sensors. **The student is not refitted**; only the goal it is
given changes (variant `v32-target-search` = v29 plus a `target:` block).

- **What and which way: the camera.** The horizon band the student already computes every step (DINOv2 features of
  patch rows 11-14, averaged per patch column: 46 columns of 1.35 degrees, 384 numbers each) is matched to the
  target's **signature** by cosine similarity on all 384 numbers. The signature comes from `just vision-signature`
  (`vision/target_signature.py`): band columns that show the target in full60's 48 training layouts, centred on the
  mean background column, grouped into 3 prototypes. The lidar has no appearance information.
- **How far: the depth camera** (nearest valid pixel of the matched columns, band rows at or above the horizon, so
  no floor), or the lidar where it sees the target and the depth camera agrees (`range_source: lidar | auto`).
  The pillar is invisible to the lidar by design, so it is ranged with depth.
- **Where to look:** `target.candidates` locations (4), each with a probability: in the test the true target's
  viewing point (0.8-1.5 m from it, in open space, so walking there does not reach it by itself) plus decoys at
  random free places at least 1 m apart. The order comes from a `Schedule` (`random` now; `probability` as a
  second example); another model plugs in as another Schedule.
- **The search:** walk to the next location, look around (a scripted turn on the spot, about 20 steps a turn),
  then the next; a match confirmed by 2 agreeing detections (on the way or while looking) becomes the goal, fused
  over steps, far jumps need confirming again. A look-around marks every location within 2.5 m checked.
  An estimate that turns out empty is a false positive: it is rejected and the search goes on. Every location
  checked without a match ends the episode as `not_found`.
- **What changes in the rollout:** the search's goal replaces the true target in the student's goal columns and in
  the lidar-map plan; the look-around turns replace the student's command. Stored rows, the expert's labels,
  goal_x/goal_y and success stay the true target. Every row logs the estimate, its source, the match and its error.

| check | result |
|---|---|
| signature, full60 held-out layouts (12): band columns | precision 98.2%, recall 76.9% (one plain mean: 98.4% / 42.1%) |
| signature, held-out pictures with the pillar | found in 81.5%; by distance 97% (< 1 m), 76% (1-3 m), 1 of 3 (3-4 m) |
| sim, expert walking 40 final layouts (1,348 steps): pillar in view | hit 81% (< 1 m), 89% (1-2 m), 78% (2-3 m), 61% (3-4 m), 0% beyond 4 m; median error < 1 cm |
| sim, same walks: matches more than 0.5 m off | 6 steps of 1,348 (0.45%), all one look-alike in one layout |
| sim, look-around at the true viewing point (40 layouts) | found in 92.5% at turn 0.15 (about 20 steps a turn); 75% at full turn rate 0.25 |
| sim, whole search with a stand-in driver (pure pursuit on the plan, no TabPFN), 12 final layouts | 12 of 12 reached, 0 false positives, final goal error 1-6 mm |

Commands: `just vision-signature full60`, `just target-check --set final --count 40` (the two sim checks, no
TabPFN), `python -m nav.dagger.experiment run --variant nav/dagger/variants/v32-target-search.yaml --closed-set
final --closed-max-steps 600 --closed-stuck-steps 0 --tag final` (the student with the search), then
`python -m nav.dagger.video --experiment v32-target-search-final --pick-search --copy-to videos/target-search`.
To swap the target (a person standing or lying): make her signature from labelled pictures of her and set
`target.signature`, `target.offset` and `target.range_source` in the variant; no code change. Margarete's geoms in
`sim/apartment_decor.xml` are lidar-transparent like the pillar, so `lidar`/`auto` ranging needs them made visible.

## Where the data is

Everything is in **`data/nav/` in the main checkout** (gitignored, never committed). Set `NAV_DATA_DIR`
to use another folder. A git worktree uses the main checkout's folder, so all data ends up in one place.

| Path | Contents |
|---|---|
| `data/nav/nav.duckdb` | The database |
| `data/nav/runs/<run_id>/` | Per run: `run.json` (settings, versions, git commit), `cycles.csv`, `episodes.csv`, `columns.csv` (data dictionary), `images/` (camera pictures), and for demos `demo.gif` and `path.png` |
| `data/nav/controllers/<name>/` | Per controller: `context.parquet` (exactly the examples given to `fit()`) and `controller.json` (kind, features, source runs, git commit) |

### The database

[DuckDB](https://duckdb.org/) is one file, queried with SQL from `just query`, Python
(`duckdb.connect("data/nav/nav.duckdb", read_only=True)`), Jupyter, or a SQL tool such as DBeaver.

| Table | One row per |
|---|---|
| `runs` | run: kind (`collect`, `eval`, `demo`), policy, controller name (`model`), start and end time, git commit, settings |
| `episodes` | episode: start, target, cycles, final distance, reached, fell, path length, directness |
| `cycles` | gait cycle: time, 68 robot sensor columns, recent motion, location, target, goal, command, outcome, camera image path |
| `steps` | 20 ms body pose (for path plots) |
| `controllers` | controller: kind, examples given to `fit()`, features, outputs, fit time |
| `predictions` | held-out example per controller: the true value and TabPFN's prediction |
| `policy_map` | point on a grid of goals: the command a controller picks there (its steering map) |
| `columns` | column of `cycles`: group, unit, meaning |

An older database may also hold the tables `training` and `training_log` and runs with a `round`, from a
fine-tuning experiment that was dropped; nothing uses them.

Examples:

```bash
just query "SELECT r.model AS controller, e.policy, avg(e.reached::INT) AS success, avg(e.directness) AS directness FROM episodes e JOIN runs r USING (run_id) WHERE r.kind = 'eval' AND r.round IS NULL GROUP BY ALL ORDER BY directness DESC"
```

```bash
just query "SELECT model AS controller, output, sqrt(avg((actual - predicted)^2)) AS rmse FROM predictions WHERE round IS NULL GROUP BY ALL ORDER BY ALL"
```

Only one process at a time can open the database for writing. Runs open it briefly, and `nav/db.py`
waits up to 60 s for it, so do not keep a SQL tool connected to it while a run is going.

## What we learned

- **Predicting motion beats copying commands.** Hindsight command labels teach the curved arcs of random
  walking; a motion model plus a small search walks straighter.
- **More examples did not help.** 300 examples were as good as 2,000 for the planner, and the goal
  controller got slightly worse with more.
- **The previous command leaks the label** for the goal controller: in random walks a command is held for
  several cycles, so "previous command" almost gives the answer. Its default inputs are the goal only.
- **Fine-tuning was tried and dropped.** Four fine-tuning rounds gave one small gain and none after, and
  the planner without any training beat them.

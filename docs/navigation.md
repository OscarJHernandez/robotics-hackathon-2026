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
(gitignored, never commit it). With `student.backend: local` inference stays on the machine and only the
key is checked online; with `backend: api` (the default in the configs since 2026-11-22) it runs on Prior
Labs' servers through `tabpfn-client` (see "Inputs, settings and results" below).

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

### Inputs, settings and results (2026-11-22)

Every student below is TabPFN 3.5 fitted on 8,000 rows (`max_context_rows`, spread evenly over bins of
the expert's stride and turn, with backing up in its own bin), driving alone (beta 0) in the 200 held-out
layouts with its safety layer on. Same layouts for every row of the results table, so the numbers compare.

| Config | Perception | Student columns | Safety | Goal | Collision | Timeout |
|---|---|---|---|---|---|---|
| `dagger.yaml` (DINOv2 classes) | probe | 606 (2 x 299 patch class + depth, 8 summary) | none | 60% | 40% | 0 |
| `depth.yaml`, `column_groups` b | none | 15 (goal, `dc_*`, their history) | depth stop | 77.0% | 22.0% | 1.0% |
| `depth.yaml`, `column_groups` c | none | 30 (+ `ds_*` scan) | depth stop | 82.8% | 17.2% | 0 |
| `lidar.yaml` | strip | 123 (goal, lidar, DINOv2 strip) | leg-aware lidar | 92.5% | 4.0% | 3.5% |
| `lidar2.yaml`, + DAgger round 1 with 56 recovery labels | strip | 123 | leg-aware lidar | 94.5% | 3.0% | 2.5% |
| **`lidar3.yaml`, + depth camera** | **strip** | **150** | **leg-aware lidar** | **97.5%** | **0.5%** | **2.0%** |
| the expert (round 0) | true layout | | | 99.8% | 0.2% | 0 |

The 606-column row is from a 199-episode run; (c) from 198 of 200 episodes (one API call hung).

**The 150 student columns of `lidar3.yaml`** (`student.column_groups: [goal, depth, scan, lidar,
lidar_summary, history, dino_strip]`). Distances are metres from the body centre, capped at 10.

| Group | Columns | n | Source | Meaning |
|---|---|---|---|---|
| goal | `goal_dist`, `goal_sin`, `goal_cos` | 3 | pose + target | distance to the target, and its bearing from the heading (left positive) |
| depth | `dc_ahead`, `dc_left`, `dc_right`, `dc_near` | 4 | depth camera | every 4th depth pixel back-projected with the camera pose; points 3 to 25 cm above the floor (not the target pillar) are obstacles. `dc_ahead`: nearest in a ±0.25 m corridor straight ahead; `dc_left` / `dc_right`: nearest in the left / right third of the 62° view; `dc_near`: nearest in view |
| scan | `ds_0` … `ds_14` | 15 | depth camera | nearest obstacle in 15 equal slices of the 62° view, about 4° each, `ds_0` leftmost (2.5x finer than the lidar ahead) |
| lidar | `lr_0` … `lr_35` | 36 | lidar | horizontal 360° lidar at `lidar_geom` (0.15 m above the floor, the camera's height): range of beam k at k·10° counter-clockwise from ahead (`lr_9` left, `lr_18` behind). `mj_multiRay` on geom group 0, passing through the target pillar, ignoring floor hits |
| lidar_summary | `lc_ahead`, `lc_left`, `lc_right`, `lc_rear`, `lc_near` | 5 | lidar | nearest hit in the ±0.25 m corridor ahead; in the 90° sector left / right / behind; anywhere |
| history | `h1_dc_*`, `h1_lc_*`, `h2_dc_*`, `h2_lc_*` | 18 | | the depth and lidar summaries 1 and 2 steps ago (the first step repeats the present) |
| dino_strip | `dp_0_0` … `dp_22_2` | 69 | camera + DINOv2 | DINOv2 runs on a band of 6 patch rows around the horizon only (about 6x cheaper than the whole picture); the 2 rows on the horizon, averaged in 2 x 2 patch blocks, give 23 cells across the view; a fixed PCA (`vision/strip_pca.py`, fitted on the 960 `full60` pictures, 45.7% of the variance) keeps 3 values per cell |

Labels: `y_stride`, `y_turn` = the expert's command. Stride runs from -0.15 (walking backwards, 6 cm per
cycle) to 0.30 (13 cm forward), turn from -0.25 to 0.25 rad (`nav.dagger.STRIDE`, `nav.common.TURN_RANGE`).

**Expert recovery** (`expert.recover`, `recovery_command` in `expert.py`), the labels that teach backing away:
- an obstacle within ±45° ahead closer than 0.30 m: walk backwards (-0.12) while turning the front away;
- an obstacle beside (45° to 135°) closer than 0.31 m **and** a 10 cm step forward would bring it 2 cm
  closer (heading into it): step back slightly (-0.06) while turning away from it;
- only turn when 10 cm further back is tighter. The heading condition matters: the plain expert passes
  furniture at 0.29 to 0.31 m running alongside it; it reaches 60 of 60 layouts with the rule on.

**Leg-aware safety** (`safety` with `source: lidar`, `lidar_safety` in `rollout.py`; only on the student's
commands). The legs reach 0.27 m ahead and behind, 0.26 m sideways and 0.29 m at the corners (measured
while walking and turning). Stop: no stride when the ±0.32 m corridor ahead is closer than 0.48 m. Side:
walking with something closer than 0.32 m beside the legs adds a 0.1 turn away. Sweep: a turn on the spot
swings the front legs toward the turn side and the rear legs the other way, so a turn with an obstacle
within 0.35 m in those quadrants is flipped or dropped. Rear: no backing up into something within 0.40 m.
Every row records the rule that fired (`safety_rule`).

**Commands** (on a Linux box set `MUJOCO_GL=egl`):

```bash
just strip-pca full60                                                          # the strip PCA, once
just dagger-augment --run lidar4 --rounds 0 1 --out lidar5 --config nav/dagger/lidar3.yaml  # depth columns by replay
just dagger context --run lidar5 --round 2 --config nav/dagger/lidar3.yaml
just dagger plan --run lidar5 --round heldout --weights victory=1 --config nav/dagger/lidar3.yaml
just dagger-worker --run lidar5 --round heldout --plan data/nav/dagger/lidar5/plans/round_99-victory.json \
    --context data/nav/dagger/lidar5/context_2.parquet --beta 0 --procs 1 --n-envs 100 \
    --config nav/dagger/lidar3.yaml
just dagger-video --run lidar5 --round heldout --episodes 99000012           # replay with lidar and strip panels
```

`augment.py` replays recorded episodes exactly (deterministic physics) to add columns that need the raw
depth pictures; it stops if a replay moves more than 1 cm from the recording. `student.backend: api` uses
Prior Labs' hosted TabPFN (`tabpfn-client`): the rows go to their servers, each `predict()` costs about
10,000 credits of a 50 M monthly quota, and at most 1,500 calls per hour are allowed. `backend: local`
keeps everything on the machine.

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

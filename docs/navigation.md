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

### Two kinds of controller

Both get their examples from the random walks in the database, and both only call `fit()` and `predict()`.

| Kind | TabPFN predicts | How the command is chosen | Examples given to `fit()` |
|---|---|---|---|
| `planner` (default) | the body's motion over the next cycle (forward, sideways, turn) from the last motion and a candidate command | 45 candidate commands are scored; the one whose predicted end pose is closest to b, facing it, wins | 300 recorded cycles |
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
| `planner` | 300 | 20 of 20 | **0.89** |
| `planner` | 2,000 | 20 of 20 | 0.89 (9 times slower) |
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
motion), so b enters only where candidate commands are scored. `polar` (default) scores distance to b plus a
cost for not facing it. `delta` scores b's offset from the predicted end pose, ahead and sideways (a metre
sideways counts double), with no bearing term. `hybrid` is `delta` plus a cost for not facing b, scaled by
the distance to b (`HYBRID_FACING` in `nav/dynamics.py`). Results with 300 examples:

| Score | Reached b | Directness | Mean cycles |
|---|---|---|---|
| `hybrid` | 20 of 20 | **0.96** | 9.8 |
| `polar` (default) | 20 of 20 | 0.89 | 9.1 |
| `delta` | 10 of 20 | 0.48 | n/a |
| hand-written baseline | 20 of 20 | 0.96 | 9.8 |

`hybrid` turns on the spot until it faces b, then walks almost straight: 15 of 20 tasks straighter than
`polar`, none worse, at the cost of 0.7 cycles on average for the turn. `delta` alone reaches every target
within 72 degrees of straight ahead and none more than 90 degrees away: with b behind, a turn on the spot
does not bring b closer, so the one-cycle planner never starts turning. The facing cost has to scale with
distance and be large (1.5, not `polar`'s fixed 0.08 m per radian), or turning toward b loses to the
growing sideways offset and `hybrid` stalls the same way.

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

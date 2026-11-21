# Walk the simulated JetHexa from a to b with TabPFN

The simulated robot walks from a start point a to a target point b (the red pillar) on a flat, empty floor.
[TabPFN](https://github.com/PriorLabs/TabPFN), a pretrained model for tables, picks the steering command
once per gait cycle, and its weights are **fine-tuned for this task** in rounds. Every run, episode, gait
cycle, training step and prediction is stored in one database you can query with SQL.

No obstacles yet: no lidar and no path planning.

## Run it

```bash
uv sync --group ml
```

```bash
just collect --episodes 200 --images
```

```bash
just train --name a2b-v1 --rounds 3
```

```bash
just nav --model a2b-v1 --round 3 --target 1.0 0.6 --yaw 3.0
```

```bash
just dashboard-file
```

| Command | What it does |
|---|---|
| `just collect` | Random walking; one recorded row per gait cycle (round 0 data). `--images` also saves the robot camera's picture of every cycle |
| `just train --name N` | Fine-tunes TabPFN in rounds and evaluates each round (about 6 min per round on an Apple Silicon Mac) |
| `just nav-eval --model N --round R` | Evaluates a round again on the 20 held-out tasks |
| `just nav --model N --round R --target X Y` | One a→b walk with a video (`demo.gif`) and a path plot (`path.png`) |
| `just dashboard-file` | Writes `data/nav/dashboard.html`, one file that works offline |
| `just dashboard` | The same dashboard served live at http://127.0.0.1:8770, rebuilt from the database on every load |
| `just query "SQL"` | Runs SQL on the database and prints the result |
| `just test-nav` | Tests for the navigation code |

The TabPFN v2 weights are used because they are not licence-gated, so no login is needed. They download
once (about 30 MB) into `~/Library/Caches/tabpfn`. Fine-tuning runs on the Mac's GPU (`mps`); TabPFN's
own example assumes CUDA, but `mps` works for this small model.

## How it works

### What TabPFN controls

The legs follow the tripod gait in `sim/jethexa_sim.py`. Once per gait cycle (1 s) TabPFN picks two numbers:

- `stride`: how far the legs swing, 0 to 0.3 rad (0.3 is about 0.13 m per cycle)
- `turn`: the difference between left and right swing, -0.25 to 0.25 rad (positive turns left)

Each command starts at the gait phase where both tripods are mid-sweep, so a new command never makes the
joint targets jump.

### The regression

Two TabPFN regressors (one per output) map the **goal** to the command:

| Input | Meaning |
|---|---|
| `goal_dist` | distance to b, capped at what 8 cycles at full speed can reach (1.04 m) |
| `goal_sin`, `goal_cos` | bearing to b in the robot's frame (positive = b is to the left) |
| `horizon` | cycles needed at full speed, 1 to 8 |

They are re-asked every cycle, so the robot keeps correcting.

Other columns (robot sensors, recent motion, location, time) are recorded and can be switched on with
`--groups`, but are off by default (see "What we learned").

### Where the training examples come from: hindsight relabelling

Nobody shows the robot how to reach b. In random walks a command is held for 1 to 8 cycles, and the
robot ends up somewhere. **Hindsight relabelling** turns each stretch into an example: "from here, to reach
the point you actually reached in K cycles, use this command". Random walking alone gives about 19,000
a→b examples (`nav/hindsight.py`).

### The training rounds

- **Round 0.** Fine-tune the pretrained TabPFN weights on the hindsight examples (`FinetunedTabPFNRegressor`).
- **Round r.** The round r-1 model, with a little noise on its commands, attempts 60 random a→b tasks.
  The more direct half of the episodes that reached b go into the training table, repeated more the more
  direct they were, and the model is fine-tuned again from its round r-1 weights.
- **Evaluation after every round.** The 20 held-out tasks in `nav/eval_tasks.json` are never used for
  training. Three policies run on them:
  - `tabpfn-finetuned`: the round's weights
  - `tabpfn-pretrained`: the original weights with exactly the same context rows, so the only difference
    is the weights
  - `baseline`: a hand-written steer-to-bearing controller

  Metrics: success (within 0.1 m of b in 40 cycles), cycles needed, final distance and directness
  (straight-line distance / path length).

TabPFN gets a context of only 150 examples when walking. With a large context the pretrained model already
predicts the examples almost perfectly, so fine-tuning has nothing to add. With a small one, the
fine-tuned weights carry what was learned from all the data.

## Where the data is

Everything is in **`data/nav/` in the main checkout** (gitignored, never committed). Set `NAV_DATA_DIR`
to use another folder. A git worktree uses the main checkout's folder, so data from every branch ends up
in one place.

| Path | Contents |
|---|---|
| `data/nav/nav.duckdb` | The database: every run, episode, cycle, training step and prediction |
| `data/nav/runs/<run_id>/` | Per run: `run.json` (settings, versions, git commit), `cycles.csv`, `episodes.csv`, `columns.csv` (data dictionary), `images/` (camera pictures), and for demos `demo.gif` and `path.png` |
| `data/nav/models/<name>/` | Per trained model: `round_<r>_<stride or turn>.ckpt` (TabPFN checkpoints), `round_<r>_context.parquet` (the training table of that round), `train.json`, `train.log` |
| `data/nav/dashboard.html` | The dashboard from `just dashboard-file` |

### The database

[DuckDB](https://duckdb.org/) is one file, queried with SQL from `just query`, Python
(`duckdb.connect("data/nav/nav.duckdb", read_only=True)`), Jupyter, or a SQL tool such as DBeaver.

| Table | One row per |
|---|---|
| `runs` | run: kind (`collect`, `attempts`, `eval`, `demo`), policy, model, round, start and end time, git commit, settings |
| `episodes` | episode: start, target, cycles, final distance, reached, fell, path length, directness |
| `cycles` | gait cycle: time, 68 robot sensor columns, recent motion, location, target, goal, command, outcome, camera image path |
| `steps` | 20 ms body pose (for path plots) |
| `training` | round and output: training and validation rows, validation MSE before and after, how far the weights moved, checkpoint files |
| `training_log` | every value the fine-tuner logged: loss per step, validation MSE per epoch |
| `predictions` | validation example per round: label, prediction with the start weights and with the fine-tuned weights |
| `policy_map` | point of a grid of goals: the predicted command before and after fine-tuning |
| `columns` | column of `cycles`: group, unit, meaning |

Examples:

```bash
just query "SELECT policy, round, avg(reached::INT) AS success, avg(directness) AS directness, avg(cycles) AS cycles FROM episodes JOIN runs USING (run_id, policy, model, round) WHERE kind = 'eval' GROUP BY ALL ORDER BY policy, round"
```

```bash
just query "SELECT round, output, val_mse_before, val_mse_after, weight_change FROM training WHERE model = 'a2b-v1' ORDER BY round, output"
```

```bash
just query "SELECT cycle, round(target_dist, 3) AS dist, round(stride, 3) AS stride, round(turn, 3) AS turn FROM cycles WHERE run_id = '<run_id>' AND episode = 0 ORDER BY cycle"
```

Only one process at a time can open the database for writing. Runs open it briefly, and `nav/db.py`
waits up to 60 s for it, so do not keep a SQL tool connected to it while a run is going.

## What we learned

- **The previous command leaks the label.** In random walks a command is held for several cycles, so
  "previous command" and "motion over the last cycle" almost give away the answer. With them as inputs
  the model learned to repeat its last command. They are off by default.
- **Fine-tuning needs something to learn.** The pretrained TabPFN predicts the hindsight examples almost
  exactly (validation MSE about 1e-5 for stride), so with TabPFN's early stopping on validation MSE the
  start weights were always kept (`weight_change` 0). Training now runs a fixed number of epochs
  (`--early-stopping` to change that), and the held-out a→b tasks judge the result; validation MSE is
  still logged every epoch.
- **Copying arcs is not the same as walking straight.** Hindsight examples teach the curved arcs of
  random walking. The improvement rounds add the more direct successful attempts to push toward
  straighter paths.

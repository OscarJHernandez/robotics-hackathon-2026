# Agent instructions

These rules apply to every coding agent working in this repo (Claude Code, Codex, Cursor, Copilot and others).
Humans: see [CONTRIBUTING.md](CONTRIBUTING.md).

## Before you start

1. Read [docs/LEARNINGS.md](docs/LEARNINGS.md). It holds what contributors and their agents have already
   learned about this project. Apply it, and don't repeat a dead end that is recorded there.
2. Update your local `main`: `git switch main && git pull`.

## How to change things

- Never commit or push to `main`. It is protected; a direct push is rejected.
- Work on a new branch off `main`, named `<type>/<short-name>`, where type is `feat`, `fix`, `docs` or `chore`
  (for example `feat/gripper-calibration`).
- Keep each pull request to one topic and small enough to review in a few minutes.
- Open the pull request with `gh pr create` and fill in the template. Pull requests are merged by squash.
- Never commit secrets, `.env` files or large binary files (robot logs, bag files, model weights). The repo is public.

## Before you finish

Add what you learned to [docs/LEARNINGS.md](docs/LEARNINGS.md), in the same pull request as the work:

- **Learnings**: anything non-obvious that would save the next person or agent time. Examples: a setup step
  that is easy to miss, a hardware quirk, an approach that failed and why, a correction the user gave you.
- **Suggestions**: improvements you noticed but did not make. Add them under "Suggestions (open)".

Check that the point is not already in the document, and follow the entry format at the top of that file.
If there is nothing new to record, say so in the pull request description.

Claude Code users: the `contribute` and `record-learning` skills in `.claude/skills/` walk through these steps.

## TabPFN: no fine-tuning. Use `fit()` and `predict()` only

**Do not fine-tune TabPFN in this repo.** This is Oscar's decision (2026-11-21) and it applies to every
experiment, demo, script and test.

- **"Training" TabPFN means calling its built-in `fit()`** on our recorded examples
  (`TabPFNRegressor.fit(X, y)` or `TabPFNClassifier.fit(X, y)`), then `predict()`. `fit()` stores the
  examples as context; the pretrained weights stay as they are.
- **Never change TabPFN's weights.** No `FinetunedTabPFNRegressor` or `FinetunedTabPFNClassifier`, no
  `tabpfn.finetuning`, no gradient steps, no optimizer, no custom training loop, no saved fine-tuned checkpoints.
- **To make a controller better, change what goes into `fit()`**: better or more examples, better features,
  a better choice of context rows. Do not reach for fine-tuning.
- **Build controllers with `just controller`** (`nav/controller.py`): it picks recorded examples and calls
  TabPFN's `fit()` on them, in seconds. `tests/test_nav.py` fails if fine-tuning code appears in `nav/`.
- If a task seems to need fine-tuning, stop and ask Oscar first.

## Stack and commands

The project is Python with the [MuJoCo](https://mujoco.readthedocs.io/) simulator. The simulated JetHexa robot is in
`sim/`; the full guide is [docs/simulation.md](docs/simulation.md).

### Python and the virtual environment: always uv

- [uv](https://docs.astral.sh/uv/) manages Python and the virtual environment. The environment is `.venv/` in the
  repo root, created by `uv sync` (or `just setup`) from `pyproject.toml` and `uv.lock`.
- Run Python through uv: `uv run python <script>`. It syncs `.venv/` first, so there is no need to activate it.
- Add a package with `uv add <name>` (dev-only tools: `uv add --dev <name>`); remove with `uv remove <name>`.
  Commit `pyproject.toml` and `uv.lock` together.
- Never use `pip install`, `python -m venv`, `virtualenv`, conda or a `requirements.txt`. Never commit `.venv/`.
- Python is pinned to 3.12 in `.python-version`; uv downloads it if missing. MuJoCo has no wheels for Python 3.14
  or newer, which is why `pyproject.toml` caps it.

### Commands: just

Project commands live in the `justfile` and run with [just](https://just.systems/). If you add a recurring
command, add it there with a one-line comment above it (the comment is what `just --list` shows).

| Command | What it does |
|---|---|
| `just setup` | First-time setup: `uv sync` plus the robot meshes |
| `just sync` | `uv sync`: make `.venv/` match `uv.lock` |
| `just meshes` | Download the JetHexa meshes into `sim/meshes/` (gitignored, no license: never commit) |
| `just sim [--seconds N] [--snapshot file.png]` | Walk without a window and print the end position |
| `just view [seconds]` | Live 3D window (uses `mjpython` on macOS, which a live window needs) |
| `just inspect` | MuJoCo's viewer on `sim/jethexa.xml`, to move joints by hand |
| `just snapshot [file.png]` | Save a picture of the robot after walking |
| `just test` | Run the tests (`unittest`; the camera test skips without OpenGL) |
| `just vision-data`, `just vision-features`, `just vision-probe` | Labeled arena pictures, DINOv2 patch features, train and test the patch classifier; guide: [docs/vision.md](docs/vision.md) |
| `just dagger local --run NAME [--smoke]`, `just dagger-worker`, `just dagger-merge`, `just dagger-bench`, `just test-dagger` | DAgger for the TabPFN arena policy, on one or several machines; guide: [docs/navigation.md](docs/navigation.md) ("DAgger in the arena") |
| `just app-build`, `just app [port]` | Build and run the web app (live sim + metacon chat) at http://127.0.0.1:8790; guide: [docs/app.md](docs/app.md) |
| `just student-bench [--flats N --budget-calls K]` | The v29 student in the flats next to A* + a-to-b, all legs lock-step, one TabPFN API call per step (10,000 credits each; `--policy fake` for a dry run; `--levels clean,medium,messy --driver baseline-only` for A* + a-to-b alone in the mess levels, no API); guide: [docs/navigation.md](docs/navigation.md) ("The student in the app") |
| `just search-demo`, `just search-sim`, `just test-homes` | Where is she: the Bayesian search's results and pictures (`homes/data/search_examples/`), one search with the simulated robot walking to each zone, and the tests; guide: [homes/README.md](homes/README.md) |

Lint: not set up yet.

If `just` is not installed, `uvx --from rust-just just <command>` runs it without installing.

### Code layout

| Path | Contents |
|---|---|
| `sim/jethexa.xml` | The robot with Hiwonder's meshes (needs `just meshes`) |
| `sim/jethexa_simple.xml` | The same robot with simple shapes; used automatically when the meshes are missing |
| `sim/jethexa_sim.py` | `JetHexaSim`: load, set joint targets, step, read joints, body, IMU, foot contacts and lidar scan, move the target marker, render camera; demo tripod gait with steering |
| `sim/arena.xml`, `sim/arena_simple.xml`, `sim/arena_furniture.xml` | The arena: a closed 50 m² square room with 8 movable obstacles, around the robot by `<include>` |
| `sim/apartment.xml`, `sim/apartment_simple.xml`, `sim/apartment_decor.xml` | The arena dressed as a home for the web app: zone patches, decoration and the Margarete figure, all without collisions, plus the clutter. Only the app loads them |
| `sim/clutter.xml`, `sim/arena_clutter.xml`, `sim/arena_clutter_simple.xml` | Twelve small solid clutter bodies (`obs_c0..obs_c11`, parked outside the room) for the app's Medium and Messy rooms, and the plain arena with them (the student's twin, `FlatEnv`). Never include `clutter.xml` in `arena.xml`/`arena_simple.xml`/`arena_furniture.xml`: it would change every training layout |
| `sim/demo.py` | What `just sim`, `just view` and `just snapshot` run |
| `sim/fetch_meshes.py` | Mesh download, pinned to one Hiwonder commit and checked by SHA-256 |
| `nav/` | Walking from a to b with TabPFN: `common.py` (walker, geometry), `db.py` (DuckDB recorder), `collect.py`, `hindsight.py` (goal examples), `dynamics.py` (motion examples and planner), `policy.py` (`fit()`-only TabPFN policies), `controller.py` (build and test a controller), `evaluate.py`, `run.py` (demo video), `dashboard.py`, `live_dashboard.py`. Guide: [docs/navigation.md](docs/navigation.md) |
| `nav/dagger/` | DAgger in the arena: `env.py` (`SimEnv`), `expert.py` (planner + pure pursuit), `featurize.py` (patch classes and depths), `policy.py` (TabPFN 3.5 `fit()`-only student), `rollout.py` (lock-step robots), `worker.py`, `merge.py`, `loop.py`, `bench.py`, `timing.py` (logs, ETAs), `dagger.yaml`; v2 research: `experiment.py` (collect, DAgger round, run a variant), `stage.py` (run and analyse variants), `lidarmap.py` (the robot's own plan on its lidar map), `report.py` (tables and pictures in `docs/dagger/`), `variants/` (one YAML per variant; best: `v29-lidarplan`), `export.py` (a student's context rows, columns and band PCA, exported once for fitting elsewhere), `flatbench.py` (the student in the flats). `policy.py` `backend: client` runs the student through the hosted TabPFN API (`tabpfn-client`; one stacked regressor, one `predict()` = 10,000 credits per decision; `CreditBudget`, `credits_left`). Report: [docs/dagger-report.md](docs/dagger-report.md). Needs `TABPFN_TOKEN` (env or gitignored `.env`) |
| `app/` | The web app: `robot.py` (live sim thread with the speed-up; scene `apartment` is the furnished flat (`ApartmentScene`); drivers student, then the TabPFN a-to-b controller on an A* route; contacts and the stuck fallback), `student.py` (the v29 student in the app: the plain twin it sees through, background loading, credit floor), `apartment.py` (flat, journal, belief and the search loop behind the Apartment panel), `llm.py` (metacon chat), `voice.py` (speech via Voice Control), `call.py` (voice calls shown live to every viewer), `knock.py` (Find and Call Margarete), `narrate.py` (metacon narrates the search: template line, metacon's wording, voice), `server.py` (API + page, static files with byte ranges; `POST /api/speed` x1/x2/x3/x5, `POST /api/apartment/obstacles` clean/medium/messy), `web/` (front end from Lovable; phone layout in `PhoneApp.tsx`; the "How it works" page in `components/jethexa/how/`, its numbers and their sources in `content/howItWorks.ts`, its videos in `assets/how/`). Guide: [docs/app.md](docs/app.md) |
| `vision/` | Camera perception: `scene.py` (the arena: random layouts, robot poses, class per pixel), `dataset.py` (labeled pictures), `features.py` (frozen DINOv2 patch features, patch labels), `probe.py` (patch classifier, `blocked_grid` for avoidance). Guide: [docs/vision.md](docs/vision.md) |
| `data/vision/` | Vision runs: pictures, labels, features, probes (gitignored, never commit) |
| `homes/` | Knock's "where is she": `flats.py` (the arena as a flat with five named zones: bedroom, living room, dining area, kitchen, hallway; `scatter(flat, rng, level)`: the app's mess levels clean, medium, messy, dropping the loose pieces and the `CLUTTER` at random), `residents.py` (routines, stated habits), `generate.py` (diary and check-in tables), `search.py` (Bayesian search: journal + habits + floor plan -> belief; `Search` is the loop: most likely zone, look, update, repeat), `tabpfn_belief.py` (the same belief from TabPFN: `fit()` on synthetic check-ins; the app's search starts from it), `search_demo.py` and `search_figure.py` (numbers and pictures), `sim_bridge.py` (the search drives the simulated robot through `nav/dagger`). Data in `homes/data/` is committed. Guide: [homes/README.md](homes/README.md) |
| `data/nav/` | Recorded runs, the database `nav.duckdb`, model checkpoints (gitignored, never commit) |
| `tests/` | Tests (`just test`; navigation tests: `just test-nav`) |
| `docs/knock-*.md` | Knock, the team's concept: `knock-ui.md` (the Find and Call Margarete buttons for `app/`, mockup in `docs/knock-ui/`) and `knock-routines.md` (data for the zone model). Read `knock-ui.md` before building those buttons |

Keep both model XML files in step: a change to joints, actuators, sensors or physics goes in both.

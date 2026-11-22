# Agent instructions

These rules apply to every coding agent working in this repo (Claude Code, Codex, Cursor, Copilot and others).
They also work as the onboarding guide for people. Humans: see [CONTRIBUTING.md](CONTRIBUTING.md) for the
step-by-step of making a change.

## The project in one minute

**Knock**: a simulated six-legged robot (JetHexa, in MuJoCo) checks in on Margarete, an elderly person at home.
When she does not answer, the robot searches the flat for her. [README.md](README.md) has the results.

A search has four parts, and each has its own folder:

| Step | What happens | Folder | Guide |
|---|---|---|---|
| 1. Belief | Which zone is she most likely in? TabPFN and Bayes' rule, from her habits and a journal | `homes/` | [homes/README.md](homes/README.md) |
| 2. Walk | A TabPFN model (the "student", variant v29) steers around furniture with lidar | `nav/dagger/` | [docs/navigation.md](docs/navigation.md) |
| 3. Look | A DINOv2 detector finds her in the camera picture; the depth camera gives her distance | `vision/`, `app/spotter.py` | [docs/app.md](docs/app.md) |
| 4. Approach | The robot walks up to her; its destination follows the measured distance | `app/robot.py` | [docs/app.md](docs/app.md) |

The web app in `app/` runs all four live in a furnished flat. `sim/` holds the robot and the rooms.

## Your first hour

1. Read [docs/LEARNINGS.md](docs/LEARNINGS.md). It holds what contributors and their agents have already
   learned. Apply it, and don't repeat a dead end that is recorded there.
2. Set up: `just setup` (installs Python and packages with uv, downloads the robot meshes).
3. See it work: `just view` (the robot walks), then `just test` (the tests pass).
4. Run the app: `just app-build`, then `just app`, and open http://127.0.0.1:8790.
5. Read the guide of the folder you will work in (table above).

## Before you start a change

1. Read [docs/LEARNINGS.md](docs/LEARNINGS.md), if you have not just done so.
2. Update your local `main`: `git switch main && git pull`.

## How to change things

- Never commit or push to `main`. It is protected; a direct push is rejected.
- Work on a new branch off `main`, named `<type>/<short-name>`, where type is `feat`, `fix`, `docs` or `chore`
  (for example `feat/gripper-calibration`).
- Keep each pull request to one topic and small enough to review in a few minutes.
- Open the pull request with `gh pr create` and fill in the template. Pull requests are merged by squash.
- Never commit secrets, `.env` files or large binary files (robot logs, bag files, model weights). The repo is public.
- After a squash merge, a branch that was stacked on the merged one conflicts with `main` everywhere. Move the
  work to a new branch off `main` instead of merging (see "Process" in the learnings).

## Before you finish

Add what you learned to [docs/LEARNINGS.md](docs/LEARNINGS.md), in the same pull request as the work:

- **Learnings**: anything non-obvious that would save the next person or agent time. Examples: a setup step
  that is easy to miss, a hardware quirk, an approach that failed and why, a correction the user gave you.
- **Suggestions**: improvements you noticed but did not make. Add them under "Suggestions (open)".

Check that the point is not already in the document, and follow the entry format at the top of that file.
If there is nothing new to record, say so in the pull request description.

Claude Code users: the `contribute` and `record-learning` skills in `.claude/skills/` walk through these steps.

## Three rules that are easy to break

### 1. TabPFN: no fine-tuning. Use `fit()` and `predict()` only

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

### 2. Python: always uv

- [uv](https://docs.astral.sh/uv/) manages Python and the virtual environment. The environment is `.venv/` in the
  repo root, created by `uv sync` (or `just setup`) from `pyproject.toml` and `uv.lock`.
- Run Python through uv: `uv run python <script>`. It syncs `.venv/` first, so there is no need to activate it.
- Add a package with `uv add <name>` (dev-only tools: `uv add --dev <name>`); remove with `uv remove <name>`.
  Commit `pyproject.toml` and `uv.lock` together.
- Never use `pip install`, `python -m venv`, `virtualenv`, conda or a `requirements.txt`. Never commit `.venv/`.
- Python is pinned to 3.12 in `.python-version`; uv downloads it if missing. MuJoCo has no wheels for Python 3.14
  or newer, which is why `pyproject.toml` caps it.

### 3. Data and secrets stay out of git

- `data/` (recorded runs, features, signatures, model exports) and `sim/meshes/` are gitignored. Never commit them.
  The meshes have no license. `homes/data/` is the exception: it is small, synthetic and committed.
- `TABPFN_TOKEN` lives in the environment or in the gitignored `.env`. Never print it, commit it or put it in a
  page or pull request. Each student decision through the TabPFN API costs 10,000 credits, so say what a run
  will cost before you start a long one.

## Commands: just

Project commands live in the `justfile` and run with [just](https://just.systems/). If you add a recurring
command, add it there with a one-line comment above it (the comment is what `just --list` shows).
If `just` is not installed, `uvx --from rust-just just <command>` runs it without installing.

**Basics**

| Command | What it does |
|---|---|
| `just setup` | First-time setup: `uv sync` plus the robot meshes |
| `just sync` | `uv sync`: make `.venv/` match `uv.lock` |
| `just meshes` | Download the JetHexa meshes into `sim/meshes/` |
| `just sim [--seconds N] [--snapshot file.png]` | Walk without a window and print the end position |
| `just view [seconds]` | Live 3D window (uses `mjpython` on macOS, which a live window needs) |
| `just inspect` | MuJoCo's viewer on `sim/jethexa.xml`, to move joints by hand |
| `just snapshot [file.png]` | Save a picture of the robot after walking |

**Tests** (lint is not set up yet)

| Command | What it covers |
|---|---|
| `just test` | Everything in `tests/` (`unittest`; tests that need OpenGL skip without it) |
| `just test-nav`, `just test-dagger`, `just test-homes` | Navigation, DAgger, and the search in `homes/` |

**The app and the search**

| Command | What it does |
|---|---|
| `just app-build`, `just app [port]` | Build and run the web app at http://127.0.0.1:8790; guide: [docs/app.md](docs/app.md) |
| `just search-demo`, `just search-sim` | The Bayesian search: numbers and pictures (`homes/data/search_examples/`); one search with the robot walking to each zone |
| `just person-data` | Pictures of Margarete in the app's flat, for her DINOv2 signature |
| `just search-bench` | The app's whole search with the robot's camera, many times, no browser. Uses the A* walker and no real-time pauses, so **also test live with the student** before calling a search change done |
| `just student-bench [--flats N --budget-calls K]` | The v29 student in the flats next to A* + a-to-b. One TabPFN API call per step (10,000 credits). `--policy fake` is a dry run; `--levels clean,medium,messy --driver baseline-only` uses no API |

**Research**

| Command | What it does |
|---|---|
| `just vision-data`, `just vision-features`, `just vision-probe`, `just vision-signature` | Labeled arena pictures, DINOv2 patch features, the patch classifier, a target's signature; guide: [docs/vision.md](docs/vision.md) |
| `just dagger local --run NAME [--smoke]`, `just dagger-worker`, `just dagger-merge`, `just dagger-bench` | DAgger for the TabPFN arena policy, on one or several machines; guide: [docs/navigation.md](docs/navigation.md) ("DAgger in the arena") |
| `just collect`, `just controller`, `just nav-eval`, `just dashboard` | The a-to-b controller: record walks, build with `fit()`, test, compare |

## Code layout

### `sim/`: the robot and the rooms ([docs/simulation.md](docs/simulation.md))

| File | Contents |
|---|---|
| `jethexa.xml`, `jethexa_simple.xml` | The robot with Hiwonder's meshes, and the same robot with simple shapes (used automatically when the meshes are missing). **Keep both in step**: a change to joints, actuators, sensors or physics goes in both |
| `jethexa_sim.py` | `JetHexaSim`: load, set joint targets, step; read joints, body, IMU, foot contacts and lidar; move the target marker; render the camera; the tripod gait with steering |
| `arena.xml`, `arena_simple.xml`, `arena_furniture.xml` | The arena: a closed 50 m² square room with 8 movable obstacles |
| `apartment.xml`, `apartment_simple.xml`, `apartment_decor.xml` | The arena dressed as a home for the app: zone patches, decoration and the Margarete figure, all without collisions. Only the app loads them |
| `clutter.xml`, `arena_clutter.xml`, `arena_clutter_simple.xml` | Twelve small solid clutter bodies for the app's Medium and Messy rooms. **Never include `clutter.xml` in the `arena*.xml` files**: it would change every training layout |
| `demo.py`, `fetch_meshes.py` | What `just sim`, `view` and `snapshot` run; the mesh download (pinned commit, SHA-256 checked) |

### `nav/`: walking with TabPFN ([docs/navigation.md](docs/navigation.md))

| Path | Contents |
|---|---|
| `nav/*.py` | Walking from a to b on an empty floor: `common.py` (walker, geometry), `db.py` (DuckDB recorder), `collect.py`, `hindsight.py`, `dynamics.py`, `policy.py` (`fit()`-only policies), `controller.py` (build and test a controller), `evaluate.py`, `run.py`, `dashboard.py`, `live_dashboard.py` |
| `nav/dagger/` core | Walking around furniture, learned with DAgger: `env.py` (`SimEnv`), `expert.py` (planner + pure pursuit, the teacher), `featurize.py`, `policy.py` (the TabPFN 3.5 `fit()`-only student), `rollout.py`, `worker.py`, `merge.py`, `loop.py`, `bench.py`, `timing.py`, `dagger.yaml` |
| `nav/dagger/` research | `experiment.py`, `stage.py`, `variants/` (one YAML per variant; best: `v29-lidarplan`), `lidarmap.py` (the robot's own plan on its lidar map), `report.py` (tables and pictures in `docs/dagger/`). Report: [docs/dagger-report.md](docs/dagger-report.md) |
| `nav/dagger/` target search | `target.py` (`Detector`: match a DINOv2 signature and range it; `Estimate`: fuse matches into a position; `TargetSearch`: the arena test), `target_check.py`, `video.py` |
| `nav/dagger/` for the app | `export.py` (a student's examples, exported once), `flatbench.py` (the student in the flats). `policy.py` with `backend: client` runs the student through the hosted TabPFN API (`tabpfn-client`; needs `TABPFN_TOKEN`) |

### `vision/`: what the camera sees ([docs/vision.md](docs/vision.md))

| File | Contents |
|---|---|
| `scene.py`, `dataset.py` | The arena with random layouts and robot poses; labeled pictures (class per pixel) |
| `features.py`, `probe.py` | Frozen DINOv2 patch features; the patch classifier and its `blocked_grid` |
| `target_signature.py` | A target's DINOv2 signature for the horizon band (the pillar, and Margarete) |

### `homes/`: where is she ([homes/README.md](homes/README.md))

| File | Contents |
|---|---|
| `flats.py` | The arena as a flat with five zones (bedroom, living room, dining area, kitchen, hallway); `scatter(flat, rng, level)` makes the app's mess levels clean, medium, messy |
| `residents.py`, `generate.py` | Routines and stated habits; diary and check-in tables |
| `search.py` | The Bayesian search: journal + habits + floor plan → belief. `Search` is the loop: most likely zone, look, update, repeat |
| `tabpfn_belief.py` | The same belief from TabPFN (`fit()` on synthetic check-ins); the app's search starts from it |
| `search_demo.py`, `search_figure.py`, `sim_bridge.py` | Numbers and pictures; the search driving the simulated robot |

### `app/`: the web app ([docs/app.md](docs/app.md))

| File | Contents |
|---|---|
| `server.py` | The API under `/api` (documented at the top of the file) and the built page |
| `robot.py` | `LiveRobot`: one thread owns the simulator, steers the robot and renders the views. Scene `apartment` is the furnished flat (`ApartmentScene`). Drivers: the student, then the a-to-b controller on an A* route. Commands: `goto`, `turn`, `look` (turn once, looking), `watch`, `approach` (walk up to her) |
| `student.py` | The v29 student in the app: loads in the background, sees through a plain copy of the room, keeps a credit floor |
| `spotter.py` | The robot's own eyes: the DINOv2 person detector the search looks with |
| `apartment.py` | The flat, journal, belief and the search loop behind the Apartment panel |
| `person_data.py`, `searchbench.py` | Pictures for Margarete's signature; the whole search many times, no browser |
| `narrate.py`, `llm.py`, `voice.py`, `call.py`, `knock.py` | The spoken commentary; the metacon chat; speech; voice calls; Find and Call Margarete |
| `web/` | The front end (React, from Lovable). Phone layout: `PhoneApp.tsx`. The "How it works" page: `components/jethexa/how/`, with its numbers and sources in `content/howItWorks.ts` |

### Everything else

| Path | Contents |
|---|---|
| `tests/` | Tests, one file per area |
| `data/` | Gitignored: `data/nav/` (recorded runs, `nav.duckdb`, student exports), `data/vision/` (pictures, features, signatures) |
| `deploy/vendetta/` | How the public copy of the app runs on Oscar's Mac ([README](deploy/vendetta/README.md)) |
| `docs/knock-*.md` | Knock, the team's concept: `knock-ui.md` (the Find and Call Margarete buttons, mockup in `docs/knock-ui/`) and `knock-routines.md`. Read `knock-ui.md` before building those buttons |

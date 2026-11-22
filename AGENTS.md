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
| `just dagger local --run NAME [--smoke]`, `just dagger-worker`, `just dagger-merge`, `just dagger-bench`, `just dagger-video`, `just dagger-augment`, `just strip-pca`, `just test-dagger` | DAgger for the TabPFN arena policy, on one or several machines; videos, adding columns by replay, the DINOv2 strip PCA; guide: [docs/navigation.md](docs/navigation.md) ("DAgger in the arena", "Inputs, settings and results") |
| `just app-build`, `just app [port]` | Build and run the web app (live sim + metacon chat) at http://127.0.0.1:8790; guide: [docs/app.md](docs/app.md) |

Lint: not set up yet.

If `just` is not installed, `uvx --from rust-just just <command>` runs it without installing.

### Code layout

| Path | Contents |
|---|---|
| `sim/jethexa.xml` | The robot with Hiwonder's meshes (needs `just meshes`) |
| `sim/jethexa_simple.xml` | The same robot with simple shapes; used automatically when the meshes are missing |
| `sim/jethexa_sim.py` | `JetHexaSim`: load, set joint targets, step, read joints, body, IMU and foot contacts, move the target marker, render camera; demo tripod gait with steering |
| `sim/arena.xml`, `sim/arena_simple.xml`, `sim/arena_furniture.xml` | The arena: a closed 50 m² square room with 8 movable obstacles, around the robot by `<include>` |
| `sim/demo.py` | What `just sim`, `just view` and `just snapshot` run |
| `sim/fetch_meshes.py` | Mesh download, pinned to one Hiwonder commit and checked by SHA-256 |
| `nav/` | Walking from a to b with TabPFN: `common.py` (walker, geometry), `db.py` (DuckDB recorder), `collect.py`, `hindsight.py` (goal examples), `dynamics.py` (motion examples and planner), `policy.py` (`fit()`-only TabPFN policies), `controller.py` (build and test a controller), `evaluate.py`, `run.py` (demo video), `dashboard.py`, `live_dashboard.py`. Guide: [docs/navigation.md](docs/navigation.md) |
| `nav/dagger/` | DAgger in the arena: `env.py` (`SimEnv`, 360° lidar), `expert.py` (planner + pure pursuit + edge recovery), `featurize.py` (patch classes and depths, depth-camera distances and scan, lidar, DINOv2 horizon strip, history), `policy.py` (TabPFN 3.5 `fit()`-only student, local or `tabpfn-client` API), `rollout.py` (lock-step robots, leg-aware safety), `worker.py`, `merge.py`, `loop.py`, `bench.py`, `augment.py` (add columns by replay), `repool.py`, `video.py`, `timing.py` (logs, ETAs); configs `dagger.yaml`, `depth.yaml`, `lidar.yaml`, `lidar2.yaml`, `lidar3.yaml` (best: 97.5%). Needs `TABPFN_TOKEN` (env or gitignored `.env`) |
| `app/` | The web app: `robot.py` (live sim thread), `llm.py` (metacon chat), `voice.py` (speech via Voice Control), `call.py` (voice calls shown live to every viewer), `knock.py` (Find and Call Margarete), `server.py` (API + page), `web/` (front end from Lovable; phone layout in `PhoneApp.tsx`). Guide: [docs/app.md](docs/app.md) |
| `vision/` | Camera perception: `scene.py` (the arena: random layouts, robot poses, class per pixel), `dataset.py` (labeled pictures), `features.py` (frozen DINOv2 patch features, patch labels), `probe.py` (patch classifier, `blocked_grid` for avoidance). Guide: [docs/vision.md](docs/vision.md) |
| `data/vision/` | Vision runs: pictures, labels, features, probes (gitignored, never commit) |
| `data/nav/` | Recorded runs, the database `nav.duckdb`, model checkpoints (gitignored, never commit) |
| `tests/` | Tests (`just test`; navigation tests: `just test-nav`) |
| `docs/knock-*.md` | Knock, the team's concept: `knock-ui.md` (the Find and Call Margarete buttons for `app/`, mockup in `docs/knock-ui/`) and `knock-routines.md` (data for the zone model). Read `knock-ui.md` before building those buttons |

Keep both model XML files in step: a change to joints, actuators, sensors or physics goes in both.

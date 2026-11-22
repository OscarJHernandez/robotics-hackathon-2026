# Commands for the simulated JetHexa. Run `just` to list them. Needs uv (https://docs.astral.sh/uv/).

# The live window needs mjpython on macOS; plain python elsewhere.
viewer_python := if os() == "macos" { "mjpython" } else { "python" }

# List the commands
default:
    @just --list

# First-time setup: Python, the virtual environment (.venv), packages and the robot meshes
setup: sync meshes

# Create or update the virtual environment from uv.lock
sync:
    uv sync

# Download the JetHexa meshes into sim/meshes/ (skips files already there)
meshes:
    uv run python sim/fetch_meshes.py

# Walk without a window and print where the robot ended up: just sim --seconds 10
sim *args:
    uv run python sim/demo.py {{ args }}

# Watch the robot walk in a live 3D window: just view 30
view seconds="20":
    uv run {{ viewer_python }} sim/demo.py --view --seconds {{ seconds }}

# Open the model in MuJoCo's own viewer to move the joints by hand
inspect:
    uv run python -m mujoco.viewer --mjcf=sim/jethexa.xml

# Save a picture of the robot after walking: just snapshot walk.png
snapshot file="walk.png":
    uv run python sim/demo.py --snapshot {{ file }}

# Run the tests
test:
    uv run python -m unittest discover -s tests -v

# Walk at random and record every gait cycle into data/nav/nav.duckdb: just collect --episodes 200 --images
collect *args:
    uv run --group ml python nav/collect.py {{ args }}

# Build a TabPFN controller (TabPFN's fit() on recorded examples, no training) and test it: just controller --name default
controller *args:
    uv run --group ml python nav/controller.py {{ args }}

# Test a controller again on the 20 held-out tasks, next to the baseline: just nav-eval --controller default
nav-eval *args:
    uv run --group ml python nav/evaluate.py {{ args }}

# Walk from a to b with a controller and save a video: just nav --controller default --target 1.0 0.6 --yaw 3.0
nav *args:
    uv run --group ml python nav/run.py {{ args }}

# Run SQL on the navigation database: just query "SELECT kind, count(*) FROM runs GROUP BY kind"
query sql:
    uv run --group ml python -c 'import sys; sys.path.insert(0, "nav"); from db import query; import pandas as pd; pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); print(query(sys.argv[1]).to_string())' "{{ sql }}"

# Write the controller dashboard to data/nav/dashboard.html (one file, works offline)
dashboard-file *args:
    uv run --group ml python nav/dashboard.py {{ args }}

# Run the navigation tests (needs the ml packages)
test-nav:
    uv run --group ml python -m unittest discover -s tests -p "test_nav*.py" -v

# Live controller dashboard at http://127.0.0.1:8770 (rebuilds from the database on every load)
dashboard port="8770":
    uv run --group ml python nav/live_dashboard.py --port {{ port }}

# DAgger for the TabPFN arena policy: just dagger local --run try1 [--smoke] (see docs/navigation.md, "DAgger")
dagger *args:
    uv run --group ml --group vision python -m nav.dagger.loop {{ args }}

# One machine's share of a DAgger round: just dagger-worker --run try1 --round 0 --plan plan.json --procs 8
dagger-worker *args:
    uv run --group ml --group vision python -m nav.dagger.worker {{ args }}

# Merge a DAgger round's parts and print timings: just dagger-merge --run try1 --round 0 --timings
dagger-merge *args:
    uv run --group ml --group vision python -m nav.dagger.merge {{ args }}

# Evaluation harness for DAgger (plans, summaries, comparisons, offline metrics): just dagger-eval plan --run try1 --set screen-50 --host vendetta
dagger-eval *args:
    uv run --group ml --group vision python -m nav.dagger.evaluate {{ args }}

# Measure this machine's DAgger speed and project the round's duration: just dagger-bench --procs 1,4,8
dagger-bench *args:
    uv run --group ml --group vision python -m nav.dagger.bench {{ args }}

# Run the DAgger tests
test-dagger:
    uv run --group ml --group vision python -m unittest tests.test_dagger -v

# Labeled camera pictures in the 50 m² arena: just vision-data --episodes 60 --poses 16
vision-data *args:
    uv run --group vision python vision/dataset.py {{ args }}

# DINOv2 patch features and patch labels for a vision run: just vision-features latest
vision-features *args:
    uv run --group vision python vision/features.py {{ args }}

# Train and test the patch classifier: just vision-probe latest [--hidden 256]
vision-probe *args:
    uv run --group vision python vision/probe.py {{ args }}

# Build the web app's front end (app/web, made with Lovable) into app/web/dist
app-build:
    cd app/web && npm ci && npm run build

# Run the web app (the robot in the apartment, the search, metacon chat) at http://127.0.0.1:8790: just app 8790 [--scene floor] [--driver planner]
app port="8790" *args:
    uv run --group app --group ml python app/server.py --port {{ port }} {{ args }}

# Where is she? Bayesian search (journal + habits + floor plan): results table and 6 examples in homes/data/search_examples/
search-demo *args:
    cd homes && uv run --group ml python search_demo.py {{ args }}

# One search in MuJoCo: the Bayesian search picks the zone, the simulated JetHexa walks there (nav/dagger expert)
search-sim *args:
    cd homes && uv run --group ml python sim_bridge.py {{ args }}

# Run the tests for homes/ (Knock data, arena flats, Bayesian search, does TabPFN's belief make sense)
test-homes:
    cd homes && uv run --group ml python -m unittest test_homes.py test_tabpfn_belief.py -v

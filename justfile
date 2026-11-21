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

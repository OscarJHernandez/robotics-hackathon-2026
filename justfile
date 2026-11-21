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

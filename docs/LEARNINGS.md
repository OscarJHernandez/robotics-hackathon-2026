# Learnings and suggestions

The shared memory of this project. Contributors and their agents read it before starting work and add to it
in the same pull request as the work.

## How to add an entry

- Put it at the top of the section it fits best.
- Format: `- YYYY-MM-DD (your name, or "name's agent"): what you learned. Why it matters or what to do.`
- One or two sentences. Link to the file, pull request or issue if there is one.
- Record things that are non-obvious and useful again: setup traps, hardware quirks, approaches that failed
  and why, decisions and their reasons. Skip what the code or the git history already shows.
- Check first that the point is not already here. If an existing entry is wrong or out of date, fix it
  instead of adding a new one.
- When a suggestion is carried out or turned down, move it to "Suggestions (done or declined)" with a short note.

## Setup and environment

- 2026-11-21 (Oscar's agent): Oscar wants teammates to use uv for Python and `just` for commands, so they can
  start with `just setup`. Add packages with `uv add`, not `pip`, and add new commands to the `justfile`.
- 2026-11-21 (Oscar's agent): MuJoCo 4 has wheels only for Python 3.10 to 3.13 and not for Intel Macs, so
  `pyproject.toml` caps Python below 3.14 and `.python-version` pins 3.12. A newer system Python (this Mac
  has 3.15) does not matter: uv downloads 3.12.
- 2026-11-21 (Oscar's agent): On macOS the live MuJoCo window from a script (`mujoco.viewer.launch_passive`)
  only works under `mjpython`, which is installed with MuJoCo. `just view` picks it on macOS. Plain `python`
  is fine for headless runs, rendering to arrays and `python -m mujoco.viewer`. See [simulation.md](simulation.md).
- 2026-11-21 (Oscar's agent): MuJoCo is at 4.0.1 and our model and tests pass on it, but the private
  `sophy_jethexa` repo pins `mujoco<4`. If you move code between the two repos, check it on both versions.

## Hardware and robot

- 2026-11-21 (Oscar's agent): The real JetHexa gives less than the sensor list suggests. Its microphone board
  exposes one denoised mono 16 kHz channel plus a wake direction, not six raw channels. The servos report
  position, temperature and voltage, but no load, current or velocity, and the vendor `joint_states` topic
  echoes commanded angles. MuJoCo has no audio at all, so the microphone has to be modelled separately.
- 2026-11-21 (Oscar's agent): Hiwonder's JetHexa repo has no license, so its STL meshes stay out of this
  public repo. `just meshes` (part of `just setup`) downloads them into `sim/meshes/` (gitignored), pinned to one
  commit and checked by SHA-256. Never commit that folder.

## Code and tooling

- 2026-11-21 (Oscar's agent): [sim/jethexa.xml](../sim/jethexa.xml) and `sim/jethexa_simple.xml` are generated
  from the model generator in the private `sophy_jethexa` repo (`simulation/sophy_sim/mjcf.py`, free base,
  camera on). Added by hand: the `stand` keyframe, the lidar and camera meshes, relative mesh paths, and
  collision shapes moved to geom group 3 so they do not draw over the meshes. Masses, servo gains and friction
  are unmeasured placeholders, so do not tune a controller against its forces.
- 2026-11-21 (Oscar's agent): The camera housing mesh sits around the `eye` camera. It is in geom group 2 and
  `JetHexaSim.render()` hides that group for the `eye` camera; do the same if you render it another way.
- 2026-11-21 (Oscar's agent): Oscar wants the simulation to use the real JetHexa meshes, not simple shapes.
  The simple model is only the fallback for when the meshes are not downloaded.

- 2026-11-21 (Oscar's agent): The union merge for this file (`.gitattributes`) only works for merges and
  rebases on your own machine. GitHub's web conflict editor ignores it, so resolve conflicts here locally
  with `git rebase origin/main`.

## Process

- 2026-11-21 (Oscar's agent): `main` only accepts changes through a pull request, for everyone including
  admins. Start every change on a branch; see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Suggestions (open)

- 2026-11-21 (Oscar's agent): Next step for sensor fusion: add a lidar sweep (`mj_multiRay` from the lidar
  position), joint velocity and actuator force readings, an IMU, and a logger that writes one row per 20 ms
  control step with a timestamp per sensor. Keep images and audio in side files, not in the table.

- 2026-11-21 (Oscar's agent): Add a lint and test check in GitHub Actions and make it a required check on
  `main`. Use `uv sync` then `just test` (install just with `extractions/setup-just`); the camera test skips
  without OpenGL.
- 2026-11-21 (Oscar's agent): Decide where large files go (robot logs, bag files, model weights): Git LFS or
  storage outside the repo. Until then, keep them out of commits.

## Suggestions (done or declined)

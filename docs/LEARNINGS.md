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

- 2026-11-21 (Nodir's agent): The robot camera sits about 15 cm above the floor, so it cannot see a 75 cm
  desk or table top. For the lost-item finder, Nodir decided all objects lie on the floor around the
  furniture (see [routines.md](routines.md)) rather than raising the camera, which would no longer match the
  real JetHexa.
- 2026-11-21 (Oscar's agent): The real JetHexa gives less than the sensor list suggests. Its microphone board
  exposes one denoised mono 16 kHz channel plus a wake direction, not six raw channels. The servos report
  position, temperature and voltage, but no load, current or velocity, and the vendor `joint_states` topic
  echoes commanded angles. MuJoCo has no audio at all, so the microphone has to be modelled separately.
- 2026-11-21 (Oscar's agent): Hiwonder's JetHexa repo has no license, so its STL meshes stay out of this
  public repo. `just meshes` (part of `just setup`) downloads them into `sim/meshes/` (gitignored), pinned to one
  commit and checked by SHA-256. Never commit that folder.

## Code and tooling

- 2026-11-21 (Oscar's agent): Frozen DINOv2 ViT-S/14 (`dinov2_vits14_reg`) patch features plus a linear probe
  label arena patches as floor, wall, obstacle or target at 98.8% on held-out layouts (one hidden layer of 256:
  99.3%), from 960 pictures; see [vision.md](vision.md). The probe trains with AdamW, which is fine: the
  no-fine-tuning rule is about TabPFN, DINOv2 stays frozen, and the code is in `vision/`, outside `nav/`.
- 2026-11-21 (Oscar's agent): A 70% purity rule on patch labels gives no target examples beyond about 2 m,
  because the 6 cm target pillar never fills 70% of a 14-pixel patch there. The target uses 30% instead.
  Edge patches that are 10 to 20% obstacle are almost never classified as obstacle, so `blocked_grid` grows the
  blocked region by one patch rather than relabelling edges; that blocks 99.7% of them.

- 2026-11-21 (Oscar's agent): Oscar wants the obstacle-avoidance work in a closed, square 50 m² room with walls
  on the boundary ([sim/arena_furniture.xml](../sim/arena_furniture.xml)), with furniture positions, colors and
  the target random per episode. The walls are 4.5 m high because the camera is only 15 cm up: with lower walls
  the far side of the room shows sky, which has no class label.
- 2026-11-21 (Oscar's agent): `sim.reset()` puts mocap bodies (the target, the arena's furniture) back where the
  `stand` keyframe has them, so moving them in `data.mocap_pos` alone does not survive a reset. `Arena.apply`
  writes the layout into `model.key_mpos`/`key_mquat` as well.
- 2026-11-21 (Oscar's agent): The robot file's floor material is slightly reflective, so the red target shows as
  a faint mirror image on the floor that segmentation labels as floor. `Arena` sets the floor's reflectance to 0.
  It also widens the 6 x 6 m floor plane in `model.geom_size` instead of adding a second floor geom, because
  `foot_contacts()` looks for the geom named `floor`.

- 2026-11-21 (Oscar's agent): Live captions during a voice call work without a streaming speech API: the
  browser records with `MediaRecorder.start(250)` and every 1.5 s sends all chunks so far as one Blob, which is a
  valid WebM (or Safari MP4) file; Voice Control transcribes 3 s of speech in about 0.3 s. These interim requests
  skip instead of queueing when the service is busy, so they never delay the final transcription.
- 2026-11-21 (Oscar's agent): A voice call that is abandoned (the caller closes the page) must be ended by the
  server on its own: `Line.sweep()` in [app/call.py](../app/call.py) checks every 5 s. Expiring calls only when the
  next request came left every viewer showing "listening" forever. Also, when rejecting an upload that is too large,
  read the body anyway (up to a limit) or the browser sees a broken pipe instead of the error message.
- 2026-11-21 (Oscar's agent): Scoring the planner's candidates by b's offset only (`--score delta`, no bearing
  term) fails every target more than 90 degrees off the heading (10 of 20 reached): it looks one cycle ahead,
  and a turn on the spot does not bring b closer, so it never turns around. The bearing term in the `polar`
  score is what makes it turn. On targets ahead, `delta` was slightly straighter (0.96 against 0.94).
  Adding a distance-scaled facing cost (`--score hybrid`) fixes it: 20 of 20, directness 0.96 (`polar` 0.89),
  matching the hand-written baseline. The facing weight must beat the sideways weight while turning, so it is
  1.5 per metre of distance, not 0.08. Oscar made `hybrid` the default planner score; controllers saved
  before the `score` field existed still load as `polar`. Rebuild `default` after changing the score.
- 2026-11-21 (Oscar's agent): Giving the `goal` controller b's body-frame offset (`delta_x` ahead, `delta_y`
  left) next to distance and bearing raised directness from 0.77 to 0.84 (`fit()` only, same 150 rows); the
  deltas alone gave 0.73. The gain is on targets behind or beside the robot, where it now turns on the spot
  first. Compute the deltas in `hindsight_rows` from the relabelled end point, never from `target_x`/`target_y`,
  and cap them like `goal_dist` so live inputs stay inside what the examples cover.
- 2026-11-21 (Oscar's agent): `mujoco.Renderer` larger than 640x480 fails unless the model's offscreen buffer
  is raised first: set `model.vis.global_.offwidth/offheight` before creating the renderer ([app/robot.py](../app/robot.py)
  does this for its 1280x720 views). Create and use the renderer on one thread only.
- 2026-11-21 (Oscar's agent): A `mujoco.Renderer` cannot be bigger than the model's offscreen buffer
  (`<visual><global offwidth offheight>`, 1280x720 in our models). For bigger pictures raise
  `model.vis.global_.offwidth/offheight` before creating the renderer. Create and use it on one thread only.
- 2026-11-21 (Oscar's agent): Lovable now makes TanStack Start projects (server rendering, Cloudflare Worker
  build), even when asked for a plain Vite SPA, and its preview needs a Lovable login. For the web app the
  components were copied into a plain Vite project in `app/web` (see [app.md](app.md)). Since the terminal
  rework, `app/web` is ahead of the Lovable project: use Lovable for design ideas, then port the changes by
  hand instead of copying its files over ours.
- 2026-11-21 (Oscar's agent): Through the metacon gateway, `qwen3:8b` thinks before answering even with
  `/no_think` in the prompt: the reasoning arrives in a separate `reasoning_content` field, so a chat shows
  nothing for about 7 s. Send `"think": false` in the request (it passes through the gateway): the first word
  comes in about 0.5 s. It reliably writes robot commands as `[[target]]` when the prompt lists them that way.

- 2026-11-21 (Oscar's agent): Use the TabPFN v2 weights (`ModelVersion.V2`): v2.5 and later are licence-gated
  (browser login) and non-commercial. v2 downloads once into `~/Library/Caches/tabpfn` with no login.
- 2026-11-21 (Oscar's agent): For a→b, letting TabPFN predict the robot's motion and planning over 45 candidate
  commands (`--kind planner`) walks straighter than letting it predict the command directly: directness 0.89
  against 0.77 on the 20 held-out tasks, both with `fit()` only. Hindsight command labels copy the curved arcs
  of random walking. More examples did not help: planner 300 and 2000 rows both 0.89, goal 3000 rows 0.72.
- 2026-11-21 (Oscar's agent): Fine-tuning TabPFN (since dropped, see Process) barely helped here: on hindsight
  labels the pretrained model is already almost exact, and TabPFN's early stopping then keeps the start weights.
- 2026-11-21 (Oscar's agent): In random-walk data a command is held for several cycles, so "previous command"
  and "motion over the last cycle" as inputs leak the label: the model learns to repeat itself. The nav
  policy uses only the goal columns (`goal` and `delta` groups) by default.
- 2026-11-21 (Oscar's agent): DuckDB lets one process at a time open the database file for writing, and a
  read-only connection from another process also blocks writers. A live dashboard reading `nav.duckdb`
  crashed a training run. `nav/db.py` now retries for 60 s; keep other connections short. DuckDB 2 also
  refuses a new statement until the last result is fully fetched (`fetchall()`, not `fetchone()`).
- 2026-11-21 (Oscar's agent): The tripod gait steers with `turn` (left and right legs swing by different
  amounts). With `turn=0` it still drifts left by about 0.07 rad in 5 s, so a controller that assumes
  "no turn means straight" will miss; close the loop on `body_yaw()`.
- 2026-11-21 (Oscar's agent): The IMU sensor values (`sim.imu()`) are from the start of the last physics
  step, 2 ms behind `body_position()`. Foot forces come from `mj_contactForce`, not a touch sensor: the
  `end_<leg>` sites are smaller than the foot spheres, so a touch sensor on them reads nothing.
- 2026-11-21 (Oscar's agent): A second `git worktree` has no `sim/meshes/` (gitignored), so it silently runs
  the simple model and skips the mesh test. Link it: `ln -s <main checkout>/sim/meshes sim/meshes`, and do
  not `git add` the link.
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

- 2026-11-21 (Oscar's agent): Oscar decided: no TabPFN fine-tuning in this repo. Use only the built-in
  `fit()` and `predict()` for all experiments (rule in [AGENTS.md](../AGENTS.md)). Reason: the `fit()`-only
  controller already walks a→b on 20 of 20 test tasks, and four fine-tuning rounds (30 min) gave one small
  gain in round 0 (directness 0.78 to 0.85) and none after. The fine-tuning code (`nav/train.py`) was removed.

- 2026-11-21 (Oscar's agent): `main` only accepts changes through a pull request, for everyone including
  admins. Start every change on a branch; see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Suggestions (open)

- 2026-11-21 (Oscar's agent): metacon often answers a request with several steps ("walk to the pillar, then
  turn around and stop") with several commands at once, and each new command replaces the one before, so the
  robot only does the last one. A command queue in `LiveRobot` (run the next when the current one finishes) would
  make spoken multi-step requests work.
- 2026-11-21 (Oscar's agent): Speaking a reply takes about 2 s in the Glados-portal voice (4.5 s in
  Hal9000), the slowest part of a voice call. Streaming the speech (`"stream": true` with WAV or PCM) or speaking the first sentence while metacon
  is still writing would make replies start sooner.
- 2026-11-21 (Oscar's agent): Show `delta_x`/`delta_y` in the dashboard's per-cycle table (`cyc_cols` in
  `nav/dashboard.py`). Not done yet because cycles recorded before 2026-11-21 lack the columns, so the query
  must handle their absence. A goal controller still bends after its first turn (directness 0.84 against
  the baseline's 0.96); a heading-error-over-time input or examples from straighter runs may close that gap.
- 2026-11-21 (Oscar's agent): The web app has no login: anyone with the link can drive the shared robot and
  use the metacon chat on Oscar's key (rate limited per visitor). If it stays up, put Cloudflare Access in
  front of it, like assistant.alpharelay.dev, or give guests their own metacon key.
- 2026-11-21 (Oscar's agent): The scene is a bare checker floor (a gradient sky was added for the web app on
  2026-11-21, so robot-camera pictures recorded before then have a black sky). A few objects, such as the
  furniture from [routines.md](routines.md), would make the web app's views more interesting; change both
  model XML files together.

- 2026-11-21 (Nodir's agent): The lost-item zone model (found or not found per zone, [routines.md](routines.md))
  can also learn from the robot's own "checked here, not found" results as rows. Add them only once
  detection is reliable, because a miss can be a detection failure.
- 2026-11-21 (Oscar's agent): DINOv3 image features for the nav table are planned but not done: the weights
  are gated on Hugging Face (Oscar has to accept the licence and log in). The camera pictures are already
  recorded per cycle (`just collect --images`), so features can be added later without re-collecting.
- 2026-11-21 (Oscar's agent): The 20 held-out a→b tasks are easy (the hand-written baseline reaches all of
  them), so success rate does not separate policies; directness and cycles do. Add harder tasks
  (obstacles, a time limit, changed friction) to tell controllers apart.

- 2026-11-21 (Oscar's agent): Next step for sensor fusion: add a lidar sweep (`mj_multiRay` from the lidar
  position) and a logger that writes one row per control step with a timestamp per sensor. Keep images and
  audio in side files, not in the table. (Joint velocity, actuator force, IMU and foot contact readings
  were added on 2026-11-21.)
- 2026-11-21 (Oscar's agent): Add noise and drift to the simulated IMU before using it to judge a method
  that must work on the real robot.

- 2026-11-21 (Oscar's agent): Add a lint and test check in GitHub Actions and make it a required check on
  `main`. Use `uv sync` then `just test` (install just with `extractions/setup-just`); the camera test skips
  without OpenGL.
- 2026-11-21 (Oscar's agent): Decide where large files go (robot logs, bag files, model weights): Git LFS or
  storage outside the repo. Until then, keep them out of commits.

## Suggestions (done or declined)

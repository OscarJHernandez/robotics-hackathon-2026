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

- 2026-11-22 (Oscar's agent): A new git worktree has no `sim/meshes/` (gitignored), so everything renders the
  stick-figure robot of `jethexa_simple.xml`. Run `just meshes` in the worktree before recording pictures or videos.
  The README's pictures in `docs/readme/` are cut from one recorded app search (orbit and robot-eye frames).
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

- 2026-11-22 (Oscar's agent): https://jethexa.alpharelay.dev is served by the laptop (the `relay` tunnel runs
  there despite its name `vendetta-mac-apps`), so it goes down when the laptop sleeps. The always-on copy is
  https://jethexa-vendetta.alpharelay.dev on vendetta; see [deploy/vendetta/README.md](../deploy/vendetta/README.md).
- 2026-11-22 (Oscar's agent): A fresh host needs the gitignored `data/nav/controllers/default` (328 KB) and
  `data/nav/dagger/v2/student/v29` (23 MB) copied in, or the app falls back to weaker drivers without an error.
  Run from a worktree, the app looks for them (and `.env`) in the main checkout.
- 2026-11-22 (Oscar's agent): On a Mac with `~/.cloudflared/config.yml`, `cloudflared tunnel route dns <name> <host>`
  routes to the tunnel in that file, not to `<name>`: our new record first pointed at the `voice` tunnel. Pass
  `--config <the new tunnel's yml>` to every `cloudflared tunnel` command.

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

- 2026-11-22 (Oscar's agent): Test the app's search live with the student (realtime, API) before calling it done:
  `just search-bench` uses the A* walker and no realtime pauses, so it missed a race that broke every live search.
  While the student decides, `LiveRobot` publishes the paused scene before it reads new commands; that state said
  "idle" for a command still queued, so the search took a new `approach` or `look` for finished. `_publish` now
  keeps `mode` and `approach` while the queue is not empty (`tests/test_person_search.py` `PendingCommandTest`).
- 2026-11-22 (Oscar's agent): Range Margarete with the depth camera, not the lidar: her geoms have no collisions,
  so the lidar passes through her (that is what makes the look-alike check work). Walking up to her, recompute the
  destination every gait cycle from the detector's estimate (`approach` in `app/robot.py`); the student needs no
  restart for a new end point, since its lidar planner plans from the goal it is given on every call.
- 2026-11-22 (Oscar's agent): The app felt slow because each student decision blocked its one simulator thread:
  about 0.6 s of every decision is the TabPFN API round trip (the local inputs, DINOv2 included, take 48 ms; a 1280x720
  render plus JPEG 4 ms). The streams fell to 2 to 3 pictures per second and x3 ran at about x1. Render while waiting,
  and use `--student-backend local` where there is a GPU (0.18 s per predict on vendetta). See docs/app.md.
- 2026-11-22 (Oscar's agent): Run the app under launchd with `ProcessType` `Interactive`: without it macOS throttled
  the agent's timers and the camera streams gave about 5 pictures/s, while the same code started from a shell gave
  11.5 (measured on vendetta at x3 with three views). Test speed fixes on the launchd-run app, not only a shell copy.
- 2026-11-22 (Oscar's agent): Run the whole suite (`just test`) before a PR, not a hand-picked list: #31 shadowed a
  parameter in `LiveRobot._publish` and rendered every view on every step (x3 ran at x1.3 to x1.8);
  `tests/test_robot_speed.py` caught it but was not in the list. Fixed in #33.
- 2026-11-22 (Oscar's agent): Oscar decided the deployed apps use the TabPFN API for the student ("use the API, you
  have the key"), not `--student-backend local`, even though local is faster per decision and free. Don't switch a
  deployment to local to save time or credits without asking him.

- 2026-11-22 (Oscar's agent): In the apartment scene the robot camera could not see anything closer than 1.1 m:
  MuJoCo's near clipping plane is `vis.map.znear` x the model's extent, and the decor and clutter parked 40-50 m away
  make the extent 112 m. Margarete, 0.6 m from the viewing pose, was invisible to the camera and its depth.
  `ApartmentScene` now sets the near plane to 5 cm. Check `model.stat.extent` before trusting close-up pictures of a
  model with parked bodies.
- 2026-11-22 (Oscar's agent): Make a detector's signature in the scene it will run in. Margarete's from the dressed
  flat (`just person-data`, 3,000 pictures at every mess level) found her in 97.7% of held-out pictures with no false
  alarms; a signature made before the clutter existed would have met it only live. Lying and standing need their own
  prototypes (3 each).
- 2026-11-22 (Oscar's agent): A close-up check of a 1.6 m long person must accept matches anywhere on her body. With
  0.5 m (the pillar's fuse radius) the first live search called her a false alarm twice: the first sighting was her
  feet, up close the camera matched her middle. The check now counts matches within 1.0 m of the spotted point.
- 2026-11-22 (Oscar's agent): The target detector's margin rule (the best place must beat everything else in view)
  rejected a person filling the view up close. Compare only against columns outside the matched stretch.
- 2026-11-22 (Oscar's agent): New obstacle bodies must never go into `sim/arena_furniture.xml` (or `arena.xml`,
  `arena_simple.xml`): `vision/scene.py` `Arena.randomize` places every `obs_` body, so one more body changes every
  recorded DAgger layout. The app's clutter lives in `sim/clutter.xml`, included only by the apartment files and the
  twin `sim/arena_clutter*.xml`; `tests/test_student_twin.py` pins a `randomize` layout to catch a change.
- 2026-11-22 (Oscar's agent): A `mujoco.Renderer` must be closed on the thread that made it. If a `LiveRobot` sim
  thread ends and the renderer is garbage-collected on the main thread, macOS hangs in CGL `make_current` (no error,
  0% CPU). `LiveRobot(max_steps=...)` closes both renderers before the thread ends.
- 2026-11-22 (Oscar's agent): The app's speed-up (x2, x3, x5) only paces the loop; each TabPFN decision still
  blocks it, so the reached rate is lower than asked when TabPFN is slow. Read `speed_actual`, not `speed`, when
  timing anything at a higher speed. Measured on this Mac with the local a-to-b controller walking: x1 0.84, x2 1.4,
  x5 2.4; idle x5 reaches 5.0.
- 2026-11-22 (Oscar's agent): Since the mess levels, `flats.scatter(flat, rng)` defaults to "medium" (the four loose
  pieces plus 4 clutter items) and checks reachability at the app's ROUTE_MARGIN (0.42 m), so flatbench's
  "scattered" legs now include clutter: numbers recorded before 2026-11-22 for "scattered" had the four pieces only.

- 2026-11-22 (Oscar's agent): TabPFN API (tabpfn-client 0.10.1) costs, measured with the v29 export (31,586 rows
  stacked to 63,172 x 117): `fit()` is free; with `fit_mode="fit_with_cache"` a `predict()` costs 10,000 credits and
  takes 0.6 to 3.5 s; with the server default ("fit_preprocessors") every `predict()` re-runs the whole context:
  13.4 s and about 11,400 credits. Use the cache (`ClientTabPFNPolicy` does) and put many robots in one call.
- 2026-11-22 (Oscar's agent): A policy fitted in the plain arena should not look at the app's dressed scene (wood
  floor, low walls, zones, decor, the person): `app/student.py` renders its camera in a second, plain MuJoCo model
  of the same layout and pose (the "plain twin"), and casts the lidar in the real scene, which ignores geoms without
  collisions. `tests/test_student_twin.py` proves the inputs equal `FlatEnv`'s; reuse the idea for any dressed view.
- 2026-11-22 (Oscar's agent): After the DAgger v2 merge, `SimEnv.step` reads `best_dist`/`since_gain` that only
  `SimEnv.reset` sets, so `homes/sim_bridge.FlatEnv` (which never calls reset) crashed in `just test-homes`;
  `FlatEnv.load`/`set_goal` now set them. A subclass that skips `reset()` must set every field `step()` reads.
- 2026-11-22 (Oscar's agent): Contact checks every 20 ms count a foot tapping the same wall many times per second;
  `ContactCounter` counts a new touch only after 1 s without contact.

- 2026-11-22 (Oscar's agent): Asked to "say the same thing" in one sentence, qwen3:8b on metacon returns the draft
  word for word; told "do not copy the draft" it rewords, but once said "I'm not in the bedroom" (the robot, not
  Margarete) and once turned "found her" into "didn't find her". So `app/narrate.py` keeps a template and checks every
  reply (same zones, no new numbers, same outcome, no "I'm not in"); test prompt changes against those cases.
- 2026-11-22 (Oscar's agent): The search can look in the same zone twice (the look misses her 20% of the time), so a
  line like "Not in the bedroom, the bedroom is now most likely" happens; `narrate.template` says "I'll look again".
- 2026-11-22 (Oscar's agent): Safari on iPhone will not play a `<video>` from a server that ignores `Range`, and
  `app/server.py` used to send every file whole with 200. `static()` now answers single ranges with 206 (tests in
  `tests/test_server_static.py`); check with `curl -H "Range: bytes=0-1"` (206, 2 bytes) before adding a video.
- 2026-11-22 (Oscar's agent): The server caches everything but `index.html` for a year (`immutable`), so media must
  be imported from TypeScript (Vite adds a content hash to the name), never put under a fixed name in `public/`.
  A missing file with an extension now gets 404 instead of `index.html`, so an old hashed name fails loudly
  instead of being cached as HTML.
- 2026-11-22 (Oscar's agent): The "How it works" page's sources disagree in small ways: v29's screening result is
  98.5% goals and 1.5% collisions (3 of 200) in `docs/dagger/promotion.md` but "98%, 2%" in STATUS.md and
  navigation.md, and the "stop and turn" share is 6.6% in LEARNINGS but 6.5% in `dagger-report.md`. The page uses
  the exact per-run table (`promotion.md`). Every number it shows is in `app/web/src/content/howItWorks.ts` with its
  source.
- 2026-11-22 (Oscar's agent): For a target the lidar cannot see (the pillar), use that as a check: a DINOv2 match
  the lidar also sees at the depth camera's range is solid furniture. With the real student, v32 drove into such
  look-alikes in 9 of 13 collisions; the check rejected 29 of 53 false matches and 0 of 1,452 true ones
  (`target.lidar_sees_target`). A lidar-visible target (a person) needs another check.
- 2026-11-22 (Oscar's agent): `experiment run` drives only 100 layouts unless `--closed-count 200` is given
  (CLOSED_DEFAULT); v29's final test used 200. Finished episodes are skipped, so rerun with the count to add the rest.
- 2026-11-22 (Oscar's agent): To match the target in the DINOv2 horizon band, centre the band columns on the mean
  background column and keep several prototypes. One plain mean of the target columns found the pillar in no held-out
  picture beyond 2 m (column recall 42%); centring plus 3 k-means prototypes reached 77% recall at 98% precision and
  76% of pictures at 2-3 m. See `vision/target_signature.py`.
- 2026-11-22 (Oscar's agent): A look-around at full turn rate (0.25, 0.53 rad per gait cycle) keeps the target in the
  62 degree view for only about 2 steps, too few to confirm a match: 75% found at the viewing point. Turning at 0.15
  (about 20 steps a turn) found 92.5%. Debounce with agreeing detections, not with "2 of the last 3 steps".
- 2026-11-22 (Oscar's agent): A search must not start its "is it really here?" look-around farther from its
  estimate than the episode's goal radius (0.25 m). At 0.35 m it spun next to the true pillar, could not confirm it so
  close, and rejected a correct estimate as a false positive in most episodes.
- 2026-11-22 (Oscar's agent): The pasted target-search design named `vision/strip_pca.py`, `extract(band=6)`, 23 cells,
  a leg-aware safety layer and `augment.py`. Those are on `feat/dagger-lidar-depth-recovery`, not on PR 22's
  `feat/dagger-v2-fixpass`. On PR 22 the band is patch rows 11-14 in `featurize.py` (46 columns), there is no safety
  layer and no reverse (stride >= 0), and v29 runs through `rollout.run_v2`, not `run_episodes`.
- 2026-11-22 (Oscar's agent): 8 of v29's 9 final-test collisions are into the table's legs (cylinders 5 cm across):
  the simulated lidar's 36 rays are 10 degrees apart, 17 cm at 1 m, so a leg falls between rays until it is close and is
  missing from the robot's lidar map. Thin obstacles need a denser lidar; see [dagger-report.md](dagger-report.md).
- 2026-11-22 (Oscar's agent): Adding every row to v29's context (81k, mostly student-driven DAgger rows) made it
  worse (97% -> 92% on screening). A large mix of student-driven rows dilutes the clean expert context.
- 2026-11-22 (Oscar's agent): DAgger v2 in the arena: the best student (98% goals, 2% collisions on 200 fresh layouts,
  two start seeds) is TabPFN 3.5 on all round 0 rows with a 360 degree lidar and the robot's own plan on its lidar map
  (`nav/dagger/lidarmap.py`). The camera-only students plateaued at 58-70%; see docs/navigation.md "DAgger v2".
- 2026-11-22 (Oscar's agent): A privileged ceiling experiment (the expert's look-ahead bearing as input) found the
  real gap in one run: 100% vs a 95% plateau, so the student lacked path information, not obstacle sensing or model
  capacity. Run a ceiling before tuning features or outputs.
- 2026-11-22 (Oscar's agent): For TabPFN students more real rows helped (8k -> 31k: 89% -> 95%), while re-weighting
  the same rows (balanced or lidar-distance quotas), gating, history and 8 estimators did not; beyond 31k rows (52k,
  61k, incl. DAgger rounds) nothing more. Compact inputs (114 columns) make 31k-row fits take about 4 min on an M4 Pro.
- 2026-11-22 (Oscar's agent): Hand-written lidar safety stops made it worse (stuck, spinning): the JetHexa touches
  furniture while turning on the spot, so a reflex at contact distance is too late.
- 2026-11-22 (Oscar's agent): In zsh `$h:Projects` applies the `:P` modifier, so `scp file $h:Projects/...` wrote to a
  local folder `vendettarojects/` with exit code 0. Quote as `"${h}:..."`; `nav/dagger/transfer.py` uses argument lists.
- 2026-11-22 (Oscar's agent): Split DAgger work by machine: a 31k-row TabPFN fit takes about 4 min on vendetta (M4 Pro,
  MPS) and 20 min on metacon (CPU), so fits and variant tests go to vendetta and simulation-heavy jobs (data, DAgger
  rounds) to metacon with 8 processes. The v2 search used 8.2 machine-hours in 5 h; see docs/dagger/compute.md.
- 2026-11-22 (Oscar's agent): TabPFN 3.5 on metacon's CPU is fine for compact inputs: 8k x 114 fit in 142 s, predict
  for 32 rows 0.2 s (faster than MPS for small batches). DINOv2 on CPU (about 115 ms per picture) is then the bottleneck.
- 2026-11-22 (Oscar's agent): The offline "stop recall at stride < 0.02" metric is unstable for plain regressors (two
  identical students: 0% vs 55%); judge with closed-loop runs, paired on the same layouts.

- 2026-11-22 (Oscar's agent): TabPFN ignored a strong journal when it only had the 13 raw journal-share columns
  (ten dining-area afternoons lowered "dining area"), although the examples hold the pattern. Adding the counting
  posterior as six features (`prior_*` in `homes/tabpfn_belief.py`) fixed it; `homes/test_tabpfn_belief.py` checks
  that and nine other "does it make sense" behaviours. Adding her own journal records (x4) to the synthetic
  examples in `fit()` is not worth it: on 12 test residents the most likely zone was right 73.8% → 74.0% and log
  loss went 0.809 → 0.792, but it needs a new fit of about 17 s per resident, so it was not adopted.
- 2026-11-22 (Oscar's agent): TabPFN's "where is she" belief went from 33% to 71% right (level with the counting
  model) by changing what goes into `fit()`, not the model: the examples are now synthetic **check-ins** instead of
  diary records, and the row has the same inputs as the counting belief (`homes/tabpfn_belief.py`). Diary records,
  one per part of the day, never show the case "heard a few minutes ago in this part of the day", which is half of
  all check-ins. Oscar: the search must use TabPFN; the app's default is now `--belief tabpfn`.
- 2026-11-22 (Oscar's agent): The TabPFN API key lives in a worktree's gitignored `.env` (`TABPFN_TOKEN=...`), and
  `nav/dagger/envfile.py` only looks in this worktree and the main checkout. A new worktree needs the file copied
  in (or put it in the main checkout once); without it everything silently falls back to the v2 weights.
- 2026-11-22 (Oscar's agent): The web app's page keeps long-lived connections open (picture streams, `/api/live`),
  and a browser allows six per host. Two `EventSource`s plus three streams left one for all ordinary requests, and an
  `<img>` stream is not released when the element is removed unless its `src` is emptied first, so adding a polled
  endpoint made the page drop to demo mode. Now `/api/live` is shared (`app/web/src/lib/live.ts`) and `Stream.tsx`
  empties `src` on removal. Count the connections before adding another stream or poll.
- 2026-11-22 (Oscar's agent): Anything that waits for `LiveRobot` to be "idle" after a command must first see it
  leave idle: the simulator thread can be busy for a moment (route planning, TabPFN predicting) and still publish
  the old state. `Apartment._settle` in `app/apartment.py` does this; the first version ended every search at once.
- 2026-11-22 (Oscar's agent): The app's home look and the Margarete figure live in `sim/apartment_decor.xml`, which
  only `sim/apartment.xml` includes, so `arena.xml`, the vision labels and the DAgger scenes are unchanged. The
  TabPFN a-to-b controller reads body motion and the target's direction, no camera, so the look cannot affect it.
  The DINOv2 probe and the DAgger student were made on the plain arena: do not expect them to work on the dressed
  scene's camera pictures without checking.
- 2026-11-22 (Oscar's agent): A git worktree has no `sim/meshes/` (gitignored), so the app silently draws the
  simple-shape robot. Copy them from the main checkout or run `just meshes` in the worktree.
- 2026-11-22 (Oscar's agent): `homes/search.py` is a counting Bayesian search (population share + stated habits + her
  journal + last heard) that checks zones from the most to the least likely: Oscar wants the order by probability,
  not by distance. On 120 test residents it finds her after 1.54 zones against 3.30 for "nearest first" and 2.12 for
  "last heard first", and before she is heard its belief is within 1 point of her true routine. It is the baseline a
  TabPFN belief has to beat (the TabPFN smoke test still loses to "last heard": 0.33 against 0.60).
- 2026-11-22 (Oscar's agent): In the search loop a zone the robot looked at must not go to 0: with a look that misses
  her 20% of the time, ruling zones out finds her in 80% of searches, the Bayes update (`after_miss` with a miss rate)
  in 99%, because it returns to a zone when the others are empty (25% of searches).
- 2026-11-22 (Oscar's agent): The search can drive the simulated robot today: `homes/sim_bridge.py` subclasses
  `nav.dagger.env.SimEnv` (`FlatEnv`), loads a flat with `arena.apply(flats.arena_layout(flat))`, and walks to each
  `Search.goal()` with the DAgger `Expert`; 330 of 330 walks arrived. `SimEnv` has no method to change the goal
  mid-episode, so `FlatEnv.set_goal` sets `goal`, `steps` and `done` itself, and it skips rendering unless `render=True`.
- 2026-11-22 (Oscar's agent): Check plot text with code, not by eye: `search_figure.text_problems()` measures every text
  box and found 18 overlaps in pictures that looked fine at a glance. Oscar had to point out the mushed words twice.
- 2026-11-22 (Oscar's agent): Two traps when making "where is she" data look good: stated habits copied from her true
  routine make the journal useless (it can only repeat the form), and one week of journal (3 records a day) adds only
  1-2 points. `stated_habits()` now names a wrong zone in 30% of the parts of the day, and the demo uses 4 weeks.
- 2026-11-21 (Oscar's agent): DAgger round 1 student (TabPFN 3.5 fitted on 32k expert rows) matched the expert
  closely at beta 0.5 (100/100 goals, mean |d turn| 0.05) but collided in 3 of 3 held-out layouts when driving alone:
  it keeps walking when the expert says stop and turn on the spot, a label only 6.6% of the rows have. Judge a
  student with beta 0 runs, and consider a context balanced over the labels (`max_context_rows`).
- 2026-11-21 (Oscar's agent): TabPFN 3.5 needs `TABPFN_TOKEN` plus a one-time licence acceptance on
  ux.priorlabs.ai; the key alone gives "license ... has not been accepted". With 2,400 columns on an M4 Pro, `fit()`
  took 89 s at 5k rows, 395 s at 20k and 696 s at 32k (two regressors); `predict()` for 16 robots took 1.2-1.6 s.
- 2026-11-21 (Oscar's agent): The JetHexa's middle feet reach about 0.2 m sideways, so a planner margin of 0.22 m
  made the expert touch furniture in 3 of 4 episodes; 0.32 m gave 12/12 and 998/1000 goals.
- 2026-11-21 (Oscar's agent): victory and metacon had no OpenGL libraries, so MuJoCo could not render. After
  `sudo apt-get install libegl1 libgl1 libgbm1 libegl-mesa0 mesa-libgallium libgl1-mesa-dri` and the `render`
  group, `MUJOCO_GL=egl` renders on the Radeon iGPU at about 1 ms per picture. OSMesa did not work.
- 2026-11-21 (Oscar's agent): On the CPU-only Linux boxes DINOv2 perception is 90-95% of a DAgger step; 8
  processes give only twice the speed of 1 (about 15 robot-steps/s). vendetta (MPS) peaks at 4 processes, 44/s.
- 2026-11-21 (Oscar's agent): A 32-process benchmark ran metacon out of memory; earlyoom then stopped Ollama and
  the user's systemd services (Minerva, Sophy). `nav/dagger/guard.py` now caps processes to free memory and stops
  workers under a reserve; a `pkill -f` pattern must not match its own ssh command (use `[r]obotics...`).
- 2026-11-21 (Oscar's agent): `multiprocessing.Pool.map` waits forever if one worker is killed; use
  `ProcessPoolExecutor` (raises `BrokenProcessPool`), as `nav/dagger/worker.py` does.
- 2026-11-21 (Oscar's agent): Copies to the homelab machines sometimes did not arrive although rsync/scp
  exited 0 (cause unknown). Check the file on the other side before starting a job that needs it.
- 2026-11-21 (Nodir's agent): `homes/` has the synthetic Knock data: 8,500 flats (since 2026-11-22: the MuJoCo arena with five named zones, bedroom, living room, dining area, kitchen and hallway, built around the arena's furniture), a weekly routine table per resident, 3 diary records a day for 7 days (102,000 training rows, `homes/data/train.parquet`), test check-ins and a cold-start zone prior; see [homes/README.md](../homes/README.md). With only 3 records a day TabPFN learns little about one person (mostly the population prior plus the slot and where she was last heard), so judge it on held-out residents and keep the `out` rows in the context. In the smoke tests TabPFN does not beat "last heard" on check-ins with 3 records a day (0.43 against 0.53) or with 12 (0.48 against 0.67), so do not promise it does. Zones are names, not `z1`-`z6` as in [knock-routines.md](knock-routines.md); `app/knock.py` still uses its old six names (`bed`, `bathroom`, ...), switching it to `homes/flats.py` and `homes/search.py` is open.

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

- 2026-11-21 (Nodir's agent): For Knock's zone model, how often the robot hears her decides whether TabPFN has
  anything to beat. In a quick simulation of [knock-routines.md](knock-routines.md), 0.6 heard events an hour
  gave 55% best possible top-1 against 51% for "usual place at this hour"; 1 an hour gave 59% against 51%.
  Columns that never change in the context rows (like `answered_robot` in heard rows) teach TabPFN nothing, so
  leave them out of `fit()`.
- 2026-11-21 (Oscar's agent): metacon (qwen3:8b) read "walk/move towards the target" and "get closer to the
  target" as a small step, `[[forward 0.13]]` to `[[forward 0.5]]`, while "go to the target" gave `[[target]]`; the
  0.13 came from the prompt's walking speed. So on voice calls the robot stopped short. The prompt now says any
  request to go to, towards or closer to the pillar is `[[target]]` and never forward or goto, and no longer gives
  the speed: 66 of 66 answers right across 11 phrasings, voice and text, but only with an empty chat history.
  Check a prompt change with several phrasings against metacon, not one, and also with old answers in the history.
- 2026-11-21 (Oscar's agent): qwen3:8b copies commands from its own earlier answers over the system prompt: after
  the prompt fix above, a page that still held one old "[[forward 0.13]]" answer got `forward 0.13` again 3 of 3
  times, and a reminder after the history did not help. Removing the commands from past answers made it stop
  writing commands at all. What works: send only the newest user message as a message, and put the earlier turns,
  without their commands, in the system prompt as background (`earlier_text` in [app/llm.py](../app/llm.py)).
  Follow-ups ("now turn around", "do that again") still work.
- 2026-11-21 (Oscar's agent): In the web app every `goto` and `forward` used to move the red pillar to its end
  point, so when metacon answered "walk to the target" with `[[forward 1]]` (likely with spoken words on the phone)
  the pillar jumped to just in front of the robot. Walking no longer moves the pillar (only `random` does), and the
  prompt says target, goal and end point all mean `[[target]]`.
- 2026-11-21 (Oscar's agent): `Hub.current()` in [app/call.py](../app/call.py) replays only the voice call in
  progress, and `useLiveCall` drops events without a call id. Anything else on `/api/live` (the Knock card) needs
  its own snapshot for pages that open late (`Knock.latest()`, written first by `live()`) and its own listener.
- 2026-11-21 (Oscar's agent): Live captions during a voice call work without a streaming speech API: the
  browser records with `MediaRecorder.start(250)` and every 1.5 s sends all chunks so far as one Blob, which is a
  valid WebM (or Safari MP4) file; Voice Control transcribes 3 s of speech in about 0.3 s. These interim requests
  skip instead of queueing when the service is busy, so they never delay the final transcription.
- 2026-11-21 (Oscar's agent): A voice call that is abandoned (the caller closes the page) must be ended by the
  server on its own: `Line.sweep()` in [app/call.py](../app/call.py) checks every 5 s. Expiring calls only when the
  next request came left every viewer showing "listening" forever. Also, when rejecting an upload that is too large,
  read the body anyway (up to a limit) or the browser sees a broken pipe instead of the error message.
- 2026-11-21 (Nodir's agent): The one-room scene ([sim/room_furniture.xml](../sim/room_furniture.xml)) is added
  to the robot with MJCF `<include>` in `sim/room.xml` and `sim/room_simple.xml`, so the generated robot files
  stay untouched. Load it with `JetHexaSim(SIM_DIR / "room_simple.xml")`. The obstacles are solid, and the a→b
  controllers do not avoid them yet.
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

- 2026-11-22 (Oscar's agent): After PR #27 was squash-merged, PR #28 (stacked on it) showed 15 conflicting files
  against `main` though `main`'s tree equalled #27's head: the squash drops the shared history. Oscar wants such
  work moved to a new branch off `main`, without merging: `git restore --source=<old branch> --staged --worktree
  -- .`, add back anything `main` has that the old branch lacked, check `git diff --cached origin/main`, commit.
- 2026-11-21 (Oscar's agent): Oscar decided: no TabPFN fine-tuning in this repo. Use only the built-in
  `fit()` and `predict()` for all experiments (rule in [AGENTS.md](../AGENTS.md)). Reason: the `fit()`-only
  controller already walks a→b on 20 of 20 test tasks, and four fine-tuning rounds (30 min) gave one small
  gain in round 0 (directness 0.78 to 0.85) and none after. The fine-tuning code (`nav/train.py`) was removed.

- 2026-11-21 (Oscar's agent): `main` only accepts changes through a pull request, for everyone including
  admins. Start every change on a branch; see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Suggestions (open)

- 2026-11-22 (Oscar's agent): Measure the student itself in the Medium and Messy rooms (`just student-bench --levels
  medium,messy`, real student) once the TabPFN API's daily limit allows; so far only A* + a-to-b was run there.
- 2026-11-22 (Oscar's agent): The clutter has fixed household colours, while the student was trained on random
  furniture colours; if it misses clutter in the camera, give the clutter random colours in the twin as in training.

- 2026-11-22 (Oscar's agent): Run `just student-bench` (40 flats x home/scattered) on vendetta with the real student
  before claiming numbers for the app; the flats are tighter than the training layouts (0.10 m furniture gaps).

- 2026-11-22 (Oscar's agent): Target search: make Margarete lidar-visible (her geoms are contype 0 like the pillar) and
  build a signature from labelled pictures of her lying and standing, then rerun `v32-target-search` with
  `range_source: auto`. Also plug `homes/search.py`'s belief in as a Schedule instead of the random order.
- 2026-11-22 (Oscar's agent): Try a denser simulated lidar (for example 360 rays at 1 degree) aimed at the table-leg
  crashes: a new round 0 (about 15 min) and one refit of v29.
- 2026-11-22 (Oscar's agent): Check the simulated lidar against the real JetHexa lidar (rays, range, noise) and
  test `lidarmap.py` with real odometry drift before trusting the 98% on the robot.
- 2026-11-22 (Oscar's agent): The `stage analyse` printout says "expert on screen" also for the confirmation and final
  sets; rename it to the set's name.
- 2026-11-22 (Oscar's agent): Knock: switch `app/knock.py` and the phone UI to the five arena zones and use
  `homes/search.Search` for `probs` and the next zone. In `homes/sim_bridge.py`, turn the robot to `heading_deg` at the
  goal and replace the mock look with the vision probe's "target" class (the pillar already stands where she is). Add a bathroom to the arena (a piece or a
  marked floor area in both arena XML files) so falls in the bathroom can be part of the story.

- 2026-11-21 (Oscar's agent): DAgger: try a label-balanced context (`max_context_rows: 20000`) from round 2 on,
  and run DINOv2 on the Radeon iGPUs (ROCm torch) to speed up the Linux workers.
- 2026-11-21 (Oscar's agent): `vision/scene.py` places the target only 0.15 m from furniture; both expert
  collisions in DAgger round 0 were targets in such gaps. Use about 0.35 m. The chase camera in
  `nav/dagger/video.py` clips into walls near corners.

- 2026-11-21 (Oscar's agent): Knock on the desktop page (mockup screens 1-4): the phone part is built, and
  `KnockCard.tsx` and `useKnock` are ready to reuse under `ViewList`. The zone poses in `app/knock.py` still wait
  for the room layout to be agreed between `sim/room_furniture.xml` and [knock-routines.md](knock-routines.md), and
  the app does not load the room yet. Then: the zone model's `predict()` in place of `PRIORS`, the person detector
  in place of `KNOCK_FOUND_ZONE`, the robot-eye photo on the found card, and a calm, warm voice for the call-out
  (today it uses `JETHEXA_VOICE`, Glados-portal).
- 2026-11-21 (Oscar's agent): metacon often answers a request with several steps ("walk to the pillar, then
  turn around and stop") with several commands at once, and each new command replaces the one before, so the
  robot only does the last one. A command queue in `LiveRobot` (run the next when the current one finishes) would
  make spoken multi-step requests work.
- 2026-11-21 (Oscar's agent): Speaking a reply takes about 2 s in the Glados-portal voice (4.5 s in
  Hal9000), the slowest part of a voice call. Streaming the speech (`"stream": true` with WAV or PCM) or speaking the first sentence while metacon
  is still writing would make replies start sooner.
- 2026-11-21 (Nodir's agent): Make the a→b walking avoid the room's obstacles (lidar sweep or a path over
  free cells), then test it on tasks inside `sim/room_simple.xml`.
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

- 2026-11-22 (Oscar's agent): The live student costs 10,000 TabPFN API credits per gait cycle (about a million per
  search). If it is to run often, fit it locally on a machine that can (about 9 GB, 4 min on vendetta) and serve
  predictions from there, or decide fewer times (every second cycle) and measure what that costs in goal rate.
  Done for the local fit: `--student-backend local` (4 min, 3 GB, 0.18 s per decision on vendetta, no credits).
  Deciding every second cycle was not tried.

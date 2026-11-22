# JetHexa Live: the web app

A web page that shows the simulated JetHexa walking live and a terminal-style chat with a metacon LLM that can
drive it. Everyone who opens the page sees and steers the same robot.

```bash
just app-build     # once, and after front-end changes: npm install + build app/web into app/web/dist
just app           # http://127.0.0.1:8790
```

**Knock** (Find Margarete and Call Margarete) is built for the phone layout; the desktop screens are still to do.
Spec: [knock-ui.md](knock-ui.md); mockup: [knock-ui/mockup.html](knock-ui/mockup.html). See [Knock](#knock) below.

## Pieces

| Path | What it is |
|---|---|
| `app/robot.py` | `LiveRobot`: one thread steps MuJoCo in real time, steers the robot (goto, forward, turn, stop, reset) and moves the end point (`random`, also the dice buttons; walking never moves it) and renders JPEG pictures of an outside view and the robot camera |
| `app/llm.py` | Chat with the metacon gateway (`http://metacon:4000/v1`, model `qwen3:8b`); the model moves the robot by writing `[[goto 1 0.5]]` in its answer |
| `app/voice.py` | Speech through Oscar's Voice Control service: transcription of the caller and the robot's spoken replies (voice `Glados-portal`) |
| `app/call.py` | Voice calls: one caller at a time holds the line; every event of the call goes to all viewers |
| `app/knock.py` | Knock: one check-in at a time that calls Margarete through the robot and walks zone to zone to find her; every viewer sees the same card |
| `app/apartment.py` | The apartment: flat, journal, belief and the search loop that walks the robot from zone to zone; `GET /api/apartment` is what the Apartment panel draws |
| `app/server.py` | Standard-library HTTP server: the API under `/api` (documented at the top of the file) and the built front end |
| `app/web/` | The front end: React, Vite, Tailwind, built with Lovable (project "JetHexa Live Studio"). Full-screen view with switchable cameras (chase, orbit, overhead, robot eye), a chat bar and a red/green terminal. Without the API it runs a demo mode |

The steering is a simple rule (turn toward the target, slow down when facing away), the same as the `baseline`
policy in `nav/policy.py`, so the app needs no TabPFN and no recorded data.

## The apartment: the search in the live room

`just app` now starts the robot in the **apartment** scene: the 50 m² arena with its furniture, dressed as a home
(wooden floor, bedding, cushions, rugs, a door), the five zones of `homes/flats.py` tinted on the floor, and
Margarete as a figure lying where she really is. On a computer the **Apartment** panel (button top right) shows:

| Part | What it shows |
|---|---|
| Check-in | The weekday and time of the check-in (the real time, or one you pick), **Find Margarete**, and where she really is (the simulation has to know: by her routine, a zone, or out) |
| Floor plan | The zones shaded by the belief with their percentage, the furniture as obstacles, door, dock, the robot with its heading, the path walked, and each look as a numbered chip (green when it found her) |
| Belief | P(zone) and P(out), step by step: population, + habits, + journal, now. Under it her stated habits |
| Journal | Where she was heard. Add an entry (zone, day, time) and the belief changes at once; entries can be removed |
| Log | Every step of the loop: proposal, observation, update, found / nobody home |

**Find Margarete** runs the loop of `homes/search.py` (`Search`): the most likely zone → the robot walks to its
viewing pose and turns to face it → a look → Bayes' update → the next zone, until she is found or `out` reaches
95%. Finding her adds a journal entry. The look is the robot's own camera (below); with `--no-detector`, or while
the detector loads, it is the old mock (a die roll that misses her 20% of the time), and the panel says which.

### The robot's own eyes: finding her with DINOv2

`app/spotter.py` (`Spotter`), on the simulator thread, once per gait cycle while the robot walks, looks around or
watches:

1. The robot camera's picture and depth of the dressed flat (what "Robot eye" shows), DINOv2 on the picture, and
   the horizon strip: patch rows 11-14 averaged per patch column, 46 columns of 1.35 degrees.
2. Each column is compared with **her signature** (cosine similarity, after centring on the average background
   column): 3 typical looks lying and 3 standing. The best stretch must pass the threshold (0.65) and clearly beat
   everything else in view. The matched columns give the direction, the nearest depth pixel in them the distance
   (+0.10 m to her middle), the pose turns that into a floor point.
3. Her geoms have no collisions, so the lidar does not see her: a match the lidar also sees at that range is
   furniture and is dropped (`nav/dagger/target.py`, `lidar_sees_target: false`).
4. Two matches within 0.5 m confirm someone there (`Estimate`). The search then sends `approach`: the robot walks
   up to her and **its destination follows the camera's range**. Every gait cycle the destination is recomputed
   from the latest estimate: 0.6 m short of her on the side the robot comes from, moved round her to a spot 0.3 m
   clear of walls and furniture when needed (`approach_point`). It is updated when it moves more than 0.15 m (A*
   replans; the student just gets the new end point, its lidar map stays). During the approach older matches fade
   faster (`fuse_decay` 0.4 instead of 0.7), so the closer, better-ranged matches win over a first range that
   landed on the furniture behind her. At the destination it turns to face her. No match for 20 cycles: `lost`.
5. At the destination it watches 4 cycles: the close-up check counts matches of 0.75 or more, from 1.0 m or
   closer, within 1.0 m of the current estimate (she is 1.6 m long). 2 of them and she is **found**, in the zone
   of the final estimate. Otherwise, or lost on the way: a **false alarm**, that spot is ignored from then on, and
   the search goes on. Matches on the way in do not count: that let look-alikes the robot passed close by through
   (bench seeds 7422, 7453), and from 1.3-1.5 m a sighting ranged on something in front of her passed (7431). `search.robot_to_person` is the robot's true distance to
   her at the end (panel and bench only).

At each zone the robot **looks around**: the new command `look` turns it once on the spot (turn 0.15, about 20
cycles, the setting the arena test found best). It stops early when she is confirmed, and skips the turn when the
lidar shows something within 0.33 m. She can be spotted while walking, during a look-around, or in a neighbouring
zone; the search then ends in the zone where she was seen (`Search.report_found`). The Bayesian update assumes a
look overlooks her 10% of the time (`DETECT_MISS_RATE`, measured with `just search-bench`). The robot never reads
where the app put her; `search.person` and `found_error` are only for the panel and the bench.

**How well it works** (`just search-bench`, run on vendetta in 6 processes: 60 whole searches, new flats at every
mess level, A* walker, about 16 s each):

| | result |
|---|---|
| found, in the zone she is in | 53 of 57 (93%, 95% CI 83-97%); 32 of them seen while walking, before any look-around |
| standing | 25 of 26 found within 1 m of her (median 0.11 m) |
| lying | 21 of 31 within 1 m (median 0.56 m); 3 found in the wrong zone (1.5-3.1 m off) |
| not found | 1 (gave up after 12 looks) |
| nobody home | 3 of 3 not "found" (1 nobody home, 2 gave up after 12 looks) |
| look-arounds in her zone that missed her | 4 of 57 (7%; the update assumes 10%) |
| contacts, fallbacks | 0, 0 |

With the student walking (live, TabPFN API, ×3): she was found lying in the dining area of a messy flat (3 looks,
86 decisions, 0.20 m off) and standing in the bedroom (seen on the way, 38 decisions, 0.11 m off), with no
contacts and no fallbacks.

Open: lying she is 1.6 m long and lies 0.3 m in front of furniture, so a match can be ranged on the furniture
behind her or on another thing in the zone: 10 of 31 lying finds were more than 1 m off.

What the bench found and what was changed for her (the pillar test is unchanged):
the lidar look-alike check called her furniture when she stood in front of the bed (tolerance 0.25 -> 0.12 m);
a dark blue roll of clutter matched her trousers (height check: the nearest thing, floor removed, must reach 0.2
m); up close her two legs were two matches and each failed the other's margin (margin only against places more
than 0.8 m to the side); a weak look-alike passed the close-up (it now needs matches of 0.75 or more).

Her signature comes from the app's own scene: `just person-data` renders 3,000 pictures (100 random flats at every
mess level, her lying or standing at zone spots, the robot 0.5-4 m away; a fifth with nobody home) with her pixels
labelled by MuJoCo, then `python vision/target_signature.py person-v2 --band --name margarete`. On 20 flats it had
never seen: found in 97.7% of the pictures that show her (lying 98.6%, standing 96.6%; 0-4 m: 98-100%, beyond 4 m:
58%), false alarms in 0 of 213 pictures without her. Both live in `data/vision/person-v2/` (gitignored); without the
signature the detector stays off and the look is the mock.

**She can stand.** "She is" next to "She really is": lying on the floor, standing, or either at random
(`POST /api/apartment/search {"pose"}`). Standing is the same figure turned upright (`ApartmentScene.person`).

**The camera now sees close up.** MuJoCo puts the near clipping plane at 0.01 × the model's size, and the decor
parked 40-50 m away made the apartment model 112 m big: nothing within 1.1 m of the robot camera was drawn (nor in
its depth), so it was blind at the viewing pose, 0.6 m from her. `ApartmentScene` sets the near plane to 5 cm.

**Who says where she is: TabPFN.** The belief the search starts from comes from TabPFN
(`homes/tabpfn_belief.py`, `--belief tabpfn`, the default): `fit()` on 2,000 synthetic check-ins, then
`predict_proba` for the row that describes now (time, her stated routines, her journal as shares per zone, where
and how long ago she was last heard, and the counting posterior of those). The panel says it twice: the pipeline
strip (TabPFN orchestrates the Bayesian search) and, under the floor plan, "Search pattern: TabPFN 3.5" beside
"Robot control: TabPFN a→b". It uses TabPFN 3.5 when the API key is there (`TABPFN_TOKEN`, or the repo's
gitignored `.env`, as for `nav/dagger`), else the v2 weights. Her routines and journal are features of that row,
not examples, so changing them changes the belief without a new `fit()`. After each look the belief is updated
with Bayes' rule. At night, or with `--belief counting`, the counting model of `homes/search.py` is used; its
steps (population, + habits, + journal) are shown beside TabPFN's for comparison.

**Custom inputs.** In the Journal card, "Custom inputs" takes one JSON object, and "Copy current" puts the live
one in the box and on the clipboard:

```json
{
 "routines": {
  "weekdays": {"morning": "kitchen", "afternoon": "dining area", "evening": "living room"},
  "weekends": {"morning": "bedroom", "afternoon": "living room", "evening": "living room"},
  "outings": [{"day": "Thu", "part": "afternoon"}]
 },
 "journal": [
  {"day": "Mon", "time": "14:10", "zone": "dining area"},
  {"day": "Thu", "time": "14:30", "zone": "out"}
 ]
}
```

`routines` replaces her stated habits, `journal` replaces her journal (oldest first; a day is a weekday, a date,
`today` or `yesterday`, and weekdays count back from the check-in time); either can be left out. Plain lines such
as `Mon 08:10 kitchen` work for a journal, with "Add lines" to append. The same through the API:
`GET /api/apartment/inputs.json`, `POST /api/apartment/inputs`.

**Random example** (in the Belief card): new random routines (a new kind of resident), a new two-week journal
drawn from her routine, or both, to see how the belief and the search change. The flat stays.

**Reset** (in the status strip after a search) forgets the search, takes Margarete out of the room and puts the
robot back on its dock; the flat and the journal stay. "New flat" starts over with another layout and resident.

**Clean · Medium · Messy** (in the floor plan's header, and on the phone card) sets how cluttered the room is
(`scatter(flat, rng, level)` in `homes/flats.py`):

| Level | On the floor |
|---|---|
| Clean | Only the fixed furniture (bed, couch, cabinet, table); the stool, armchair, crate and plant are taken out |
| Medium | The four loose pieces at random spots, plus 4 of the 12 clutter items (8 loose things) |
| Messy | The four loose pieces plus as many of the 12 clutter items as fit, at least 10 (up to 16 loose things) |

The clutter is twelve small household things (laundry basket, bin, boxes, books, pouf, suitcase, bag, vacuum,
bucket, toy crate and chest; `sim/clutter.xml`, `CLUTTER` in `homes/flats.py`), solid, 0.2 to 0.45 m across and at
least 0.3 m tall so the lidar and the A* map both see them, parked outside the room until a level drops them.
Every dropped thing lands at a random spot, any heading: the pieces keep 0.5 m from every other piece and wall
(the training rule), the clutter 0.5 m in Medium and 0.4 m in Messy; everything keeps 0.5 m from every viewing
spot, the dock and the doorway, and every viewing spot stays reachable from the dock on the app's A* map
(`ROUTE_MARGIN`, 0.42 m). The current level is highlighted; pressing it again draws a new arrangement. The zones, the
resident and the journal stay; the robot goes back to its dock and the log says how many things are on the floor.
Refused while a search runs; "New flat" puts everything back as built (`obstacles: "home"`). The floor plan draws
the loose pieces and the clutter in a second (amber) tone. `POST /api/apartment/obstacles {"level": "messy"}` (no
level: medium); `GET /api/apartment` says the level in `obstacles`.

**Speed: x1 · x2 · x3 · x5** (under the camera list on a computer, at the end of the view row on a phone) runs the
simulation that many times faster than real time: the loop waits CONTROL_DT / speed of wall time per 20 ms physics
step, and renders each view every VIEWS[view] x speed steps, so the streams keep their picture rate. One robot, so
the speed is shared by every viewer and kept until someone changes it ("Reset" and "New flat" keep it). Every
steering decision makes the loop wait (the a-to-b controller's local `predict()`, the student's API call of about
0.7 s per simulated second), so the asked speed is not always reached: the robot's state carries `speed` and
`speed_actual` (simulated seconds per wall second over the last 3 s), and the control shows "x5 (running x3.4)"
when they differ by more than 10%. The search's walk and turn timeouts are wall-clock and only get easier.
`POST /api/speed {"speed": 2}` (rate-limited); `/api/state` and `/api/health` report it.

**Who walks.** First choice is the **TabPFN student** (`--driver student`, the default): the obstacle-avoiding
student of `nav/dagger` (v29, "lidar-map plan"; see [navigation.md](navigation.md)), through the hosted TabPFN API.
It walks straight to the end point with its own sensors: a 360 degree lidar (36 rays), the camera's horizon band
(DINOv2 + the patch probe) and its own map of lidar hits with a planned bearing (`LidarPlanner`). It is not told
where the furniture is. The A* route over the known furniture is still computed and drawn as a dashed line on the
floor plan: what a map-knowing planner would do.

| `--driver` | What it is |
|---|---|
| `student` (default) | The v29 student (`app/student.py`), TabPFN 3.5 through the TabPFN API: `fit()` once on the exported 31,586 examples, one `predict()` per gait cycle. Needs `TABPFN_TOKEN` (environment or the gitignored `.env`) and the export in `data/nav/dagger/v2/student/v29/` |
| `tabpfn` | The a-to-b controller built with `just controller` (`--controller default`): TabPFN picks stride and turn every gait cycle along a route of straight legs around the furniture (`plan_route`, the grid and A* of `nav/dagger/expert.py`). It does not see obstacles; the route keeps it clear of them. Needs the `ml` packages and `data/nav/controllers/` |
| `planner` | The `nav/dagger` expert (path planner + pure pursuit) |
| `rule` | The old steer-to-bearing rule |

A driver that cannot be loaded falls back to the next one; the panel says which one walks. With the a-to-b
TabPFN controller the physics waits about 0.25 s every gait cycle while it predicts.

Under the student, the first a-to-b driver that loads (tabpfn, planner, rule) walks **while the student loads**
(about 20 s at server start: the export, the API `fit()`, DINOv2) and **takes over a walk** when the student gets
stuck (the trainer's rule: not 0.2 m closer to the end point in 40 gait cycles) or the API fails: that leg is
handed to A* + a-to-b, the log says "Fallback: ..." and metacon says it once. The search treats this as normal
walking. The roles card shows "student loading…", a fallback, or why the student is off (`student` in
`/api/apartment` and `/api/health`: `loading | ready | failed | no_credits`, with the reason).

**Contacts.** Every control step checks the trainer's collision rule (any robot geom touching a `wall_` or `obs_`
geom). The app does not end the walk on a touch; it counts it (`contacts` in the robot state, a red cross on the
floor plan, a "Contact" log line). A foot tapping the same wall step after step counts once (1 s of no contact
starts a new touch).

**The plain twin: why the home look does not confuse the student.** The student was fitted in the plain training
arena (grey checker floor, 4.5 m walls, the red pillar at the goal); the apartment has a wood floor, 0.7 m walls,
zone patches, rugs, cushions, a door and Margarete, and the camera feeds 75 of the student's 116 inputs. So the
student never looks at the apartment: `app/student.py` keeps a second MuJoCo model of the plain arena, and before
each decision copies the robot's joint state and the poses of every obstacle body (the eight pieces of furniture
and the twelve clutter items: the twin is `sim/arena_clutter.xml`, the plain arena plus `sim/clutter.xml`) into it,
puts the red pillar at the goal, and renders the camera picture and depth there. The lidar is cast in the real scene (rays skip geoms without
collisions, so zones, decor and Margarete are invisible to it). `tests/test_student_twin.py` checks, for 20 flats
and poses (as built and at each mess level, with clutter), that the twin mirrors every obstacle body, that these inputs equal those of the plain `FlatEnv` (`homes/sim_bridge.py`) at
the same layout and pose, that the lidar in the dressed scene equals the twin's, and that painting the zones,
moving Margarete and moving the decor change none of them (while they do change the app's own camera picture).

**TabPFN API credits.** Every `predict()` costs 10,000 credits (measured 2026-11-22; `fit()` is free) and the
student makes one per gait cycle, so a 5 m walk is about 40 calls = 400,000 credits, and a whole search can use
over a million. With `fit_mode="fit_with_cache"` (the default) a call takes 0.6 to 3.5 s; without the cache it
re-runs the whole 63,000-row context every call (13 s, 11,400 credits). At start the app checks the account: under
150,000 credits left the student is not started (`no_credits`), and its budget stops it before the account drops
below that; `--student-calls N` caps the calls further. The a-to-b drivers run locally and cost nothing. The
physics waits while the student decides (0.6 to 4 s per gait cycle through the API), so a walk is slower than real
time; the camera streams keep their rate meanwhile (the paused scene is rendered while the decision is made).

**Local student (`--student-backend local`).** The same student with TabPFN's `fit()` and `predict()` run by the
`tabpfn` package on the server instead of the API (`TabPFNPolicy`, one regressor for stride and one for turn; still no
weight changes). On vendetta (M4 Pro, MPS) `fit()` takes about 4 min at start (the a-to-b driver walks meanwhile) and
3 GB, then a `predict()` takes 0.18 s and costs no credits. Measured on vendetta 2026-11-22 with the same walks: the
chase stream stays at 11 pictures per second while the student walks (2 to 3 before this change), and x3 reaches
x1.5 to x1.7 (about x1 through the API). Use it on a machine with a GPU; on a laptop CPU the fit is too slow.

`--scene floor` gives the old empty floor (the panel then moves a marker on the plan only). `--flat N` picks the
layout and resident. The home look and Margarete are in `sim/apartment_decor.xml`, which only `sim/apartment.xml`
includes: `arena.xml`, the vision data and the DAgger scenes do not change. The JetHexa meshes must be in
`sim/meshes/` (`just meshes`), else the robot is drawn with simple shapes.

**Where she really is** can be her routine, a zone, out, or **somewhere random** (a random zone and a random spot in
it; the dice button on the phone).

**On a phone** the apartment is a card under the camera: status, the floor plan with the belief, a chip per look,
Find Margarete, the dice, and **Details**, which opens the whole panel (time, belief steps, journal, log) as a
full-screen sheet. It replaces the old Knock card while the server runs the apartment scene; the Knock check-in
(`app/knock.py`, with its own zones and the call-out) is still what `--scene floor` shows.

### metacon narrates the search

While the search runs, metacon says what is happening in one short line at each step (`app/narrate.py`): the three
most likely zones at the start, where it goes first, "Not in the kitchen. The bedroom is now most likely at 35%, so
I'm heading there.", and the end (found, nobody home, gave up, stopped, interrupted). The search only hands over the
facts (`Apartment(on_event=...)`); one worker thread writes the lines, so the search never waits for metacon or the
voice.

- **The words.** Each line starts as a template built from the search's own numbers. metacon (`llm.complete`, the
  same gateway and model as the chat, 6 s at most) rewords it; the reply is used only if it is at most 30 words,
  names exactly the same zones, has no number the template lacks and keeps the outcome (found or not). Otherwise,
  on any error, or when the 2 chat slots are taken (it does not wait), the template is used. `source` in the event
  says which (`"metacon"` only when the wording really changed).
- **Where it shows.** Every page gets the line on `/api/live` (`{"type": "narration", "id", "kind", "text",
  "source"}`): in the terminal and the phone chat as a metacon line tagged "search", in the Apartment log tagged
  metacon, and for 9 s in the header (above the chat bar while the Apartment panel is open; on the phone's
  apartment card). A page that opens within 2 minutes gets the latest line first.
- **The voice.** On a second thread, if the voice service is up and nobody is on a voice call, the line is spoken
  (3 s wait for the service at most) and published again with `"audio": "/api/narration/audio/<id>.mp3"` (the
  last 6 are kept). Phones with the speaker on play it once, one line after another, never over a call's reply;
  the Find and dice buttons unlock sound on the tap (iPhone needs that). If speech falls behind, the oldest waiting
  line that is not an ending is skipped; text is never held back by speech.
- **Off:** `--no-narration`. Without a metacon key the lines are the templates; without the voice service, text only.

## The "How it works" page

A long-form page that explains the flow of Find Margarete and the three TabPFN models (where is she, walking
a→b, avoiding furniture, the last marked **In progress**). Open it with **How it works** in the header (desktop) or
the **i** button in the phone's top bar; it is also a link: `/#how-it-works` (with `?phone=1` kept).

- `App.tsx` owns the `#how-it-works` hash and renders the page **instead of** the stage, so the picture streams
  unmount and their connections are free for the videos. The robot-state and apartment polling pause while it is
  open; the call, Knock and `/api/live` hooks stay mounted. Close (Back button, Escape or the browser's Back) returns
  to the robot; after a direct link, closing uses `history.replaceState`, so it never leaves the site. The section
  nav scrolls with `scrollIntoView` and never changes the hash.
- The page calls no API: it works with the simulator stopped.
- Code: `app/web/src/components/jethexa/how/` (`HowItWorks.tsx`, `FlowDiagram.tsx`, `BarChart.tsx`, `Clip.tsx`,
  `Section.tsx`, `StatusChip.tsx`). **Every number lives in `app/web/src/content/howItWorks.ts`**, each with a comment
  naming its source; change numbers there only. The flow diagram's steps are data (`NODES`, `EDGES` in
  `FlowDiagram.tsx`): a new step is one node and one edge.
- Chart colours: one teal hue in two steps (`#23aa8a` for the bar the chart is about, `#366e6b` for the others),
  checked with the dataviz skill's `validate_palette.js --ordinal --mode dark --surface "#111518"`.

**Media** (`app/web/src/assets/how/`, imported from TypeScript so Vite gives them content-hashed names, which is
what the server's one-year `immutable` cache needs). 1.7 MB in all; sources are in the main checkout's gitignored
`data/`:

| File | Size | Source | What it shows |
|---|---|---|---|
| `a2b.mp4` | 140 kB | `data/nav/runs/20261121-163413-demo/demo.gif` (controller `default`, (0, 0) to (1.6, 0.2)) | The a→b controller walking to the pillar |
| `dagger-first.mp4` | 720 kB | `data/nav/dagger/dagger1/videos/round50-episode50000001.mp4` (dagger1 round 50, scene 100001, student alone) | The camera-only student walking past the target into a blue box beside it |
| `dagger-student.mp4` | 576 kB | `data/nav/dagger/v2/videos/v29-student-final-episode102093.mp4` (v29, final test layout 102093, student alone) | The current student going round a long piece of furniture to the pillar |
| `*-poster.jpg` | 9-35 kB | a frame of each clip | Shown before a clip plays |
| `search-loop.jpg` | 147 kB | `homes/data/search_examples/loop_3.png` | A missed sighting, and the robot comes back |

```bash
F=/opt/homebrew/bin/ffmpeg   # 9.0.2
$F -i data/nav/runs/20261121-163413-demo/demo.gif -an -c:v libx264 -pix_fmt yuv420p -crf 28 -preset slow -movflags +faststart -vf "scale=640:-2:flags=lanczos,fps=10" a2b.mp4
$F -i data/nav/dagger/dagger1/videos/round50-episode50000001.mp4 -an -c:v libx264 -pix_fmt yuv420p -crf 28 -preset slow -movflags +faststart -vf "scale=960:-2:flags=lanczos" dagger-first.mp4
$F -i data/nav/dagger/v2/videos/v29-student-final-episode102093.mp4 -an -c:v libx264 -pix_fmt yuv420p -crf 28 -preset slow -movflags +faststart -vf "scale=960:-2:flags=lanczos" dagger-student.mp4
$F -i homes/data/search_examples/loop_3.png -vf "scale=1600:-2:flags=lanczos,format=yuvj444p" -q:v 4 search-loop.jpg
$F -ss 6 -i a2b.mp4 -frames:v 1 -q:v 5 a2b-poster.jpg              # -ss 21 for dagger-first, -ss 12 for dagger-student
```

**Server:** `static()` in `app/server.py` answers single byte ranges (`Range: bytes=a-b`, `a-`, `-n`) with `206`,
`Content-Range` and `Accept-Ranges: bytes`, reading only those bytes; a range past the end gets `416`; several
ranges or a malformed header get the whole file (`200`). Safari on iPhone does not play a video without this. A
missing file with an extension (an old hashed asset, a video) gets `404` instead of the page; paths without one
still get `index.html`. Tests: `tests/test_server_static.py`.

## The terminal

Typing happens in the terminal's own prompt line when it is open (the white chat bar shows only while it is
closed). Plain text goes to metacon; `/target`, `/goto X Y`, `/forward M`, `/turn DEG`, `/stop`, `/random`,
`/reset` go straight to the robot; `/help` and `/clear` work locally. The title bar shows the robot's status, and
the terminal prints a line when the robot arrives, finishes a turn or falls over.

## Knock

The phone page starts with the **Margarete card** (`KnockCard.tsx`) under the camera: Find Margarete and Call
Margarete. What follows is the same on every viewer's page, because the server runs the check-in
([app/knock.py](../app/knock.py)) and publishes the whole card on `GET /api/live` as `{"type": "knock", ...}`:

- **Call**: the robot says "Margarete, it's Thomas. Are you all right?" in its voice (played on every page whose
  speaker is on), then listens for 20 s with a countdown, Stop and Find her now. Nobody answers yet, so the search
  starts by itself.
- **Find**: the card shows the flat as a 3 × 2 zone map with each zone's probability. The robot walks to the zone
  with the highest probability per metre of walking, turns to face it, looks for 2 s, and on a miss marks it
  checked and shares its probability out over the others. The camera switches to the robot eye.
- **Found**: the robot eye fills the top, the card turns red, and Call 112, Call Frau Weber and Talk to her
  replace the talk bar. The calls are simulated; Talk to her ends the check-in and brings back hold-to-talk.

Until the zone model and the person detector exist, the probabilities are the placeholders from
[knock-ui.md](knock-ui.md) (the card says "placeholder" instead of "TabPFN") and `KNOCK_FOUND_ZONE` says where she
is (default `couch`; `none` gives "She doesn't seem to be home"). The zone poses are in `ZONES` in `app/knock.py`
and use the 2 × 3 grid of [knock-routines.md](knock-routines.md), on the robot's empty floor. Without the server
the page plays the same screens on timers (demo mode).

## Voice calls

People can talk to the robot. On a phone the page is a call screen (`PhoneApp.tsx`): the robot's camera on top,
the conversation in the middle and a big **hold-to-talk** button at the bottom; the keyboard button switches to
typing, and **New chat** above the conversation empties it and what metacon remembers. On a computer the mic button sits in the chat bar and the terminal prompt (hold it with the mouse, or hold
Space or Enter on it). Sliding up while holding cancels.

```
hold:     POST /api/call/start  -> call id (409 while someone else is on the line)
          recording in the browser (WebM/Opus, or MP4/AAC on Safari); every 1.5 s the recording so far goes to
          POST /api/transcribe?final=0 -> live caption for everyone
release:  POST /api/transcribe?final=1 -> the words, shown in the conversation
          POST /api/chat with the call id -> metacon answers in one short spoken sentence and moves the robot
          POST /api/speak -> MP3 in the Glados-portal voice, played on the caller's phone
          POST /api/call/end
```

Everyone else on the page follows the call live through `GET /api/live` (server-sent events): a "Live call" card
shows the caller's words as they are recognised, metacon's answer as it is written, and the robot commands. Only
one person can talk at a time; the line is freed when the call ends, after 30 s without activity, or after 2 minutes.

Speech comes from Oscar's **Voice Control** service (see the `use-voice-service` skill), which keeps audio in the
homelab. The server calls it; the browser never does, so its key stays on the server. Measured 2026-11-21 over
Tailscale: transcribing a 3.6 s command takes about 0.3 s, metacon answers in about 1.5 s, and speaking one sentence
in the Glados-portal voice takes about 2 s (the Hal9000 voice about 4.5 s, a built-in voice about 2.5 s). The service does one job at a time.

- Settings: `VOICE_CONTROL_URL` (default `http://127.0.0.1:8765`; on the laptop use vendetta's Tailscale address),
  `VOICE_CONTROL_API_KEY`, `JETHEXA_VOICE` (voice name or id, default `Glados-portal`; empty = the service default),
  `JETHEXA_LANGUAGE` (default `en`; `auto` detects the language but is slower).
- Without the voice service the mic buttons are hidden and the rest of the page works as before.
- Limits per visitor: 20 calls and 30 spoken replies per 10 minutes, 15 live captions per call, recordings up to
  15 s (1.5 MB).
- Browsers only allow the microphone on HTTPS or `localhost`, so a phone needs the published address. Phones only
  play sound a touch started: the page plays a moment of silence on the press so the reply can play later.
- To try it without a microphone, `?phone=1` forces the phone layout on a computer (`?phone=0` the desktop one).

## The metacon key

The server reads the gateway key from `$METACON_LLM_KEY`, else from the macOS Keychain (service `metacon-llm`, see
the metacon skill). It stays on the server: the browser never sees it, and the chat runs through the server.
Without a key the page still works and the chat says it is offline. The machine needs Tailscale to reach metacon.

## Publishing at alpharelay.dev

The app runs on Oscar's Mac and is published through its Cloudflare tunnel with the `alpharelay-deploy` skill:
`relay deploy jethexa 8790` (live at https://jethexa.alpharelay.dev since 2026-11-21). The server must be running on
the Mac; it does not restart after a reboot. It is then public without a login, and every chat uses Oscar's metacon key, so the
server limits each visitor (by Cloudflare's `Cf-Connecting-Ip`) to 20 chats per 10 minutes and 60 commands per
minute, and runs at most 2 chats at once. The server must have the voice settings above in its environment for voice calls.

A second copy runs on vendetta, Oscar's always-on Mac, at https://jethexa-vendetta.alpharelay.dev (since
2026-11-22; `main` at `76455c3`). It has its own Cloudflare tunnel and launchd agents, so it restarts after a crash or
reboot. It runs the student with `--student-backend local` (TabPFN on vendetta's GPU, no API credits; see "Local
student" above), so it does not draw on the laptop copy's TabPFN API credits. Setup, updates and teardown:
[deploy/vendetta/README.md](../deploy/vendetta/README.md).

## Settings

`METACON_URL` (default `http://metacon:4000/v1`), `METACON_MODEL` (default `qwen3:8b`), `METACON_LLM_KEY`.
Server flags: `--no-narration` turns off metacon's lines about the search (see "metacon narrates the search").

## Front-end development

`cd app/web && npm run dev` serves the page with hot reload on http://localhost:5173 and passes `/api` to the
Python server on :8790 (start it with `just app`). Each picture stream is about 0.5 MB/s, so a visitor with three
views open uses about 1.5 MB/s of the Mac's upload; views nobody watches are not rendered.

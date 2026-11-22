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
95%. Finding her adds a journal entry. The look is still a mock (it misses her 20% of the time).

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

**Who walks.** A goto in the apartment is a route of straight legs around the furniture (`plan_route` in
`app/robot.py`, the grid and A* of `nav/dagger/expert.py`); a driver walks each leg:

| `--driver` | What it is |
|---|---|
| `tabpfn` (default) | The a-to-b controller built with `just controller` (`--controller default`): TabPFN picks stride and turn every gait cycle. It does not see obstacles; the route keeps it clear of them. Needs the `ml` packages and `data/nav/controllers/` |
| `planner` | The `nav/dagger` expert (path planner + pure pursuit) |
| `rule` | The old steer-to-bearing rule |

A driver that cannot be loaded falls back to the next one; the panel says which one walks. With TabPFN the physics
waits about 0.25 s every gait cycle while it predicts, so a walk takes about a quarter longer than real time.

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

## Settings

`METACON_URL` (default `http://metacon:4000/v1`), `METACON_MODEL` (default `qwen3:8b`), `METACON_LLM_KEY`.
Server flags: `--no-narration` turns off metacon's lines about the search (see "metacon narrates the search").

## Front-end development

`cd app/web && npm run dev` serves the page with hot reload on http://localhost:5173 and passes `/api` to the
Python server on :8790 (start it with `just app`). Each picture stream is about 0.5 MB/s, so a visitor with three
views open uses about 1.5 MB/s of the Mac's upload; views nobody watches are not rendered.

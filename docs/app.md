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

Not done yet: the phone page still shows the old Knock card with its own zones (`app/knock.py`); the Apartment
panel is desktop only.

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

## Front-end development

`cd app/web && npm run dev` serves the page with hot reload on http://localhost:5173 and passes `/api` to the
Python server on :8790 (start it with `just app`). Each picture stream is about 0.5 MB/s, so a visitor with three
views open uses about 1.5 MB/s of the Mac's upload; views nobody watches are not rendered.

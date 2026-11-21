# Knock buttons in JetHexa Live: UI spec

For Oscar and his agent. Two new buttons in the web app ([app.md](app.md)), **Find Margarete** and **Call
Margarete**, and the screens that follow them. Knock is the team's concept: a family member can't reach an older
woman living alone, so the robot calls out to her, works out where she probably is, walks there and reports what it
finds. The story is on the [Knock User Journey page](https://claude.ai/artifact/DEh7egdji2AmNuHswoP3pR), the plan
on the [Knock Architecture page](https://claude.ai/artifact/LNmANYSUocKoTr41kJCnG5) and the data in
[knock-routines.md](knock-routines.md).

## The mockup

- **[knock-ui/mockup.html](knock-ui/mockup.html)**: all 7 screens as one static page. Open it in a browser, or read
  the markup: every size, colour and font is inline. Colours, fonts and radii are copied from
  `app/web/src/styles.css` (`--primary`, `--term-red`, `--term-amber`, `--term-green`, `glass`, `pixel`).
  The desktop screens are drawn over [knock-ui/jethexa-live.jpg](knock-ui/jethexa-live.jpg), a screenshot of the
  live page, so they show where each card goes.
- The same screens as an editable design canvas (private, ask Nodir for access):
  [Knock Buttons Mockup](https://claude.ai/artifact/8QPFDN79fsTChDgDAAEBQc).

| # | Screen | Where it goes in the code |
|---|---|---|
| 1 | Desktop, idle: the Margarete card | `App.tsx`, top right under `ViewList` (the "Random end point" row) |
| 2 | Desktop, calling | the card changes state; the top caption shows a live-call bar like `LiveCallCard` |
| 3 | Desktop, finding | the card shows the zone map and steps; the top caption says which zone is next |
| 4 | Desktop, found | the card turns red with the photo and three choices |
| 5 | Phone, idle | `PhoneApp.tsx`, between the camera view and the talk bar, where the suggestions are now |
| 6 | Phone, finding | the same card, compact |
| 7 | Phone, found | the robot eye fills the top; the alert and buttons below |

The card and its states should show on every viewer's page, like a voice call does now.

## What each button does

### Call Margarete

1. The robot says the message in its own voice: *"Margarete, it's Thomas. Are you all right?"*. Use
   `voice.speak()` as `/api/speak` does. Today that audio plays only on the caller's device; every viewer should
   hear it when their speaker is on, because in the story it comes out of the robot in the flat.
2. It listens for 20 seconds. The card shows a countdown, **Stop** and **Find her now**.
3. Margarete is not in the simulation yet, so nobody answers. After 20 s the card says "No answer" and Find starts
   by itself. (Later: a viewer can answer as Margarete with hold-to-talk, and "I'm fine" ends the check-in.)

The voice: `JETHEXA_VOICE` is `Glados-portal` today. For an 81-year-old woman, pick a calm, warm voice.

### Find Margarete

1. Get a probability for each zone. Until Krishna's zone model exists, use these placeholders (Tuesday 10:35 in
   [knock-routines.md](knock-routines.md)): couch 42%, kitchen 18%, armchair 16%, bed 13%, bathroom 9%,
   entrance 2%. Later: `predict(context)` from the zone model, same shape.
2. Pick the next zone by probability ÷ walking distance from the robot, and walk to that zone's viewing pose
   (`goto x y` on the robot, then a turn to face the zone).
3. Look. Later: the person detector (`render(segmentation=True)` from pull request #14, is her body in view).
   Until then, every zone is "not there", or "found" in the zone where the scene places her.
4. Not there: mark the zone checked, lower its probability and share it out over the others (bathroom 30% → 5%,
   the rest rises), then go to step 2. Found: show screen 4. Every zone checked: "She doesn't seem to be home."
5. **Stop search** sends `stop`. **Call her again** runs Call without leaving the search.

The viewing poses must be on open floor: the a→b walker does not avoid obstacles yet. Take the zone positions
from the scene the demo uses (`sim/room_furniture.xml` in pull request #12, still being agreed with
[knock-routines.md](knock-routines.md)).

### Found her

- The card shows the robot-eye photo (`frame("eye")`), where she is, what it saw ("moved her arm") and heard
  ("a sound, no words"), and when she was last heard.
- **Call 112**, **Call Frau Weber** (her neighbour, who has a key) and **Talk to her**. In the demo the first
  two only show "Calling… (simulated)". **Talk to her** opens the existing hold-to-talk call, and the robot plays
  the caller's voice.
- The robot never calls 112 by itself. The person on the phone decides.

## Suggested server shape

Nothing here is built. A shape that fits `app/server.py` and the existing `Hub`:

```text
POST /api/knock/call                 start a call-out; 409 while a check-in runs
POST /api/knock/find                 start a search; 409 while one runs
POST /api/knock/stop                 stop either and send `stop` to the robot
POST /api/knock/choice {"choice": "call_112" | "call_neighbour" | "talk"}   simulated
GET  /api/knock/audio/<id>.mp3       the call-out in the robot's voice, for every viewer

Events on /api/live, so every page shows the same card:
{"type": "knock", "state": "calling", "text": "Margarete, it's Thomas. Are you all right?", "audio": "/api/knock/audio/<id>.mp3"}
{"type": "knock", "state": "no_answer"}
{"type": "knock", "state": "predict", "probs": {"couch": 0.42, "kitchen": 0.18, ...}}
{"type": "knock", "state": "goto", "zone": "kitchen"}
{"type": "knock", "state": "miss", "zone": "kitchen", "probs": {...}}
{"type": "knock", "state": "found", "zone": "couch", "pose": "lying", "seen": "moved her arm", "heard": "a sound, no words"}
{"type": "knock", "state": "not_home"}
{"type": "knock", "state": "end"}
```

One check-in at a time, like the voice-call line in `app/call.py`.

## Not in this step

- Real phone calls, 112 or an alarm service.
- Margarete's body in the scene and the person detector: separate pull requests.
- The zone model: Krishna, from [knock-routines.md](knock-routines.md).

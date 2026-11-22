// Every number on the "How it works" page (components/jethexa/how/), each with the file it comes from.
// Paths are repo paths. "PR #22" means branch feat/dagger-v2-fixpass (not merged yet when this was written);
// "STATUS.md" is data/nav/dagger/v2/STATUS.md in the main checkout (gitignored). Check a number against its
// source before changing it, and change it here only.
import a2bClip from "@/assets/how/a2b.mp4";
import a2bPoster from "@/assets/how/a2b-poster.jpg";
import firstClip from "@/assets/how/dagger-first.mp4";
import firstPoster from "@/assets/how/dagger-first-poster.jpg";
import studentClip from "@/assets/how/dagger-student.mp4";
import studentPoster from "@/assets/how/dagger-student-poster.jpg";
import searchLoop from "@/assets/how/search-loop.jpg";

export type Status = "live" | "progress";

export interface SectionInfo { id: string; nav: string; status?: Status }

/** The page's sections, in order: the nav and the scroll targets. */
export const SECTIONS: SectionInfo[] = [
  { id: "how-flow", nav: "The flow" },
  { id: "how-search", nav: "Where is she", status: "live" },
  { id: "how-walk", nav: "Walking a → b", status: "live" },
  { id: "how-avoid", nav: "Avoiding furniture", status: "progress" },
  { id: "how-next", nav: "Limits and next" },
];

/** Under the flow diagram: what the "metacon narrates" step does (app/narrate.py). */
export const NARRATION =
  "While it searches, metacon (the qwen3:8b chat model behind the app's terminal) says what is happening: where the robot goes, " +
  "what it did not find there and where it goes next, and when it finds her. The line is built from the search's own " +
  "numbers, and metacon only rewords it; if its wording names another room or changes a number, the plain line is " +
  "used. It shows on every screen, and the phone says it aloud when its speaker is on.";

/** "One example per task": the three uses of TabPFN side by side. */
export const TASKS = [
  {
    task: "Where is she", section: "how-search",
    tabpfn: "3.5 classifier, predict_proba()", // homes/tabpfn_belief.py (v2 without the API key)
    row: "Time, last heard, journal, stated habits, the counting model's estimate",
    answer: "Where the simulated resident really was",
  },
  {
    task: "Walk from a to b", section: "how-walk",
    tabpfn: "v2, three motion regressors", // nav/dynamics.py MOTION; docs/navigation.md (v2 weights)
    row: "Last motion, previous command, a candidate command",
    answer: "The motion measured after that command",
  },
  {
    task: "Avoid furniture", section: "how-avoid",
    tabpfn: "3.5, two command regressors", // PR #22 nav/dagger/policy.py (stride, turn)
    row: "Goal, lidar, camera horizon band, its own plan",
    answer: "The expert's stride and turn",
  },
];

// --- 1. Where is she ----------------------------------------------------------------------------------------
// homes/tabpfn_belief.py FEATURES: 4 + 8 + 7 + 5 + 1 + 6 = 31 columns; 6 classes (5 zones + out).
export const SEARCH_INPUTS = [
  { group: "Time", n: 4, what: "Hour, weekday, weekend or not, part of the day" },
  { group: "Last heard", n: 8, what: "Where she was last heard (one column per class), how long ago, and whether it was this part of today" },
  { group: "Her journal", n: 7, what: "Her share of records per class in this part of the day, and how many records there are" },
  { group: "Stated habits", n: 5, what: "The zone she says she is usually in now" },
  { group: "Outing", n: 1, what: "Whether she said she goes out now" },
  { group: "Counting estimate", n: 6, what: "The counting Bayes model's probability per class (the prior_* columns)" },
];

export const SEARCH = {
  classes: 6, // homes/flats.py CLASSES: bedroom, living room, dining area, kitchen, hallway, out
  features: 31, // homes/tabpfn_belief.py FEATURES
  examples: 2000, // homes/tabpfn_belief.py CONTEXT_ROWS; homes/README.md
  journalDays: [7, 14, 28], // homes/README.md "1, 2 and 4 weeks of journal"
  missRate: 20, // homes/search.py MISS_RATE = 0.2
  stopOut: 95, // homes/search.py STOP_OUT = 0.95
  maxLooks: 12, // homes/search.py MAX_LOOKS
  testResidents: 120, // homes/data/search_results.json "residents"
  checkIns: 4800, // search_results.json all.check_ins
  atHome: 4101, // search_results.json all.at_home
  top1: 72, // search_results.json all.tabpfn.belief_top1 = 0.724 (out included)
  zonesChecked: 1.58, // all.tabpfn.zones_checked (perfect look)
  foundWithMisses: 98.5, // loop.tabpfn.found = 0.985 (20% misses, at-home check-ins)
  looksWhenFound: 2.23, // loop.tabpfn.looks
  countingFound: 98.9, // loop.bayes.found
};

/** Zones checked until found, perfect look, 4,101 at-home check-ins (homes/data/search_results.json "all"). */
export const ZONES_CHECKED = [
  { label: "TabPFN belief", value: 1.58, focus: true }, // all.tabpfn.zones_checked
  { label: "Counting Bayes model", value: 1.54 }, // all.full.zones_checked
  { label: "Last heard first", value: 2.12 }, // all.last.zones_checked
  { label: "Nearest zone first", value: 3.30 }, // all.nearest.zones_checked
];

export const SEARCH_FAILED = [
  {
    title: "Diary records as examples",
    body: "Right 33% of the time. A diary has one record per part of the day, so it never shows the common case \"heard a few minutes ago\". Check-ins as examples fixed it.",
    // docs/LEARNINGS.md 2026-11-22 "went from 33% to 71%"
  },
  {
    title: "Only the raw journal columns",
    body: "A strong journal lowered the right zone: ten dining-area afternoons took \"dining area\" from 28% to 24%. Adding the counting model's estimate as six more columns fixed it.",
    // homes/README.md, docs/LEARNINGS.md
  },
  {
    title: "Her own records in fit()",
    body: "Most likely zone right 73.8% → 74.0% on 12 test residents, but every resident needs a new fit() of about 17 s. Not adopted.",
    // docs/LEARNINGS.md 2026-11-22
  },
];

// --- 2. Walking from a to b ---------------------------------------------------------------------------------
// nav/dynamics.py STATE, COMMAND, MOTION, CANDIDATES; data/nav/controllers/default/controller.json;
// nav.duckdb controllers.fit_seconds (default: 5.4 s); docs/navigation.md lines 74-137.
export const WALK_INPUTS = [
  { name: "prev_stride, prev_turn", what: "The command of the last cycle" },
  { name: "fwd_1, left_1, dyaw_1", what: "How the body moved in the last cycle" },
  { name: "stride, turn", what: "The candidate command to try" },
];
export const WALK_OUTPUTS = [
  { name: "m_fwd", what: "metres forward" },
  { name: "m_left", what: "metres to the left" },
  { name: "m_dyaw", what: "turn, in radians" },
];

export const WALK = {
  episodes: 200, // data/nav/runs/20261121-134654-collect/run.json settings.episodes
  cycles: 6000, // same run: 200 x 30 cycles; docs/navigation.md "6,000 recorded cycles"
  examples: 5826, // controller.json total_examples (full cycles, no falls)
  context: 300, // controller.json context_rows
  fitSeconds: 5, // nav.duckdb controllers: default 5.4 s
  strides: 5, turns: 9, candidates: 45, // nav/dynamics.py CANDIDATES
  reached: "20 of 20", // docs/navigation.md results, hybrid score
  directness: 0.96, // docs/navigation.md; hand-written baseline also 0.96
  baseline: 0.96,
  noFacing: "10 of 20", // docs/navigation.md: score "delta" (no facing term)
};

export const WALK_VARIANTS = [
  {
    title: "Predict the command directly",
    body: "TabPFN given b's distance and bearing, asked for the command: directness 0.77 (0.84 with b's offset added). Its examples are random walks relabelled in hindsight, so it copies their curved arcs. Giving it the previous command made it worse: in random walks a command is held for several cycles, so the previous one almost gives the answer away.",
    // docs/navigation.md results table; docs/LEARNINGS.md 2026-11-21
  },
  {
    title: "More examples",
    body: "2,000 examples instead of 300: the same directness (0.89 with the older score), nine times slower.",
    // docs/navigation.md results table
  },
  {
    title: "Fine-tuning",
    body: "Tried once: four rounds (30 min) gave one small gain (directness 0.78 → 0.85) and none after. Dropped, and now a project rule: fit() and predict() only.",
    // docs/LEARNINGS.md Process; AGENTS.md
  },
];

// --- 3. Avoiding furniture (in progress) --------------------------------------------------------------------
// PR #22: docs/navigation.md, docs/dagger-report.md, docs/dagger/promotion.md, docs/LEARNINGS.md; STATUS.md;
// data/nav/dagger/dagger1/rounds.json and timings.json (main checkout); docs/vision.md.
export const FIRST_STUDENT = {
  grid: "26 × 46", // PR #22 docs/navigation.md featurize.py row
  probeAccuracy: 99.3, // docs/vision.md, one hidden layer, held-out layouts
  columns: 2400, // docs/navigation.md
  episodes: 1000, // dagger1/rounds.json round 0
  machines: 4, // dagger1/timings.json: laptop, metacon, vendetta, victory
  rows: 32455, // dagger1/rounds.json round 0
  minutes: 7, // dagger1/timings.json: 20:58-21:05 on the laptop, the slowest part
  fitSeconds: 696, // docs/LEARNINGS.md (32k rows, two regressors)
  crashed: "3 of 3", // dagger1/round_50.json outcomes: collision 3
};

export const HARD = [
  "The camera sees only 62° ahead, and most crashes were into furniture beside the robot.", // dagger-report.md §3, §4
  "Camera-only students stayed at 58–70% of goals, however the picture was summarised: the patch grid, eight summary columns, a horizon scan or a horizon band.", // PR #22 docs/LEARNINGS.md, docs/dagger/experiments.md (v03, v00, v02, v09, v10)
  "\"Stop and turn on the spot\" is the label in only 6.6% of the rows, and it is the one that prevents crashes.", // docs/LEARNINGS.md (dagger-report.md says 6.5%)
  "The expert almost never gets close to furniture: 18 of 41,084 rows are within 0.26 m.", // dagger-report.md §4
  "Hand-written safety stops made it worse (94% and 91% of goals against 95% without them, on the same 100 layouts): a hexapod's legs touch furniture while it turns, and a reflex near contact comes too late.", // docs/dagger/experiments.md v27, v28 against v15
];

export const NIGHT = [
  { time: "01:59", text: "Expert data collection starts on two machines, vendetta and metacon." }, // STATUS.md "Running (started 01:59)"
  { time: "02:12", text: "Round 0 merged: 1,000 of 1,000 expert episodes reach the goal, 31,586 rows." }, // STATUS.md "Data so far"
  { time: "05:15", text: "New direction: the robot plans its own path on a map built from its lidar." }, // STATUS.md "New direction (05:15)"
  { time: "06:33", text: "The agreed bar is met on fresh layouts with two start seeds. The search stops." }, // STATUS.md "RESULT ... (06:33)"
];

/** Student alone on the same 200 screening layouts (PR #22 docs/dagger/promotion.md, screen rows). */
export const LADDER = [
  { label: "Camera patch grid", note: "606 columns", value: 63.5, collisions: 36.0 }, // v03-reference
  { label: "+ 360° lidar", note: "114 columns", value: 90.5, collisions: 9.0 }, // v11-lidar
  { label: "+ all 31,586 rows", note: "instead of 8,000", value: 94.5, collisions: 5.5 }, // v15-lidar-allrows
  { label: "+ its own plan", note: "116 columns (v29)", value: 98.5, collisions: 1.5, focus: true }, // v29-lidarplan
];
export const EXPERT_SCREEN = 100; // promotion.md: expert anchor on screen, 100.0%

/** The final check: 200 fresh layouts, 300 steps, used once (promotion.md final and final_b). */
export const FINAL = [
  { run: "Start seed 99", goal: "98.0%", ci: "95.0–99.2", collisions: "2.0%" },
  { run: "Start seed 98", goal: "97.5%", ci: "94.3–98.9", collisions: "2.5%" },
  { run: "The expert (reference)", goal: "99.5% / 100%", ci: "", collisions: "0.5% / 0%" },
];

export const AVOID_LIMITS = [
  "Simulation only.",
  "The plan uses the simulator's true pose. Real odometry drifts, and the map would smear.",
  "The lidar's 36 rays, 0.15–8 m range and lack of noise are assumptions, not the JetHexa lidar's datasheet.",
  "8 of the 9 collisions in the final check were into table legs 5 cm across, which fall between rays 10° apart.",
  "Not merged (PR #22), not in the live app, not on the real robot.",
]; // dagger-report.md §6, §7

// --- Clips and pictures ---------------------------------------------------------------------------------
// Source run and episode of each; the ffmpeg commands are in docs/app.md ("How it works" page).
export const MEDIA = {
  a2b: {
    src: a2bClip, poster: a2bPoster,
    source: "data/nav/runs/20261121-163413-demo/demo.gif · controller default, from (0, 0) to (1.6, 0.2)",
  },
  first: {
    src: firstClip, poster: firstPoster,
    source: "data/nav/dagger/dagger1/videos/round50-episode50000001.mp4 · scene 100001, student alone",
  },
  student: {
    src: studentClip, poster: studentPoster,
    source: "data/nav/dagger/v2/videos/v29-student-final-episode102093.mp4 · final test, student alone",
  },
  searchLoop: {
    src: searchLoop,
    source: "homes/data/search_examples/loop_3.png (homes/search_figure.py)",
  },
};

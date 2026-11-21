import type { ChatEvent, KnockState, RobotState } from "./api";

export function demoState(t: number): RobotState {
  const a = t * 0.25;
  const x = Math.cos(a) * 1.2, y = Math.sin(a) * 1.2;
  const yaw = ((a * 180) / Math.PI + 90) % 360;
  return { mode: "goto", x, y, height: 0.08, yaw_deg: yaw > 180 ? yaw - 360 : yaw,
    target_x: Math.cos(a + 0.4) * 1.2, target_y: Math.sin(a + 0.4) * 1.2, sim_time: t };
}

const answers = [
  { text: "Sure. Walking forward half a metre now [[forward 0.5]]. The path ahead looks clear.", cmd: "forward 0.50" },
  { text: "Turning around for you [[turn 180]]. I'll stop once I'm facing the other way.", cmd: "turn 180.0" },
  { text: "I can see the red pillar ahead and slightly left. Heading there [[goto 1.2 0.8]].", cmd: "goto 1.20 0.80" },
];
let n = 0;

export async function demoChat(onEvent: (e: ChatEvent) => void) {
  const a = answers[n++ % answers.length]!;
  await new Promise((r) => setTimeout(r, 600));
  for (const ch of a.text) {
    onEvent({ type: "token", text: ch });
    await new Promise((r) => setTimeout(r, 18));
  }
  onEvent({ type: "command", text: a.cmd, ok: true, error: null });
  onEvent({ type: "done" });
}

// ---- Knock without the server: the same states, on timers (app/knock.py is the real one) ----

export const KNOCK_IDLE: KnockState = {
  type: "knock", state: "idle", text: "", checked: [], next: null, steps: [], source: "placeholder",
  probs: { couch: 0.42, kitchen: 0.18, armchair: 0.16, bed: 0.13, bathroom: 0.09, entrance: 0.02 },
  last_heard: { zone: "kitchen", time: "09:10" },
};

export function demoKnock(set: (k: KnockState) => void) {
  let timers: number[] = [];
  let card = KNOCK_IDLE;
  const clock = () => new Date().toTimeString().slice(0, 5);
  const show = (k: Partial<KnockState>) => { card = { ...card, ...k }; set(card); };
  const step = (text: string) => [...card.steps, { text, time: clock() }];
  const later = (ms: number, fn: () => void) => { timers.push(window.setTimeout(fn, ms)); };
  const clear = () => { timers.forEach(clearTimeout); timers = []; };
  const search = () => {
    clear();
    show({ state: "finding", text: "Walking to the kitchen (18%).", next: "kitchen", listen_s: null });
    later(3500, () => show({ text: "Looking at the kitchen…" }));
    later(5500, () => show({
      steps: step("Kitchen: not there"), checked: ["kitchen"], next: "couch",
      probs: { couch: 0.51, kitchen: 0, armchair: 0.2, bed: 0.16, bathroom: 0.11, entrance: 0.02 },
      text: "Not in the kitchen. Walking to the couch (51%).",
    }));
    later(10000, () => show({ text: "Looking at the couch…" }));
    later(12000, () => show({
      state: "found", zone: "couch", pose: "lying", next: null, seen: "moved her arm", heard: "a sound, no words",
      time: clock(), steps: step("Found her in front of the couch"),
      text: "On the floor in front of the couch. She moved her arm but didn't answer clearly.",
    }));
  };
  return {
    call: () => {
      if (card.state === "finding") { show({ steps: step("Called her name again") }); return; }
      clear();
      card = KNOCK_IDLE;
      show({ state: "calling", text: "Margarete, it's Thomas. Are you all right?", listen_s: 8 });
      later(8000, () => show({ state: "no_answer", listen_s: null, steps: step("Called her name, no answer") }));
      later(9500, search);
    },
    find: () => {
      if (card.state === "calling") card = { ...card, steps: step("Called her name, no answer") };
      else card = KNOCK_IDLE;
      search();
    },
    stop: () => { clear(); card = KNOCK_IDLE; set(card); },
    choose: (c: string) => {
      if (c === "talk") { clear(); card = KNOCK_IDLE; set(card); return; }
      show({ choice: c === "call_112" ? "call_112" : "call_neighbour",
        steps: step(c === "call_112" ? "Calling 112… (simulated)" : "Calling Frau Weber… (simulated)") });
    },
  };
}

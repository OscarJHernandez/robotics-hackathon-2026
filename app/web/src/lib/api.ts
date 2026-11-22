export const API: string = import.meta.env["VITE_API_BASE"] ?? "";

export type Mode = "idle" | "goto" | "turn" | "fallen";
export type View = "chase" | "orbit" | "overhead" | "eye";
export const VIEWS: View[] = ["chase", "orbit", "overhead", "eye"];
export interface Health {
  ok: boolean; model: string; llm: boolean; commands: string; views?: View[]; voice?: boolean;
  scene?: "apartment" | "floor"; apartment?: boolean;
}
export interface RobotState {
  mode: Mode; x: number; y: number; height: number; yaw_deg: number;
  target_x: number; target_y: number; sim_time: number; turn_to_deg?: number;
  target_dist?: number; target_bearing_deg?: number;
}
export interface ChatMessage { role: "user" | "assistant"; content: string }
export type ChatEvent =
  | { type: "token"; text: string }
  | { type: "command"; text: string; ok: boolean; error: string | null }
  | { type: "error"; text: string }
  | { type: "done" };
export interface CommandResult { ok: boolean; command?: string; error?: string }

export const streamUrl = (v: View) => `${API}/api/stream/${v}`;
export const frameUrl = (v: View) => `${API}/api/frame/${v}.jpg`;

export async function getHealth(signal?: AbortSignal | undefined): Promise<Health> {
  const r = await fetch(`${API}/api/health`, { signal: signal ?? null });
  if (!r.ok) throw new Error(`health ${r.status}`);
  const j = (await r.json()) as Health;
  if (!j || typeof j.ok !== "boolean") throw new Error("bad health");
  return j;
}

export async function getState(signal?: AbortSignal | undefined): Promise<RobotState> {
  const r = await fetch(`${API}/api/state`, { signal: signal ?? null });
  if (!r.ok) throw new Error(`state ${r.status}`);
  return (await r.json()) as RobotState;
}

export async function sendCommand(text: string): Promise<CommandResult> {
  try {
    const r = await fetch(`${API}/api/command`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const j = (await r.json().catch(() => ({}))) as CommandResult;
    if (!r.ok) return { ok: false, error: j.error ?? `HTTP ${r.status}` };
    return j;
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : "network error" };
  }
}

export async function streamChat(
  messages: ChatMessage[], onEvent: (e: ChatEvent) => void, signal?: AbortSignal | undefined, call?: string | undefined,
): Promise<void> {
  const r = await fetch(`${API}/api/chat`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(call ? { messages, call } : { messages }), signal: signal ?? null,
  });
  if (!r.ok || !r.body) {
    const j = (await r.json().catch(() => ({}))) as { error?: string };
    onEvent({ type: "error", text: j.error ?? `HTTP ${r.status}` });
    onEvent({ type: "done" });
    return;
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let finished = false;
  const emit = onEvent;
  onEvent = (e) => { if (e.type === "done") finished = true; emit(e); };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
      for (const line of chunk.split("\n")) {
        if (!line.startsWith("data:")) continue;
        try { onEvent(JSON.parse(line.slice(5).trim()) as ChatEvent); } catch { /* skip */ }
      }
    }
  }
  if (!finished) { // the server went away mid-answer (restart, network): say so instead of ending silently
    onEvent({ type: "error", text: "connection lost before metacon finished, please try again" });
    onEvent({ type: "done" });
  }
}

/** Hide [[command]] markers from displayed assistant text. */
export const stripMarkers = (s: string) =>
  s.replace(/\[\[[^\]]*\]\]/g, "").replace(/\[\[[^\]]*$/, "").replace(/[ \t]{2,}/g, " ").replace(/ ([.,!?;:])/g, "$1");

// ---- voice calls (see app/call.py) ----

/** What every page hears on /api/live while someone is on a voice call. */
export type LiveEvent =
  | { type: "call"; call: string; state: "start" | "end"; seq: number }
  | { type: "heard"; call: string; text: string; final: boolean; seq: number }
  | { type: "token"; call: string; text: string; seq: number }
  | { type: "command"; call: string; text: string; ok: boolean; seq: number }
  | { type: "error"; call: string; text: string; seq: number }
  | { type: "speaking"; call: string; text: string; seq: number };

async function postJson(path: string, body: unknown): Promise<{ status: number; json: Record<string, unknown> }> {
  try {
    const r = await fetch(`${API}${path}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    return { status: r.status, json: (await r.json().catch(() => ({}))) as Record<string, unknown> };
  } catch {
    return { status: 0, json: { error: "network error" } };
  }
}

export async function startCall(): Promise<{ call?: string; error?: string; busy?: boolean }> {
  const { status, json } = await postJson("/api/call/start", {});
  if (status === 200 && typeof json["call"] === "string") return { call: json["call"] };
  return { error: String(json["error"] ?? `HTTP ${status}`), busy: status === 409 };
}

export function endCall(call: string) {
  void postJson("/api/call/end", { call });
}

/** Words in a recording. final=false is a live caption: null when the server skipped it. */
export async function transcribe(audio: Blob, call: string, final: boolean): Promise<{ text: string | null; error?: string }> {
  try {
    const r = await fetch(`${API}/api/transcribe?call=${encodeURIComponent(call)}&final=${final ? 1 : 0}`, {
      method: "POST", headers: { "Content-Type": audio.type || "audio/webm" }, body: audio,
    });
    const j = (await r.json().catch(() => ({}))) as { text?: string | null; error?: string };
    if (!r.ok) return { text: null, error: j.error ?? `HTTP ${r.status}` };
    return { text: j.text ?? null };
  } catch {
    return { text: null, error: "network error" };
  }
}

/** The reply in the robot's voice (MP3), or null. */
export async function speak(text: string, call: string): Promise<Blob | null> {
  try {
    const r = await fetch(`${API}/api/speak`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, call }),
    });
    return r.ok ? await r.blob() : null;
  } catch {
    return null;
  }
}

export const liveUrl = () => `${API}/api/live`;

// ---- Knock: the robot checks on Margarete (see app/knock.py, docs/knock-ui.md) ----

export type KnockZone = "bed" | "bathroom" | "kitchen" | "couch" | "armchair" | "entrance";
export const KNOCK_ZONES: { id: KnockZone; label: string }[] = [
  { id: "bed", label: "Bed" }, { id: "bathroom", label: "Bathroom" }, { id: "kitchen", label: "Kitchen" },
  { id: "couch", label: "Couch" }, { id: "armchair", label: "Armchair" }, { id: "entrance", label: "Entrance" },
];
export const zoneLabel = (z: string | null | undefined) => KNOCK_ZONES.find((k) => k.id === z)?.label ?? (z ?? "");

/** The Knock card as every viewer sees it: the latest {"type": "knock"} event on /api/live. */
export interface KnockState {
  type: "knock";
  state: "idle" | "calling" | "no_answer" | "finding" | "found" | "not_home";
  text: string;
  probs: Partial<Record<KnockZone, number>>;
  checked: KnockZone[];
  next: KnockZone | null;
  steps: { text: string; time: string }[];
  last_heard: { zone: KnockZone; time: string };
  source: "placeholder" | "tabpfn";
  audio?: string; // the call-out in the robot's voice, only in the event that brings it
  listen_s?: number | null; // seconds left for her to answer
  zone?: KnockZone; pose?: string; seen?: string; heard?: string; time?: string;
  choice?: "call_112" | "call_neighbour";
}

export type KnockChoice = "call_112" | "call_neighbour" | "talk";

/** null when it worked, else the error to show. */
async function knockPost(path: string, body: Record<string, unknown> = {}): Promise<string | null> {
  const { status, json } = await postJson(`/api/knock/${path}`, body);
  return status === 200 ? null : String(json["error"] ?? `HTTP ${status}`);
}
export const knockCall = () => knockPost("call");
export const knockFind = () => knockPost("find");
export const knockStop = () => knockPost("stop");
export const knockChoice = (choice: KnockChoice) => knockPost("choice", { choice });
export const knockAudioUrl = (path: string) => `${API}${path}`;

// ---- The apartment: zones, obstacles, journal, belief and the search loop (app/apartment.py) ----

export type ZoneName = "bedroom" | "living_room" | "dining_area" | "kitchen" | "hallway";
export type ZoneClass = ZoneName | "out";
export const ZONE_CLASSES: ZoneClass[] = ["bedroom", "living_room", "dining_area", "kitchen", "hallway", "out"];
export const ZONE_LABEL: Record<ZoneClass, string> = {
  bedroom: "Bedroom", living_room: "Living room", dining_area: "Dining area", kitchen: "Kitchen", hallway: "Hallway", out: "Out",
};
/** Zone colors: categorical slots 1-5 of the chart palette (dark steps, validated as a set); "out" is a neutral. */
export const ZONE_COLOR: Record<ZoneClass, string> = {
  bedroom: "#3987e5", living_room: "#d95926", dining_area: "#199e70", kitchen: "#c98500", hallway: "#d55181", out: "#898781",
};
export type Belief = Record<ZoneClass, number>;
export type Stage = "population" | "habits" | "journal" | "last_heard";
export interface ApartmentZone {
  name: ZoneName; label: string; x: number; y: number; size_x: number; size_y: number;
  view_x: number; view_y: number; person_x: number; person_y: number;
}
export interface ApartmentObstacle { name: string; zone: ZoneName; x: number; y: number; size_x: number; size_y: number }
export interface JournalEntry {
  id: number; day: string; weekday: number; minute: number; time: string; slot: number | null;
  zone: ZoneClass; source: "robot" | "entered" | "search";
}
export interface SearchStep { zone: ZoneName; p: number; seen: boolean; metres: number }
export interface LogLine { id: number; time: string; text: string; kind: string }
export interface ApartmentState {
  seed: number; name: string;
  flat: {
    half: number; door: { wall: string; x: number; y: number; width: number }; dock: { x: number; y: number };
    zones: ApartmentZone[]; obstacles: ApartmentObstacle[];
  };
  habits: { usual: Record<"0" | "1", ZoneName[]>; outings: [number, number][] };
  now: { weekday: number; minute: number; time: string; day: string; slot: number | null; real: boolean };
  journal: JournalEntry[]; journal_total: number;
  last_heard: { zone: ZoneClass; minutes: number } | null;
  stages: Record<Stage, Belief>; belief: Belief; plan: ZoneName[];
  search: {
    status: "idle" | "starting" | "walking" | "looking" | "found" | "nobody_home" | "gave_up" | "stopped" | "failed";
    running: boolean; goal: ZoneName | null; truth: ZoneClass | null; miss_rate: number;
    path: [number, number][]; metres: number; steps: SearchStep[];
  };
  robot: { x: number; y: number; yaw_deg?: number };
  driver: string;
  log: LogLine[];
}

export async function getApartment(signal?: AbortSignal | undefined): Promise<ApartmentState> {
  const r = await fetch(`${API}/api/apartment`, { signal: signal ?? null });
  if (!r.ok) throw new Error(`apartment ${r.status}`);
  return (await r.json()) as ApartmentState;
}

async function apartmentPost(action: string, body: object = {}): Promise<{ ok: boolean; error?: string }> {
  try {
    const r = await fetch(`${API}/api/apartment/${action}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const j = (await r.json().catch(() => ({}))) as { ok?: boolean; error?: string };
    return r.ok ? { ok: true } : { ok: false, error: j.error ?? `HTTP ${r.status}` };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : "network error" };
  }
}
export const apartmentAddEntry = (zone: ZoneClass, time: string, days_ago: number) =>
  apartmentPost("journal", { zone, time, days_ago });
export const apartmentRemoveEntry = (id: number) => apartmentPost("journal/remove", { id });
export const apartmentClock = (weekday: number | null, time: string | null) =>
  apartmentPost("clock", weekday === null && time === null ? {} : { weekday, time });
export const apartmentSearch = (where: string) => apartmentPost("search", { where });
export const apartmentStop = () => apartmentPost("stop");
export const apartmentReset = (seed: number) => apartmentPost("reset", { seed });

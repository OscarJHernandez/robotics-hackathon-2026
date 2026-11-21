export const API: string = import.meta.env["VITE_API_BASE"] ?? "";

export type Mode = "idle" | "goto" | "turn" | "fallen";
export type View = "chase" | "orbit" | "overhead" | "eye";
export const VIEWS: View[] = ["chase", "orbit", "overhead", "eye"];
export interface Health { ok: boolean; model: string; llm: boolean; commands: string; views?: View[]; voice?: boolean }
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

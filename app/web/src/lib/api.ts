export const API: string = import.meta.env["VITE_API_BASE"] ?? "";

export type Mode = "idle" | "goto" | "turn" | "fallen";
export type View = "chase" | "orbit" | "overhead" | "eye";
export const VIEWS: View[] = ["chase", "orbit", "overhead", "eye"];
export interface Health { ok: boolean; model: string; llm: boolean; commands: string; views?: View[] }
export interface RobotState {
  mode: Mode; x: number; y: number; height: number; yaw_deg: number;
  target_x: number; target_y: number; sim_time: number; turn_to_deg?: number;
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
  messages: ChatMessage[], onEvent: (e: ChatEvent) => void, signal?: AbortSignal | undefined,
): Promise<void> {
  const r = await fetch(`${API}/api/chat`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }), signal: signal ?? null,
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
}

/** Hide [[command]] markers from displayed assistant text. */
export const stripMarkers = (s: string) =>
  s.replace(/\[\[[^\]]*\]\]/g, "").replace(/\[\[[^\]]*$/, "").replace(/[ \t]{2,}/g, " ").replace(/ ([.,!?;:])/g, "$1");

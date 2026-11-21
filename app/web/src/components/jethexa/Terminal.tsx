import { useCallback, useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { sendCommand, streamChat, stripMarkers, type ChatEvent, type ChatMessage, type RobotState } from "@/lib/api";
import { demoChat } from "@/lib/demo";
import type { Conn } from "@/hooks/useRobotState";

export type Line =
  | { id: number; kind: "user"; text: string }
  | { id: number; kind: "ai"; text: string; pending?: boolean }
  | { id: number; kind: "cmd"; text: string; ok: boolean; error?: string | null | undefined }
  | { id: number; kind: "event"; text: string; tone: "ok" | "warn" }
  | { id: number; kind: "info"; text: string };
type NoId<T> = T extends unknown ? Omit<T, "id"> : never;

const WELCOME = 'metacon online. Ask me to move the robot, e.g. "walk to the red pillar" or "turn around". Type /help for direct commands.';
const HELP = "/target · /goto X Y · /forward M · /turn DEG · /stop · /random · /reset · /clear · /help";
let uid = 0;

/** All metacon terminal behaviour: chat stream, slash commands, input history, robot events. */
export function useMetacon(conn: Conn, llm: boolean) {
  const [lines, setLines] = useState<Line[]>([{ id: uid++, kind: "ai", text: WELCOME }]);
  const [busy, setBusy] = useState(false);
  const hist = useRef<ChatMessage[]>([]);
  const inputs = useRef<string[]>([]);
  const hIdx = useRef(-1);
  const live = conn === "live";
  const chatOff = live && !llm;

  const push = useCallback((l: NoId<Line>) => setLines((p) => [...p, { ...l, id: uid++ } as Line]), []);
  const clear = useCallback(() => setLines([]), []);
  /** A line from the robot itself (arrived, fell, ...). */
  const note = useCallback((text: string, tone: "ok" | "warn" = "ok") => push({ kind: "event", text, tone }), [push]);

  const history = (dir: -1 | 1): string | null => {
    const h = inputs.current; if (!h.length) return null;
    let i = hIdx.current === -1 ? h.length : hIdx.current;
    i = dir === -1 ? Math.max(0, i - 1) : i + 1;
    if (i >= h.length) { hIdx.current = -1; return ""; }
    hIdx.current = i; return h[i]!;
  };

  const send = async (raw0: string) => {
    const text = raw0.trim(); hIdx.current = -1;
    if (!text || busy) return;
    inputs.current.push(text);
    if (text === "/clear") { setLines([]); return; }
    push({ kind: "user", text });
    if (text === "/help") { push({ kind: "info", text: HELP }); return; }
    if (text.startsWith("/")) {
      const cmd = text.slice(1);
      if (!live) { push({ kind: "cmd", text: cmd, ok: true }); return; }
      const r = await sendCommand(cmd);
      push({ kind: "cmd", text: r.command ?? cmd, ok: r.ok, error: r.error });
      return;
    }
    if (chatOff) { push({ kind: "info", text: "chat offline — use /help for direct commands" }); return; }
    hist.current.push({ role: "user", content: text });
    setBusy(true);
    const aiId = uid++;
    setLines((p) => [...p, { id: aiId, kind: "ai", text: "", pending: true }]);
    let raw = "";
    let ran = false;
    const onEvent = (e: ChatEvent) => {
      if (e.type === "token") {
        raw += e.text;
        setLines((p) => p.map((l) => (l.id === aiId ? { ...l, text: stripMarkers(raw).trim(), pending: false } : l)));
      } else if (e.type === "command") { ran = true; push({ kind: "cmd", text: e.text, ok: e.ok, error: e.error }); }
      else if (e.type === "error") { ran = true; push({ kind: "cmd", text: "error", ok: false, error: e.text }); }
    };
    try {
      if (live) await streamChat(hist.current.slice(-20), onEvent);
      else await demoChat(onEvent);
    } catch (e) {
      onEvent({ type: "error", text: e instanceof Error ? e.message : "connection lost" });
    }
    if (raw) hist.current.push({ role: "assistant", content: raw });
    else hist.current.pop(); // nothing came back: do not keep an unanswered question in the history
    setLines((p) => p.filter((l) => !(l.id === aiId && !l.text)));
    if (!stripMarkers(raw).trim() && !ran) push({ kind: "info", text: "metacon did not answer, please try again" });
    setBusy(false);
  };

  return { lines, busy, send, clear, history, chatOff, note };
}

function statusText(s: RobotState | null) {
  if (!s) return "";
  if (s.mode === "goto") return `walking · ${(s.target_dist ?? 0).toFixed(1)} m to go`;
  if (s.mode === "turn") return `turning · ${Math.round(s.yaw_deg)}°`;
  if (s.mode === "fallen") return "fallen · /reset";
  return `idle · x ${s.x.toFixed(2)} y ${s.y.toFixed(2)}`;
}

export function TerminalPanel({ lines, busy, chatOff, live, state, onSend, onHistory, onClose, onClear }: {
  lines: Line[]; busy: boolean; chatOff: boolean; live: boolean; state: RobotState | null;
  onSend: (t: string) => void; onHistory: (dir: -1 | 1) => string | null; onClose: () => void; onClear: () => void;
}) {
  const scroll = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const [v, setV] = useState("");
  useEffect(() => { scroll.current?.scrollTo({ top: scroll.current.scrollHeight, behavior: "smooth" }); }, [lines, busy]);
  useEffect(() => { if (!busy) input.current?.focus(); }, [busy]);
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.ctrlKey && e.key.toLowerCase() === "l") { e.preventDefault(); onClear(); } };
    window.addEventListener("keydown", k); return () => window.removeEventListener("keydown", k);
  }, [onClear]);
  const moving = state?.mode === "goto" || state?.mode === "turn";
  return (
    <section className="terminal rise-in flex h-[44vh] min-h-[260px] flex-col" onClick={() => input.current?.focus()}>
      <div className="terminal-bar">
        <span className="dot bg-term-red" /><span className="dot bg-term-amber" /><span className="dot bg-term-green" />
        <span className="ml-2 hidden sm:inline">metacon@jethexa: ~</span>
        <span className="flex-1" />
        <span className={`flex items-center gap-1.5 text-[11px] ${state?.mode === "fallen" ? "text-term-amber" : moving ? "text-term-green" : ""}`}>
          {live && <span className={`h-1.5 w-1.5 rounded-full ${moving ? "bg-term-green animate-pulse" : state?.mode === "fallen" ? "bg-term-amber" : "bg-term-dim"}`} />}
          {chatOff ? "chat offline" : live ? statusText(state) : "demo"}
        </span>
        <button onClick={(e) => { e.stopPropagation(); onClose(); }} aria-label="Close terminal" className="ml-3 opacity-60 hover:opacity-100"><X className="h-3.5 w-3.5" /></button>
      </div>
      <div ref={scroll} className="flex-1 space-y-1.5 overflow-y-auto p-4 font-mono text-[13px] leading-relaxed">
        {lines.map((l) => {
          if (l.kind === "user") return <div key={l.id} className="whitespace-pre-wrap text-term-green"><span className="opacity-70">you@jethexa:~$ </span>{l.text}</div>;
          if (l.kind === "ai") return <div key={l.id} className="whitespace-pre-wrap text-term-red"><span className="font-semibold">metacon&gt; </span>{l.text}{l.pending && <span className="cursor-blink">▍</span>}</div>;
          if (l.kind === "cmd") return <div key={l.id} className="pl-4 text-term-dim">⚙ {l.text} {l.ok ? <span className="text-term-green">✓</span> : <><span className="text-term-red">✗</span> <span className="text-term-amber">{l.error}</span></>}</div>;
          if (l.kind === "event") return <div key={l.id} className={`pl-4 ${l.tone === "ok" ? "text-term-green/80" : "text-term-amber"}`}>{l.tone === "ok" ? "●" : "▲"} {l.text}</div>;
          return <div key={l.id} className="text-term-dim">{l.text}</div>;
        })}
        <label className={`flex items-center text-term-green ${busy ? "opacity-40" : ""}`}>
          <span className="shrink-0 opacity-70">you@jethexa:~$&nbsp;</span>
          <input ref={input} value={v} disabled={busy} spellCheck={false} autoComplete="off" aria-label="Message metacon"
            placeholder={busy ? "metacon is answering…" : "type a message or /help"}
            onChange={(e) => setV(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") { e.preventDefault(); if (v.trim()) { onSend(v); setV(""); } }
              else if (e.key === "Escape") onClose();
              else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
                const r = onHistory(e.key === "ArrowUp" ? -1 : 1);
                if (r !== null) { e.preventDefault(); setV(r); }
              }
            }}
            className="min-w-0 flex-1 bg-transparent caret-term-green outline-none placeholder:text-term-dim/50" />
        </label>
      </div>
    </section>
  );
}

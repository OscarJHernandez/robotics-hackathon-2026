import { useCallback, useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { sendCommand, streamChat, stripMarkers, type ChatEvent, type ChatMessage } from "@/lib/api";
import { demoChat } from "@/lib/demo";
import type { Conn } from "@/hooks/useRobotState";

export type Line =
  | { id: number; kind: "user"; text: string }
  | { id: number; kind: "ai"; text: string; pending?: boolean }
  | { id: number; kind: "cmd"; text: string; ok: boolean; error?: string | null | undefined }
  | { id: number; kind: "info"; text: string };
type NoId<T> = T extends unknown ? Omit<T, "id"> : never;

const WELCOME = 'metacon online. Ask me to move the robot, e.g. "walk to the red pillar" or "turn around". Type /help for direct commands.';
const HELP = "/goto X Y · /forward M · /turn DEG · /stop · /random · /reset · /clear · /help";
let uid = 0;

/** All metacon terminal behaviour: chat stream, slash commands, input history. */
export function useMetacon(conn: Conn, llm: boolean) {
  const [lines, setLines] = useState<Line[]>([{ id: uid++, kind: "ai", text: WELCOME }]);
  const [busy, setBusy] = useState(false);
  const hist = useRef<ChatMessage[]>([]);
  const inputs = useRef<string[]>([]);
  const hIdx = useRef(-1);
  const live = conn === "live";
  const chatOff = live && !llm;

  const push = (l: NoId<Line>) => setLines((p) => [...p, { ...l, id: uid++ } as Line]);
  const clear = useCallback(() => setLines([]), []);

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
    const onEvent = (e: ChatEvent) => {
      if (e.type === "token") {
        raw += e.text;
        setLines((p) => p.map((l) => (l.id === aiId ? { ...l, text: stripMarkers(raw), pending: false } : l)));
      } else if (e.type === "command") push({ kind: "cmd", text: e.text, ok: e.ok, error: e.error });
      else if (e.type === "error") push({ kind: "cmd", text: "error", ok: false, error: e.text });
    };
    try {
      if (live) await streamChat(hist.current.slice(-20), onEvent);
      else await demoChat(onEvent);
    } catch (e) {
      onEvent({ type: "error", text: e instanceof Error ? e.message : "connection lost" });
    }
    if (raw) hist.current.push({ role: "assistant", content: raw });
    setLines((p) => p.filter((l) => !(l.id === aiId && !l.text)));
    setBusy(false);
  };

  return { lines, busy, send, clear, history, chatOff };
}

export function TerminalPanel({ lines, busy, chatOff, live, onClose, onClear }: {
  lines: Line[]; busy: boolean; chatOff: boolean; live: boolean; onClose: () => void; onClear: () => void;
}) {
  const scroll = useRef<HTMLDivElement>(null);
  useEffect(() => { scroll.current?.scrollTo({ top: scroll.current.scrollHeight, behavior: "smooth" }); }, [lines]);
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.ctrlKey && e.key.toLowerCase() === "l") { e.preventDefault(); onClear(); } };
    window.addEventListener("keydown", k); return () => window.removeEventListener("keydown", k);
  }, [onClear]);
  return (
    <section className="terminal rise-in flex h-[40vh] min-h-[240px] flex-col">
      <div className="terminal-bar">
        <span className="dot bg-term-red" /><span className="dot bg-term-amber" /><span className="dot bg-term-green" />
        <span className="flex-1 text-center">metacon@jethexa: ~</span>
        <span className="text-[10px] uppercase tracking-wider">{chatOff ? "chat offline" : live ? "" : "demo"}</span>
        <button onClick={onClose} aria-label="Close terminal" className="ml-1 opacity-60 hover:opacity-100"><X className="h-3.5 w-3.5" /></button>
      </div>
      <div ref={scroll} className="flex-1 space-y-1.5 overflow-y-auto p-4 font-mono text-[13px] leading-relaxed">
        {lines.map((l) => {
          if (l.kind === "user") return <div key={l.id} className="whitespace-pre-wrap text-term-green"><span className="opacity-80">you@jethexa:~$ </span>{l.text}</div>;
          if (l.kind === "ai") return <div key={l.id} className="whitespace-pre-wrap text-term-red"><span className="font-semibold">metacon&gt; </span>{l.text}{l.pending && <span className="cursor-blink">▍</span>}</div>;
          if (l.kind === "cmd") return <div key={l.id} className="text-term-dim">⚙ {l.text} {l.ok ? <span className="text-term-green">✓</span> : <><span className="text-term-red">✗</span> <span className="text-term-amber">{l.error}</span></>}</div>;
          return <div key={l.id} className="text-term-dim">{l.text}</div>;
        })}
        {!busy && <div className="text-term-green"><span className="opacity-80">you@jethexa:~$ </span><span className="cursor-blink">█</span></div>}
      </div>
    </section>
  );
}

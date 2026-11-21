import { ArrowUp, SquareTerminal } from "lucide-react";
import { useRef, useState } from "react";

export function ChatBar({ onSend, onHistory, busy, termOpen, onToggleTerm }: {
  onSend: (t: string) => void; onHistory: (dir: -1 | 1, current: string) => string | null;
  busy: boolean; termOpen: boolean; onToggleTerm: () => void;
}) {
  const [v, setV] = useState("");
  const ref = useRef<HTMLInputElement>(null);
  const send = () => { if (busy || !v.trim()) return; onSend(v); setV(""); };
  return (
    <div className="flex w-full items-center gap-2">
      <button onClick={onToggleTerm} aria-pressed={termOpen}
        className="term-toggle flex h-14 shrink-0 items-center gap-2 rounded-full px-4 text-sm font-medium max-sm:w-14 max-sm:justify-center max-sm:px-0">
        <SquareTerminal className="h-4 w-4" /><span className="max-sm:hidden">Terminal</span>
      </button>
      <div className="chat-pill flex h-14 flex-1 items-center pl-5 pr-2">
        <input ref={ref} value={v} onChange={(e) => setV(e.target.value)} placeholder="Tell metacon what to do…"
          aria-label="Message metacon" spellCheck={false}
          onKeyDown={(e) => {
            if (e.key === "Enter") { e.preventDefault(); send(); }
            else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
              const r = onHistory(e.key === "ArrowUp" ? -1 : 1, v);
              if (r !== null) { e.preventDefault(); setV(r); }
            }
          }}
          className="h-full flex-1 bg-transparent text-[15px] text-card-foreground outline-none placeholder:text-muted-foreground" />
        <button onClick={send} disabled={busy || !v.trim()} aria-label="Send"
          className="flex h-10 w-10 items-center justify-center rounded-full bg-primary text-primary-foreground transition-all duration-200 hover:brightness-110 disabled:opacity-40">
          <ArrowUp className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

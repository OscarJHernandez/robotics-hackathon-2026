// The page on a phone: a call screen with the robot's camera on top, the conversation in the middle and one big
// hold-to-talk button at the bottom, like a walkie-talkie. Same state and hooks as the desktop page (App.tsx).
import { useEffect, useRef, useState } from "react";
import { ArrowUp, Keyboard, Mic } from "lucide-react";
import type { Conn } from "@/hooks/useRobotState";
import type { LiveCall } from "@/hooks/useLiveCall";
import { phaseText, type Phase } from "@/hooks/useVoiceCall";
import type { RobotState, View } from "@/lib/api";
import { HexMark } from "./Brand";
import { LiveCallCard } from "./LiveCallCard";
import { Stream, VIEW_LABEL } from "./Stream";
import { SpeakerToggle, TalkButton } from "./TalkButton";
import { statusText, type Line } from "./Terminal";

const SUGGESTIONS = ["Walk to the red pillar", "Turn around", "Walk forward one metre", "Stop"];

export interface VoiceProps {
  supported: boolean; phase: Phase; interim: string;
  press: () => void; release: () => void; cancel: () => void;
}

export function PhoneApp({ conn, views, state, lines, busy, chatOff, voice, other, speakerOn, onToggleSpeaker, onSend }: {
  conn: Conn; views: View[]; state: RobotState | null; lines: Line[]; busy: boolean; chatOff: boolean;
  voice: VoiceProps; other: LiveCall | null; speakerOn: boolean; onToggleSpeaker: () => void;
  onSend: (text: string) => void;
}) {
  const live = conn === "live";
  const [view, setView] = useState<View>("chase");
  const [typing, setTyping] = useState(false); // without voice the text box shows anyway (canTalk)
  const [text, setText] = useState("");
  const [cancelArmed, setCancelArmed] = useState(false);
  const scroll = useRef<HTMLDivElement>(null);
  const lineBusy = !!other && !other.ended && voice.phase === "idle";
  const canTalk = voice.supported && !chatOff;

  useEffect(() => {
    scroll.current?.scrollTo({ top: scroll.current.scrollHeight, behavior: "smooth" });
  }, [lines, voice.interim, voice.phase, other]);

  const status = lineBusy ? "Someone else is on a call…"
    : voice.phase === "idle" && busy ? "metacon is answering…" : phaseText(voice.phase, cancelArmed);
  const submit = () => { const t = text.trim(); if (t && !busy) { onSend(t); setText(""); } };
  const recording = voice.phase === "recording" || voice.phase === "understanding";

  return (
    <div className="phone-app">
      {/* The robot's view */}
      <div className="relative h-[42dvh] shrink-0 overflow-hidden rounded-b-[28px] shadow-[0_18px_40px_-16px_oklch(0_0_0/70%)]">
        <div key={view} className="absolute inset-0 crossfade"><Stream view={view} live={live} /></div>
        <div className="stage-vignette" />
        <div className="absolute inset-x-0 top-0 flex items-center gap-2 px-4 pt-[max(12px,env(safe-area-inset-top))]">
          <HexMark />
          <span className="text-shadow-soft font-semibold tracking-tight">JetHexa</span>
          <span className="glass pixel flex h-5 items-center gap-1.5 rounded-full px-2 text-[9px]">
            <span className={live ? "live-dot" : "h-1.5 w-1.5 rounded-full bg-video/50"} />{live ? "LIVE" : "DEMO"}
          </span>
          <span className="text-shadow-soft ml-auto truncate font-mono text-[11px] text-video/85">{live ? statusText(state) : "demo"}</span>
        </div>
        <div className="absolute inset-x-0 bottom-0 flex gap-1.5 overflow-x-auto px-3 pb-3 [scrollbar-width:none]">
          {views.map((v) => (
            <button key={v} onClick={() => setView(v)}
              className={`glass-btn h-8 shrink-0 px-3 text-xs font-medium ${v === view ? "glass-btn-on" : ""}`}>{VIEW_LABEL[v]}</button>
          ))}
        </div>
      </div>

      {/* The conversation */}
      <div ref={scroll} className="flex-1 space-y-2 overflow-y-auto px-4 py-4">
        {other && !other.ended && <LiveCallCard call={other} className="sticky top-0 z-10" />}
        {lines.map((l) => <Bubble key={l.id} line={l} />)}
        {voice.phase !== "idle" && voice.phase !== "thinking" && voice.phase !== "speaking" && (
          <div className="flex justify-end">
            <div className="max-w-[85%] rounded-2xl rounded-br-md border border-dashed border-video/30 px-3.5 py-2 text-[15px] text-video/80">
              {voice.interim || (recording ? "…" : "")}{voice.phase === "recording" && <span className="cursor-blink">▍</span>}
            </div>
          </div>
        )}
        {lines.length <= 1 && voice.phase === "idle" && (
          <div className="flex flex-wrap justify-center gap-2 pt-2">
            {SUGGESTIONS.map((s) => (
              <button key={s} onClick={() => onSend(s)} disabled={busy}
                className="glass-btn h-9 px-3.5 text-sm">{s}</button>
            ))}
          </div>
        )}
      </div>

      {/* Talk */}
      <div className="shrink-0 px-5 pt-2 pb-[max(16px,env(safe-area-inset-bottom))]">
        {typing || !canTalk ? (
          <div className="flex items-center gap-2">
            {canTalk && (
              <button onClick={() => setTyping(false)} aria-label="Talk instead" className="glass-btn h-12 w-12"><Mic className="h-5 w-5" /></button>
            )}
            <div className="chat-pill flex h-12 flex-1 items-center pl-4 pr-1.5">
              <input value={text} onChange={(e) => setText(e.target.value)} placeholder={chatOff ? "chat offline" : "Tell metacon what to do…"}
                aria-label="Message metacon" enterKeyHint="send" disabled={chatOff}
                onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); submit(); } }}
                className="h-full min-w-0 flex-1 bg-transparent text-[16px] text-card-foreground outline-none placeholder:text-muted-foreground" />
              <button onClick={submit} disabled={busy || !text.trim()} aria-label="Send"
                className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-primary-foreground disabled:opacity-40">
                <ArrowUp className="h-4 w-4" />
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className={`mb-3 text-center text-sm ${cancelArmed ? "text-term-amber" : voice.phase === "recording" ? "text-term-red" : "text-video/70"}`}>
              {status}
            </div>
            <div className="flex items-center justify-between">
              <button onClick={() => setTyping(true)} aria-label="Type instead" className="glass-btn h-12 w-12"><Keyboard className="h-5 w-5" /></button>
              <TalkButton phase={voice.phase} disabled={lineBusy || (busy && voice.phase === "idle")}
                onPress={voice.press} onRelease={voice.release} onCancel={voice.cancel} onCancelArmed={setCancelArmed} />
              <SpeakerToggle on={speakerOn} onToggle={onToggleSpeaker} className={`glass-btn h-12 w-12 ${speakerOn ? "" : "opacity-60"}`} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Bubble({ line: l }: { line: Line }) {
  if (l.kind === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-3.5 py-2 text-[15px] text-primary-foreground">
          {l.voice && <Mic className="mr-1 inline h-3.5 w-3.5 opacity-70" />}{l.text}
        </div>
      </div>
    );
  }
  if (l.kind === "ai") {
    if (!l.text && !l.pending) return null;
    return (
      <div className="flex justify-start">
        <div className="glass max-w-[85%] rounded-2xl rounded-bl-md px-3.5 py-2 text-[15px] leading-snug">
          <div className="pixel mb-0.5 text-[9px] tracking-[0.2em] text-term-red">metacon</div>
          {l.text}{l.pending && <span className="cursor-blink">▍</span>}
        </div>
      </div>
    );
  }
  if (l.kind === "cmd") {
    return (
      <div className="flex justify-center">
        <span className={`rounded-full px-2.5 py-1 font-mono text-[11px] ${l.ok ? "bg-term-green/15 text-term-green" : "bg-term-amber/15 text-term-amber"}`}>
          ⚙ {l.text} {l.ok ? "✓" : `✗ ${l.error ?? ""}`}
        </span>
      </div>
    );
  }
  if (l.kind === "event") {
    return <div className={`text-center text-xs ${l.tone === "ok" ? "text-term-green/80" : "text-term-amber"}`}>{l.text}</div>;
  }
  return <div className="text-center text-xs text-video/50">{l.text}</div>;
}

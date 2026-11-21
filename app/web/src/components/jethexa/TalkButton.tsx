import { useRef, useState } from "react";
import { Loader2, Mic, Square, Volume2, VolumeX } from "lucide-react";
import type { Phase } from "@/hooks/useVoiceCall";

const CANCEL_PX = 90; // slide this far up while holding to cancel

/**
 * Hold to talk: pointer down starts, pointer up sends, sliding up and letting go cancels. While the reply is
 * spoken, a tap stops it. Works with touch, mouse and pen (pointer events, captured so a finger that slides off
 * the button still ends the recording).
 */
export function TalkButton({ phase, disabled, size = "lg", onPress, onRelease, onCancel, onCancelArmed }: {
  phase: Phase; disabled?: boolean; size?: "lg" | "sm";
  onPress: () => void; onRelease: () => void; onCancel: () => void; onCancelArmed?: (armed: boolean) => void;
}) {
  const startY = useRef<number | null>(null);
  const [armed, setArmed] = useState(false);
  const arm = (a: boolean) => { if (a !== armed) { setArmed(a); onCancelArmed?.(a); } };
  const recording = phase === "recording" || phase === "starting";
  const working = phase === "understanding" || phase === "thinking";
  const speaking = phase === "speaking";
  const lg = size === "lg";

  const end = (cancel: boolean) => {
    if (startY.current === null) return;
    startY.current = null;
    arm(false);
    if (cancel) onCancel(); else onRelease();
  };

  const label = speaking ? "Stop speaking" : recording ? "Release to send" : "Hold to talk";
  return (
    <button
      type="button" aria-label={label} title={label} disabled={disabled || working}
      onPointerDown={(e) => {
        if (disabled || working || e.button > 0) return;
        e.preventDefault();
        if (speaking) { onPress(); return; }
        try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* not a live pointer */ }
        startY.current = e.clientY;
        onPress();
      }}
      onPointerMove={(e) => { if (startY.current !== null) arm(startY.current - e.clientY > CANCEL_PX); }}
      onPointerUp={() => end(armed)}
      onPointerCancel={() => end(true)}
      onLostPointerCapture={() => end(armed)}
      onContextMenu={(e) => e.preventDefault()}
      onKeyDown={(e) => { if ((e.key === " " || e.key === "Enter") && !e.repeat && startY.current === null) { e.preventDefault(); startY.current = 0; onPress(); } }}
      onKeyUp={(e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); end(false); } }}
      className={`talk-btn relative flex shrink-0 items-center justify-center rounded-full transition-all duration-150 disabled:opacity-40
        ${lg ? "h-[88px] w-[88px]" : "h-10 w-10"}
        ${recording ? (armed ? "bg-term-amber text-stage" : "bg-term-red text-video talk-live") : speaking ? "bg-primary text-primary-foreground" : lg ? "bg-video text-[var(--stage)]" : "bg-secondary text-secondary-foreground hover:brightness-95"}
        ${recording && lg ? "scale-110" : ""}`}
    >
      {working ? <Loader2 className={`${lg ? "h-8 w-8" : "h-4 w-4"} animate-spin`} />
        : speaking ? <Square className={lg ? "h-7 w-7" : "h-3.5 w-3.5"} fill="currentColor" />
        : <Mic className={lg ? "h-9 w-9" : "h-4 w-4"} />}
    </button>
  );
}

export function SpeakerToggle({ on, onToggle, className = "" }: { on: boolean; onToggle: () => void; className?: string }) {
  const label = on ? "Spoken replies on" : "Spoken replies off";
  return (
    <button type="button" onClick={onToggle} aria-pressed={on} aria-label={label} title={label}
      className={`flex shrink-0 items-center justify-center rounded-full ${className}`}>
      {on ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
    </button>
  );
}

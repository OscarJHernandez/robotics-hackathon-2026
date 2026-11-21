import { Phone, PhoneOff } from "lucide-react";
import type { LiveCall } from "@/hooks/useLiveCall";

/** Someone's voice call, shown live to everyone else on the page: what they say and what metacon answers. */
export function LiveCallCard({ call, className = "" }: { call: LiveCall; className?: string }) {
  const listening = !call.final && !call.ended;
  return (
    <div className={`glass live-call rise-in w-full rounded-2xl px-4 py-3 text-left ${className}`} aria-live="polite">
      <div className="pixel flex items-center gap-2 text-[10px] tracking-[0.2em] text-video/75">
        {call.ended ? <PhoneOff className="h-3 w-3" /> : <span className="call-dot" />}
        {call.ended ? "Call ended" : call.speaking ? "Live call · robot speaking" : listening ? "Live call · listening" : "Live call"}
        {!call.ended && <Phone className="ml-auto h-3 w-3 opacity-60" />}
      </div>
      <p className="mt-1.5 text-[15px] leading-snug text-video">
        <span className="text-video/60">Caller: </span>
        {call.heard ? <>“{call.heard}”</> : <span className="text-video/50">speaking…</span>}
        {listening && <span className="cursor-blink">▍</span>}
      </p>
      {(call.reply || call.commands.length > 0) && (
        <p className="mt-1 text-[15px] leading-snug text-term-red">
          <span className="font-semibold">metacon: </span>{call.reply}
          {call.commands.map((c, i) => (
            <span key={i} className={`ml-2 inline-block rounded-full px-2 py-0.5 font-mono text-[11px] ${c.ok ? "bg-term-green/15 text-term-green" : "bg-term-amber/15 text-term-amber"}`}>
              ⚙ {c.text}
            </span>
          ))}
        </p>
      )}
    </div>
  );
}

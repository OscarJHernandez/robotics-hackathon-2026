// The Knock card: Find Margarete and Call Margarete, and what follows (docs/knock-ui.md, mockup screens 5-7).
// Every viewer sees the same card: it is drawn from the latest knock event (useKnock).
import { useEffect, useState } from "react";
import { Home, Mic, Phone, Search, Square, Volume2 } from "lucide-react";
import type { Knock } from "@/hooks/useKnock";
import { KNOCK_ZONES, zoneLabel, type KnockState } from "@/lib/api";

/** True while a check-in is on screen (anything but the idle card). */
export const knockActive = (k: KnockState) => k.state !== "idle";

export function KnockCard({ k: { knock, call, find, stop, choose } }: { k: Knock }) {
  switch (knock.state) {
    case "idle": return <Idle knock={knock} onFind={find} onCall={call} />;
    case "calling":
    case "no_answer": return <Calling knock={knock} onStop={stop} onFind={find} />;
    case "finding": return <Finding knock={knock} onStop={stop} onCall={call} />;
    case "found": return <Found knock={knock} onChoose={choose} onDone={stop} />;
    case "not_home": return <NotHome knock={knock} onFind={find} onDone={stop} />;
  }
}

function Idle({ knock, onFind, onCall }: { knock: KnockState; onFind: () => void; onCall: () => void }) {
  return (
    <section aria-label="Margarete" className="knock-card rise-in flex flex-col gap-3.5">
      <div className="flex items-center gap-3">
        <div aria-hidden className="knock-avatar">M</div>
        <div className="flex min-w-0 flex-col gap-0.5">
          <span className="text-lg font-semibold">Margarete</span>
          <span className="font-mono text-xs text-video/70">Last heard: {zoneLabel(knock.last_heard.zone).toLowerCase()} · {knock.last_heard.time}</span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <button type="button" onClick={onFind} className="knock-btn knock-primary h-14 flex-col gap-[3px]">
          <Search className="h-[18px] w-[18px]" />Find Margarete
        </button>
        <button type="button" onClick={onCall} className="knock-btn knock-secondary h-14 flex-col gap-[3px]">
          <Volume2 className="h-[18px] w-[18px]" />Call Margarete
        </button>
      </div>
      <p className="text-[12.5px] leading-[1.45] text-video/70">
        Call speaks your message through the robot. Find walks to where she most likely is.
      </p>
    </section>
  );
}

function Calling({ knock, onStop, onFind }: { knock: KnockState; onStop: () => void; onFind: () => void }) {
  const left = useCountdown(knock);
  const noAnswer = knock.state === "no_answer";
  return (
    <section aria-label="Calling Margarete" aria-live="polite" className="knock-card rise-in flex flex-col gap-3">
      <Header dot={noAnswer ? "amber" : "call"} label={noAnswer ? "No answer" : "Calling Margarete"}
        right={!noAnswer && left !== null ? `${left} s` : undefined} />
      <p className="text-base leading-snug">“{knock.text}”</p>
      <p className="text-[13px] text-video/70">
        {noAnswer ? "Starting the search…" : left === null ? "The robot is speaking…" : "Listening for her answer…"}
      </p>
      <div className="grid grid-cols-2 gap-2">
        <button type="button" onClick={onStop} className="knock-btn knock-secondary h-12 gap-2"><Square className="h-4 w-4" />Stop</button>
        <button type="button" onClick={onFind} disabled={noAnswer} className="knock-btn knock-primary h-12 gap-2 disabled:opacity-50">
          <Search className="h-4 w-4" />Find her now
        </button>
      </div>
    </section>
  );
}

function Finding({ knock, onStop, onCall }: { knock: KnockState; onStop: () => void; onCall: () => void }) {
  return (
    <section aria-label="Finding Margarete" aria-live="polite" className="knock-card rise-in flex flex-col gap-3">
      <Header dot="amber" label="Finding Margarete" chip={knock.source === "tabpfn" ? "TabPFN" : "placeholder"} />
      <p className="text-base leading-snug">{knock.text}</p>
      <ZoneMap knock={knock} />
      <div className="grid grid-cols-2 gap-2">
        <button type="button" onClick={onStop} className="knock-btn knock-secondary h-12 gap-2"><Square className="h-4 w-4" />Stop search</button>
        <button type="button" onClick={onCall} className="knock-btn knock-secondary h-12 gap-2"><Volume2 className="h-4 w-4" />Call her again</button>
      </div>
    </section>
  );
}

function Found({ knock, onChoose, onDone }: {
  knock: KnockState; onChoose: (c: "call_112" | "call_neighbour" | "talk") => void; onDone: () => void;
}) {
  return (
    <>
      <section aria-label="Found Margarete" aria-live="assertive" className="knock-card knock-alert rise-in flex flex-col gap-3">
        <div className="pixel flex items-center gap-2 text-[10px] tracking-[0.2em] text-[oklch(0.85_0.09_22)]">
          <span className="h-2 w-2 rounded-full bg-term-red" />Found Margarete{knock.time ? ` · ${knock.time}` : ""}
          <button type="button" onClick={onDone} className="ml-auto font-sans text-[11px] normal-case tracking-normal text-video/70 underline-offset-2 hover:underline">Done</button>
        </div>
        <p className="text-[17px] font-medium leading-snug">{knock.text}</p>
        <p className="text-[13px] text-video/70">Last heard: {zoneLabel(knock.last_heard.zone).toLowerCase()}, {knock.last_heard.time}</p>
      </section>
      <button type="button" onClick={() => onChoose("call_112")} className="knock-btn knock-danger h-14 gap-2.5 text-base">
        <Phone className="h-5 w-5" />{knock.choice === "call_112" ? "Calling 112… (simulated)" : "Call 112"}
      </button>
      <div className="grid grid-cols-2 gap-2">
        <button type="button" onClick={() => onChoose("call_neighbour")} className="knock-btn knock-secondary h-[52px] gap-2">
          <Home className="h-4 w-4" />{knock.choice === "call_neighbour" ? "Calling…" : "Call Frau Weber"}
        </button>
        <button type="button" onClick={() => onChoose("talk")} className="knock-btn knock-secondary h-[52px] gap-2">
          <Mic className="h-4 w-4" />Talk to her
        </button>
      </div>
      <p className="text-xs leading-[1.45] text-video/60">
        Frau Weber is her neighbour and has a key. The robot stays beside her with its light on. In the demo, calls are simulated.
      </p>
    </>
  );
}

function NotHome({ knock, onFind, onDone }: { knock: KnockState; onFind: () => void; onDone: () => void }) {
  return (
    <section aria-label="Margarete not found" aria-live="polite" className="knock-card rise-in flex flex-col gap-3">
      <Header dot="amber" label="Search finished" />
      <p className="text-base leading-snug">{knock.text}</p>
      <ZoneMap knock={knock} />
      <div className="grid grid-cols-2 gap-2">
        <button type="button" onClick={onDone} className="knock-btn knock-secondary h-12">Done</button>
        <button type="button" onClick={onFind} className="knock-btn knock-primary h-12 gap-2"><Search className="h-4 w-4" />Search again</button>
      </div>
    </section>
  );
}

/** What the robot has done so far, under the card while it searches. */
export function KnockSteps({ knock }: { knock: KnockState }) {
  if (!knock.steps.length) return null;
  return (
    <ol className="flex flex-col gap-1.5 px-1 text-[13px]">
      {knock.steps.map((s, i) => (
        <li key={i} className="flex gap-2.5">
          <span className="text-term-green">✓</span><span className="flex-1">{s.text}</span>
          <span className="font-mono text-video/60">{s.time}</span>
        </li>
      ))}
    </ol>
  );
}

function Header({ dot, label, chip, right }: { dot: "amber" | "call"; label: string; chip?: string; right?: string }) {
  return (
    <div className="pixel flex items-center gap-2 text-[10px] tracking-[0.2em] text-video/70">
      {dot === "call" ? <span className="call-dot" /> : <span className="h-2 w-2 rounded-full bg-term-amber" />}
      {label}
      {chip && <span className="knock-chip ml-auto">{chip}</span>}
      {right && <span className="ml-auto font-mono text-xs normal-case tracking-normal text-video">{right}</span>}
    </div>
  );
}

/** The flat as a 3 x 2 grid (as in knock-routines.md): back row bed, bathroom, kitchen; front row couch, armchair, entrance. */
function ZoneMap({ knock }: { knock: KnockState }) {
  const cells = KNOCK_ZONES.map(({ id, label }) => {
    const p = knock.probs[id] ?? 0;
    const checked = knock.checked.includes(id);
    return { id, label, p, checked, next: knock.next === id };
  });
  const summary = cells.map((c) => (c.checked ? `${c.label} checked` : `${c.label} ${Math.round(c.p * 100)} percent`)).join(", ");
  return (
    <div role="img" aria-label={`Floor plan: ${summary}`} className="grid grid-cols-3 gap-1 rounded-xl bg-[oklch(1_0_0/6%)] p-1">
      {cells.map((c) => (
        <div key={c.id} className={`knock-cell ${c.checked ? "knock-cell-checked" : ""} ${c.next ? "knock-cell-next" : ""}`}
          style={c.checked ? undefined : { background: `oklch(0.75 0.17 55 / ${Math.min(70, 8 + c.p * 106).toFixed(0)}%)` }}>
          <span className={`text-[11px] ${c.next ? "font-semibold" : "text-video/80"}`}>{c.label}{c.next ? " · next" : ""}</span>
          {c.checked ? <span className="font-mono text-[11px] text-term-red">✗ checked</span>
            : <span className="font-mono text-xs">{Math.round(c.p * 100)}%</span>}
        </div>
      ))}
    </div>
  );
}

/** Seconds left to answer, counted down from the event's listen_s; null while the robot is still speaking. */
function useCountdown(knock: KnockState): number | null {
  const [left, setLeft] = useState<number | null>(null);
  useEffect(() => {
    if (knock.listen_s == null) { setLeft(null); return; }
    const end = Date.now() + knock.listen_s * 1000;
    const tick = () => setLeft(Math.max(0, Math.ceil((end - Date.now()) / 1000)));
    tick();
    const t = window.setInterval(tick, 250);
    return () => clearInterval(t);
  }, [knock]);
  return left;
}

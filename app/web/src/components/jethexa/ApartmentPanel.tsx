// The Apartment panel: the flat's zones and obstacles with the belief of where Margarete is, the search loop
// (proposal, walk, look, update), the journal viewers can add to, and the log. Drawn from GET /api/apartment
// (useApartment), so every viewer sees the same thing. The 3D view behind shows the same flat in MuJoCo.
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { BrainCircuit, Check, ChevronRight, ClipboardPaste, Clock, Copy, RotateCcw, Dices, Eye, EyeOff, Footprints, Plus, ScanEye, Search, Square, Trash2, X } from "lucide-react";
import {
  apartmentAddEntry, apartmentClear, apartmentClock, apartmentRandom, apartmentImportJournal, apartmentInputsJson, apartmentSetInputs, apartmentRemoveEntry, apartmentReset,
  apartmentSearch, apartmentStop,
  ZONE_CLASSES, ZONE_COLOR, ZONE_LABEL,
  type ApartmentState, type Belief, type Stage, type ZoneClass, type ZoneName,
} from "@/lib/api";
import { unlockAudio } from "@/lib/audio";
import type { Narration } from "@/hooks/useNarration";

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const PARTS = ["morning", "afternoon", "evening"];
const STAGES: { id: Stage | "live"; label: string; hint: string }[] = [
  { id: "population", label: "Population", hint: "Where people are at this time of day" },
  { id: "habits", label: "+ Habits", hint: "Plus her stated habits" },
  { id: "journal", label: "+ Journal", hint: "Plus her own journal" },
  { id: "live", label: "Now", hint: "Plus last heard, and every look of the search" },
];
const pct = (p: number) => `${Math.round(100 * p)}%`;
const ago = (m: number) => (m >= 60 ? `${Math.floor(m / 60)} h ${String(Math.round(m % 60)).padStart(2, "0")} min` : `${Math.round(m)} min`);

export function statusText(a: ApartmentState): { text: string; tone: "idle" | "busy" | "good" | "warn" } {
  const s = a.search;
  const goal = s.goal ? ZONE_LABEL[s.goal].toLowerCase() : "";
  switch (s.status) {
    case "starting": return { text: "Starting the search", tone: "busy" };
    case "walking": return { text: `Walking to the ${goal}`, tone: "busy" };
    case "looking": return { text: `Looking in the ${goal}`, tone: "busy" };
    case "found": return { text: `Found in the ${ZONE_LABEL[s.steps[s.steps.length - 1]!.zone].toLowerCase()}`, tone: "good" };
    case "nobody_home": return { text: "Nobody home", tone: "warn" };
    case "gave_up": return { text: "Not found", tone: "warn" };
    case "failed": return { text: "Search interrupted", tone: "warn" };
    case "stopped": return { text: "Stopped", tone: "idle" };
    default: return { text: "Ready", tone: "idle" };
  }
}

export function ApartmentPanel({ apartment: a, onClose, onError, onChanged }: {
  apartment: ApartmentState; onClose: () => void; onError: (m: string) => void; onChanged: () => void;
}) {
  const [stage, setStage] = useState<Stage | "live">("live");
  const [showHer, setShowHer] = useState(false);
  const [where, setWhere] = useState("routine");
  const status = statusText(a);
  const belief: Belief = stage === "live" ? a.belief : a.stages[stage];
  const run = async (p: Promise<{ ok: boolean; error?: string }>) => {
    const r = await p;
    if (!r.ok && r.error) onError(r.error);
    onChanged();
  };
  const part = a.now.slot === null ? "night" : PARTS[a.now.slot];
  const found = a.search.status === "found";

  return (
    <aside aria-label="Apartment" className="apt-panel rise-in flex h-full w-full flex-col">
      {/* Who and when */}
      <header className="px-5 pt-4 pb-3">
        <div className="flex items-center gap-3">
          <div aria-hidden className="knock-avatar">{a.name[0]}</div>
          <div className="flex min-w-0 flex-1 flex-col">
            <span className="truncate text-[17px] font-semibold leading-tight">Where is {a.name}?</span>
            <span className="truncate text-xs text-video/60">
              {a.last_heard ? `Last heard in the ${ZONE_LABEL[a.last_heard.zone].toLowerCase()}, ${ago(a.last_heard.minutes)} ago` : "Not heard yet"}
            </span>
          </div>
          <button onClick={onClose} aria-label="Close the apartment panel" className="apt-icon-btn"><X className="h-4 w-4" /></button>
        </div>
        <div role="status" className={`apt-strip apt-strip-${status.tone} mt-3`}>
          {status.tone === "busy" ? <span className="live-dot" /> : status.tone === "good" ? <Check className="h-4 w-4" /> : null}
          <span className="min-w-0 truncate font-semibold">{status.text}</span>
          {a.search.steps.length > 0 && (
            <span className="ml-auto shrink-0 whitespace-nowrap font-mono text-[11.5px] tabular-nums opacity-80">
              {a.search.steps.length} look{a.search.steps.length === 1 ? "" : "s"} · {a.search.metres.toFixed(1)} m
            </span>
          )}
          <ResetButton a={a} run={run} />
        </div>
        <Pipeline a={a} />
      </header>

      <div className="flex-1 space-y-4 overflow-y-auto px-5 pb-5 [scrollbar-width:thin]">
        {/* The check-in: when, where she really is (the mock needs to know), start and stop */}
        <section className="apt-card flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <Clock className="h-4 w-4 text-video/50" aria-hidden />
            <select aria-label="Weekday of the check-in" className="apt-input" value={a.now.weekday} disabled={a.search.running}
              onChange={(e) => void run(apartmentClock(Number(e.target.value), a.now.time))}>
              {DAYS.map((d, i) => <option key={d} value={i}>{d}</option>)}
            </select>
            <input aria-label="Time of the check-in" type="time" className="apt-input" value={a.now.time} disabled={a.search.running}
              onChange={(e) => e.target.value && void run(apartmentClock(a.now.weekday, e.target.value))} />
            <span className="text-xs text-video/55">{part}</span>
            {!a.now.real && (
              <button className="apt-link ml-auto" disabled={a.search.running} onClick={() => void run(apartmentClock(null, null))}>Use the time now</button>
            )}
          </div>
          <div className="flex flex-wrap items-end gap-2">
            {a.search.running ? (
              <button onClick={() => void run(apartmentStop())} className="knock-btn knock-secondary h-11 min-w-[170px] flex-1 gap-2 whitespace-nowrap">
                <Square className="h-4 w-4" />Stop the search
              </button>
            ) : (
              <button onClick={() => { unlockAudio(); void run(apartmentSearch(where)); }} className="knock-btn knock-primary h-11 min-w-[170px] flex-1 gap-2 whitespace-nowrap">
                <Search className="h-4 w-4" />Find {a.name}
              </button>
            )}
            <label className="flex min-w-[170px] flex-1 flex-col gap-0.5 text-[10px] uppercase tracking-wider text-video/45">
              She really is
              <select aria-label="Where she really is, for the simulation" className="apt-input w-full" value={where} disabled={a.search.running}
                onChange={(e) => setWhere(e.target.value)}>
                <option value="routine">where her routine says</option>
                <option value="random">somewhere random</option>
                {ZONE_CLASSES.map((z) => <option key={z} value={z}>{z === "out" ? "out of the flat" : `in the ${ZONE_LABEL[z].toLowerCase()}`}</option>)}
              </select>
            </label>
          </div>
        </section>

        {/* The flat */}
        <section className="apt-card">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="apt-title">Floor plan {a.model?.name === "tabpfn" && <span className="ml-1 font-normal normal-case tracking-normal text-primary">· search by TabPFN</span>}</h3>
            <div className="flex items-center gap-1">
              <button className="apt-link flex items-center gap-1" onClick={() => setShowHer((s) => !s)}
                title="Show where she really is (the simulation knows; the robot does not)">
                {showHer ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}{showHer ? "Hide her" : "Show her"}
              </button>
              <button className="apt-link flex items-center gap-1" disabled={a.search.running}
                onClick={() => void run(apartmentReset(a.seed + 1))} title="A new random flat and resident">
                <Dices className="h-3.5 w-3.5" />New flat
              </button>
            </div>
          </div>
          <Plan a={a} belief={belief} showHer={showHer || found} />
          <Legend />
          <Looks a={a} />
          <div className="mt-3 grid grid-cols-2 gap-2">
            <div className={`apt-role ${a.model?.name === "tabpfn" ? "apt-role-on" : ""}`}>
              <span className="apt-role-k">Search pattern</span>
              <span className="apt-role-v">{a.model?.name === "tabpfn" ? a.model.label : "Counting model"}</span>
              <span className="apt-role-d">{a.model?.name === "tabpfn" ? "picks the order of the zones: most likely first" : "orders the zones, most likely first"}</span>
            </div>
            <div className={`apt-role ${a.driver === "tabpfn" ? "apt-role-on" : ""}`}>
              <span className="apt-role-k">Robot control</span>
              <span className="apt-role-v">{a.driver === "tabpfn" ? "TabPFN a→b" : a.driver === "marker" ? "Marker" : a.driver}</span>
              <span className="apt-role-d">{a.driver === "tabpfn" ? "stride and turn every step, along a route around the furniture"
                : a.driver === "marker" ? "no robot in this scene" : "walks each leg of the route"}</span>
            </div>
          </div>
        </section>

        {/* The belief */}
        <section className="apt-card">
          <div className="mb-2.5 flex items-center justify-between gap-2">
            <h3 className="apt-title">Belief</h3>
            <div role="tablist" aria-label="Belief step" className="apt-tabs">
              {STAGES.map((s) => (
                <button key={s.id} role="tab" aria-selected={stage === s.id} onClick={() => setStage(s.id)}
                  title={s.id === "live" && a.model ? `${a.model.label}: ${a.model.detail}, then every look of the search` : s.hint}
                  className={`apt-tab ${stage === s.id ? "apt-tab-on" : ""}`}>{s.id === "live" && a.model?.name === "tabpfn" ? "TabPFN" : s.label}</button>
              ))}
            </div>
          </div>
          {a.model && (
            <p className="mb-2.5 flex items-center gap-1.5 text-[11.5px] text-video/60">
              <span className={`h-1.5 w-1.5 rounded-full ${a.model.name === "tabpfn" ? "bg-primary" : "bg-video/40"}`} />
              {stage === "live"
                ? <><span className="font-semibold text-video/85">{a.model.label}</span> · {a.model.detail}{a.search.steps.length > 0 ? " · updated after every look" : ""}</>
                : <>Counting step, for comparison: {STAGES.find((s) => s.id === stage)?.hint.toLowerCase()}</>}
            </p>
          )}
          <div className="flex flex-col gap-1.5">
            {ZONE_CLASSES.map((z) => (
              <div key={z} className="grid grid-cols-[92px_1fr_40px] items-center gap-2 text-[13px]">
                <span className={z === a.search.goal ? "font-semibold" : "text-video/80"}>{ZONE_LABEL[z]}</span>
                <div className="h-2.5 overflow-hidden rounded-full bg-video/8" title={`${ZONE_LABEL[z]}: ${pct(belief[z])}`}>
                  <div className="h-full rounded-full transition-[width] duration-500" style={{ width: pct(belief[z]), background: ZONE_COLOR[z] }} />
                </div>
                <span className="text-right font-mono text-xs tabular-nums text-video/80">{pct(belief[z])}</span>
              </div>
            ))}
          </div>
          <Habits a={a} />
          <RandomInputs a={a} run={run} />
        </section>

        <Journal a={a} run={run} onError={onError} onChanged={onChanged} />
        <Log a={a} />
      </div>
    </aside>
  );
}

/**
 * Who does what in the search, with the step that is running lit up: the model gives the belief from routines and
 * journal, the most likely zone is picked, the robot walks there, looks, and Bayes' rule updates the belief.
 */
function Pipeline({ a }: { a: ApartmentState }) {
  const tabpfn = a.model?.name === "tabpfn";
  const s = a.search.status;
  const active = s === "starting" ? 0 : s === "walking" ? 1 : s === "looking" ? 2 : -1;
  const steps = [
    { label: "Belief", by: tabpfn ? a.model!.label : "Counting", hint: `Where she probably is, from her routines and journal${tabpfn ? `: ${a.model!.label}, ${a.model!.detail}` : ""}` },
    { label: "Walk", by: a.driver === "tabpfn" ? "TabPFN a→b" : a.driver, hint: "The robot walks to the most likely zone" },
    { label: "Look", by: "mock", hint: "A look at the zone; it misses her 20% of the time" },
    { label: "Update", by: "Bayes", hint: "Nobody seen: that zone gets less likely, the others more; then the next most likely zone" },
  ];
  return (
    <div className="mt-2.5">
      <div className="mb-1.5 flex items-center gap-1.5 text-[11px] text-video/60">
        <BrainCircuit className={`h-3.5 w-3.5 ${tabpfn ? "text-primary" : ""}`} aria-hidden />
        <span>{tabpfn ? <><b className="font-semibold text-video/90">{a.model!.label}</b> orchestrates the Bayesian search</> : "Bayesian search (counting belief)"}</span>
      </div>
      <ol aria-label="The steps of the search" className="flex items-stretch gap-1">
        {steps.map((step, i) => (
          <li key={step.label} title={step.hint} className="flex min-w-0 flex-1 items-center gap-1">
            <div className={`apt-step ${i === active ? "apt-step-on" : ""} ${i === 0 && tabpfn ? "apt-step-model" : ""}`}>
              <span className="truncate text-[11.5px] font-semibold leading-tight">{step.label}</span>
              <span className="truncate text-[10px] leading-tight opacity-70">{step.by}</span>
            </div>
            {i < steps.length - 1 && <ChevronRight className="h-3 w-3 shrink-0 text-video/30" aria-hidden />}
          </li>
        ))}
      </ol>
    </div>
  );
}

/** After a search: forget it, take Margarete out of the room and put the robot back on its dock. */
function ResetButton({ a, run }: { a: ApartmentState; run: (p: Promise<{ ok: boolean; error?: string }>) => Promise<void> }) {
  if (a.search.status === "idle" || a.search.running) return null;
  return (
    <button type="button" onClick={() => void run(apartmentClear())} title={`Take ${a.name} out of the room and put the robot back on its dock`}
      className={`flex shrink-0 items-center gap-1 rounded-full border border-current/30 px-2 py-0.5 text-[11.5px] font-medium hover:bg-video/10 ${a.search.steps.length > 0 ? "" : "ml-auto"}`}>
      <RotateCcw className="h-3 w-3" />Reset
    </button>
  );
}

/** The flat from above: zones shaded by the belief, obstacles, door, dock, the robot, its path and the looks. */
export function Plan({ a, belief, showHer }: { a: ApartmentState; belief: Belief; showHer: boolean }) {
  const { half, zones, obstacles, door, dock } = a.flat;
  const pad = 0.12;
  const looks = useMemo(() => {
    const at = new Map<ZoneName, { n: number[]; seen: boolean }>();
    a.search.steps.forEach((s, i) => {
      const e = at.get(s.zone) ?? { n: [], seen: false };
      e.n.push(i + 1); e.seen = e.seen || s.seen;
      at.set(s.zone, e);
    });
    return at;
  }, [a.search.steps]);
  const path = [...a.search.path, [a.robot.x, a.robot.y] as [number, number]];
  const truth = a.search.truth && a.search.truth !== "out" ? zones.find((z) => z.name === a.search.truth) : undefined;
  const doorFlat = door.wall === "ymin" || door.wall === "ymax";

  return (
    <svg viewBox={`${-half - pad} ${-half - pad} ${2 * (half + pad)} ${2 * (half + pad)}`} role="img"
      aria-label="Floor plan of the flat with the belief per zone" className="aspect-square w-full select-none">
      <rect x={-half} y={-half} width={2 * half} height={2 * half} rx={0.06} fill="oklch(0.25 0.012 255)" />
      {zones.map((z) => {
        const p = belief[z.name];
        const x0 = z.x - z.size_x / 2, y0 = -z.y - z.size_y / 2;
        const narrow = z.size_x < 1.9;
        return (
          <g key={z.name}>
            <rect x={x0 + 0.03} y={y0 + 0.03} width={z.size_x - 0.06} height={z.size_y - 0.06} rx={0.08}
              fill={ZONE_COLOR[z.name]} fillOpacity={0.16 + 0.6 * p}
              stroke={z.name === a.search.goal ? "white" : ZONE_COLOR[z.name]} strokeOpacity={z.name === a.search.goal ? 0.95 : 0.55}
              strokeWidth={z.name === a.search.goal ? 0.05 : 0.02} style={{ transition: "fill-opacity .5s" }}>
              <title>{`${z.label}: ${pct(p)}`}</title>
            </rect>
            {(() => {
              // The label goes in the corner of the zone with the most room: away from its furniture and viewing pose.
              const busy = [...obstacles.filter((o) => o.zone === z.name).map((o) => [o.x, o.y]), [z.view_x, z.view_y], [z.person_x, z.person_y]];
              if (z.name === "hallway") busy.push([dock.x, dock.y]);
              let best = { sx: -1, sy: 1, d: -1 };
              for (const sy of [1, -1]) for (const sx of [-1, 1]) {
                const cx = z.x + sx * z.size_x / 2, cy = z.y + sy * z.size_y / 2;
                const d = Math.min(...busy.map(([bx, by]) => Math.hypot(cx - bx!, cy - by!)));
                if (d > best.d + 1e-6) best = { sx, sy, d };
              }
              const lx = z.x + best.sx * (z.size_x / 2 - 0.16);
              const up = best.sy > 0; // the top of the zone on screen
              const nameY = up ? y0 + 0.34 : y0 + z.size_y - 0.5;
              return (
                <g textAnchor={best.sx < 0 ? "start" : "end"} fill="white">
                  <text x={lx} y={nameY} fontSize={0.2} fontWeight={600} fillOpacity={0.92}>{z.label}</text>
                  <text x={lx} y={nameY + 0.33} fontSize={narrow ? 0.26 : 0.3} fontWeight={700}
                    style={{ fontVariantNumeric: "tabular-nums" }}>{pct(p)}</text>
                </g>
              );
            })()}
          </g>
        );
      })}
      {obstacles.map((o) => (
        <g key={o.name}>
          <rect x={o.x - o.size_x / 2} y={-o.y - o.size_y / 2} width={o.size_x} height={o.size_y} rx={0.06}
            fill="oklch(0.17 0.012 255)" fillOpacity={0.55} stroke="white" strokeOpacity={0.5} strokeWidth={0.025}>
            <title>{`Obstacle: ${o.name}`}</title>
          </rect>
          {Math.min(o.size_x, o.size_y) > 0.55 && (
            <text x={o.x} y={-o.y + 0.05} textAnchor="middle" fontSize={0.15} fill="white" fillOpacity={0.6}
              transform={o.size_y > o.size_x * 1.3 ? `rotate(-90 ${o.x} ${-o.y})` : undefined}>{o.name}</text>
          )}
        </g>
      ))}
      <rect x={-half} y={-half} width={2 * half} height={2 * half} rx={0.06} fill="none" stroke="white" strokeOpacity={0.55} strokeWidth={0.05} />
      {/* the door */}
      <line x1={doorFlat ? door.x - door.width / 2 : door.x} y1={doorFlat ? -door.y : -door.y - door.width / 2}
        x2={doorFlat ? door.x + door.width / 2 : door.x} y2={doorFlat ? -door.y : -door.y + door.width / 2}
        stroke="oklch(0.82 0.15 80)" strokeWidth={0.09} strokeLinecap="round"><title>Door</title></line>
      {/* the walk so far */}
      {path.length > 1 && (
        <polyline points={path.map(([x, y]) => `${x},${-y}`).join(" ")} fill="none" stroke="white" strokeOpacity={0.75}
          strokeWidth={0.045} strokeLinejoin="round" strokeLinecap="round" strokeDasharray="0.02 0.11" />
      )}
      {/* the dock */}
      <rect x={dock.x - 0.13} y={-dock.y - 0.13} width={0.26} height={0.26} rx={0.05} fill="oklch(0.17 0.012 255)" stroke="white" strokeOpacity={0.8} strokeWidth={0.03}>
        <title>The robot's dock</title>
      </rect>
      {/* the looks, numbered */}
      {zones.map((z) => {
        const l = looks.get(z.name);
        if (!l) return null;
        const label = l.n.join("·");
        const w = 0.34 + 0.14 * Math.max(0, label.length - 1);
        return (
          <g key={`look-${z.name}`} transform={`translate(${z.view_x} ${-z.view_y})`}>
            <rect x={-w / 2} y={-0.19} width={w} height={0.38} rx={0.19} fill={l.seen ? "oklch(0.79 0.18 150)" : "oklch(0.17 0.012 255)"}
              stroke="white" strokeWidth={0.035} />
            <text textAnchor="middle" y={0.075} fontSize={0.21} fontWeight={700} fill={l.seen ? "oklch(0.2 0.03 150)" : "white"}>{label}</text>
            <title>{`Look ${label.replace(/·/g, ", ")}: ${l.seen ? "found her" : "nobody seen"}`}</title>
          </g>
        );
      })}
      {/* Margarete, when shown */}
      {showHer && truth && (
        <g transform={`translate(${a.search.person?.[0] ?? truth.person_x} ${-(a.search.person?.[1] ?? truth.person_y)})`}>
          <circle r={0.2} fill="oklch(0.68 0.19 25)" stroke="white" strokeWidth={0.035} />
          <text textAnchor="middle" y={0.075} fontSize={0.2} fontWeight={700} fill="white">{a.name[0]}</text>
          <title>{`${a.name} is here`}</title>
        </g>
      )}
      {/* the robot */}
      <g transform={`translate(${a.robot.x} ${-a.robot.y}) rotate(${-(a.robot.yaw_deg ?? 0)})`} style={{ transition: "transform .4s linear" }}>
        <circle r={0.27} fill="oklch(0.78 0.13 172)" fillOpacity={0.28} />
        <path d="M0.24 0 L-0.14 0.15 L-0.06 0 L-0.14 -0.15 Z" fill="oklch(0.78 0.13 172)" stroke="oklch(0.17 0.012 255)" strokeWidth={0.02} />
        <title>The robot</title>
      </g>
    </svg>
  );
}

function Legend() {
  const item = "flex items-center gap-1.5";
  return (
    <div className="mt-2 flex flex-wrap gap-x-3.5 gap-y-1 text-[11px] text-video/55">
      <span className={item}><svg viewBox="-1 -1 2 2" className="h-3 w-3"><path d="M0.8 0 L-0.5 0.55 L-0.2 0 L-0.5 -0.55 Z" fill="oklch(0.78 0.13 172)" /></svg>robot</span>
      <span className={item}><span className="h-2.5 w-2.5 rounded-[3px] border border-video/80 bg-[oklch(0.17_0.012_255)]" />dock</span>
      <span className={item}><span className="h-2.5 w-3.5 rounded-[3px] border border-video/50 bg-[oklch(0.17_0.012_255)]/60" />obstacle</span>
      <span className={item}><span className="flex h-3.5 w-3.5 items-center justify-center rounded-full border border-video text-[8px] font-bold">1</span>look</span>
      <span className={item}><span className="h-[3px] w-3.5 rounded-full bg-term-amber" />door</span>
      <span className="ml-auto">darker zone = more likely</span>
    </div>
  );
}

/** The search so far, one chip per look: where the robot looked, how likely that zone was, what it saw. */
export function Looks({ a }: { a: ApartmentState }) {
  const s = a.search;
  if (s.steps.length === 0 && !s.running) return null;
  return (
    <ol aria-label="The looks of this search" className="mt-3 flex flex-wrap gap-1.5">
      {s.steps.map((step, i) => (
        <li key={i} className={`apt-look ${step.seen ? "apt-look-found" : ""}`}>
          <span className="apt-look-n">{i + 1}</span>
          <span className="h-2 w-2 rounded-full" style={{ background: ZONE_COLOR[step.zone] }} aria-hidden />
          {ZONE_LABEL[step.zone]}
          <span className="font-mono text-[11px] tabular-nums opacity-70">{pct(step.p)}</span>
          {step.seen ? <Check className="h-3.5 w-3.5" aria-label="found" /> : <X className="h-3.5 w-3.5 opacity-60" aria-label="nobody seen" />}
        </li>
      ))}
      {s.running && s.goal && (
        <li className="apt-look apt-look-next">
          <span className="apt-look-n">{s.steps.length + 1}</span>
          {s.status === "looking" ? <ScanEye className="h-3.5 w-3.5" /> : <Footprints className="h-3.5 w-3.5" />}
          {s.status === "looking" ? "Looking in the" : "Walking to the"} {ZONE_LABEL[s.goal].toLowerCase()}
        </li>
      )}
    </ol>
  );
}

/** New random examples, to see how the belief changes: routines, a journal drawn from her routine, or both. */
function RandomInputs({ a, run }: { a: ApartmentState; run: (p: Promise<{ ok: boolean; error?: string }>) => Promise<void> }) {
  const off = a.search.running;
  const items: { what: "routines" | "journal" | "both"; label: string; hint: string }[] = [
    { what: "routines", label: "Routines", hint: "New stated habits and outings (a new kind of resident); the journal stays" },
    { what: "journal", label: "Journal", hint: "Two new weeks of journal, drawn from her routine" },
    { what: "both", label: "Both", hint: "A new resident: new routines and a journal drawn from them" },
  ];
  return (
    <div className="mt-3 flex items-center gap-2 rounded-xl border border-dashed border-video/15 px-2.5 py-2">
      <Dices className="h-4 w-4 shrink-0 text-video/55" aria-hidden />
      <span className="mr-auto text-[11.5px] text-video/60">Random example</span>
      {items.map((i) => (
        <button key={i.what} type="button" disabled={off} title={i.hint} onClick={() => void run(apartmentRandom(i.what))}
          className="apt-chip">{i.label}</button>
      ))}
    </div>
  );
}

function Habits({ a }: { a: ApartmentState }) {
  const outings = a.habits.outings.map(([d, s]) => `${DAYS[d]} ${PARTS[s]}`).join(", ");
  return (
    <div className="mt-3 border-t border-video/10 pt-2.5">
      <div className="mb-1.5 text-[10px] uppercase tracking-wider text-video/45">Stated habits</div>
      <div className="grid grid-cols-[62px_repeat(3,1fr)] gap-1 text-[11px]">
        <span />
        {PARTS.map((p) => <span key={p} className="text-center capitalize text-video/45">{p}</span>)}
        {(["0", "1"] as const).map((wk) => (
          <div key={wk} className="contents">
            <span className="self-center text-video/60">{wk === "0" ? "Weekdays" : "Weekends"}</span>
            {a.habits.usual[wk].map((z, i) => {
              const now = a.now.slot === i && (a.now.weekday >= 5) === (wk === "1");
              return (
                <span key={i} className={`rounded-md px-1 py-1 text-center ${now ? "font-semibold ring-1 ring-video/70" : ""}`}
                  style={{ background: `color-mix(in oklab, ${ZONE_COLOR[z]} 28%, transparent)` }}>{ZONE_LABEL[z]}</span>
              );
            })}
          </div>
        ))}
      </div>
      <div className="mt-1.5 text-[11px] text-video/50">Regular outings: {outings || "none"}</div>
    </div>
  );
}

const PASTE_EXAMPLE = `{
 "routines": {
  "weekdays": {"morning": "kitchen", "afternoon": "living room", "evening": "living room"},
  "weekends": {"morning": "bedroom", "afternoon": "living room", "evening": "living room"},
  "outings": [{"day": "Tue", "part": "afternoon"}]
 },
 "journal": [
  {"day": "Mon", "time": "08:10", "zone": "kitchen"},
  {"day": "Mon", "time": "14:30", "zone": "out"}
 ]
}`;

/** Custom inputs: paste routines and a journal (JSON, or journal lines) to change what the belief is built from;
 *  "Copy current" gives the live ones to edit. */
function PasteJournal({ a, onError, onChanged }: { a: ApartmentState; onError: (m: string) => void; onChanged: () => void }) {
  const [text, setText] = useState("");
  const [note, setNote] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const send = async (replace: boolean) => {
    if (!text.trim() || busy) return;
    setBusy(true);
    const json = text.trim().startsWith("{"); // routines and journal as one object; else journal lines
    const r: { ok: boolean; added?: number; routines?: boolean; problems?: string[]; error?: string } =
      json ? await apartmentSetInputs(text) : await apartmentImportJournal(text, replace);
    setBusy(false);
    if (!r.ok) { onError(r.error ?? "Could not read the inputs"); return; }
    const skipped = r.problems?.length ?? 0;
    const parts = [
      r.routines ? "routines replaced" : "",
      r.added ? `${r.added} journal record${r.added === 1 ? "" : "s"} ${replace || json ? "are now the journal" : "added"}` : "",
    ].filter(Boolean);
    setNote(`${parts.join(" · ")}. The belief is updated.` +
      (skipped ? ` ${skipped} line${skipped === 1 ? "" : "s"} skipped: ${r.problems![0]}` : ""));
    setText("");
    onChanged();
  };
  const copy = async () => {
    const all = await apartmentInputsJson();
    if (all === null) { onError("Could not get the inputs"); return; }
    setText(all);
    try { await navigator.clipboard.writeText(all); setNote("The current routines and journal are in the box and on the clipboard"); }
    catch { setNote("The current routines and journal are in the box (the browser did not allow the clipboard)"); }
  };
  return (
    <div className="mb-3 rounded-xl border border-video/10 bg-video/[0.03] p-2.5">
      <textarea aria-label="Custom inputs: routines and journal as JSON, or journal lines" value={text} rows={9} spellCheck={false}
        onChange={(e) => { setText(e.target.value); setNote(null); }} placeholder={PASTE_EXAMPLE}
        className="apt-input h-auto w-full resize-y py-1.5 font-mono text-[11.5px] leading-[1.5] placeholder:text-video/30" />
      <p className="mt-1.5 text-[11px] leading-snug text-video/50">
        JSON with <b>routines</b> (her stated habits and outings) and <b>journal</b> (where she was heard, oldest first);
        either can be left out. Plain lines work for a journal too: <span className="font-mono">Mon 08:10 kitchen</span>.
        Days: Mon, a date, today, yesterday (weekdays count back from the check-in time). Zones: bedroom, living room,
        dining area, kitchen, hallway, out.
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <button type="button" className="apt-add" disabled={busy || a.search.running || !text.trim()} onClick={() => void send(true)}
          title="The pasted routines and journal replace hers: the belief is built from them">Apply</button>
        <button type="button" className="apt-ghost" disabled={busy || a.search.running || !text.trim() || text.trim().startsWith("{")}
          onClick={() => void send(false)} title="For journal lines: add them to her journal instead of replacing it">Add lines</button>
        <button type="button" className="apt-ghost ml-auto" onClick={() => void copy()} title="Put the current journal in the box and on the clipboard">
          <Copy className="h-3.5 w-3.5" />Copy current
        </button>
      </div>
      {note && <p role="status" className="mt-2 text-[11.5px] text-term-green">{note}</p>}
    </div>
  );
}

function Journal({ a, run, onError, onChanged }: {
  a: ApartmentState; run: (p: Promise<{ ok: boolean; error?: string }>) => Promise<void>;
  onError: (m: string) => void; onChanged: () => void;
}) {
  const [paste, setPaste] = useState(false);
  const [zone, setZone] = useState<ZoneClass>("living_room");
  const [daysAgo, setDaysAgo] = useState(0);
  const [time, setTime] = useState("");
  const dayLabel = (n: number) => (n === 0 ? "Today" : n === 1 ? "Yesterday" : `${n} days ago`);
  const add = () => void run(apartmentAddEntry(zone, time, daysAgo)).then(() => setTime(""));
  return (
    <section className="apt-card">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="apt-title">Journal <span className="ml-1 font-normal normal-case tracking-normal text-video/45">{a.journal_total} entries</span></h3>
        <button type="button" className="apt-link flex items-center gap-1" aria-expanded={paste} onClick={() => setPaste((o) => !o)}>
          <ClipboardPaste className="h-3.5 w-3.5" />{paste ? "Hide" : "Custom inputs"}
        </button>
      </div>
      {paste && <PasteJournal a={a} onError={onError} onChanged={onChanged} />}
      <form className="mb-2.5 flex flex-wrap items-center gap-1.5" onSubmit={(e) => { e.preventDefault(); add(); }}>
        <select aria-label="Zone" className="apt-input min-w-0 flex-1" value={zone} onChange={(e) => setZone(e.target.value as ZoneClass)}>
          {ZONE_CLASSES.map((z) => <option key={z} value={z}>{ZONE_LABEL[z]}</option>)}
        </select>
        <select aria-label="Day" className="apt-input" value={daysAgo} onChange={(e) => setDaysAgo(Number(e.target.value))}>
          {[0, 1, 2, 3, 4, 5, 6].map((n) => <option key={n} value={n}>{dayLabel(n)}</option>)}
        </select>
        <input aria-label="Time (empty: the check-in time)" type="time" className="apt-input" value={time} onChange={(e) => setTime(e.target.value)} />
        <button type="submit" className="apt-add" aria-label="Add the journal entry"><Plus className="h-4 w-4" />Add</button>
      </form>
      <ul className="max-h-[168px] space-y-px overflow-y-auto pr-1 [scrollbar-width:thin]">
        {a.journal.map((e) => (
          <li key={e.id} className="group flex items-center gap-2 rounded-md px-1.5 py-1 text-[12.5px] hover:bg-video/6">
            <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: ZONE_COLOR[e.zone] }} aria-hidden />
            <span className="w-[118px] shrink-0 whitespace-nowrap font-mono text-[11.5px] tabular-nums text-video/60">{DAYS[e.weekday]} {e.day.slice(8)}.{e.day.slice(5, 7)} {e.time}</span>
            <span className="flex-1 truncate">{ZONE_LABEL[e.zone]}</span>
            <span className="text-[10.5px] text-video/40">{e.source === "robot" ? "heard" : e.source === "search" ? "seen by robot" : e.source === "pasted" ? "pasted" : "entered"}</span>
            <button onClick={() => void run(apartmentRemoveEntry(e.id))} aria-label={`Remove the entry of ${e.time}`}
              className="text-video/35 opacity-0 transition-opacity hover:text-term-red group-hover:opacity-100 focus:opacity-100">
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </li>
        ))}
        {a.journal.length === 0 && <li className="px-1.5 py-2 text-[12.5px] text-video/45">No entries before this time.</li>}
      </ul>
    </section>
  );
}

const LOG_TONE: Record<string, string> = {
  start: "text-video", proposal: "text-video", belief: "text-video/60", update: "text-video/60", miss: "text-term-amber",
  seen: "text-term-green", found: "text-term-green font-semibold", out: "text-term-amber font-semibold",
  journal: "text-video/55", info: "text-video/45", metacon: "text-video/85",
};

function Log({ a }: { a: ApartmentState }) {
  const box = useRef<HTMLDivElement>(null);
  const last = a.log[a.log.length - 1]?.id;
  useEffect(() => { box.current?.scrollTo({ top: box.current.scrollHeight, behavior: "smooth" }); }, [last]);
  return (
    <section className="apt-card">
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="apt-title">Log</h3>
        <span className="text-[11px] text-video/45">
          {a.search.steps.length > 0 ? `${a.search.steps.length} look${a.search.steps.length === 1 ? "" : "s"} · ${a.search.metres.toFixed(1)} m` : "proposal → look → update"}
        </span>
      </div>
      <div ref={box} role="log" aria-live="polite" className="max-h-[190px] overflow-y-auto pr-1 font-mono text-[11.5px] leading-[1.55] [scrollbar-width:thin]">
        {a.log.map((l) => (
          <div key={l.id} className="flex gap-2">
            <span className="shrink-0 tabular-nums text-video/35">{l.time}</span>
            <span className={LOG_TONE[l.kind] ?? "text-video/70"}>
              {l.kind === "metacon" && <span className="apt-tag-metacon">metacon</span>}{l.text}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

/**
 * The apartment on a phone: who and where, the status, the floor plan with the belief, the looks so far, and
 * Find Margarete. "Details" opens the whole panel (time, belief steps, journal, log) as a full-screen sheet.
 */
export function ApartmentPhoneCard({ apartment: a, onError, onChanged, narration }: {
  apartment: ApartmentState; onError: (m: string) => void; onChanged: () => void;
  /** metacon's latest line about the search, while it is recent. */
  narration?: Narration | null;
}) {
  const [sheet, setSheet] = useState(false);
  const status = statusText(a);
  const found = a.search.status === "found";
  const run = async (p: Promise<{ ok: boolean; error?: string }>) => {
    const r = await p;
    if (!r.ok && r.error) onError(r.error);
    onChanged();
  };
  return (
    <section aria-label={`Where is ${a.name}?`} className="knock-card rise-in flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <div aria-hidden className="knock-avatar">{a.name[0]}</div>
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="text-lg font-semibold leading-tight">Where is {a.name}?</span>
          <span className="truncate font-mono text-xs text-video/70">
            {a.last_heard ? `Last heard: ${ZONE_LABEL[a.last_heard.zone].toLowerCase()} · ${ago(a.last_heard.minutes)} ago` : "Not heard yet"}
          </span>
        </div>
        <span className="shrink-0 font-mono text-[11px] text-video/55">{DAYS[a.now.weekday]} {a.now.time}</span>
      </div>
      <div role="status" className={`apt-strip apt-strip-${status.tone}`}>
        {status.tone === "busy" ? <span className="live-dot" /> : status.tone === "good" ? <Check className="h-4 w-4" /> : null}
        <span className="min-w-0 truncate font-semibold">{status.text}</span>
        {a.search.steps.length > 0 && (
          <span className="ml-auto shrink-0 whitespace-nowrap font-mono text-[11.5px] tabular-nums opacity-80">
            {a.search.steps.length} look{a.search.steps.length === 1 ? "" : "s"} · {a.search.metres.toFixed(1)} m
          </span>
        )}
        <ResetButton a={a} run={run} />
      </div>
      {narration && (
        <div key={narration.id} role="status" aria-live="polite" className="apt-narration rise-in">
          <span className="pixel text-[9px] tracking-[0.2em] text-term-red">metacon</span>
          <span className="text-[14.5px] leading-snug">{narration.text}</span>
        </div>
      )}
      <Pipeline a={a} />
      <Plan a={a} belief={a.belief} showHer={found} />
      <Looks a={a} />
      <div className={`grid gap-2 ${a.search.running ? "grid-cols-[1fr_auto]" : "grid-cols-[1fr_auto_auto]"}`}>
        {a.search.running ? (
          <button type="button" onClick={() => void run(apartmentStop())} className="knock-btn knock-secondary h-12 gap-2">
            <Square className="h-4 w-4" />Stop the search
          </button>
        ) : (
          <>
            <button type="button" onClick={() => { unlockAudio(); void run(apartmentSearch("routine")); }} className="knock-btn knock-primary h-12 gap-2">
              <Search className="h-[18px] w-[18px]" />Find {a.name}
            </button>
            <button type="button" onClick={() => { unlockAudio(); void run(apartmentSearch("random")); }} className="knock-btn knock-secondary h-12 w-12"
              aria-label={`Put ${a.name} somewhere random and find her`} title="Put her somewhere random, then search">
              <Dices className="h-5 w-5" />
            </button>
          </>
        )}
        <button type="button" onClick={() => setSheet(true)} className="knock-btn knock-secondary h-12 px-4">Details</button>
      </div>
      {/* In the page body, not in the card: the card's entrance animation would trap a fixed element inside it. */}
      {sheet && createPortal(
        <div role="dialog" aria-modal="true" aria-label="Apartment details"
          className="fixed inset-0 z-50 bg-[var(--stage)] p-2 pt-[max(8px,env(safe-area-inset-top))] pb-[max(8px,env(safe-area-inset-bottom))]">
          <ApartmentPanel apartment={a} onClose={() => setSheet(false)} onError={onError} onChanged={onChanged} />
        </div>, document.body)}
    </section>
  );
}

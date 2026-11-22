// The Apartment panel: the flat's zones and obstacles with the belief of where Margarete is, the search loop
// (proposal, walk, look, update), the journal viewers can add to, and the log. Drawn from GET /api/apartment
// (useApartment), so every viewer sees the same thing. The 3D view behind shows the same flat in MuJoCo.
import { useEffect, useMemo, useRef, useState } from "react";
import { Check, Clock, Dices, Eye, EyeOff, Footprints, Plus, ScanEye, Search, Square, Trash2, X } from "lucide-react";
import {
  apartmentAddEntry, apartmentClock, apartmentRemoveEntry, apartmentReset, apartmentSearch, apartmentStop,
  ZONE_CLASSES, ZONE_COLOR, ZONE_LABEL,
  type ApartmentState, type Belief, type Stage, type ZoneClass, type ZoneName,
} from "@/lib/api";

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

function statusText(a: ApartmentState): { text: string; tone: "idle" | "busy" | "good" | "warn" } {
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
          <span className="font-semibold">{status.text}</span>
          {a.search.steps.length > 0 && (
            <span className="ml-auto font-mono text-[11.5px] tabular-nums opacity-80">
              {a.search.steps.length} look{a.search.steps.length === 1 ? "" : "s"} · {a.search.metres.toFixed(1)} m
            </span>
          )}
        </div>
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
          <div className="flex items-center gap-2">
            {a.search.running ? (
              <button onClick={() => void run(apartmentStop())} className="knock-btn knock-secondary h-11 flex-1 gap-2">
                <Square className="h-4 w-4" />Stop the search
              </button>
            ) : (
              <button onClick={() => void run(apartmentSearch(where))} className="knock-btn knock-primary h-11 flex-1 gap-2">
                <Search className="h-4 w-4" />Find {a.name}
              </button>
            )}
            <label className="flex flex-col gap-0.5 text-[10px] uppercase tracking-wider text-video/45">
              She really is
              <select aria-label="Where she really is, for the simulation" className="apt-input" value={where} disabled={a.search.running}
                onChange={(e) => setWhere(e.target.value)}>
                <option value="routine">where her routine says</option>
                {ZONE_CLASSES.map((z) => <option key={z} value={z}>{z === "out" ? "out of the flat" : `in the ${ZONE_LABEL[z].toLowerCase()}`}</option>)}
              </select>
            </label>
          </div>
        </section>

        {/* The flat */}
        <section className="apt-card">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="apt-title">Floor plan</h3>
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
          <p className="mt-2.5 text-[11px] leading-snug text-video/45">
            {a.driver === "tabpfn" ? "The TabPFN a-to-b controller walks each leg of the route."
              : a.driver === "marker" ? "No robot in this scene: a marker walks in straight lines." : `The ${a.driver} walks each leg of the route.`}
          </p>
        </section>

        {/* The belief */}
        <section className="apt-card">
          <div className="mb-2.5 flex items-center justify-between gap-2">
            <h3 className="apt-title">Belief</h3>
            <div role="tablist" aria-label="Belief step" className="apt-tabs">
              {STAGES.map((s) => (
                <button key={s.id} role="tab" aria-selected={stage === s.id} title={s.hint} onClick={() => setStage(s.id)}
                  className={`apt-tab ${stage === s.id ? "apt-tab-on" : ""}`}>{s.label}</button>
              ))}
            </div>
          </div>
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
        </section>

        <Journal a={a} run={run} />
        <Log a={a} />
      </div>
    </aside>
  );
}

/** The flat from above: zones shaded by the belief, obstacles, door, dock, the robot, its path and the looks. */
function Plan({ a, belief, showHer }: { a: ApartmentState; belief: Belief; showHer: boolean }) {
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
        <g transform={`translate(${truth.person_x} ${-truth.person_y})`}>
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
function Looks({ a }: { a: ApartmentState }) {
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

function Journal({ a, run }: { a: ApartmentState; run: (p: Promise<{ ok: boolean; error?: string }>) => Promise<void> }) {
  const [zone, setZone] = useState<ZoneClass>("living_room");
  const [daysAgo, setDaysAgo] = useState(0);
  const [time, setTime] = useState("");
  const dayLabel = (n: number) => (n === 0 ? "Today" : n === 1 ? "Yesterday" : `${n} days ago`);
  const add = () => void run(apartmentAddEntry(zone, time, daysAgo)).then(() => setTime(""));
  return (
    <section className="apt-card">
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="apt-title">Journal</h3>
        <span className="text-[11px] text-video/45">{a.journal_total} entries</span>
      </div>
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
            <span className="text-[10.5px] text-video/40">{e.source === "robot" ? "heard" : e.source === "search" ? "seen by robot" : "entered"}</span>
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
  journal: "text-video/55", info: "text-video/45",
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
            <span className={LOG_TONE[l.kind] ?? "text-video/70"}>{l.text}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

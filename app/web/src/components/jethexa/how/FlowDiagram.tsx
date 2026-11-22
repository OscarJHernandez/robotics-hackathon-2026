import { useId } from "react";

// The flow of "Find Margarete", drawn twice: a wide two-row loop (900 px and up) and a tall stack (phones).
// To add a step, add a node to NODES with its place in both layouts and an edge to EDGES; an edge can be moved off
// the boxes' centre lines (dy in the wide layout, dx in the tall one) so it does not run along another line.

type Kind = "input" | "step" | "model" | "progress";
interface Node {
  id: string; title: string; sub: string; kind: Kind;
  wide: { x: number; y: number; w?: number };
  /** Position in the phone stack; the in-progress branch comes after the loop. */
  tall: number;
}

const NODES: Node[] = [
  { id: "inputs", title: "Journal + habits", sub: "+ the floor plan", kind: "input", wide: { x: 0, y: 30 }, tall: 0 },
  { id: "belief", title: "TabPFN belief", sub: "P(each zone), P(out)", kind: "model", wide: { x: 172, y: 30 }, tall: 1 },
  { id: "zone", title: "Most likely zone", sub: "highest probability", kind: "step", wide: { x: 344, y: 30 }, tall: 2 },
  { id: "spot", title: "Viewing spot", sub: "from the floor plan", kind: "step", wide: { x: 516, y: 30 }, tall: 3 },
  { id: "route", title: "A* route", sub: "around known furniture", kind: "step", wide: { x: 688, y: 30 }, tall: 4 },
  { id: "walk", title: "TabPFN a → b", sub: "walks each waypoint", kind: "model", wide: { x: 688, y: 176 }, tall: 5 },
  { id: "look", title: "Look", sub: "simulated sighting", kind: "step", wide: { x: 516, y: 176 }, tall: 6 },
  { id: "update", title: "Bayes update", sub: "lower the zone looked at", kind: "step", wide: { x: 344, y: 176 }, tall: 7 },
  { id: "narrate", title: "metacon narrates", sub: "text; voice on the phone", kind: "step", wide: { x: 0, y: 176, w: 196 }, tall: 8 },
  { id: "sensors", title: "Lidar + camera", sub: "its own sensors", kind: "progress", wide: { x: 344, y: 336 }, tall: 9 },
  { id: "student", title: "TabPFN student", sub: "live · drives the walk", kind: "progress", wide: { x: 560, y: 336, w: 280 }, tall: 10 },
];
interface Shift { dy?: number; dx?: number }
const EDGES: [string, string, Shift?][] = [
  ["inputs", "belief"], ["belief", "zone"], ["zone", "spot"], ["spot", "route"], ["route", "walk"],
  ["walk", "look"], ["look", "update"],
  // below the loop's line back to the belief (wide); beside the "not seen" note (tall)
  ["update", "narrate", { dy: 20, dx: -112 }],
  ["sensors", "student"],
];
/** The steps the student would replace, drawn inside a dashed bracket. */
const REPLACED = ["route", "walk"];

const W = 860, NW = 152, NH = 66;
const T_W = 340, T_NW = 268, T_NH = 58, T_GAP = 26, T_X = 22;

const STYLE: Record<Kind, { fill: string; stroke: string; title: string; dash?: string }> = {
  input: { fill: "oklch(1 0 0 / 4%)", stroke: "oklch(1 0 0 / 22%)", title: "oklch(1 0 0 / 92%)" },
  step: { fill: "oklch(1 0 0 / 6%)", stroke: "oklch(1 0 0 / 16%)", title: "oklch(1 0 0 / 92%)" },
  model: { fill: "oklch(0.78 0.13 172 / 16%)", stroke: "oklch(0.78 0.13 172 / 75%)", title: "oklch(0.93 0.06 172)" },
  progress: { fill: "oklch(0.82 0.15 80 / 9%)", stroke: "oklch(0.82 0.15 80 / 70%)", title: "oklch(0.92 0.09 80)", dash: "5 4" },
};
const LINE = "oklch(1 0 0 / 45%)";
const AMBER = "oklch(0.82 0.15 80 / 80%)";

interface Box { x: number; y: number; w: number; h: number }

function NodeBox({ n, b }: { n: Node; b: Box }) {
  const s = STYLE[n.kind];
  return (
    <g>
      <rect x={b.x} y={b.y} width={b.w} height={b.h} rx={12} fill={s.fill} stroke={s.stroke} strokeWidth={1.2} strokeDasharray={s.dash} />
      <text x={b.x + 12} y={b.y + b.h / 2 - 4} fontSize={14} fontWeight={600} fill={s.title}>{n.title}</text>
      <text x={b.x + 12} y={b.y + b.h / 2 + 14} fontSize={12} fill="oklch(1 0 0 / 62%)">{n.sub}</text>
    </g>
  );
}

function Markers({ id }: { id: string }) {
  return (
    <defs>
      <marker id={`${id}-a`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,1 L9,5 L0,9 z" fill={LINE} />
      </marker>
      <marker id={`${id}-p`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,1 L9,5 L0,9 z" fill={AMBER} />
      </marker>
    </defs>
  );
}

/** A straight or one-bend arrow between the facing sides of two boxes; `dy` moves a same-row arrow down, `dx` a
 *  same-column arrow right. */
function arrow(a: Box, b: Box, dy = 0, dx = 0): string {
  const ac = { x: a.x + a.w / 2, y: a.y + a.h / 2 }, bc = { x: b.x + b.w / 2, y: b.y + b.h / 2 };
  if (Math.abs(ac.y - bc.y) < 4) { // same row
    const right = bc.x > ac.x;
    return `M${right ? a.x + a.w : a.x},${ac.y + dy} H${right ? b.x - 2 : b.x + b.w + 2}`;
  }
  const down = bc.y > ac.y;
  if (Math.abs(ac.x - bc.x) < 4) return `M${ac.x + dx},${down ? a.y + a.h : a.y} V${down ? b.y - 2 : b.y + b.h + 2}`;
  return `M${ac.x},${down ? a.y + a.h : a.y} V${bc.y} H${bc.x > ac.x ? b.x - 2 : b.x + b.w + 2}`;
}

function bracket(boxes: Box[], pad: number): Box {
  const x = Math.min(...boxes.map((b) => b.x)) - pad, y = Math.min(...boxes.map((b) => b.y)) - pad;
  const x2 = Math.max(...boxes.map((b) => b.x + b.w)) + pad, y2 = Math.max(...boxes.map((b) => b.y + b.h)) + pad;
  return { x, y, w: x2 - x, h: y2 - y };
}

function Wide({ id }: { id: string }) {
  const box = (n: Node): Box => ({ x: n.wide.x, y: n.wide.y, w: n.wide.w ?? NW, h: NH });
  const at = Object.fromEntries(NODES.map((n) => [n.id, box(n)])) as Record<string, Box>;
  const br = bracket(REPLACED.map((r) => at[r]!), 9);
  const belief = at.belief!, update = at.update!, student = at.student!, look = at.look!;
  const H = student.y + NH + 6;
  const loopX = belief.x + belief.w / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-labelledby={`${id}-wt`}
      style={{ fontFamily: "var(--font-sans)" }}>
      <title id={`${id}-wt`}>The flow of Find Margarete, as a loop</title>
      <Markers id={id} />
      <rect x={br.x} y={br.y} width={br.w} height={br.h} rx={16} fill="none" stroke={AMBER} strokeWidth={1.2} strokeDasharray="5 4" />
      {EDGES.map(([f, t, shift]) => {
        const progress = f === "sensors";
        return <path key={f + t} d={arrow(at[f]!, at[t]!, shift?.dy)} fill="none" stroke={progress ? AMBER : LINE} strokeWidth={1.5}
          strokeDasharray={progress ? "5 4" : undefined} markerEnd={`url(#${id}-${progress ? "p" : "a"})`} />;
      })}
      {/* back to the belief after a look that did not see her */}
      <path d={`M${update.x},${update.y + NH / 2} H${loopX} V${belief.y + NH + 2}`} fill="none" stroke={LINE} strokeWidth={1.5} markerEnd={`url(#${id}-a)`} />
      <text x={loopX - 10} y={(belief.y + NH + update.y) / 2 + 4} textAnchor="end" fontSize={12} fill="oklch(1 0 0 / 60%)">not seen: look again</text>
      <text x={look.x + 14} y={look.y + NH + 20} fontSize={12} fill="oklch(0.88 0.12 150)">seen: found her, stop</text>
      <text x={0} y={update.y + NH + 24} fontSize={12} fill="oklch(1 0 0 / 50%)">
        <tspan x={0}>It says each step and the end.</tspan>
        <tspan x={0} dy={22}>Also stops when P(out) ≥ 95%</tspan><tspan x={0} dy={16}>or after 12 looks.</tspan>
      </text>
      {/* the student would replace the route and the controller */}
      <path d={`M${student.x + student.w / 2 + 60},${student.y} V${br.y + br.h + 2}`} fill="none" stroke={AMBER} strokeWidth={1.5}
        strokeDasharray="5 4" markerEnd={`url(#${id}-p)`} />
      <text x={student.x + student.w / 2 + 70} y={br.y + br.h + 30} fontSize={12} fill="oklch(0.9 0.1 80)">drives instead</text>
      {NODES.map((n) => <NodeBox key={n.id} n={n} b={at[n.id]!} />)}
    </svg>
  );
}

function Tall({ id }: { id: string }) {
  const loopEnd = Math.max(...NODES.filter((n) => n.kind !== "progress").map((n) => n.tall));
  const afterUpdate = NODES.find((n) => n.id === "update")!.tall;
  // room under the update for the "not seen" note, and before the in-progress branch
  const yOf = (i: number) => 8 + i * (T_NH + T_GAP) + (i > afterUpdate ? 22 : 0) + (i > loopEnd ? 40 : 0);
  const at = Object.fromEntries(NODES.map((n) => [n.id, { x: T_X, y: yOf(n.tall), w: T_NW, h: T_NH }])) as Record<string, Box>;
  const br = bracket(REPLACED.map((r) => at[r]!), 7);
  const belief = at.belief!, update = at.update!, student = at.student!;
  const H = student.y + T_NH + 8;
  const rx = T_X + T_NW + 26;
  return (
    <svg viewBox={`0 0 ${T_W} ${H}`} className="block h-auto w-full" role="img" aria-labelledby={`${id}-tt`}
      style={{ fontFamily: "var(--font-sans)" }}>
      <title id={`${id}-tt`}>The flow of Find Margarete, step by step</title>
      <Markers id={`${id}t`} />
      <rect x={br.x} y={br.y} width={br.w} height={br.h} rx={14} fill="none" stroke={AMBER} strokeWidth={1.2} strokeDasharray="5 4" />
      {EDGES.map(([f, t, shift]) => {
        const progress = f === "sensors";
        return <path key={f + t} d={arrow(at[f]!, at[t]!, 0, shift?.dx)} fill="none" stroke={progress ? AMBER : LINE} strokeWidth={1.5}
          strokeDasharray={progress ? "5 4" : undefined} markerEnd={`url(#${id}t-${progress ? "p" : "a"})`} />;
      })}
      <path d={`M${update.x + update.w},${update.y + T_NH / 2} H${rx} V${belief.y + T_NH / 2} H${belief.x + belief.w + 2}`}
        fill="none" stroke={LINE} strokeWidth={1.5} markerEnd={`url(#${id}t-a)`} />
      <text x={T_X + 34} y={update.y + T_NH + 22} fontSize={12} fill="oklch(1 0 0 / 55%)">Not seen: back to the belief. Seen: stop.</text>
      <path d={`M${student.x},${student.y + T_NH / 2} H${6} V${br.y + br.h / 2} H${br.x - 2}`} fill="none" stroke={AMBER}
        strokeWidth={1.5} strokeDasharray="5 4" markerEnd={`url(#${id}t-p)`} />
      {NODES.map((n) => <NodeBox key={n.id} n={n} b={at[n.id]!} />)}
    </svg>
  );
}

export function FlowDiagram() {
  const id = useId().replace(/:/g, "");
  return (
    <figure className="how-figure">
      <div className="hidden min-[900px]:block p-1"><Wide id={id} /></div>
      <div className="mx-auto max-w-[400px] min-[900px]:hidden"><Tall id={id} /></div>
      <figcaption className="how-caption flex flex-wrap items-center gap-x-5 gap-y-2">
        <span className="inline-flex items-center gap-2"><span className="inline-block h-3 w-5 rounded-[4px] border border-[oklch(0.78_0.13_172/75%)] bg-[oklch(0.78_0.13_172/16%)]" />TabPFN model</span>
        <span className="inline-flex items-center gap-2"><span className="inline-block h-3 w-5 rounded-[4px] border border-video/20 bg-video/5" />Step</span>
        <span className="inline-flex items-center gap-2"><span className="inline-block h-3 w-5 rounded-[4px] border border-dashed border-[oklch(0.82_0.15_80/70%)] bg-[oklch(0.82_0.15_80/9%)]" />Obstacle-avoiding student, in the app (fallback: A* + a → b)</span>
      </figcaption>
    </figure>
  );
}

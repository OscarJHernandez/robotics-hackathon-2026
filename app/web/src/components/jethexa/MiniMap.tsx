import type { RobotState } from "@/lib/api";

const clamp = (v: number) => Math.max(-2.6, Math.min(2.6, v));

export function MiniMap({ s, onGoto }: { s: RobotState | null; onGoto: (x: number, y: number) => void }) {
  const click = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * 6 - 3;
    const y = 3 - ((e.clientY - r.top) / r.height) * 6;
    onGoto(+clamp(x).toFixed(2), +clamp(y).toFixed(2));
  };
  const lines = [-2, -1, 0, 1, 2];
  return (
    <svg viewBox="-3 -3 6 6" onClick={click} className="aspect-square w-full cursor-crosshair rounded-lg bg-video/5">
      {lines.map((v) => (
        <g key={v} className={v === 0 ? "stroke-video/30" : "stroke-video/10"} strokeWidth={0.015}>
          <line x1={v} y1={-3} x2={v} y2={3} /><line x1={-3} y1={v} x2={3} y2={v} />
        </g>
      ))}
      {s && <circle cx={s.target_x} cy={-s.target_y} r={0.09} className="fill-destructive" />}
      {s && (
        <g transform={`translate(${s.x} ${-s.y}) rotate(${-s.yaw_deg})`} style={{ transition: "transform .45s linear" }}>
          <circle r={0.22} className="fill-primary/25" />
          <path d="M0.2 0 L-0.12 0.12 L-0.05 0 L-0.12 -0.12 Z" className="fill-primary" />
        </g>
      )}
    </svg>
  );
}

export function MiniMapCard({ s, onGoto }: { s: RobotState | null; onGoto: (x: number, y: number) => void }) {
  return (
    <div className="glass fade-in w-[220px] rounded-2xl p-2.5">
      <MiniMap s={s} onGoto={onGoto} />
      <div className="mt-2 text-center font-mono text-[11px] text-video/80">
        {s ? `x ${s.x.toFixed(2)} y ${s.y.toFixed(2)} · ${Math.round(s.yaw_deg)}° · ${s.mode}` : "no state"}
      </div>
    </div>
  );
}

import { useEffect, useState } from "react";
import { streamUrl, type View } from "@/lib/api";

export const VIEW_LABEL: Record<View, string> = { chase: "Chase", orbit: "Orbit", overhead: "Overhead", eye: "Robot eye" };

export function Stream({ view, live, className = "" }: { view: View; live: boolean; className?: string }) {
  const [key, setKey] = useState(0);
  const [err, setErr] = useState(false);
  useEffect(() => {
    if (!err) return;
    const t = setTimeout(() => { setErr(false); setKey((k) => k + 1); }, 3000);
    return () => clearTimeout(t);
  }, [err]);
  useEffect(() => { setErr(false); }, [view]);
  if (!live || err) return <DemoView view={view} className={className} />;
  return <img key={`${view}-${key}`} src={`${streamUrl(view)}?r=${key}`} onError={() => setErr(true)}
    alt={VIEW_LABEL[view]} className={`h-full w-full object-cover ${className}`} />;
}

/** Animated placeholder used when the simulator is offline. */
export function DemoView({ view, className = "" }: { view: View; className?: string }) {
  if (view === "overhead") {
    return (
      <div className={`demo-view h-full w-full ${className}`}>
        <svg viewBox="-80 -45 160 90" preserveAspectRatio="xMidYMid slice" className="h-full w-full">
          <g className="stroke-primary/20" strokeWidth={0.3}>
            {Array.from({ length: 17 }, (_, i) => <line key={`v${i}`} x1={(i - 8) * 10} y1={-45} x2={(i - 8) * 10} y2={45} />)}
            {Array.from({ length: 9 }, (_, i) => <line key={`h${i}`} x1={-80} y1={(i - 4) * 10} x2={80} y2={(i - 4) * 10} />)}
          </g>
          <circle cx={28} cy={-16} r={3} className="fill-destructive/80" />
          <g className="demo-orbit"><polygon points="20,0 13,-4 13,4" className="fill-primary" /></g>
        </svg>
      </div>
    );
  }
  const tilt = view === "orbit" ? -12 : 0;
  const pillarX = view === "eye" ? 78 : view === "orbit" ? 118 : 108;
  return (
    <div className={`demo-view relative h-full w-full overflow-hidden ${className}`}>
      <svg viewBox="0 0 160 90" preserveAspectRatio="xMidYMid slice" className="h-full w-full">
        <g className="demo-floor stroke-primary/35" strokeWidth={0.3} transform={`skewX(${tilt})`}>
          {Array.from({ length: 18 }, (_, i) => <line key={i} x1={80 + (i - 9) * 4} y1={48} x2={80 + (i - 9) * 26} y2={95} />)}
          {[52, 58, 67, 80].map((y) => <line key={y} x1={-40} y1={y} x2={200} y2={y} />)}
        </g>
        <rect x={pillarX} y={view === "eye" ? 14 : 26} width={view === "eye" ? 12 : 6} height={view === "eye" ? 40 : 26} rx={1.5} className="fill-destructive/75" />
        {view !== "eye" && (
          <g transform={`translate(80 ${view === "orbit" ? 64 : 62})`}><g className="demo-bot">
            {[-1, 1].flatMap((s) => [-1, 0, 1].map((k) => (
              <line key={`${s}${k}`} x1={0} y1={k * 4} x2={s * 14} y2={k * 6 + 4} className={`stroke-video/70 leg leg-${(k + 1 + (s > 0 ? 1 : 0)) % 2}`} strokeWidth={1.2} strokeLinecap="round" />
            )))}
            <polygon points="-7,-6 7,-6 10,0 7,6 -7,6 -10,0" className="fill-primary" />
            <circle cx={0} cy={-2} r={1.6} className="fill-primary-foreground" />
          </g></g>
        )}
      </svg>
    </div>
  );
}

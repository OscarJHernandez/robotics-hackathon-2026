import { useEffect, useRef, useState } from "react";

// Colours (dataviz skill): one teal hue in two steps, the focus bar brighter. Checked with the skill's
// validate_palette.js --ordinal --mode dark --surface "#111518" (the figure surface): all checks pass.
const FOCUS = "#23aa8a"; // oklch(0.66 0.12 172)
const CONTEXT = "#366e6b"; // oklch(0.50 0.06 190)
const GRID = "oklch(1 0 0 / 9%)";
const TEXT = "oklch(1 0 0 / 90%)";
const MUTED = "oklch(1 0 0 / 55%)";

export interface Bar { label: string; note?: string | undefined; value: number; focus?: boolean | undefined }

/**
 * Horizontal bars with the value printed on every bar (no hover needed), drawn at the container's real width so
 * text stays at its true size on a phone. Wide: labels on the left. Narrow: each label above its bar.
 */
export function BarChart({ title, bars, max, ticks, format, inside = false, reference, summary }: {
  title: string; bars: Bar[]; max: number; ticks: number[]; format: (v: number) => string;
  /** Print the value inside the bar end (for bars that all reach far to the right). */
  inside?: boolean;
  /** A labelled reference line, such as the expert's 100%. */
  reference?: { value: number; label: string } | undefined;
  /** One sentence for screen readers. */
  summary: string;
}) {
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => { if (entry) setWidth(Math.round(entry.contentRect.width)); });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const wide = width >= 560;
  const labelW = wide ? 228 : 0;
  const notes = bars.some((b) => b.note);
  const right = inside ? 8 : 52;
  const plotW = Math.max(60, width - labelW - right);
  const barH = 22;
  const rowH = wide ? 50 : notes ? 74 : 56;
  const top = reference ? 26 : 6;
  const axisH = 26;
  const height = top + bars.length * rowH + axisH;
  const x = (v: number) => labelW + (Math.max(0, Math.min(v, max)) / max) * plotW;

  return (
    <div ref={box} className="w-full">
      {width > 0 && (
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title}. ${summary}`}
          style={{ display: "block", fontFamily: "var(--font-sans)" }}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={top - 4} y2={height - axisH + 2} stroke={GRID} strokeWidth={1} />
              <text x={x(t)} y={height - 8} textAnchor={t === 0 ? "start" : "middle"} fontSize={11} fill={MUTED}
                style={{ fontFamily: "var(--font-mono)", fontVariantNumeric: "tabular-nums" }}>{format(t)}</text>
            </g>
          ))}
          {reference && (
            <g>
              <line x1={x(reference.value)} x2={x(reference.value)} y1={top - 8} y2={height - axisH + 2} stroke="oklch(1 0 0 / 55%)" strokeWidth={1} />
              <text x={x(reference.value)} y={12} textAnchor="end" fontSize={11.5} fill={MUTED}>{reference.label}</text>
            </g>
          )}
          {bars.map((b, i) => {
            const y0 = top + i * rowH + (wide ? (rowH - barH) / 2 : rowH - barH - 8);
            const x0 = labelW;
            const x1 = Math.max(x0 + 6, x(b.value));
            const r = 4;
            const path = `M${x0},${y0} H${x1 - r} A${r},${r} 0 0 1 ${x1},${y0 + r} V${y0 + barH - r} A${r},${r} 0 0 1 ${x1 - r},${y0 + barH} H${x0} Z`;
            const value = format(b.value);
            const labelY = wide ? y0 + barH / 2 : y0 - (b.note ? 26 : 9);
            return (
              <g key={b.label}>
                <title>{`${b.label}${b.note ? ` (${b.note})` : ""}: ${value}`}</title>
                <text x={0} y={wide && b.note ? labelY - 3 : labelY + (wide ? 4 : 0)} fontSize={13.5}
                  fontWeight={b.focus ? 600 : 500} fill={TEXT}>{b.label}</text>
                {b.note && <text x={0} y={labelY + (wide ? 13 : 16)} fontSize={11.5} fill={MUTED}>{b.note}</text>}
                <path d={path} fill={b.focus ? FOCUS : CONTEXT} />
                {inside ? (
                  <text x={x1 - 8} y={y0 + barH / 2 + 4.5} textAnchor="end" fontSize={12.5} fontWeight={600}
                    fill={b.focus ? "#0b0b0b" : "#ffffff"} style={{ fontVariantNumeric: "tabular-nums" }}>{value}</text>
                ) : (
                  <text x={x1 + 8} y={y0 + barH / 2 + 4.5} fontSize={13} fontWeight={600} fill={TEXT}
                    style={{ fontVariantNumeric: "tabular-nums" }}>{value}</text>
                )}
              </g>
            );
          })}
        </svg>
      )}
      <table className="sr-only">
        <caption>{title}</caption>
        <tbody>{bars.map((b) => <tr key={b.label}><th scope="row">{b.label}{b.note ? ` (${b.note})` : ""}</th><td>{format(b.value)}</td></tr>)}</tbody>
      </table>
    </div>
  );
}

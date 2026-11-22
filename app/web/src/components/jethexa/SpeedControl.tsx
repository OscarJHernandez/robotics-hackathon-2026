// x1 · x2 · x3 · x5: how much faster than real time the shared simulation runs (POST /api/speed). One robot, so
// the speed is the same for every viewer: the highlighted option is read from the robot's state. Every TabPFN
// decision makes the loop wait, so the asked speed is not always reached; then the actual rate is shown.
import { useEffect, useState } from "react";
import { Gauge } from "lucide-react";
import { SPEEDS, setSpeed, type RobotState, type Speed } from "@/lib/api";

/** "x5 (running x3.4)" when the measured rate is more than 10% off the asked one, else null. */
export function speedNote(state: RobotState | null): string | null {
  const asked = state?.speed, actual = state?.speed_actual;
  if (!asked || actual === undefined || actual === null) return null;
  if (Math.abs(actual - asked) <= 0.1 * asked) return null;
  return `x${asked} (running x${actual < 10 ? actual.toFixed(1) : Math.round(actual)})`;
}

export function SpeedControl({ state, live, onError, className = "" }: {
  state: RobotState | null; live: boolean; onError?: ((m: string) => void) | undefined; className?: string;
}) {
  const [pending, setPending] = useState<Speed | null>(null);
  const current = (state?.speed ?? 1) as Speed;
  // A click shows at once; the next state that carries it (or 3 s) ends the wait.
  useEffect(() => { if (pending !== null && current === pending) setPending(null); }, [current, pending]);
  useEffect(() => {
    if (pending === null) return;
    const id = setTimeout(() => setPending(null), 3000);
    return () => clearTimeout(id);
  }, [pending]);
  const shown = pending ?? current;
  const note = pending === null ? speedNote(state) : null;
  const pick = async (s: Speed) => {
    if (s === shown) return;
    setPending(s);
    const r = await setSpeed(s);
    if (!r.ok) { setPending(null); onError?.(r.error ?? "could not change the speed"); }
  };
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <div role="group" aria-label="Simulation speed" className="speed-seg"
        title="How much faster than real time the robot moves (the same for every viewer)">
        <Gauge aria-hidden className="mx-1 h-3.5 w-3.5 text-video/60" />
        {SPEEDS.map((s) => (
          <button key={s} type="button" aria-pressed={s === shown} disabled={!live} onClick={() => void pick(s)}
            className={`speed-opt ${s === shown ? "speed-opt-on" : ""}`}>x{s}</button>
        ))}
      </div>
      {note && <span role="status" className="speed-note text-shadow-soft" title="Each TabPFN decision makes the simulation wait, so it runs slower than asked">{note}</span>}
    </div>
  );
}

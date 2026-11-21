import { useEffect, useState } from "react";
import { getHealth, getState, type Health, type RobotState } from "@/lib/api";
import { demoState } from "@/lib/demo";

export type Conn = "checking" | "live" | "demo";

export function useHealth() {
  const [health, setHealth] = useState<Health | null>(null);
  const [conn, setConn] = useState<Conn>("checking");
  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const h = await getHealth(AbortSignal.timeout(2500));
        if (alive) { setHealth(h); setConn("live"); }
      } catch { if (alive) { setHealth(null); setConn("demo"); } }
    };
    check();
    const id = setInterval(check, 5000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  return { health, conn };
}

export function useRobotState(conn: Conn) {
  const [state, setState] = useState<RobotState | null>(null);
  useEffect(() => {
    if (conn === "checking") return;
    let alive = true;
    const start = performance.now();
    const tick = async () => {
      if (conn === "demo") { setState(demoState((performance.now() - start) / 1000)); return; }
      try { const s = await getState(AbortSignal.timeout(1500)); if (alive) setState(s); } catch { /* keep last */ }
    };
    tick();
    const id = setInterval(tick, 500);
    return () => { alive = false; clearInterval(id); };
  }, [conn]);
  return state;
}

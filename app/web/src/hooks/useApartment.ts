import { useEffect, useRef, useState } from "react";
import { getApartment, type ApartmentState } from "@/lib/api";

/**
 * The apartment (zones, obstacles, journal, belief, search, log), polled from the server: the same for every
 * viewer. Fast while a search runs, slow otherwise; nothing is fetched while `enabled` is false.
 */
export function useApartment(enabled: boolean): { apartment: ApartmentState | null; refresh: () => void } {
  const [apartment, setApartment] = useState<ApartmentState | null>(null);
  const running = useRef(false);
  const kick = useRef<() => void>(() => {});

  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const next = await getApartment();
        if (!alive) return;
        running.current = next.search.running;
        setApartment(next);
      } catch { /* the server is restarting: keep the last picture and try again */ }
      if (alive) timer = setTimeout(tick, running.current ? 350 : 1500);
    };
    kick.current = () => { if (timer) clearTimeout(timer); void tick(); };
    void tick();
    return () => { alive = false; if (timer) clearTimeout(timer); };
  }, [enabled]);

  return { apartment, refresh: () => kick.current() };
}

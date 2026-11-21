import { Dices, Map } from "lucide-react";
import type { View } from "@/lib/api";
import { VIEW_LABEL } from "./Stream";

export function ViewList({ views, active, onPick, mapOpen, onToggleMap, onRandom }: {
  views: View[]; active: View; onPick: (v: View) => void; mapOpen: boolean; onToggleMap: () => void;
  onRandom: () => void;
}) {
  return (
    <div className="flex flex-col items-start gap-2 sm:items-end">
      <ul className="flex flex-wrap gap-1.5 sm:flex-col sm:items-end sm:gap-1">
        {views.map((v) => {
          const on = v === active;
          return (
            <li key={v}>
              <button onClick={() => onPick(v)}
                className={`pixel flex items-center gap-2 text-xs transition-colors duration-200 text-shadow-soft max-sm:glass max-sm:rounded-full max-sm:px-2.5 max-sm:py-1 max-sm:text-[10px] ${on ? "text-video" : "text-video/50 hover:text-video/80"}`}>
                {VIEW_LABEL[v]}
                <span className={`h-2 w-2 rounded-full ${on ? "bg-video" : "bg-video/30"}`} />
              </button>
            </li>
          );
        })}
      </ul>
      <div className="mt-1 flex items-center gap-2">
        <button onClick={onRandom} title="Move the red pillar (the end point) to a random spot"
          className="glass pixel flex h-9 items-center gap-2 rounded-full px-3.5 text-[10px] transition-colors duration-200 hover:bg-video/10">
          <Dices className="h-4 w-4" />Random end point
        </button>
      <button onClick={onToggleMap} aria-label="Toggle mini map" aria-pressed={mapOpen}
        className={`glass-btn h-9 w-9 max-sm:hidden ${mapOpen ? "glass-btn-on" : ""}`}><Map className="h-4 w-4" /></button>
      </div>
    </div>
  );
}

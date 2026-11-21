import { Eye, EyeOff, X } from "lucide-react";
import type { View } from "@/lib/api";
import { Stream, VIEW_LABEL } from "./Stream";

export function CameraCards({ views, live, visible, onToggle, onPick, onClose }: {
  views: View[]; live: boolean; visible: boolean; onToggle: () => void; onPick: (v: View) => void; onClose: (v: View) => void;
}) {
  return (
    <div className="flex flex-col items-start gap-2.5">
      {visible && views.map((v) => (
        <div key={v} className="cam-card group fade-in">
          <button className="absolute inset-0" onClick={() => onPick(v)} aria-label={`Show ${VIEW_LABEL[v]} as main view`}>
            <Stream view={v} live={live} />
          </button>
          <span className="pixel pointer-events-none absolute left-2.5 top-2 text-[9px] text-video text-shadow-soft">::: {VIEW_LABEL[v]}</span>
          <button onClick={() => onClose(v)} aria-label="Close" className="glass-btn absolute right-2 top-2 h-6 w-6"><X className="h-3 w-3" /></button>
        </div>
      ))}
      <button onClick={onToggle} className="glass pixel flex h-8 items-center gap-2 rounded-full px-3.5 text-[10px] transition-colors duration-200 hover:bg-video/10">
        {visible ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}{visible ? "Hide cameras" : "Show cameras"}
      </button>
    </div>
  );
}

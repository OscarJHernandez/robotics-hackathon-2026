import { useEffect, useState } from "react";
import type { Narration } from "@/hooks/useNarration";

const SUGGESTIONS = ["Walk to the red pillar", "Turn around and come back", "Walk forward one metre", "Go to the far corner", "Spin 180 degrees"];

/** The header line: metacon's latest word on the search while it is recent, otherwise a suggestion to send. */
export function Caption({ onPick, narration }: { onPick: (s: string) => void; narration?: Narration | null }) {
  const [i, setI] = useState(0);
  const [show, setShow] = useState(true);
  useEffect(() => {
    const id = setInterval(() => {
      setShow(false);
      setTimeout(() => { setI((n) => (n + 1) % SUGGESTIONS.length); setShow(true); }, 300);
    }, 8000);
    return () => clearInterval(id);
  }, []);
  if (narration) {
    return (
      <div key={narration.id} role="status" aria-live="polite" className="rise-in text-center">
        <div className="pixel text-[10px] tracking-[0.3em] text-video/70 text-shadow-soft">metacon · searching</div>
        <div className="mt-1 line-clamp-3 text-lg font-medium leading-snug tracking-tight text-video text-shadow-soft sm:text-2xl">
          {narration.text}
        </div>
      </div>
    );
  }
  const s = SUGGESTIONS[i]!;
  return (
    <button onClick={() => onPick(s)} className="group text-center" title="Send to metacon">
      <div className="pixel text-[10px] tracking-[0.3em] text-video/70 text-shadow-soft">Suggested by metacon</div>
      <div className={`mt-1 text-2xl font-medium tracking-tight text-video text-shadow-soft transition-all duration-300 group-hover:opacity-80 sm:text-4xl ${show ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-1"}`}>
        “{s}”
      </div>
    </button>
  );
}

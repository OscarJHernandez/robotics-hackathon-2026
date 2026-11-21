import { useEffect, useState } from "react";

const SUGGESTIONS = ["Walk to the red pillar", "Turn around and come back", "Walk forward one metre", "Go to the far corner", "Spin 180 degrees"];

export function Caption({ onPick }: { onPick: (s: string) => void }) {
  const [i, setI] = useState(0);
  const [show, setShow] = useState(true);
  useEffect(() => {
    const id = setInterval(() => {
      setShow(false);
      setTimeout(() => { setI((n) => (n + 1) % SUGGESTIONS.length); setShow(true); }, 300);
    }, 8000);
    return () => clearInterval(id);
  }, []);
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

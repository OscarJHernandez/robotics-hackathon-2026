import { useEffect, useRef, useState } from "react";
import { narrationAudioUrl, type NarrationEvent } from "@/lib/api";
import { isPlaying, playAudio } from "@/lib/audio";
import { openLive } from "@/lib/live";

export interface Narration { id: string; kind: string; text: string; source: "metacon" | "template"; at: number }

const SHOW_MS = 9000; // a line stays "recent" (header caption, phone card) this long
const MAX_QUEUE = 2; // spoken lines waiting behind the one playing; older ones are skipped

/**
 * metacon's lines about the search, from /api/live (app/narrate.py). Each line arrives once as text (`onLine` is
 * called once per id) and, when the voice service spoke it, again with its audio. The audio plays once per id when
 * `speak` says so (the speaker is on and no voice call is going on), one line after the other, and never over a
 * call's reply: if something else is playing, the line is not spoken.
 */
export function useNarration(live: boolean, opts: { speak: () => boolean; onLine: (n: Narration) => void }) {
  const [recent, setRecent] = useState<Narration | null>(null);
  const seen = useRef(new Set<string>());
  const played = useRef(new Set<string>());
  const queue = useRef<string[]>([]);
  const mine = useRef(false); // a narration line is playing right now
  const optsRef = useRef(opts);
  optsRef.current = opts;
  const hide = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (!live || typeof EventSource === "undefined") return;
    const playNext = () => {
      const path = queue.current.shift();
      if (!path) return;
      if (!optsRef.current.speak() || (isPlaying() && !mine.current)) { queue.current = []; return; }
      mine.current = true;
      void fetch(narrationAudioUrl(path)).then((r) => (r.ok ? r.blob() : null))
        .then((b) => (b && optsRef.current.speak() && !isPlaying() ? playAudio(b) : undefined))
        .catch(() => { /* no sound: the words are on screen */ })
        .finally(() => { mine.current = false; playNext(); });
    };
    const source = openLive();
    source.onmessage = (m) => {
      let e: NarrationEvent;
      try { e = JSON.parse(m.data as string) as NarrationEvent; } catch { return; }
      if (e.type !== "narration" || !e.id) return;
      if (!seen.current.has(e.id)) {
        seen.current.add(e.id);
        const n: Narration = { id: e.id, kind: e.kind, text: e.text, source: e.source, at: Date.now() };
        optsRef.current.onLine(n);
        setRecent(n);
        clearTimeout(hide.current);
        hide.current = window.setTimeout(() => setRecent((r) => (r?.id === n.id ? null : r)), SHOW_MS);
      }
      if (e.audio && !played.current.has(e.id)) {
        played.current.add(e.id);
        if (!optsRef.current.speak()) return;
        if (isPlaying() && !mine.current) return; // a call's reply is playing: never talk over it
        queue.current.push(e.audio);
        queue.current = queue.current.slice(-MAX_QUEUE);
        if (!mine.current) playNext();
      }
    };
    return () => { source.close(); clearTimeout(hide.current); queue.current = []; };
  }, [live]);

  return { recent };
}

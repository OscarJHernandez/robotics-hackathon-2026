import { useCallback, useEffect, useRef, useState } from "react";
import {
  knockAudioUrl, knockCall, knockChoice, knockFind, knockStop, liveUrl,
  type KnockChoice, type KnockState,
} from "@/lib/api";
import { playAudio, unlockAudio } from "@/lib/audio";
import { demoKnock, KNOCK_IDLE } from "@/lib/demo";

export interface Knock {
  knock: KnockState;
  call: () => void; find: () => void; stop: () => void; choose: (c: KnockChoice) => void;
}

/**
 * The Knock check-in, the same on every viewer's page: the latest knock event from /api/live. The call-out plays
 * on every page whose speaker is on. Without the server (demo mode) the same states are played through locally.
 */
export function useKnock(live: boolean, speakerOn: boolean, onError: (m: string) => void): Knock {
  const [knock, setKnock] = useState<KnockState>(KNOCK_IDLE);
  const speaker = useRef(speakerOn);
  speaker.current = speakerOn;
  const played = useRef(new Set<string>());
  const demo = useRef<ReturnType<typeof demoKnock> | null>(null);

  useEffect(() => {
    if (!live || typeof EventSource === "undefined") return;
    const source = new EventSource(liveUrl());
    source.onmessage = (m) => {
      let e: KnockState;
      try { e = JSON.parse(m.data as string) as KnockState; } catch { return; }
      if (e.type !== "knock") return;
      setKnock(e);
      if (e.audio && !played.current.has(e.audio)) {
        played.current.add(e.audio);
        if (speaker.current) {
          void fetch(knockAudioUrl(e.audio)).then((r) => (r.ok ? r.blob() : null)).then((b) => b && playAudio(b))
            .catch(() => { /* no sound, the card still shows the words */ });
        }
      }
    };
    return () => source.close();
  }, [live]);

  useEffect(() => {
    if (live) return;
    demo.current = demoKnock(setKnock);
    return () => { demo.current?.stop(); demo.current = null; };
  }, [live]);

  const send = useCallback((request: () => Promise<string | null>, local: () => void) => {
    unlockAudio(); // inside the tap, so the call-out can play when it arrives
    if (!live) { local(); return; }
    void request().then((error) => { if (error) onError(error); });
  }, [live, onError]);

  return {
    knock,
    call: () => send(knockCall, () => demo.current?.call()),
    find: () => send(knockFind, () => demo.current?.find()),
    stop: () => send(knockStop, () => demo.current?.stop()),
    choose: (c) => send(() => knockChoice(c), () => demo.current?.choose(c)),
  };
}

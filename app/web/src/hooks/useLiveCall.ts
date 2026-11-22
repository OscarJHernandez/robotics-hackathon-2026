import { useEffect, useRef, useState } from "react";
import { liveUrl, stripMarkers, type LiveEvent } from "@/lib/api";
import { openLive } from "@/lib/live";

/** A voice call as every viewer sees it, built from the /api/live events. */
export interface LiveCall {
  id: string;
  heard: string; // what the caller said (live caption until final)
  final: boolean;
  reply: string; // metacon's answer so far, without [[command]] markers
  commands: { text: string; ok: boolean }[];
  speaking: boolean;
  ended: boolean;
}

const LINGER_MS = 7000; // a finished call stays on screen this long

/** The latest voice call on the site (anyone's), or null. Listens only while `on`. */
export function useLiveCall(on: boolean): LiveCall | null {
  const [call, setCall] = useState<LiveCall | null>(null);
  const raw = useRef("");
  const hide = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (!on || typeof EventSource === "undefined") return;
    const source = openLive();
    source.onmessage = (m) => {
      let e: LiveEvent;
      try { e = JSON.parse(m.data as string) as LiveEvent; } catch { return; }
      if (e.type === "call" && e.state === "start") {
        clearTimeout(hide.current);
        raw.current = "";
        setCall({ id: e.call, heard: "", final: false, reply: "", commands: [], speaking: false, ended: false });
        return;
      }
      setCall((c) => {
        if (!c || c.id !== e.call) return c;
        switch (e.type) {
          case "heard": return c.final && !e.final ? c : { ...c, heard: e.text, final: e.final };
          case "token": raw.current += e.text; return { ...c, reply: stripMarkers(raw.current).trim() };
          case "command": return { ...c, commands: [...c.commands, { text: e.text, ok: e.ok }] };
          case "speaking": return { ...c, speaking: true, reply: c.reply || e.text };
          case "call":
            clearTimeout(hide.current);
            hide.current = window.setTimeout(() => setCall((x) => (x?.id === c.id ? null : x)), LINGER_MS);
            return { ...c, ended: true, speaking: false };
          default: return c;
        }
      });
    };
    return () => { source.close(); clearTimeout(hide.current); };
  }, [on]);

  return call;
}

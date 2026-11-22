import { liveUrl } from "@/lib/api";

/**
 * One shared connection to /api/live for every hook that listens to it. A browser allows six connections per
 * host and the picture streams already hold three, so each hook opening its own EventSource starved the
 * ordinary requests (state, health, apartment). `openLive()` looks like an EventSource: set `onmessage`, call
 * `close()`; the real connection closes when the last listener does.
 */
export interface LiveSource { onmessage: ((m: { data: unknown }) => void) | null; close: () => void }

let source: EventSource | null = null;
const listeners = new Set<LiveSource>();

export function openLive(): LiveSource {
  const mine: LiveSource = {
    onmessage: null,
    close: () => {
      listeners.delete(mine);
      if (listeners.size === 0 && source) { source.close(); source = null; }
    },
  };
  listeners.add(mine);
  if (!source) {
    source = new EventSource(liveUrl());
    source.onmessage = (m) => { listeners.forEach((l) => l.onmessage?.({ data: m.data })); };
  }
  return mine;
}

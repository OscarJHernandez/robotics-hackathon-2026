// Recording and playback for voice calls.

let player: HTMLAudioElement | null = null;
let unlocked = false;
let silentUrl = "";
let current: { url: string; done: () => void } | null = null;

/** 0.1 s of silence as a WAV, used to unlock playback. */
function silence(): string {
  if (silentUrl) return silentUrl;
  const n = 800, buf = new ArrayBuffer(44 + n), v = new DataView(buf);
  const str = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + n, true); str(8, "WAVE"); str(12, "fmt "); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, 8000, true); v.setUint32(28, 8000, true);
  v.setUint16(32, 1, true); v.setUint16(34, 8, true); str(36, "data"); v.setUint32(40, n, true);
  for (let i = 0; i < n; i++) v.setUint8(44 + i, 128);
  silentUrl = URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
  return silentUrl;
}

function audio(): HTMLAudioElement {
  if (!player) { player = new Audio(); player.preload = "auto"; }
  return player;
}

/**
 * Call inside a touch or click handler. Phones (iOS above all) only let a page play sound that a touch started;
 * playing silence on the press lets the spoken reply play later, after the network round trip.
 */
export function unlockAudio() {
  if (unlocked) return;
  const a = audio();
  a.src = silence();
  a.play().then(() => { unlocked = true; }).catch(() => { /* try again on the next press */ });
}

/** Plays an audio blob; resolves when it ends or is stopped. A new call stops the one before. */
export function playAudio(blob: Blob): Promise<void> {
  stopAudio();
  const a = audio();
  const url = URL.createObjectURL(blob);
  return new Promise<void>((resolve) => {
    const done = () => {
      a.onended = a.onerror = a.onpause = null;
      if (current?.url === url) current = null;
      URL.revokeObjectURL(url);
      resolve();
    };
    current = { url, done };
    a.onended = a.onerror = a.onpause = done;
    a.src = url;
    a.play().catch(done);
  });
}

export function stopAudio() {
  if (!current) return;
  const c = current;
  current = null;
  audio().pause();
  c.done();
}

/** The recording type this browser can make: WebM/Opus (Chrome, Firefox, Android) or MP4/AAC (Safari, iOS). */
export function recordingType(): string {
  if (typeof MediaRecorder === "undefined") return "";
  for (const t of ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"]) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

export const canRecord = () =>
  typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== "undefined";

export const buzz = (ms = 12) => { try { navigator.vibrate?.(ms); } catch { /* not supported */ } };

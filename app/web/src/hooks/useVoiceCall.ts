import { useCallback, useEffect, useRef, useState } from "react";
import { endCall, speak, startCall, transcribe } from "@/lib/api";
import { buzz, canRecord, playAudio, recordingType, stopAudio, unlockAudio } from "@/lib/audio";

/** Where a hold-to-talk call is: press → recording → understanding → thinking (metacon) → speaking → idle. */
export type Phase = "idle" | "starting" | "recording" | "understanding" | "thinking" | "speaking";

const INTERIM_MS = 1500; // a live caption of the recording so far, this often
const MAX_MS = 15000; // a recording stops by itself after this
const MIN_MS = 400; // shorter presses are taps, not speech

export function phaseText(phase: Phase, cancelArmed = false): string {
  if (cancelArmed) return "Release to cancel";
  switch (phase) {
    case "starting": return "Connecting…";
    case "recording": return "Listening… release to send";
    case "understanding": return "Understanding…";
    case "thinking": return "metacon is answering…";
    case "speaking": return "Speaking… tap to stop";
    default: return "Hold to talk";
  }
}

/**
 * One voice call per press: claim the line, record while the button is held (sending live captions), then
 * transcribe, ask metacon through `ask` (which returns the reply text), speak the reply and hang up.
 */
export function useVoiceCall({ enabled, speakerOn, ask, onError }: {
  enabled: boolean; speakerOn: boolean;
  ask: (text: string, call: string) => Promise<string>;
  onError: (message: string) => void;
}) {
  const [phase, setPhaseState] = useState<Phase>("idle");
  const [interim, setInterim] = useState("");
  const phaseRef = useRef<Phase>("idle");
  const held = useRef(false);
  const hush = useRef(false); // the caller tapped to stop the spoken reply
  const own = useRef(new Set<string>());
  const rec = useRef<{ recorder: MediaRecorder; stream: MediaStream; chunks: Blob[]; type: string; call: string;
    started: number; timers: number[]; captioning: boolean } | null>(null);
  const speakerRef = useRef(speakerOn);
  speakerRef.current = speakerOn;
  const askRef = useRef(ask);
  askRef.current = ask;
  const errRef = useRef(onError);
  errRef.current = onError;

  const setPhase = (p: Phase) => { phaseRef.current = p; setPhaseState(p); };
  const supported = enabled && canRecord();

  const hangUp = useCallback((call: string | null) => {
    if (call) endCall(call);
    setInterim("");
    setPhase("idle");
  }, []);

  const stopRecording = (keep: boolean): Promise<Blob | null> => {
    const r = rec.current;
    rec.current = null;
    if (!r) return Promise.resolve(null);
    r.timers.forEach((t) => clearTimeout(t));
    return new Promise((resolve) => {
      r.recorder.onstop = () => {
        r.stream.getTracks().forEach((t) => t.stop());
        resolve(keep ? new Blob(r.chunks, { type: r.type }) : null);
      };
      if (r.recorder.state !== "inactive") r.recorder.stop(); else r.recorder.onstop(new Event("stop"));
    });
  };

  const caption = async () => {
    const r = rec.current;
    if (!r || r.captioning || !r.chunks.length) return;
    r.captioning = true;
    const res = await transcribe(new Blob(r.chunks, { type: r.type }), r.call, false);
    r.captioning = false;
    if (rec.current === r && res.text) setInterim(res.text);
  };

  const finish = async (call: string, blob: Blob | null, took: number) => {
    if (!blob || took < MIN_MS) {
      hangUp(call);
      if (took < MIN_MS) errRef.current("Hold the button while you speak");
      return;
    }
    setPhase("understanding");
    const heard = await transcribe(blob, call, true);
    if (!heard.text) { hangUp(call); errRef.current(heard.error ?? "Heard nothing, try again"); return; }
    setInterim(heard.text);
    setPhase("thinking");
    let reply = "";
    try { reply = await askRef.current(heard.text, call); } catch { /* the terminal shows the error */ }
    setInterim("");
    if (reply && speakerRef.current) {
      hush.current = false;
      setPhase("speaking");
      const audio = await speak(reply, call);
      if (audio && !hush.current) await playAudio(audio);
    }
    hangUp(call);
  };

  const press = useCallback(async () => {
    if (!supported) return;
    if (phaseRef.current === "speaking") { hush.current = true; stopAudio(); return; }
    if (phaseRef.current !== "idle") return;
    held.current = true;
    unlockAudio();
    setPhase("starting");
    setInterim("");
    let stream: MediaStream | null = null;
    const [line, mic] = await Promise.all([
      startCall(),
      navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } })
        .then((s) => { stream = s; return null; })
        .catch((e: unknown) => (e instanceof Error && e.name === "NotAllowedError"
          ? "Microphone blocked: allow it in the browser to talk" : "No microphone found")),
    ]);
    const s = stream as MediaStream | null;
    if (!line.call || mic || !s) {
      s?.getTracks().forEach((t) => t.stop());
      if (line.call) endCall(line.call);
      setPhase("idle");
      errRef.current(line.busy ? "Someone else is talking to the robot, wait a moment"
        : mic ?? line.error ?? "Voice is offline");
      return;
    }
    own.current.add(line.call);
    if (!held.current) {
      // Released while the browser asked for the microphone (the first press): nothing was said yet.
      s.getTracks().forEach((t) => t.stop());
      hangUp(line.call);
      errRef.current("Microphone ready: hold the button while you speak");
      return;
    }
    const type = recordingType();
    const recorder = type ? new MediaRecorder(s, { mimeType: type }) : new MediaRecorder(s);
    const r = { recorder, stream: s, chunks: [] as Blob[], type: (recorder.mimeType || type || "audio/webm"),
      call: line.call, started: performance.now(), timers: [] as number[], captioning: false };
    recorder.ondataavailable = (e) => { if (e.data.size) r.chunks.push(e.data); };
    recorder.start(250);
    rec.current = r;
    buzz();
    setPhase("recording");
    r.timers.push(window.setInterval(() => void caption(), INTERIM_MS));
    r.timers.push(window.setTimeout(() => void release(), MAX_MS));
  }, [supported]); // eslint-disable-line react-hooks/exhaustive-deps

  const release = useCallback(async () => {
    held.current = false;
    const r = rec.current;
    if (!r || phaseRef.current !== "recording") return;
    buzz(8);
    const took = performance.now() - r.started;
    const blob = await stopRecording(true);
    await finish(r.call, blob, took);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const cancel = useCallback(async () => {
    held.current = false;
    const r = rec.current;
    if (!r) return;
    await stopRecording(false);
    hangUp(r.call);
  }, [hangUp]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => () => { void cancel(); stopAudio(); }, [cancel]);

  const isMine = useCallback((call: string) => own.current.has(call) || phaseRef.current === "starting", []);

  return { supported, phase, interim, press, release, cancel, isMine };
}

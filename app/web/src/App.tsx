// The page. Made with Lovable (originally src/routes/index.tsx in a TanStack Start project),
// moved into a plain Vite single-page app so the Python server can serve it as static files.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { Toaster } from "@/components/ui/sonner";
import { Brand } from "@/components/jethexa/Brand";
import { Caption } from "@/components/jethexa/Caption";
import { ViewList } from "@/components/jethexa/ViewList";
import { CameraCards } from "@/components/jethexa/CameraCards";
import { MiniMapCard } from "@/components/jethexa/MiniMap";
import { ApartmentPanel } from "@/components/jethexa/ApartmentPanel";
import { QuickActions } from "@/components/jethexa/QuickActions";
import { ChatBar } from "@/components/jethexa/ChatBar";
import { TerminalPanel, useMetacon } from "@/components/jethexa/Terminal";
import { Stream } from "@/components/jethexa/Stream";
import { LiveCallCard } from "@/components/jethexa/LiveCallCard";
import { PhoneApp } from "@/components/jethexa/PhoneApp";
import { HowItWorks } from "@/components/jethexa/how/HowItWorks";
import { SpeakerToggle, TalkButton } from "@/components/jethexa/TalkButton";
import { useHealth, useRobotState } from "@/hooks/useRobotState";
import { useIsPhone } from "@/hooks/useIsPhone";
import { useLiveCall } from "@/hooks/useLiveCall";
import { useKnock } from "@/hooks/useKnock";
import { useApartment } from "@/hooks/useApartment";
import { useNarration } from "@/hooks/useNarration";
import { phaseText, useVoiceCall } from "@/hooks/useVoiceCall";
import { sendCommand, VIEWS, type View } from "@/lib/api";

const HOW = "#how-it-works";

export function App() {
  // The "How it works" page: App owns the #how-it-works hash alone. It renders instead of the stage, so the
  // picture streams unmount; polling that only feeds the screen pauses while it is open.
  const [howOpen, setHowOpen] = useState(() => window.location.hash === HOW);
  const howPushed = useRef(false); // opened here, so a history entry for the robot is right behind it
  useEffect(() => {
    const sync = () => {
      const open = window.location.hash === HOW;
      if (!open) howPushed.current = false;
      setHowOpen(open);
    };
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);
  const openHow = useCallback(() => {
    if (window.location.hash === HOW) return;
    howPushed.current = true;
    window.location.hash = HOW; // a new history entry (Back returns to the robot); ?phone=1 stays
  }, []);
  const closeHow = useCallback(() => {
    if (howPushed.current) {
      window.history.back(); // the entry behind it is the robot; hashchange closes the page
    } else { // opened by a direct link: never go back to another site
      window.history.replaceState(window.history.state, "", window.location.pathname + window.location.search);
      setHowOpen(false);
    }
  }, []);
  // Focus returns to the button that opens the page (desktop header or phone top bar) once the stage is back.
  const wasOpen = useRef(howOpen);
  useEffect(() => {
    if (wasOpen.current && !howOpen) document.querySelector<HTMLElement>("[data-how-opener]")?.focus({ preventScroll: true });
    wasOpen.current = howOpen;
  }, [howOpen]);

  const { health, conn } = useHealth();
  const state = useRobotState(conn, !howOpen);
  const live = conn === "live";
  const views = health?.views?.length ? health.views : VIEWS;
  const [main, setMain] = useState<View>("chase");
  const [closed, setClosed] = useState<View[]>([]);
  const [camsOn, setCamsOn] = useState(true);
  const [mapOpen, setMapOpen] = useState(false);
  const [termOpen, setTermOpen] = useState(false);
  const term = useMetacon(conn, health?.llm ?? false);
  const phone = useIsPhone();
  // The apartment: zones, obstacles, journal, search and log, in a panel on the right (desktop).
  // On a phone it is a card under the camera, with the whole panel as a full-screen sheet.
  const hasApartment = live && !!health?.apartment;
  const [aptOpen, setAptOpen] = useState(true);
  const { apartment, refresh: refreshApartment } = useApartment(hasApartment && (aptOpen || phone) && !howOpen);
  const apt = hasApartment && !phone && aptOpen && apartment ? apartment : null;
  useEffect(() => { if (health?.scene === "apartment") setMain("orbit"); }, [health?.scene]);

  // Voice: hold to talk, the reply spoken in the robot's voice, and everyone else's calls shown live.
  const [speakerOn, setSpeakerOn] = useState(() => { try { return localStorage.getItem("jethexa-speaker") !== "off"; } catch { return true; } });
  const toggleSpeaker = () => setSpeakerOn((on) => {
    try { localStorage.setItem("jethexa-speaker", on ? "off" : "on"); } catch { /* private mode */ }
    return !on;
  });
  const { send: termSend } = term;
  const ask = useCallback((text: string, call: string) => {
    if (!phone) setTermOpen(true);
    return termSend(text, call);
  }, [termSend, phone]);
  const voiceOn = live && !!health?.voice && !!health?.llm;
  const voice = useVoiceCall({ enabled: voiceOn, speakerOn, ask, onError: (m) => toast(m) });
  const liveCall = useLiveCall(live);
  const other = liveCall && !voice.isMine(liveCall.id) ? liveCall : null;
  const knockError = useCallback((m: string) => toast(m), []);
  const knock = useKnock(live, speakerOn, knockError); // Find and Call Margarete; the phone layout shows it for now

  // metacon narrates the search: a metacon line in the conversation (terminal or phone chat), the header caption for
  // a few seconds, and on the phone its voice, when the speaker is on and nobody is on a call.
  const { narrate } = term;
  const narration = useNarration(live && !!health?.narration, {
    speak: () => phone && speakerOn && voice.phase === "idle" && !(liveCall && !liveCall.ended),
    onLine: (n) => narrate(n.text),
  });

  // The robot reports in the terminal when a walk or turn ends, or when it falls.
  const lastMode = useRef<string | null>(null);
  const { note } = term;
  useEffect(() => {
    if (!state || !live) return;
    const before = lastMode.current;
    lastMode.current = state.mode;
    if (!before || before === state.mode) return;
    if (state.mode === "fallen") note("the robot fell over · /reset puts it back", "warn");
    else if (state.mode === "idle" && before === "goto")
      note(`arrived at x ${state.x.toFixed(2)} y ${state.y.toFixed(2)}${(state.target_dist ?? 9) < 0.2 ? " · at the pillar" : ""}`);
    else if (state.mode === "idle" && before === "turn") note(`turned · now facing ${Math.round(state.yaw_deg)}°`);
  }, [state, live, note]);

  const pips = useMemo(() => {
    const pick = (["eye", "overhead"] as View[]).map((v) => (v === main ? "orbit" : v));
    return pick.filter((v) => views.includes(v) && !closed.includes(v));
  }, [main, closed, views]);

  const say = (t: string) => {
    setTermOpen(true);
    void term.send(t);
  };
  const run = async (text: string) => {
    if (!live) {
      toast(`Demo: ${text}`);
      return;
    }
    const r = await sendCommand(text);
    if (r.ok) toast.success(r.command ?? text);
    else toast.error(r.error ?? "Command failed");
  };

  if (howOpen) {
    return (
      <>
        <HowItWorks onClose={closeHow} />
        <Toaster position="top-center" />
      </>
    );
  }

  if (phone) {
    return (
      <>
        <PhoneApp conn={conn} views={views} state={state} lines={term.lines} busy={term.busy} chatOff={term.chatOff} onClear={term.clear}
          apartment={hasApartment ? apartment : null} onApartmentChanged={refreshApartment} onError={(m) => toast.error(m)}
          voice={voice} other={other} knock={knock} speakerOn={speakerOn} onToggleSpeaker={toggleSpeaker} onSend={(t) => void term.send(t)}
          onHowItWorks={openHow} narration={narration.recent} />
        <Toaster position="top-center" />
      </>
    );
  }

  const otherBusy = !!other && !other.ended && voice.phase === "idle";
  const mic = voice.supported ? (
    <>
      <SpeakerToggle on={speakerOn} onToggle={toggleSpeaker}
        className={`h-8 w-8 text-current ${speakerOn ? "opacity-80" : "opacity-40"} hover:opacity-100`} />
      <TalkButton size="sm" phase={voice.phase} disabled={otherBusy || (term.busy && voice.phase === "idle")}
        onPress={voice.press} onRelease={voice.release} onCancel={voice.cancel} />
    </>
  ) : undefined;
  const listening = voice.phase === "starting" || voice.phase === "recording" || voice.phase === "understanding";

  return (
    <div className="stage">
      <div key={main} className="absolute inset-0 crossfade">
        <Stream view={main} live={live} />
      </div>
      <div className="stage-vignette" />
      <div className="accent-line" />

      <header className={`absolute left-0 top-0 flex flex-col gap-3 p-4 sm:flex-row sm:items-start sm:justify-between sm:p-6 ${apt ? "right-[468px]" : "right-0"}`}>
        <Brand conn={conn} />
        {(!apt || other) && (
          <div className="max-sm:hidden sm:absolute sm:left-1/2 sm:top-6 sm:w-[min(560px,calc(100vw-760px))] sm:min-w-[340px] sm:-translate-x-1/2 sm:text-center">
            {other ? <LiveCallCard call={other} /> : <Caption onPick={say} narration={narration.recent} />}
          </div>
        )}
        <div className="flex flex-col items-end gap-3">
          <ViewList views={views} active={main} onPick={setMain} mapOpen={mapOpen} onToggleMap={() => setMapOpen((o) => !o)}
            onRandom={() => run("random")} apartment={hasApartment ? aptOpen : undefined} onToggleApartment={() => setAptOpen((o) => !o)}
            scene={health?.scene} onHowItWorks={openHow} state={state} live={live} onError={(m) => toast.error(m)} />
          {mapOpen && !apt && <MiniMapCard s={state} onGoto={(x, y) => run(`goto ${x} ${y}`)} />}
        </div>
      </header>

      {apt && (
        <div className="absolute bottom-4 right-4 top-4 z-20 w-[min(452px,calc(100vw-32px))]">
          <ApartmentPanel apartment={apt} onClose={() => setAptOpen(false)} onError={(m) => toast.error(m)} onChanged={refreshApartment} />
        </div>
      )}

      <div className="absolute bottom-24 left-3 sm:bottom-6 sm:left-6">
        <CameraCards
          views={pips}
          live={live}
          visible={camsOn}
          onToggle={() => setCamsOn((o) => !o)}
          onPick={(v) => setMain(v)}
          onClose={(v) => setClosed((c) => [...c, v])}
        />
      </div>

      {!apt && (
        <div className="absolute bottom-24 right-3 sm:bottom-6 sm:right-6">
          <QuickActions onRun={run} />
        </div>
      )}

      <div className={apt
        ? "absolute bottom-6 left-[340px] right-[484px] flex min-w-[300px] flex-col gap-3"
        : "absolute inset-x-3 bottom-3 mx-auto flex max-w-[900px] flex-col gap-3 sm:bottom-6 md:max-w-[min(900px,calc(100vw-720px))] md:min-w-[520px]"}>
        {apt && narration.recent && !listening && (
          /* With the apartment panel open the header has no room: metacon's line sits above the chat instead. */
          <div key={narration.recent.id} role="status" aria-live="polite"
            className="glass rise-in max-w-[640px] self-center rounded-2xl px-4 py-2.5 text-center">
            <div className="pixel text-[9px] tracking-[0.25em] text-term-red">metacon · searching</div>
            <div className="mt-0.5 text-[15px] leading-snug">{narration.recent.text}</div>
          </div>
        )}
        {listening && (
          <div className="glass rise-in self-center rounded-full px-4 py-2 text-sm">
            <span className="call-dot mr-2 inline-block align-middle" />
            {voice.interim ? <>“{voice.interim}”</> : phaseText(voice.phase)}
          </div>
        )}
        {termOpen ? (
          <TerminalPanel
            lines={term.lines}
            busy={term.busy}
            chatOff={term.chatOff}
            live={live}
            state={state}
            onSend={say}
            onHistory={(d) => term.history(d)}
            onClose={() => setTermOpen(false)}
            onClear={term.clear}
            mic={mic}
          />
        ) : (
          <ChatBar
            onSend={say}
            onHistory={(d) => term.history(d)}
            busy={term.busy}
            termOpen={termOpen}
            onToggleTerm={() => setTermOpen(true)}
            mic={mic}
            placeholder={voice.supported ? "Tell metacon what to do, or hold the mic to talk…" : undefined}
          />
        )}
      </div>
      <Toaster position="top-center" />
    </div>
  );
}

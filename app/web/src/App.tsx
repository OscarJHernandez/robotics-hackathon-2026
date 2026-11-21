// The page. Made with Lovable (originally src/routes/index.tsx in a TanStack Start project),
// moved into a plain Vite single-page app so the Python server can serve it as static files.
import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { Toaster } from "@/components/ui/sonner";
import { Brand } from "@/components/jethexa/Brand";
import { Caption } from "@/components/jethexa/Caption";
import { ViewList } from "@/components/jethexa/ViewList";
import { CameraCards } from "@/components/jethexa/CameraCards";
import { MiniMapCard } from "@/components/jethexa/MiniMap";
import { QuickActions } from "@/components/jethexa/QuickActions";
import { ChatBar } from "@/components/jethexa/ChatBar";
import { TerminalPanel, useMetacon } from "@/components/jethexa/Terminal";
import { Stream } from "@/components/jethexa/Stream";
import { useHealth, useRobotState } from "@/hooks/useRobotState";
import { sendCommand, VIEWS, type View } from "@/lib/api";

export function App() {
  const { health, conn } = useHealth();
  const state = useRobotState(conn);
  const live = conn === "live";
  const views = health?.views?.length ? health.views : VIEWS;
  const [main, setMain] = useState<View>("chase");
  const [closed, setClosed] = useState<View[]>([]);
  const [camsOn, setCamsOn] = useState(true);
  const [mapOpen, setMapOpen] = useState(false);
  const [termOpen, setTermOpen] = useState(false);
  const term = useMetacon(conn, health?.llm ?? false);

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

  return (
    <div className="stage">
      <div key={main} className="absolute inset-0 crossfade">
        <Stream view={main} live={live} />
      </div>
      <div className="stage-vignette" />
      <div className="accent-line" />

      <header className="absolute inset-x-0 top-0 flex flex-col gap-3 p-4 sm:flex-row sm:items-start sm:justify-between sm:p-6">
        <Brand conn={conn} />
        <div className="max-sm:hidden sm:absolute sm:left-1/2 sm:top-6 sm:-translate-x-1/2">
          <Caption onPick={say} />
        </div>
        <div className="flex flex-col items-end gap-3">
          <ViewList views={views} active={main} onPick={setMain} mapOpen={mapOpen} onToggleMap={() => setMapOpen((o) => !o)}
            onRandom={() => run("random")} />
          {mapOpen && <MiniMapCard s={state} onGoto={(x, y) => run(`goto ${x} ${y}`)} />}
        </div>
      </header>

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

      <div className="absolute bottom-24 right-3 sm:bottom-6 sm:right-6">
        <QuickActions onRun={run} />
      </div>

      <div className="absolute inset-x-3 bottom-3 mx-auto flex max-w-[900px] flex-col gap-3 sm:bottom-6 md:max-w-[min(900px,calc(100vw-720px))] md:min-w-[520px]">
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
          />
        ) : (
          <ChatBar
            onSend={say}
            onHistory={(d) => term.history(d)}
            busy={term.busy}
            termOpen={termOpen}
            onToggleTerm={() => setTermOpen(true)}
          />
        )}
      </div>
      <Toaster position="top-center" />
    </div>
  );
}

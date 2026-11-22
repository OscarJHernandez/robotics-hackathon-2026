// The "How it works" page: a long-form explanation of the end-to-end flow and the three TabPFN models.
// App.tsx owns the #how-it-works hash and renders this instead of the stage. The page needs no API, so it also
// works with the simulator offline. Every number comes from content/howItWorks.ts, which names its source.
import { useEffect, useRef, useState } from "react";
import { ArrowLeft, Maximize2, X } from "lucide-react";
import {
  AVOID_LIMITS, EXPERT_SCREEN, FINAL, FIRST_STUDENT, HARD, LADDER, MEDIA, NIGHT, SEARCH, SEARCH_FAILED, SEARCH_INPUTS,
  SECTIONS, TASKS, WALK, WALK_INPUTS, WALK_OUTPUTS, WALK_VARIANTS, ZONES_CHECKED,
} from "@/content/howItWorks";
import { HexMark } from "../Brand";
import { BarChart } from "./BarChart";
import { Clip } from "./Clip";
import { FlowDiagram } from "./FlowDiagram";
import { Block, Details, Failures, Section, Stats } from "./Section";
import { StatusChip } from "./StatusChip";

const n = (v: number) => v.toLocaleString("en-US");

export function HowItWorks({ onClose }: { onClose: () => void }) {
  const scroller = useRef<HTMLDivElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const [active, setActive] = useState<string>("");
  const [zoom, setZoom] = useState(false);
  const zoomRef = useRef(false);
  zoomRef.current = zoom;

  // Focus the heading when the page opens, so keyboard and screen-reader users start at the top.
  useEffect(() => { heading.current?.focus({ preventScroll: true }); }, []);

  // Escape closes the enlarged picture first, then the page.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      if (zoomRef.current) setZoom(false);
      else onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    const prev = document.title;
    document.title = "How it works · JetHexa Live";
    return () => { document.title = prev; };
  }, []);

  // The section in view, for the nav.
  useEffect(() => {
    const root = scroller.current;
    if (!root || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver((entries) => {
      const seen = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (seen[0]) setActive(seen[0].target.id);
    }, { root, rootMargin: "-20% 0px -65% 0px" });
    SECTIONS.forEach((s) => { const el = document.getElementById(s.id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);

  // Section links scroll inside the page; they never touch the hash, which App.tsx owns.
  const go = (id: string) => {
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    document.getElementById(id)?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  };

  return (
    <div ref={scroller} className="how-page">
      <header className="how-bar">
        <div className="mx-auto flex h-14 max-w-[1180px] items-center gap-3 px-4 sm:px-8">
          <button type="button" onClick={onClose} className="how-back" aria-label="Back to the robot">
            <ArrowLeft className="h-4 w-4" aria-hidden />Robot
          </button>
          <div className="ml-1 flex min-w-0 items-center gap-2">
            <HexMark />
            <span className="truncate text-[15px] font-semibold tracking-tight">JetHexa Live</span>
            <span className="text-video/35" aria-hidden>/</span>
            <span className="truncate text-[15px] text-video/70">How it works</span>
          </div>
        </div>
        <nav aria-label="Sections" className="xl:hidden">
          <div className="mx-auto flex max-w-[1180px] gap-1.5 overflow-x-auto px-4 pb-2.5 [scrollbar-width:none] sm:px-8">
            {SECTIONS.map((s) => (
              <button key={s.id} type="button" onClick={() => go(s.id)} className="how-pill" aria-current={active === s.id}>
                {s.status === "progress" && <span className="how-dot-progress" aria-hidden />}
                {s.nav}
                {s.status === "progress" && <span className="sr-only">(in progress)</span>}
              </button>
            ))}
          </div>
        </nav>
      </header>

      <div className="mx-auto max-w-[1180px] px-5 pb-28 sm:px-8 xl:grid xl:grid-cols-[200px_minmax(0,780px)] xl:justify-between xl:gap-14">
        <nav aria-label="Sections" className="hidden xl:block">
          <div className="sticky top-[88px] pt-16">
            <div className="how-eyebrow mb-4 text-video/45">On this page</div>
            {SECTIONS.map((s) => (
              <button key={s.id} type="button" onClick={() => go(s.id)} className="how-nav-link" aria-current={active === s.id}>
                <span className="min-w-0 flex-1">{s.nav}</span>
                {s.status === "progress" && <span className="apt-status apt-status-warn !h-5 !px-2 !text-[10px]">In progress</span>}
                {s.status === "live" && <span className="how-dot-live" aria-label="live" />}
              </button>
            ))}
          </div>
        </nav>

        <main className="min-w-0 max-w-[780px]">
          {/* Opening */}
          <div className="pt-12 sm:pt-16">
            <div className="how-eyebrow">How it works</div>
            <h1 ref={heading} tabIndex={-1} className="how-h1 mt-4">
              How does the robot choose a room, get there, and decide where to look next?
            </h1>
            <p className="how-lead mt-6">
              When you press <span className="how-term">Find Margarete</span>, the robot guesses where she is most likely
              to be, walks there, and looks. If it does not see her, it updates its guess and tries the next place.
              Three separate models do the thinking. All three are the same kind of model, used in different ways.
            </p>
            <div className="mt-6 inline-flex items-center gap-2 rounded-full border border-video/15 bg-video/5 px-3.5 py-1.5 font-mono text-[12.5px] text-video/85">
              TabPFN in three roles · no fine-tuning
            </div>
            <div className="how-prose mt-8">
              <p>
                <span className="how-term">TabPFN</span> is a model that was pretrained on tables. You hand it example
                rows with their answers (<code>fit()</code>), then ask it about new rows (<code>predict()</code>). It
                answers by comparing the new row with the examples. Its weights never change: making it better means
                giving it better examples, not training it.
              </p>
            </div>
          </div>

          {/* The flow */}
          <Section id="how-flow" num="01" title="The flow of “Find Margarete”"
            lede="A loop: guess, walk, look, update. Two of the three models are in the loop today; the third is being tested to replace the walking part.">
            <FlowDiagram />

            <Block title="One example per task">
              <p className="how-prose mb-5">
                Each model gets its own examples. One row is one situation; the last column is what TabPFN learns to
                predict.
              </p>
              <table className="how-table how-table-stack">
                <thead><tr><th scope="col">Task</th><th scope="col">TabPFN</th><th scope="col">One example row</th><th scope="col">The answer in that row</th></tr></thead>
                <tbody>
                  {TASKS.map((t) => (
                    <tr key={t.task}>
                      <td>{t.task}</td>
                      <td data-label="TabPFN" className="mono">{t.tabpfn}</td>
                      <td data-label="One example row">{t.row}</td>
                      <td data-label="The answer">{t.answer}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Block>

            <div className="how-note mt-8">
              <div className="text-[15px] font-semibold">fit(), not training</div>
              <p className="mt-1.5 text-[15px] leading-relaxed text-video/75">
                In this project “training” TabPFN only ever means calling <code className="font-mono text-[13px]">fit()</code> on
                recorded examples. No gradient steps, no fine-tuned weights. To make a model better we change what goes
                into <code className="font-mono text-[13px]">fit()</code>: better examples, better columns, a better choice of rows.
              </p>
            </div>
          </Section>

          {/* 1. Where is she */}
          <Section id="how-search" num="02" title="Where is she?" status="live"
            lede={<>A probability for each of the five zones of the flat, and for “out”, at the moment the search starts. The
              app uses it by default; a counting Bayes model takes over at night or when TabPFN is not available.</>}>
            <Block title="Inputs">
              <p className="how-prose mb-5">
                {SEARCH.features} columns in six groups. The last group is the answer of a simple counting model, given to
                TabPFN as a hint.
              </p>
              <table className="how-table how-table-stack">
                <thead><tr><th scope="col">Group</th><th scope="col" className="num">Columns</th><th scope="col">What they say</th></tr></thead>
                <tbody>
                  {SEARCH_INPUTS.map((g) => (
                    <tr key={g.group}><td>{g.group}</td><td data-label="Columns" className="num">{g.n}</td><td data-label="What they say">{g.what}</td></tr>
                  ))}
                </tbody>
              </table>
            </Block>

            <Block title="Examples">
              <div className="how-prose">
                <p>
                  <strong>{n(SEARCH.examples)} synthetic check-ins.</strong> Each comes from a generated resident with{" "}
                  {SEARCH.journalDays.join(", ").replace(/, (\d+)$/, " or $1")} days of journal, living in a generated flat.
                  The answer in each row is where that resident really was. The {SEARCH.classes} possible answers are the
                  five zones and “out”.
                </p>
                <p>
                  Margarete’s own journal is not an example. It reaches the model through the columns of the row being
                  asked about, so editing her journal in the app changes the belief at once, without a new{" "}
                  <code>fit()</code>.
                </p>
              </div>
            </Block>

            <Block title="How it decides">
              <ol className="how-steps how-prose">
                <li><strong>Belief.</strong> TabPFN gives a probability for each zone and for “out”.</li>
                <li><strong>Walk.</strong> The robot goes to the viewing spot of the most likely zone.</li>
                <li><strong>Look.</strong> A simulated sighting: it misses her {SEARCH.missRate}% of the time, even in the
                  right zone, and never sees her where she is not.</li>
                <li><strong>Update.</strong> If she was not seen, Bayes’ rule lowers that zone and raises the others. Then
                  back to step 2. The search stops when she is found, when “out” reaches {SEARCH.stopOut}%, or after{" "}
                  {SEARCH.maxLooks} looks.</li>
              </ol>
              <Details summary="The update, in one line">
                <div className="how-prose text-[15px]">
                  <p>
                    For the zone just looked at, with belief <em>p</em> and miss rate <em>m</em>:
                  </p>
                  <p className="my-3 overflow-x-auto rounded-xl bg-video/5 px-4 py-3 font-mono text-[14px] text-video/90">
                    P(zone | not seen) = p · m / (1 − p · (1 − m))
                  </p>
                  <p>
                    Every other class is divided by the same 1 − p · (1 − m). Because a look can miss, a zone is never
                    ruled out completely, so the robot can come back to it when the others turn out empty.
                  </p>
                </div>
              </Details>

              <figure className="how-figure">
                <button type="button" onClick={() => setZoom(true)} className="group relative block w-full overflow-hidden rounded-xl bg-white"
                  aria-label="Enlarge the picture of a search with a missed sighting">
                  <img src={MEDIA.searchLoop.src} alt="Four looks of one search: the living room starts at 51%, the robot misses her there, the belief moves to the kitchen and the dining area, and the robot returns to the living room and finds her at look 4."
                    className="block h-auto w-full" loading="lazy" width={1600} height={900} />
                  <span className="glass absolute right-2.5 top-2.5 flex h-8 items-center gap-1.5 rounded-full px-3 text-[12px] opacity-90 group-hover:opacity-100">
                    <Maximize2 className="h-3.5 w-3.5" aria-hidden />Enlarge
                  </span>
                </button>
                <figcaption>
                  <div className="how-caption">
                    A missed sighting, and the robot comes back. She is in the living room (51%), but the first look
                    misses her. The living room drops to 17%, not to zero, so after the kitchen and the dining area turn
                    out empty the robot returns and finds her at look 4.
                  </div>
                  <div className="how-source">Source: {MEDIA.searchLoop.source}</div>
                </figcaption>
              </figure>
            </Block>

            <Block title="Result">
              <Stats items={[
                { value: `${SEARCH.top1}%`, label: <>most likely answer right, over all {n(SEARCH.checkIns)} test check-ins (“out” included)</> },
                { value: `${SEARCH.zonesChecked}`, label: <>zones checked until found with a perfect look, over the {n(SEARCH.atHome)} check-ins with her at home</> },
                { value: `${SEARCH.foundWithMisses}%`, label: <>found with the {SEARCH.missRate}% miss rate, {SEARCH.looksWhenFound} looks on average when found</> },
              ]} />
              <figure className="how-figure">
                <div className="mb-3 px-1">
                  <div className="text-[15px] font-semibold">Zones checked until she is found</div>
                  <div className="mt-0.5 text-[13px] text-video/55">Fewer is better · perfect look · the same {n(SEARCH.atHome)} at-home check-ins</div>
                </div>
                <div className="px-1">
                  <BarChart title="Zones checked until she is found" bars={ZONES_CHECKED} max={3.5} ticks={[0, 1, 2, 3]}
                    format={(v) => (Number.isInteger(v) ? String(v) : v.toFixed(2))}
                    summary="TabPFN 1.58, counting model 1.54, last heard first 2.12, nearest zone first 3.30." />
                </div>
                <figcaption className="how-caption">
                  {SEARCH.testResidents} test residents the models never saw, 28 days of journal each. TabPFN is level with
                  the counting model ({SEARCH.countingFound}% found with misses against TabPFN’s {SEARCH.foundWithMisses}%)
                  and well ahead of the simple rules.
                </figcaption>
              </figure>
            </Block>

            <Block title="What did not work">
              <Failures items={SEARCH_FAILED} />
            </Block>
          </Section>

          {/* 2. Walking from a to b */}
          <Section id="how-walk" num="03" title="Walking from a to b" status="live"
            lede={<>Every second the controller picks a stride and a turn for the six legs. TabPFN does not pick them directly:
              it is a <span className="how-term">dynamics model</span>, which predicts what a command will do to the body. The
              controller then chooses the command whose result is best.</>}>
            <Clip src={MEDIA.a2b.src} poster={MEDIA.a2b.poster} label="The robot walking from a to the red pillar"
              caption="The default controller walking to the red pillar on the empty floor: it turns toward the pillar, then walks almost straight to it. Inset: the robot’s camera."
              source={MEDIA.a2b.source} />

            <Block title="What TabPFN predicts">
              <div className="grid gap-4 sm:grid-cols-[1fr_auto_1fr] sm:items-stretch">
                <div className="rounded-2xl border border-video/10 bg-video/[0.03] p-4">
                  <div className="apt-title mb-3">7 inputs</div>
                  <ul className="space-y-2.5">
                    {WALK_INPUTS.map((i) => (
                      <li key={i.name}><div className="font-mono text-[12.5px] text-video/90">{i.name}</div><div className="text-[13.5px] text-video/62">{i.what}</div></li>
                    ))}
                  </ul>
                </div>
                <div className="flex items-center justify-center font-mono text-[13px] text-video/45" aria-hidden>
                  <span className="sm:hidden">↓</span><span className="max-sm:hidden">→</span>
                </div>
                <div className="rounded-2xl border border-[oklch(0.78_0.13_172/35%)] bg-[oklch(0.78_0.13_172/7%)] p-4">
                  <div className="apt-title mb-3">3 outputs · over the next 1 s gait cycle</div>
                  <ul className="space-y-2.5">
                    {WALK_OUTPUTS.map((o) => (
                      <li key={o.name}><div className="font-mono text-[12.5px] text-video/90">{o.name}</div><div className="text-[13.5px] text-video/62">{o.what}</div></li>
                    ))}
                  </ul>
                  <p className="mt-3 text-[12.5px] leading-relaxed text-video/55">One TabPFN regressor per output.</p>
                </div>
              </div>
            </Block>

            <Block title="Examples">
              <div className="how-prose">
                <p>
                  The robot walked at random on an empty floor: {WALK.episodes} episodes, {n(WALK.cycles)} gait cycles. Every
                  full cycle without a fall is an example, {n(WALK.examples)} in all. {WALK.context} of them go into{" "}
                  <code>fit()</code>, which takes about {WALK.fitSeconds} s. More did not help (see below).
                </p>
              </div>
            </Block>

            <Block title="How it decides">
              <ol className="how-steps how-prose">
                <li>Every second, ask TabPFN about all {WALK.candidates} candidate commands ({WALK.strides} strides × {WALK.turns} turns) in one batch.</li>
                <li>For each predicted end pose, score where b would then be: how far ahead, how far to the side (counted
                  double), plus a cost for not facing b that grows with the distance to b.</li>
                <li>Send the command with the best score to the legs. Repeat.</li>
              </ol>
              <p className="how-prose mt-5">
                In the app, an <strong>A* route</strong> (a shortest path on a grid) goes around the furniture the app
                knows about, and the controller walks to each point on it. The controller itself does not sense obstacles.
              </p>
            </Block>

            <Block title="Result">
              <Stats items={[
                { value: WALK.reached, label: "held-out targets reached on an empty floor" },
                { value: WALK.directness.toFixed(2), label: <>directness (straight-line distance ÷ path length), level with the hand-written baseline’s {WALK.baseline.toFixed(2)}</> },
              ]} />
            </Block>

            <Block title="What did not work">
              <Failures items={[{
                title: "Scoring only where b ends up",
                body: `Without the facing cost, the robot reached only ${WALK.noFacing} targets. It looks one second ahead, and turning on the spot does not bring a target behind it any closer, so it never turned around.`,
              }]} />
              <Details summary="More variants">
                <div className="space-y-4">
                  {WALK_VARIANTS.map((v) => (
                    <div key={v.title}>
                      <div className="text-[14.5px] font-semibold text-video/90">{v.title}</div>
                      <p className="mt-1 text-[14.5px] leading-relaxed text-video/68">{v.body}</p>
                    </div>
                  ))}
                </div>
              </Details>
            </Block>
          </Section>

          {/* 3. Avoiding furniture */}
          <Section id="how-avoid" num="04" title="Avoiding furniture" status="progress" statusDetail="tested in simulation"
            lede={<>A student model that drives to the target through a furnished room from its own sensors, picking stride and
              turn every second. It would replace the route and the controller above. <strong className="text-video">It is not in
              the live app.</strong></>}>
            <div className="how-note-progress mt-6 text-[14.5px] leading-relaxed text-video/80">
              Tested only in simulation, on a branch that is not merged yet (pull request #22). Nothing on this part runs in
              the demo you just saw.
            </div>

            <Block title="DAgger, in four steps">
              <p className="how-prose mb-4">
                <span className="how-term">DAgger</span> (dataset aggregation) is a way to teach a student by imitation
                without it only ever seeing the teacher’s good driving.
              </p>
              <ol className="how-steps how-prose">
                <li><strong>An expert drives.</strong> It knows the true map: a shortest path to the goal over the real
                  furniture, kept 0.32 m clear, followed by looking 0.4 m ahead along it. Every second is a row.</li>
                <li><strong>The student gets the rows.</strong> TabPFN’s <code>fit()</code> on everything so far. The
                  student sees only the robot’s sensors, never the map.</li>
                <li><strong>The student drives, the expert still labels.</strong> In later rounds the student drives, and
                  for every state it gets into, the expert says what it would have done.</li>
                <li><strong>Add the rows and repeat.</strong> So the student also learns how to get out of situations
                  the expert would never have got into. It is judged only by driving alone.</li>
              </ol>
            </Block>

            <Block title="The first student: camera only">
              <div className="how-prose">
                <p>
                  The camera picture is cut into a {FIRST_STUDENT.grid} grid of small squares, called{" "}
                  <span className="how-term">patches</span>. A frozen image model (DINOv2 ViT-S/14) describes each patch,
                  and a small separately trained classifier names it floor, wall, obstacle or target ({FIRST_STUDENT.probeAccuracy}%
                  right on simulated layouts it never saw). Class and depth per patch, plus the goal, gave{" "}
                  {n(FIRST_STUDENT.columns)} columns.
                </p>
                <p>
                  Round 0: the expert drove {n(FIRST_STUDENT.episodes)} episodes on {FIRST_STUDENT.machines} machines in
                  about {FIRST_STUDENT.minutes} minutes, {n(FIRST_STUDENT.rows)} rows. <code>fit()</code> took{" "}
                  {FIRST_STUDENT.fitSeconds} s. With the expert executing about half the commands, it reached 100 of 100 goals. Driving alone in held-out layouts, it
                  crashed <strong>{FIRST_STUDENT.crashed}</strong> times.
                </p>
              </div>
              <Clip src={MEDIA.first.src} poster={MEDIA.first.poster} label="The first student crashing into a box beside it"
                caption={<>First student, camera only: collision. It walks past the target while the expert’s label keeps
                  saying “stop and turn on the spot” (stride 0), and runs into the blue box beside it. Bottom right: the student’s
                  command, which is executed, and the expert’s, which is only the label.</>}
                source={MEDIA.first.source} />
            </Block>

            <Block title="Why it was hard">
              <ul className="space-y-3">
                {HARD.map((h) => (
                  <li key={h} className="flex gap-3 text-[15.5px] leading-relaxed text-video/75">
                    <span className="mt-[11px] h-1 w-1 shrink-0 rounded-full bg-video/50" aria-hidden />{h}
                  </li>
                ))}
              </ul>
            </Block>

            <Block title="The night run">
              <p className="how-prose mb-5">
                On the night of 22 November a research loop on two machines tested one idea after another, each judged by the student
                driving alone on the same held-out layouts. The big step was a <span className="how-term">lidar</span>: a
                sensor that measures the distance to the nearest surface all around the robot. Here it is simulated with 36
                rays, one every 10°.
              </p>
              <ol className="how-timeline">
                {NIGHT.map((t) => <li key={t.time}><time>{t.time}</time><p>{t.text}</p></li>)}
              </ol>
              <figure className="how-figure">
                <div className="mb-3 px-1">
                  <div className="text-[15px] font-semibold">Goals reached, student driving alone</div>
                  <div className="mt-0.5 text-[13px] text-video/55">The same 200 screening layouts for every step · the expert reaches {EXPERT_SCREEN}%</div>
                </div>
                <div className="px-1">
                  <BarChart title="Goals reached, student driving alone" inside
                    bars={LADDER.map((l) => ({ label: l.label, note: `${l.note} · ${l.collisions.toFixed(1)}% collisions`, value: l.value, focus: l.focus }))}
                    max={100} ticks={[0, 25, 50, 75, 100]} format={(v) => (Number.isInteger(v) && v % 25 === 0 ? `${v}%` : `${v.toFixed(1)}%`)}
                    reference={{ value: EXPERT_SCREEN, label: `Expert ${EXPERT_SCREEN}%` }}
                    summary="Camera patch grid 63.5%, plus lidar 90.5%, plus all rows 94.5%, plus its own plan 98.5%; the expert reaches 100%." />
                </div>
                <figcaption className="how-caption">
                  The lidar step also made the camera inputs compact (606 → 114 columns), so not all of that gain is the
                  lidar. <strong className="text-video/90">Its own plan:</strong> lidar hits collected during the episode
                  become a map, the expert’s planner finds a path on it, and the direction of the point 0.4 m along that
                  path is two more columns. Only the lidar, the robot’s pose and the goal are used. Later DAgger rounds
                  added nothing on top.
                </figcaption>
              </figure>
            </Block>

            <Block title="The final check">
              <div className="grid gap-8 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
                <div>
                  <p className="how-prose mb-4 text-[15px]">200 fresh layouts, used once, the student driving alone for up to 300 steps.</p>
                  <table className="how-table">
                    <thead><tr><th scope="col">Run</th><th scope="col" className="num">Goals</th><th scope="col" className="num">Collisions</th></tr></thead>
                    <tbody>
                      {FINAL.map((f) => (
                        <tr key={f.run}>
                          <td>{f.run}</td>
                          <td className="num">{f.goal}{f.ci && <div className="text-[11.5px] font-normal text-video/45">95% CI {f.ci}</div>}</td>
                          <td className="num">{f.collisions}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="how-note-progress">
                  <div className="apt-title mb-2.5 !text-[oklch(0.9_0.1_80)]">Read these numbers with</div>
                  <ul className="space-y-2">
                    {AVOID_LIMITS.map((l) => <li key={l} className="text-[14px] leading-relaxed text-video/78">{l}</li>)}
                  </ul>
                </div>
              </div>
              <Clip src={MEDIA.student.src} poster={MEDIA.student.poster} label="The current student reaching the target past furniture"
                caption="The current student (v29) driving alone on a final-test layout: it goes round a long piece of furniture, close along its side, and reaches the red pillar."
                source={MEDIA.student.source} />
            </Block>
          </Section>

          {/* Limits and next */}
          <Section id="how-next" num="05" title="Limits and what is next">
            <div className="grid gap-4 sm:grid-cols-3">
              {[
                { t: "Where is she", s: "live" as const, items: ["The look is simulated: a fixed 20% miss rate, no false alarms.", "Residents, routines and habits are generated: the numbers say the method works on this data, not how well it works for a real person."] },
                { t: "Walking a → b", s: "live" as const, items: ["Tested on an empty floor.", "It avoids furniture only because the route goes around what the app knows."] },
                { t: "Avoiding furniture", s: "progress" as const, items: ["Simulation only, with the true pose and an assumed lidar.", "Not merged, not in the app, not on the real robot."] },
              ].map((c) => (
                <div key={c.t} className="apt-card">
                  <div className="mb-3 text-[14.5px] font-semibold">{c.t}</div>
                  <StatusChip status={c.s} />
                  <ul className="mt-3 space-y-2">
                    {c.items.map((i) => <li key={i} className="text-[13.5px] leading-relaxed text-video/70">{i}</li>)}
                  </ul>
                </div>
              ))}
            </div>
            <Block title="Next">
              <ol className="how-steps how-prose">
                <li>Merge pull request #22, the avoiding-furniture work.</li>
                <li>Wire the student into the app, in place of the route and the controller.</li>
                <li>Check the lidar and the odometry on the real robot, and try a denser lidar for thin table legs.</li>
              </ol>
            </Block>
            <div className="mt-14 flex justify-center">
              <button type="button" onClick={onClose} className="how-back !h-11 !px-5">
                <ArrowLeft className="h-4 w-4" aria-hidden />Back to the robot
              </button>
            </div>
          </Section>
        </main>
      </div>

      {zoom && (
        <div className="how-zoom" role="dialog" aria-modal="true" aria-label="The search picture, enlarged" onClick={() => setZoom(false)}>
          <img src={MEDIA.searchLoop.src} alt="" className="max-h-full max-w-full rounded-lg bg-white object-contain" />
          <button type="button" autoFocus onClick={() => setZoom(false)} aria-label="Close the picture"
            className="glass-btn absolute right-[max(16px,env(safe-area-inset-right))] top-[max(16px,env(safe-area-inset-top))] h-10 w-10">
            <X className="h-5 w-5" />
          </button>
        </div>
      )}
    </div>
  );
}

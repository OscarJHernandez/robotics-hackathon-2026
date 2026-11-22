import type { ReactNode } from "react";
import { ChevronRight } from "lucide-react";
import type { Status } from "@/content/howItWorks";
import { StatusChip } from "./StatusChip";

/** One section of the page: number and status, heading, a short lede, then its blocks. */
export function Section({ id, num, title, status, statusDetail, lede, children }: {
  id: string; num: string; title: string; status?: Status | undefined; statusDetail?: string | undefined;
  lede?: ReactNode; children: ReactNode;
}) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} className="how-section">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <span className="how-eyebrow">{num}</span>
        {status && <StatusChip status={status} detail={statusDetail} />}
      </div>
      <h2 id={`${id}-h`} className="how-h2 mt-3">{title}</h2>
      {lede && <div className="how-lead mt-4">{lede}</div>}
      {children}
    </section>
  );
}

/** A titled block inside a section ("Inputs", "Result", ...): the same anatomy in every model section. */
export function Block({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <h3 className="how-h3">{title}</h3>
      {children}
    </div>
  );
}

/** An expandable detail ("The update", "More variants"). */
export function Details({ summary, children }: { summary: string; children: ReactNode }) {
  return (
    <details className="how-details">
      <summary><ChevronRight className="how-chev h-4 w-4" aria-hidden />{summary}</summary>
      <div className="how-details-body">{children}</div>
    </details>
  );
}

/** Things that failed, each with a red rule on the left. */
export function Failures({ items }: { items: { title: string; body: string }[] }) {
  return (
    <div className="space-y-3">
      {items.map((f) => (
        <div key={f.title} className="how-fail">
          <div className="text-[15px] font-semibold text-video/95">{f.title}</div>
          <p className="mt-1 text-[15px] leading-relaxed text-video/70">{f.body}</p>
        </div>
      ))}
    </div>
  );
}

/** A row of headline numbers, each with what it counts. */
export function Stats({ items }: { items: { value: string; label: ReactNode }[] }) {
  return (
    <div className={`how-stats grid-cols-1 ${items.length >= 3 ? "sm:grid-cols-3" : "sm:grid-cols-2"}`}>
      {items.map((s) => (
        <div key={s.value} className="how-stat">
          <div className="how-stat-v">{s.value}</div>
          <div className="how-stat-l">{s.label}</div>
        </div>
      ))}
    </div>
  );
}

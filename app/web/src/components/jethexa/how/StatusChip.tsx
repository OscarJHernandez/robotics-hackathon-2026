import type { Status } from "@/content/howItWorks";

/** "Live" (green) or "In progress" (amber), with a dot and words, never colour alone. */
export function StatusChip({ status, detail }: { status: Status; detail?: string }) {
  const live = status === "live";
  return (
    <span className={`apt-status ${live ? "apt-status-good" : "apt-status-warn"}`}>
      <span className={live ? "how-dot-live" : "how-dot-progress"} aria-hidden />
      {live ? "Live in the app" : "In progress"}
      {detail && <span className="font-normal opacity-80">· {detail}</span>}
    </span>
  );
}

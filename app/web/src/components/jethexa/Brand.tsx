import type { Conn } from "@/hooks/useRobotState";

export function HexMark() {
  return (
    <svg viewBox="0 0 32 32" className="h-7 w-7 drop-shadow" aria-hidden>
      <path d="M16 2 28 9v14l-12 7-12-7V9z" className="fill-primary" />
      <path d="M16 9l6 3.5v7L16 23l-6-3.5v-7z" className="fill-primary-foreground/90" />
    </svg>
  );
}

export function Brand({ conn }: { conn: Conn }) {
  const label = conn === "live" ? "LIVE" : conn === "demo" ? "DEMO" : "OFFLINE";
  return (
    <div className="flex items-center gap-3">
      <HexMark />
      <span className="text-shadow-soft text-lg font-semibold tracking-tight text-video">JetHexa Live</span>
      <span className="glass pixel flex h-6 items-center gap-1.5 rounded-full px-2.5 text-[10px]">
        <span className={conn === "live" ? "live-dot" : "h-1.5 w-1.5 rounded-full bg-video/50"} />{label}
      </span>
    </div>
  );
}

import { useState } from "react";
import { ArrowUp, MoreHorizontal, RotateCcw, RotateCw, Square, RefreshCw } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

const actions = [
  ["Forward 0.5 m", "forward 0.5", ArrowUp], ["Turn left 90°", "turn 90", RotateCcw],
  ["Turn right 90°", "turn -90", RotateCw], ["Stop", "stop", Square], ["Reset", "reset", RefreshCw],
] as const;

export function QuickActions({ onRun }: { onRun: (t: string) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <TooltipProvider delayDuration={150}>
      <div className="flex flex-col items-end gap-2">
        <div className={`flex-col gap-2 ${open ? "flex" : "hidden"} sm:flex`}>
          {actions.map(([label, cmd, Icon]) => (
            <Tooltip key={cmd}>
              <TooltipTrigger asChild>
                <button onClick={() => onRun(cmd)} aria-label={label}
                  className={`glass-btn h-10 w-10 ${cmd === "stop" ? "glass-btn-danger" : ""}`}><Icon className="h-4 w-4" /></button>
              </TooltipTrigger>
              <TooltipContent side="left">{label}</TooltipContent>
            </Tooltip>
          ))}
        </div>
        <button onClick={() => setOpen((o) => !o)} aria-label="Quick actions" className="glass-btn h-10 w-10 sm:hidden">
          <MoreHorizontal className="h-4 w-4" />
        </button>
      </div>
    </TooltipProvider>
  );
}

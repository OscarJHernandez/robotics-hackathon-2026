import type { ChatEvent, RobotState } from "./api";

export function demoState(t: number): RobotState {
  const a = t * 0.25;
  const x = Math.cos(a) * 1.2, y = Math.sin(a) * 1.2;
  const yaw = ((a * 180) / Math.PI + 90) % 360;
  return { mode: "goto", x, y, height: 0.08, yaw_deg: yaw > 180 ? yaw - 360 : yaw,
    target_x: Math.cos(a + 0.4) * 1.2, target_y: Math.sin(a + 0.4) * 1.2, sim_time: t };
}

const answers = [
  { text: "Sure. Walking forward half a metre now [[forward 0.5]]. The path ahead looks clear.", cmd: "forward 0.50" },
  { text: "Turning around for you [[turn 180]]. I'll stop once I'm facing the other way.", cmd: "turn 180.0" },
  { text: "I can see the red pillar ahead and slightly left. Heading there [[goto 1.2 0.8]].", cmd: "goto 1.20 0.80" },
];
let n = 0;

export async function demoChat(onEvent: (e: ChatEvent) => void) {
  const a = answers[n++ % answers.length]!;
  await new Promise((r) => setTimeout(r, 600));
  for (const ch of a.text) {
    onEvent({ type: "token", text: ch });
    await new Promise((r) => setTimeout(r, 18));
  }
  onEvent({ type: "command", text: a.cmd, ok: true, error: null });
  onEvent({ type: "done" });
}

# JetHexa Live: the web app

A web page that shows the simulated JetHexa walking live and a terminal-style chat with a metacon LLM that can
drive it. Everyone who opens the page sees and steers the same robot.

```bash
just app-build     # once, and after front-end changes: npm install + build app/web into app/web/dist
just app           # http://127.0.0.1:8790
```

## Pieces

| Path | What it is |
|---|---|
| `app/robot.py` | `LiveRobot`: one thread steps MuJoCo in real time, steers the robot (goto, forward, turn, stop, reset) and renders JPEG pictures of an outside view and the robot camera |
| `app/llm.py` | Chat with the metacon gateway (`http://metacon:4000/v1`, model `qwen3:8b`); the model moves the robot by writing `[[goto 1 0.5]]` in its answer |
| `app/server.py` | Standard-library HTTP server: the API under `/api` (documented at the top of the file) and the built front end |
| `app/web/` | The front end: React, Vite, Tailwind, built with Lovable (project "JetHexa Live Studio"). Full-screen view with switchable cameras (chase, orbit, overhead, robot eye), a chat bar and a red/green terminal. Without the API it runs a demo mode |

The steering is a simple rule (turn toward the target, slow down when facing away), the same as the `baseline`
policy in `nav/policy.py`, so the app needs no TabPFN and no recorded data.

## The metacon key

The server reads the gateway key from `$METACON_LLM_KEY`, else from the macOS Keychain (service `metacon-llm`, see
the metacon skill). It stays on the server: the browser never sees it, and the chat runs through the server.
Without a key the page still works and the chat says it is offline. The machine needs Tailscale to reach metacon.

## Publishing at alpharelay.dev

The app runs on Oscar's Mac and is published through its Cloudflare tunnel with the `alpharelay-deploy` skill:
`relay deploy jethexa 8790` (live at https://jethexa.alpharelay.dev since 2026-11-21). The server must be running on
the Mac; it does not restart after a reboot. It is then public without a login, and every chat uses Oscar's metacon key, so the
server limits each visitor (by Cloudflare's `Cf-Connecting-Ip`) to 20 chats per 10 minutes and 60 commands per
minute, and runs at most 2 chats at once.

## Settings

`METACON_URL` (default `http://metacon:4000/v1`), `METACON_MODEL` (default `qwen3:8b`), `METACON_LLM_KEY`.

## Front-end development

`cd app/web && npm run dev` serves the page with hot reload on http://localhost:5173 and passes `/api` to the
Python server on :8790 (start it with `just app`). Each picture stream is about 0.5 MB/s, so a visitor with three
views open uses about 1.5 MB/s of the Mac's upload; views nobody watches are not rendered.

# The app on vendetta: https://jethexa-vendetta.alpharelay.dev

The web app (`app/`) runs on vendetta, Oscar's always-on Mac, and is published through its own Cloudflare tunnel
`jethexa-vendetta`. It is separate from https://jethexa.alpharelay.dev, which the laptop serves with the
`alpharelay-deploy` skill. Public, no login. Set up 2026-11-22 at `main` `66c94be` (PR 27); now at `76455c3` (PR 31).
The student uses the TabPFN API (the default backend, Oscar's choice): ready about 20 s after a start, then 0.6 to
2 s and 10,000 credits per decision, from the same account as the laptop copy. `--student-backend local` (TabPFN on
vendetta's GPU, 5 min to fit, then about 0.35 s per decision) exists but is only used if Oscar asks for it.

| Piece | Where on vendetta |
|---|---|
| Code | git worktree `~/Projects/robotics-hackathon-2026/.claude/worktrees/deploy-jethexa-vendetta` (detached at one commit) |
| App | launchd agent `dev.alpharelay.jethexa-vendetta`: `app/server.py --port 8795`, logs `~/Library/Logs/jethexa-vendetta.*.log` |
| Tunnel | launchd agent `dev.alpharelay.jethexa-vendetta-tunnel`, config `~/.cloudflared/jethexa-vendetta.yml` |
| Data | `data/nav/controllers/default`, `data/nav/dagger/v2/student/v29` and `.env` (`TABPFN_TOKEN`) in the main checkout, found from the worktree by `nav/common.py` and `nav/dagger/envfile.py` |
| Chat key | Keychain item `metacon-llm` (account = user name); without it chat is off |

The files here are templates: replace `__HOME__` (and `__TUNNEL_ID__` in the yml) before copying them into place.

## Check it

```bash
ssh vendetta 'curl -s http://127.0.0.1:8795/api/health'
ssh vendetta 'launchctl list | grep jethexa-vendetta'
```

## Update to a newer commit

```bash
ssh vendetta
cd ~/Projects/robotics-hackathon-2026/.claude/worktrees/deploy-jethexa-vendetta
git fetch origin main && git checkout --detach origin/main
export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
uv sync --group app --group ml && (cd app/web && npm ci && npm run build)
launchctl kickstart -k gui/$(id -u)/dev.alpharelay.jethexa-vendetta
```

New gitignored data a commit needs (controllers, student exports) must be copied into the main checkout's
`data/nav/` by hand.

## Take it down

```bash
launchctl bootout gui/$(id -u)/dev.alpharelay.jethexa-vendetta-tunnel
launchctl bootout gui/$(id -u)/dev.alpharelay.jethexa-vendetta
cloudflared tunnel --config ~/.cloudflared/jethexa-vendetta.yml delete jethexa-vendetta
```

Then delete the DNS record `jethexa-vendetta.alpharelay.dev` in the Cloudflare dashboard, and remove the worktree
with `git worktree remove`.

## cloudflared trap

vendetta also has `~/.cloudflared/config.yml` (the `voice` tunnel). Without `--config`, `cloudflared tunnel route
dns <name> <host>` and `tunnel info <name>` use that file's tunnel, not the name given. Always pass
`--config ~/.cloudflared/jethexa-vendetta.yml`.

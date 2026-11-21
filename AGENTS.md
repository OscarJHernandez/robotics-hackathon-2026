# Agent instructions

These rules apply to every coding agent working in this repo (Claude Code, Codex, Cursor, Copilot and others).
Humans: see [CONTRIBUTING.md](CONTRIBUTING.md).

## Before you start

1. Read [docs/LEARNINGS.md](docs/LEARNINGS.md). It holds what contributors and their agents have already
   learned about this project. Apply it, and don't repeat a dead end that is recorded there.
2. Update your local `main`: `git switch main && git pull`.

## How to change things

- Never commit or push to `main`. It is protected; a direct push is rejected.
- Work on a new branch off `main`, named `<type>/<short-name>`, where type is `feat`, `fix`, `docs` or `chore`
  (for example `feat/gripper-calibration`).
- Keep each pull request to one topic and small enough to review in a few minutes.
- Open the pull request with `gh pr create` and fill in the template. Pull requests are merged by squash.
- Never commit secrets, `.env` files or large binary files (robot logs, bag files, model weights). The repo is public.

## Before you finish

Add what you learned to [docs/LEARNINGS.md](docs/LEARNINGS.md), in the same pull request as the work:

- **Learnings**: anything non-obvious that would save the next person or agent time. Examples: a setup step
  that is easy to miss, a hardware quirk, an approach that failed and why, a correction the user gave you.
- **Suggestions**: improvements you noticed but did not make. Add them under "Suggestions (open)".

Check that the point is not already in the document, and follow the entry format at the top of that file.
If there is nothing new to record, say so in the pull request description.

Claude Code users: the `contribute` and `record-learning` skills in `.claude/skills/` walk through these steps.

## Stack and commands

Python with [MuJoCo](https://mujoco.readthedocs.io/). The simulated JetHexa is in `sim/`; the full guide is
[docs/simulation.md](docs/simulation.md).

- Install: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
- Meshes (once, not committed): `.venv/bin/python sim/fetch_meshes.py`. Without them the sim falls back to
  simple shapes.
- Run: `.venv/bin/python sim/demo.py` (no window), or `.venv/bin/mjpython sim/demo.py --view` for the live
  window (macOS needs `mjpython`; elsewhere use `python`).
- Test: `.venv/bin/python -m unittest discover -s tests -v`
- Lint: not set up yet.

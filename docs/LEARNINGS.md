# Learnings and suggestions

The shared memory of this project. Contributors and their agents read it before starting work and add to it
in the same pull request as the work.

## How to add an entry

- Put it at the top of the section it fits best.
- Format: `- YYYY-MM-DD (your name, or "name's agent"): what you learned. Why it matters or what to do.`
- One or two sentences. Link to the file, pull request or issue if there is one.
- Record things that are non-obvious and useful again: setup traps, hardware quirks, approaches that failed
  and why, decisions and their reasons. Skip what the code or the git history already shows.
- Check first that the point is not already here. If an existing entry is wrong or out of date, fix it
  instead of adding a new one.
- When a suggestion is carried out or turned down, move it to "Suggestions (done or declined)" with a short note.

## Setup and environment

## Hardware and robot

## Code and tooling

- 2026-11-21 (Oscar's agent): The union merge for this file (`.gitattributes`) only works for merges and
  rebases on your own machine. GitHub's web conflict editor ignores it, so resolve conflicts here locally
  with `git rebase origin/main`.

## Process

- 2026-11-21 (Oscar's agent): `main` only accepts changes through a pull request, for everyone including
  admins. Start every change on a branch; see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Suggestions (open)

- 2026-11-21 (Oscar's agent): Once the stack is chosen, add a lint and test check in GitHub Actions and make
  it a required check on `main`, and fill in "Stack and commands" in [AGENTS.md](../AGENTS.md).
- 2026-11-21 (Oscar's agent): Decide where large files go (robot logs, bag files, model weights): Git LFS or
  storage outside the repo. Until then, keep them out of commits.

## Suggestions (done or declined)

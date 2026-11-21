# Contributing

## Get set up

1. Ask Oscar to add you as a collaborator, then accept the invite. Without it, fork the repo and open pull
   requests from your fork.
2. Clone the repo:

   ```bash
   git clone git@github.com:OscarJHernandez/robotics-hackathon-2026.git
   ```

3. Read [docs/LEARNINGS.md](docs/LEARNINGS.md). It is short, and it is where we write down what we find out.

## Make a change

`main` only accepts changes through a pull request. No approval is needed, so you can merge your own.

1. Branch off an up-to-date `main`:

   ```bash
   git switch main && git pull
   git switch -c feat/short-name
   ```

   Use `feat/`, `fix/`, `docs/` or `chore/` as the prefix.
2. Make the change. Keep it to one topic.
3. Add anything you learned, and any improvement you noticed but did not make, to
   [docs/LEARNINGS.md](docs/LEARNINGS.md). The format is at the top of that file.
4. Push and open a pull request:

   ```bash
   git push -u origin HEAD
   gh pr create --fill
   ```

5. Merge by squash. The branch is deleted on merge.

## Working with a coding agent

Agents get their rules from [AGENTS.md](AGENTS.md); Claude Code loads it through `CLAUDE.md`. It tells them to
read the learnings first, work on a branch and add their own learnings before the pull request. Claude Code also
has two project skills in `.claude/skills/`: `contribute` and `record-learning`.

If your agent gets something wrong that others' agents would too, fix the rule in `AGENTS.md` or add an entry
to the learnings, so it is fixed for everyone.

## Keep out of the repo

- Secrets and `.env` files. The repo is public.
- Large binary files: robot logs, bag files, model weights.

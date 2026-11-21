---
name: contribute
description: The workflow for every change to this repo. Use at the start of any requested change (feature, fix, docs, config) and again when it is done. Reads the shared learnings, puts the work on a new branch off main, records new learnings and finishes with a pull request. main is protected, so direct pushes fail.
---

# Contribute a change

## At the start

1. Read `docs/LEARNINGS.md` and apply what is relevant to the task.
2. Check for uncommitted work with `git status`. If there is any that is not yours, ask the user before
   switching branches.
3. Update `main` and branch off it:

   ```bash
   git switch main && git pull
   git switch -c <type>/<short-name>
   ```

   Type is `feat`, `fix`, `docs` or `chore`, for example `fix/camera-timeout`.

A request with several unrelated tasks gets one branch and one pull request per task.

## While working

- Keep the change to one topic.
- Never commit secrets, `.env` files or large binary files (robot logs, bag files, model weights).
- If the task has an issue, mention it in the commits or the pull request ("Closes #12").

## When done

1. Run the project's checks, if `AGENTS.md` lists any under "Stack and commands".
2. Use the `record-learning` skill to add learnings and suggestions to `docs/LEARNINGS.md` on this branch.
3. Push and open the pull request:

   ```bash
   git push -u origin HEAD
   gh pr create --fill
   ```

   Fill in the template: what changed, how it was tested, and the learnings checkbox.
4. Give the user the pull request link. Merge only when the user asks; pull requests are merged by squash
   (`gh pr merge --squash`), and the branch is deleted on merge.

If `main` moved and the branch conflicts, rebase on `main` (`git fetch origin && git rebase origin/main`).
`docs/LEARNINGS.md` uses a union merge, so entries from both sides are kept; check the file reads well afterwards.

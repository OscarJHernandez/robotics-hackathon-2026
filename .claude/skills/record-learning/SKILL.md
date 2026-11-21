---
name: record-learning
description: Add learnings and suggestions to docs/LEARNINGS.md, the project's shared memory. Use before opening or updating a pull request, when a task is finished, when you hit something surprising or a dead end, when a setup or hardware step was not obvious, or when the user corrects you or says "remember this" or "note this".
---

# Record a learning or suggestion

`docs/LEARNINGS.md` is the one document every contributor and agent reads before working here. Keep it useful:
short, specific and free of duplicates.

## What to record

Record a **learning** when it is non-obvious and would save the next person or agent time:

- a setup or environment step that is easy to miss;
- a hardware or robot quirk (timing, calibration, ports, safety limits);
- an approach that failed, and why;
- a decision and the reason for it;
- a correction or preference the user gave you that applies to the whole project.

Record a **suggestion** when you noticed an improvement you did not make: a refactor, a missing test, a tool
worth adopting, a process change.

Do not record what the code, the git history or `AGENTS.md` already shows, status updates ("finished X"), or
anything that only matters to the current task.

## Steps

1. Read `docs/LEARNINGS.md`. If the point is already there, stop, or fix the existing entry if it is wrong
   or out of date.
2. Add the entry at the top of the section it fits best, in this format:

   ```
   - YYYY-MM-DD (name, or "name's agent"): what you learned. Why it matters or what to do.
   ```

   Use today's date and the git user's first name (`git config user.name`). One or two sentences. Link the
   file, pull request or issue if there is one.
3. Suggestions go under "Suggestions (open)". When one is carried out or turned down, move it to
   "Suggestions (done or declined)" with a short note.
4. Commit the change on the task's branch, so it reaches `main` in the same pull request as the work. If you
   are not on a task branch, use the `contribute` skill to make one (for example `docs/learning-<topic>`).
5. Tell the user in one line what you added. If there was nothing new to record, say so in the pull request
   description.

Never put secrets, tokens, private addresses or personal details in the document. The repo is public.

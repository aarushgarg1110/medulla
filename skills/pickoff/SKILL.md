---
name: pickoff
description: Resume work from a previous session with no context loss — read the prior session's handoff with medulla_resume, load the notes and files it points at, verify against git and the artifacts it names, then restate the plan for correction before doing anything. Use when the user says "pickoff", "/pickoff <session>", "pick up from <session>", "continue from yesterday's chat", or opens a new session to continue earlier work. Pairs with the `handoff` skill.
---

# Pickoff — resume a previous session

Accepts a session name and free-form extra instruction (e.g. `/pickoff fooPart2 start with the
tests, skip the write-up`). The extra instruction sets the FOCUS: catch up on what serves that, not
on everything.

If no session is named, take the most recent one for this project directory
(`medulla_list(project=...)`) and say which you chose, so a wrong guess is caught immediately.

## Procedure

**1. `medulla_resume("<name>")` — one call, first.** It finds the session by its /rename title (or a
session-id prefix), indexes it on the spot if it changed after the last scan (a session that just
ended usually has, and its handoff report was written after its own last index), and returns:
- the header: id, project, dates, turns. Confirm they look like the session the user means;
- **Parts**: every part of the same objective by date. If a LATER part exists than the one named,
  say so: the user may have opened the wrong one, and the later part holds the current state;
- **the handoff report**: goal, accomplishments, open items, next action, lookup table, keywords,
  suggested reading. Read it before anything else.

Do not run `medulla scan`, `medulla_list` or `medulla_session_detail` first: resume replaces all
three. If resume says the session has no handoff report, it returns the last chunks instead: read
those, then go to step 3. (In a terminal, `medulla resume <name>` prints the same report.)

**2. Follow the handoff's own pointers.** Load the notes, docs and files it names. Do not substitute
your own guesses for its list; it was written by a session that had the context.

**3. Fill gaps with targeted reads, not sequential paging.** Only when the handoff leaves a thread
unclear: `medulla_search(<handoff keyword>, expand=2)` returns the matching chunks' text in the same
call. If its footer says transcripts changed since they were indexed and the thread is missing,
repeat with `refresh=true`. Only widen if a thread comes back empty.

Conclusions cluster at the **very end** and **wherever the user asked a big question**. Chunks break
on turn boundaries, so "Let me check X" occupies a chunk just like a long synthesis: paging through
the middle mostly buys narration.

**4. VERIFY before trusting.** Notes and transcripts describe a moment in time; the world moved.
- `git log --oneline -5` and `git status` in the repos involved, in the RIGHT repo (check `pwd`; a
  stale `cd` will have you reading the wrong history and reaching confident wrong conclusions)
- confirm the artifacts the handoff names still exist (files, cloud storage, PRs, branches)
- job/run status if anything was left running

Notes are a POINTER, not a source of truth. They are unversioned and go stale silently: a note can
read "not yet submitted" for work that had already finished.

**5. Restate and invite correction, before doing any work.** Post a short summary:
- where the previous session left off
- what changed since (from step 4)
- what you understand the task to be
- a concrete gameplan
- anything ambiguous or that looks stale

Then STOP and let the user correct it. This is the highest-value step in the whole procedure: a
first restatement typically contains one or two errors, and one correction turn is far cheaper than
discovering them mid-execution.

## Rules

- **Never assert a conclusion from partial evidence.** If a thread is unresolved, say "open, no
  decision recorded". Checking a fact in the wrong place and concluding it does not exist is a real
  failure mode: verify in the right repo or location before declaring anything missing.
- **Do not re-derive what the handoff already states.** If it gives a number and a path, use them;
  re-computing is a waste unless step 4 shows they went stale.
- **Respect deferred decisions.** A handoff marking something "deferred by choice" means it was
  settled: do not reopen it unprompted.
- **Flag at-risk work.** If the handoff references uncommitted, unpushed, or `/tmp` artifacts,
  surface that immediately; `/tmp` gets cleaned and untracked files are one `git clean` from gone.

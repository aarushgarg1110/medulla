---
name: handoff
description: End-of-session handoff report, written into the chat as the final message so the next session can pick it up from the transcript. Use when the user says "handoff", "/handoff", "wrap up this session", "I'm running out of context", or is about to move to a new chat. Produces no files; the report IS the artifact, because medulla indexes the transcript and `pickoff` reads it back with `medulla_resume`. Pairs with the `pickoff` skill.
---

# Handoff — hand this session to the next one

The session that is ENDING has the context for free. The session that STARTS has to pay to
rebuild it. So the handoff is written here, now, by the side that already knows.

**Output is a chat message. Do not write any file, do not commit anything.** medulla indexes the
transcript, and `pickoff` reads the report back with `medulla_resume`, so a well-structured final
message *is* the durable artifact. Writing it to disk as well is duplication that goes stale.

Accepts free-form extra instruction after the command (e.g. `/handoff focus on the backtest
thread, skip the write-up`). Honour it: it tells you what the next session is for.

**Name the session.** The next session finds this one by its title. You cannot see it directly: read it
from the `· this session:` part of the `[context]` line medulla adds to each prompt, or, without that
line, from the header of `medulla_resume` called with your session id (`echo $CLAUDE_CODE_SESSION_ID`).
If it has none, ask the user to
run `/rename <objective>` first (or use the next free part name, e.g. `fooPart3` when `fooPart2`
exists). Parts of one objective share a stem (`foo`, `fooPart2`, `fooPart3`), and `medulla_resume`
lists them all, so the next session can see where the thread began.

## Step 0 — index and search your own transcript BEFORE writing

`medulla_resume("<this session's title>")` indexes this session on the spot (a running session can
index and read itself) and shows its parts. Then `medulla_search(..., expand=2)` your OWN session for
the threads you are about to summarize. This is not ceremony: in a long session the early turns are
no longer in your context window, so a handoff written from memory is reconstructed, not recalled,
which is exactly the kind of confident-but-wrong artifact this skill exists to prevent. Verify
numbers, paths and decisions against the actual text before asserting them.

The report you are about to write is not indexed yet. The next session's `medulla_resume` indexes it
when it opens, so neither side needs a full `medulla scan`.

Cite threads by KEYWORD, not by chunk index. Chunk boundaries are assigned by turn grouping and are
not guaranteed stable across rescans: a keyword degrades gracefully, a wrong chunk number silently
points somewhere else.

## The one rule that decides everything

**Write for the NEXT TASK, not as a session summary.** medulla can retrieve anything from this
session later on demand, so re-narrating finished work is wasted space. If a thread is closed, name
it in one line and move on. If a thread is what the next session will actually do, give it
everything: paths, numbers, the decision and its reasoning, what is already verified.

Ask yourself: *"What would I be annoyed not to have, ten minutes into the next session?"* That is the
content. Everything else is one line or omitted.

## Sections

Adapt to where the session actually ended: drop sections that are empty rather than padding them.

**1. Overarching goal.** One or two sentences: what this session was *for*. Enough that a reader who
never saw it knows why the rest matters.

**2. What we accomplished.** Bullets. Closed threads get ONE line each. Include commit SHAs, PR
numbers, and job/run identifiers, because those are the things that are painful to re-derive.

**3. Open items.** What is genuinely unfinished, each with enough state to resume: what is done so
far, what it is blocked on, and whether a decision is still owed. Mark anything *deferred by choice*
as deferred, otherwise the next session re-litigates a settled decision.

**4. Immediate next action.** ONE thing. The single next step, concrete enough to start on. Keep
this heading as written: `medulla_resume` finds the report by it.

**5. Session SSOT.** The lookup table: the identifiers that would cost real time to find again. Only
what is relevant to the open work, for example:
- file and storage paths (result files, manifests, artifacts, cloud storage prefixes)
- experiment, run, job and CI identifiers
- headline numbers with their units and what they are measured on
- repo / branch / commit state, and anything uncommitted or unpushed
- named entities: endpoints, tickets, PRs, environments

**6. Decisions + why.** Only decisions that would otherwise be re-argued. Include the *reason*, not
just the verdict; the reason is what makes it stick. Also record decisions that were reversed and
why, since the wrong turn is usually the useful part.

**7. Keywords.** 4–8 terms `pickoff` can hand to `medulla_search` to pull the right threads. Use the
distinctive nouns actually spoken in the session (file names, tool names, ids), not generic words.

**8. Suggested reading for the next session.** Specific paths, not categories:
- your agent's persistent notes that matter here (Claude Code: the memory files under
  `~/.claude/projects/<project>/memory/`; Kiro: steering files)
- project docs and skills that apply (CLAUDE.md, READMEs, design notes, `skills/*/SKILL.md`)
- say which are stale or were corrected this session, so the next one does not trust them blindly

## Hard requirements

- **Verify durability before pointing at anything.** If the handoff references work that is
  uncommitted, unpushed, or only in `/tmp`, SAY SO explicitly and flag it as at-risk. A handoff
  pointing at untracked files is a trap.
- **Never invent a conclusion.** If a thread has no decision, write "open, no decision recorded".
  Inferring one is how a guess becomes fact in the next session.
- **Distinguish measured from assumed.** Numbers that were computed get stated plainly; anything
  believed-but-unverified gets marked as such.
- **Keep it scannable.** Tables for identifiers and numbers, bullets for threads. The next session
  reads this as one block, so density matters more than prose.

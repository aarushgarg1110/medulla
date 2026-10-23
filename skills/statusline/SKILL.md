---
name: statusline
description: Set up medulla's Claude Code statusline — a context bar with its token count, which also records the window size that autohandoff reads — or wrap the user's existing statusline so it keeps its own output. Use when the user says "/statusline", "set up the statusline", "show my context in the statusline", or turns on autohandoff. Claude Code only.
---

# Statusline — show the context filling up, and record its size

Claude Code reports a session's context window size only to the statusline, never to hooks, so
autohandoff can only use the real size if medulla's statusline (or its tap) runs. Nothing changes in
the user's settings without their OK.

## Procedure

**1. Say what would change.** Run the setup in dry-run mode:
- installed as a plugin: `uv --project "${CLAUDE_PLUGIN_ROOT}" run --quiet medulla statusline setup --dry-run`
- installed as a CLI (`uv tool install`): `medulla statusline setup --dry-run`

It reports one of three cases, in plain words. Show it to the user as it is:
- **no statusline**: medulla would add its two lines;
- **their own statusline**: medulla would *wrap* it, recording the size and then printing their output
  unchanged. Their script is not edited;
- **already medulla's**: nothing to do. Stop here.

**2. Ask, then apply.** Only on a clear yes, run the same command with `--yes` instead of
`--dry-run`. It backs settings.json up once (to `settings.json.medulla-backup`) before writing.

**3. Tell them how to undo it:** `medulla statusline undo` (the plugin form: `uv --project
"${CLAUDE_PLUGIN_ROOT}" run --quiet medulla statusline undo`) removes medulla's statusline, or restores
the one it wrapped, exactly.

**If they would rather edit their own script** than have it wrapped, give them this line to add where
their script has read the statusline's JSON into `$input`, and change nothing for them:

```sh
printf '%s' "$input" | "$HOME/.medulla/bin/statusline" --tap
```

`--tap` records the size and prints nothing. Run the setup command with `--tap` (instead of
`--dry-run`) to install the launcher that line calls, without touching their settings.

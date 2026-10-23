#!/bin/sh
# medulla's autohandoff hook. Off unless turned on in /config (or MEDULLA_AUTOHANDOFF): the gate is in
# the shell, so a disabled hook costs no Python start on every prompt.
case "$(printf '%s' "${MEDULLA_AUTOHANDOFF:-$CLAUDE_PLUGIN_OPTION_AUTOHANDOFF}" | tr 'A-Z' 'a-z')" in
  1|true|yes|on) ;;
  *) exit 0 ;;
esac
py="$CLAUDE_PLUGIN_ROOT/.venv/bin/python"
[ -x "$py" ] || exit 0
exec "$py" -I -m medulla.autohandoff

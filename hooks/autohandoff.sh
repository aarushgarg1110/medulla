#!/bin/sh
# medulla's autohandoff hook. Off unless turned on in env, Claude /config, or config.toml [autohandoff]:
# the gate is in the shell, so a disabled hook costs no Python start on every prompt.
_enabled() {
  case "$(printf '%s' "$1" | tr 'A-Z' 'a-z')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}
if [ -n "${MEDULLA_AUTOHANDOFF+x}" ]; then
  _enabled "$MEDULLA_AUTOHANDOFF" || exit 0
elif [ -n "${CLAUDE_PLUGIN_OPTION_AUTOHANDOFF+x}" ]; then
  _enabled "$CLAUDE_PLUGIN_OPTION_AUTOHANDOFF" || exit 0
else
  cfg="${MEDULLA_DIR:-$HOME/.medulla}/config.toml"
  [ -f "$cfg" ] || exit 0
  awk '/^\[autohandoff\]/{f=1;next} /^\[/{f=0} f && /^[[:space:]]*enabled[[:space:]]*=[[:space:]]*true([[:space:]]*(#.*)?)?$/{found=1} END{exit !found}' "$cfg" || exit 0
fi
root="${CLAUDE_PLUGIN_ROOT:-${PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
py="$root/.venv/bin/python"
[ -x "$py" ] || exit 0
exec "$py" -I -m medulla.autohandoff

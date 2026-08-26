#!/usr/bin/env bash
# Stop orphaned Ray / Gym processes and the GCS log spam they produce.
#
#   bash gymclean.sh              show every target with its full command line, ask, then kill
#   bash gymclean.sh --dry-run    show targets, kill nothing
#   bash gymclean.sh --yes        do not ask
#
# WHY YOU NEED THIS
# `gym env start` runs a Ray head plus three servers in the foreground. Close
# that terminal without Ctrl-C -- or let `gym eval reverify` spin up its own
# cluster and exit badly -- and the workers survive with their head gone. They
# retry forever and print this into whatever terminal inherited them:
#
#   Failed to get cluster ID from GCS server: TimedOut
#   Failed to connect to GCS at address 172.x.x.x:38471 within 5 seconds
#   Failed to connect to the observability pubsub at address ...
#
# It looks cosmetic. It is not -- they still hold worker slots, and a fresh
# `gym env start` can hang or fail because of them.
#
# WHY THIS IS NOT JUST `pkill -f ray`
# `pkill -f` matches the WHOLE COMMAND LINE. A plain `pkill -9 -f 'ray::'` will
# also kill a `tail -f ray.log`, an editor with a Ray file open, a colleague's
# grep, this script, and -- verified the hard way -- PID 1 on a host whose init
# happens to carry the pattern in its argv. Every candidate below is filtered
# and shown to you before anything is signalled.
set -uo pipefail

DRY=0; YES=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --yes|-y)  YES=1 ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $a  (try --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m !!\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ok\033[0m %s\n' "$*"; }

SELF=$$
SELF_PGID="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"

# What we are looking for, in the command line.
PATTERNS=(
  'ray::' 'raylet' 'gcs_server' 'plasma_store'
  'log_monitor.py' 'runtime_env_agent' 'dashboard/agent.py'
  'ray/dashboard' 'default_worker.py'
  'resources_servers/' 'responses_api_agents/' 'responses_api_models/'
  'gym env start'
)

# A candidate must ALSO be the right KIND of process. We check the executable
# name, not the command line -- this is the check that does the real work.
#
# Checking the command line a second time is worthless and I proved it: an
# earlier version accepted anything whose cmdline contained "raylet", which
# happily matched `tail -f /var/log/raylet.log` and an editor with a file
# called notes-about-gcs_server.md open. Both were killed in testing.
#
# `ps -o comm=` gives the executable, so a log tail is `tail`, an editor is
# `vim`, and only a genuine worker is `python3` / `raylet` / `gcs_server`.
right_kind() {
  case "$1" in
    python|python3|python[0-9].[0-9]|python[0-9].[0-9][0-9]) return 0 ;;
    raylet|gcs_server|plasma_store|gym|uvicorn) return 0 ;;
    ray::*) return 0 ;;                 # Ray renames its workers
    *) return 1 ;;
  esac
}

cmdline() { tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null; }

# ------------------------------------------------------------------ collect
declare -A TARGET=()
for p in "${PATTERNS[@]}"; do
  while read -r pid; do
    [ -z "$pid" ] && continue
    [ "$pid" = "1" ] && continue                       # never init
    [ "$pid" = "$SELF" ] && continue                   # never ourselves
    [ "$pid" = "${PPID:-0}" ] && continue              # never our parent shell
    pg="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
    [ -n "$SELF_PGID" ] && [ "$pg" = "$SELF_PGID" ] && continue   # nor our group
    cl="$(cmdline "$pid")"
    [ -z "$cl" ] && continue                           # kernel thread
    case "$cl" in *gymclean*) continue ;; esac         # nor another copy of this
    comm="$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ')"
    right_kind "$comm" || continue
    TARGET["$pid"]="[$comm] $cl"
  done < <(pgrep -f -- "$p" 2>/dev/null)
done

if [ "${#TARGET[@]}" -eq 0 ]; then
  say "Nothing to clean"
  ok "no orphaned Ray or Gym processes"
  cat <<'EOF'

  Still seeing the warnings? Then that terminal is displaying output from a
  process group it owns. The messages are being written to its tty, not
  generated fresh. Close that window.
EOF
  exit 0
fi

# ------------------------------------------------------------------- show
say "${#TARGET[@]} process(es) will be stopped"
for pid in "${!TARGET[@]}"; do
  printf '  \033[1;33m%-8s\033[0m %s\n' "$pid" "$(echo "${TARGET[$pid]}" | cut -c1-96)"
done

if [ "$DRY" = 1 ]; then
  warn "--dry-run: nothing was signalled"
  exit 0
fi

if [ "$YES" = 0 ]; then
  printf '\nKill these? [y/N] '
  read -r r </dev/tty || r=""
  case "$r" in [yY]*) ;; *) echo "aborted."; exit 0 ;; esac
fi

# -------------------------------------------------------------- stop them
if command -v ray >/dev/null 2>&1; then
  ray stop --force >/dev/null 2>&1 && ok "ray stop --force" || true
fi

say "Signalling"
for pid in "${!TARGET[@]}"; do kill -TERM "$pid" 2>/dev/null; done
sleep 2
LEFT=0
for pid in "${!TARGET[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null; LEFT=$((LEFT+1)); fi
done
[ "$LEFT" -gt 0 ] && ok "$LEFT needed SIGKILL" || ok "all exited on SIGTERM"
sleep 1

# Stale session dirs keep the dead cluster's address around and confuse the
# next start. Scratch only -- no configuration lives here.
rm -rf /tmp/ray 2>/dev/null && ok "cleared /tmp/ray"

# ---------------------------------------------------------------- verify
say "Verifying"
STILL=""
for pid in "${!TARGET[@]}"; do
  kill -0 "$pid" 2>/dev/null && STILL="$STILL $pid"
done
if [ -z "${STILL// /}" ]; then
  ok "clean — the GCS warnings will stop"
  echo
  echo "  A terminal that is STILL printing them is showing output from its own"
  echo "  process group. Close that window; nothing new is being generated."
else
  warn "survived:$STILL"
  echo "  Owned by another user, or needs elevation:"
  for pid in $STILL; do echo "      sudo kill -9 $pid"; done
fi

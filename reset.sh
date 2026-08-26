#!/usr/bin/env bash
# Put the labs back to their pre-Lab-1 state.
#
#   bash reset.sh              show what goes and what stays, ask, then do it
#   bash reset.sh --dry-run    show only
#   bash reset.sh --yes        no prompt
#   bash reset.sh --deep       ALSO drop the per-server venvs (slower next start)
#
# WHAT IT DELETES — everything the labs generate
#   results/                      rollouts, reverify output, aggregate metrics, logs
#   outputs/                      stray run output
#   hf_exports/                   Lab 6's round-trip export
#   resources_servers/support_triage/   the whole Lab 5 environment, so
#                                 `gym env init` works again -- it refuses to
#                                 run when the directory already exists
#   __pycache__, .pytest_cache    stale bytecode that can shadow an edited file
#   /tmp/ray                      dead cluster state
#
# WHAT IT KEEPS — the expensive things
#   env.yaml                      YOUR CREDENTIALS. Never touched
#   ~/venv-relay, ~/venv-gym      the two main environments
#   resources_servers/mcqa/.venv  and the agent + model server venvs. 174
#                                 packages each; nothing in the labs modifies
#                                 them, so rebuilding is pure waste
#   ~/.cache/uv                   THE important one. It is what makes a venv
#                                 rebuild take seconds instead of minutes.
#                                 Even --deep leaves this alone
#   the NeMo container            37 GB. Not ours to churn
#
# Lab 5's environment is safe to delete because the canonical copy lives in
# gym/support_triage/ and RUN.md step 2 copies it back in.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

DRY=0; YES=0; DEEP=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --yes|-y)  YES=1 ;;
    --deep)    DEEP=1 ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $a  (try --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m !!\033[0m %s\n' "$*"; }
bad()  { printf '\033[1;31m XX\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ok\033[0m %s\n' "$*"; }
sz()   { [ -e "$1" ] && du -sh "$1" 2>/dev/null | cut -f1 || echo "-"; }

[ -f RUN.md ] && [ -d gym ] || { bad "this does not look like the labs repo"; exit 1; }

# ------------------------------------------------------------------ inventory
say "Will DELETE"
TARGETS=(results outputs hf_exports resources_servers/support_triage)
FOUND=0
for t in "${TARGETS[@]}"; do
  if [ -e "$t" ]; then
    printf '  %-40s %s\n' "$t" "$(sz "$t")"; FOUND=1
  fi
done
PYC=$(find . -type d \( -name __pycache__ -o -name .pytest_cache \) -not -path './.git/*' 2>/dev/null | wc -l | tr -d ' ')
[ "$PYC" != "0" ] && { printf '  %-40s %s dirs\n' "__pycache__ / .pytest_cache" "$PYC"; FOUND=1; }
[ -d /tmp/ray ] && { printf '  %-40s %s\n' "/tmp/ray" "$(sz /tmp/ray)"; FOUND=1; }

if [ "$DEEP" = 1 ]; then
  for v in resources_servers/*/.venv responses_api_agents/*/.venv responses_api_models/*/.venv; do
    [ -d "$v" ] && printf '  %-40s %s  \033[1;33m(--deep)\033[0m\n' "$v" "$(sz "$v")"
  done
fi

[ "$FOUND" = 0 ] && [ "$DEEP" = 0 ] && ok "already clean"

say "Will KEEP"
printf '  %-40s %s\n' "env.yaml  (your credentials)" "$([ -f env.yaml ] && echo present || echo MISSING)"
for v in "$HOME/venv-relay" "$HOME/venv-gym" /home/ubuntu/workspace/venv-relay /home/ubuntu/workspace/venv-gym; do
  [ -d "$v" ] && printf '  %-40s %s\n' "$(basename "$v")" "$(sz "$v")"
done
[ -d "$HOME/.cache/uv" ] && printf '  %-40s %s  <- makes rebuilds fast\n' "~/.cache/uv" "$(sz "$HOME/.cache/uv")"
if [ "$DEEP" = 0 ]; then
  # support_triage's venv goes with its parent directory, so do not count it
  # among the survivors -- claiming to keep something we delete is worse than
  # not mentioning it.
  n=0
  for v in resources_servers/*/.venv responses_api_agents/*/.venv responses_api_models/*/.venv; do
    case "$v" in *support_triage*) continue ;; esac
    [ -d "$v" ] && n=$((n+1))
  done
  [ "$n" != "0" ] && printf '  %-40s %s venvs (mcqa, agent, model)\n' "per-server venvs" "$n"
fi
command -v docker >/dev/null 2>&1 && docker image inspect nvcr.io/nvidia/nemo:25.09 >/dev/null 2>&1 \
  && printf '  %-40s %s\n' "nvcr.io/nvidia/nemo:25.09" "37 GB"

if [ "$DRY" = 1 ]; then warn "--dry-run: nothing was deleted"; exit 0; fi

if [ "$YES" = 0 ]; then
  printf '\nProceed? [y/N] '
  read -r r </dev/tty || r=""
  case "$r" in [yY]*) ;; *) echo "aborted."; exit 0 ;; esac
fi

# ------------------------------------------------- stop anything still running
# Deleting under a live server leaves half-written state and a held port.
if [ -x ./gymclean.sh ]; then
  say "Stopping any running servers first"
  bash ./gymclean.sh --yes 2>/dev/null | grep -E "will be stopped|port 11000|clean —" || true
fi

# ------------------------------------------------------------------- delete
say "Deleting"
for t in "${TARGETS[@]}"; do
  [ -e "$t" ] && { rm -rf "$t" && ok "$t"; }
done
find . -type d \( -name __pycache__ -o -name .pytest_cache \) -not -path './.git/*' \
  -prune -exec rm -rf {} + 2>/dev/null && ok "__pycache__ / .pytest_cache"
rm -rf /tmp/ray 2>/dev/null && ok "/tmp/ray"
rm -f /tmp/app.py.bak /tmp/server.log /tmp/g.log 2>/dev/null

if [ "$DEEP" = 1 ]; then
  say "Deep clean — per-server venvs"
  rm -rf resources_servers/*/.venv responses_api_agents/*/.venv responses_api_models/*/.venv 2>/dev/null
  ok "removed; the uv cache is intact so they rebuild in seconds"
fi

# `gym eval run` does not create its own output directory and dies with a bare
# FileNotFoundError on the materialized-inputs path if it is missing.
mkdir -p results
ok "results/ recreated empty"

# -------------------------------------------------------------------- verify
say "Ready for Lab 1?"
PROB=0
chk() { if eval "$2"; then ok "$1"; else bad "$1"; PROB=$((PROB+1)); fi; }

chk "env.yaml present"                 '[ -f env.yaml ]'
chk "relay venv present"               '[ -d "$HOME/venv-relay" ] || [ -d /home/ubuntu/workspace/venv-relay ]'
chk "gym venv present"                 '[ -d "$HOME/venv-gym" ]   || [ -d /home/ubuntu/workspace/venv-gym ]'
chk "results/ empty"                   '[ -z "$(ls -A results 2>/dev/null)" ]'
chk "support_triage gone (gym env init will work)" '[ ! -d resources_servers/support_triage ]'
chk "port 11000 free"                  '! ss -ltn 2>/dev/null | grep -q ":11000"'

if [ "$PROB" = 0 ]; then
  printf '\n\033[1;32m  Clean slate.\033[0m\n\n'
  cat <<'EOF'
  Before anything else, confirm the model still exists -- it can be retired
  overnight, and the failure names nothing:

      bash modelcheck.sh

  Then start at Lab 1:

      source ~/venv-relay/bin/activate
      python relay/lab1_quickstart.py

EOF
else
  printf '\n\033[1;31m  %d problem(s) above.\033[0m\n\n' "$PROB"
  exit 1
fi

#!/usr/bin/env bash
# Setup for the NVIDIA Agent Toolkit labs, on a Brev L40S box.
# (Nothing here is L40S-specific — only lab 6 touches the GPU at all.)
#
#   curl -fsSL https://raw.githubusercontent.com/willson-nv/nvidia-agent-toolkit-labs/main/setup.sh | bash
#
# or, from a checkout:
#
#   bash setup.sh
#
# Builds THREE separate environments, on purpose:
#
#   relay/   venv, Python >= 3.11    no GPU, no credentials
#   gym/     venv, Python  = 3.13.14 no GPU, needs a model endpoint
#   bridge   NGC container, Py 3.12  GPU + NGC credentials
#
# Their Python requirements are mutually exclusive. Do not try to unify them.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/willson-nv/nvidia-agent-toolkit-labs.git}"
WORK="${WORK:-/home/ubuntu/workspace}"
REPO="$WORK/nvidia-agent-toolkit-labs"
RELAY_VENV="$WORK/venv-relay"
GYM_VENV="$WORK/venv-gym"

RELAY_VER="0.7.3"      # 14 Aug 2026
GYM_VER="0.5.0"        # 7 Aug 2026 -- requires Python >= 3.13.14
NEMO_IMAGE="${NEMO_IMAGE:-nvcr.io/nvidia/nemo:25.09}"   # verify the tag before the workshop

say()  { printf '\n\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m!!\033[0m %s\n' "$*"; }
skip() { printf '    \033[1;33mskipped:\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------- 0. the box
say "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  warn "no nvidia-smi. Labs 1-5 will still work; lab 6 will not."
fi

say "Persistent paths under $WORK"
# Only /home/ubuntu/workspace survives a Brev stop. Everything expensive goes here.
mkdir -p "$WORK"
export GIT_TERMINAL_PROMPT=0

# ---------------------------------------------------------------- 1. the repo
say "Repo"
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" pull --ff-only || warn "pull failed; using the checkout on disk"
elif [ -f "$REPO/setup.sh" ]; then
  say "Using the existing checkout at $REPO"
elif git clone "$REPO_URL" "$REPO" 2>/tmp/clone.err; then
  :
else
  sed 's/^/    /' /tmp/clone.err >&2
  cat >&2 <<EOF

  Could not clone $REPO_URL

  Either the repo is private (make it public, or clone it by hand with your own
  credentials), or nothing has been pushed to main yet. Check from your laptop:

      git log --oneline origin/main..HEAD

EOF
  exit 1
fi
git -C "$REPO" log --oneline -1 2>/dev/null || true

# ------------------------------------------------------- 2. shared venv helper
make_venv() {              # make_venv <path> <python-binary>
  local path="$1" py="$2"
  if [ ! -x "$path/bin/python" ]; then
    rm -rf "$path"
    # Test for a working interpreter, not just a directory: a venv that fails at
    # ensurepip leaves the tree behind and would otherwise be accepted silently.
    if ! "$py" -m venv "$path" 2>/tmp/venv.err; then
      sed 's/^/    /' /tmp/venv.err >&2
      warn "venv creation failed -- installing the venv package and retrying"
      local v; v="$("$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
      sudo apt-get update -qq
      sudo apt-get install -y -qq "python${v}-venv" || sudo apt-get install -y -qq python3-venv
      rm -rf "$path"; "$py" -m venv "$path"
    fi
  fi
  [ -f "$path/bin/activate" ] || { echo "venv incomplete at $path" >&2; exit 1; }
}

# ------------------------------------------------------------ 3. relay (1, 2)
say "Relay environment  (labs 1 and 2)"
RELAY_PY="$(command -v python3.12 || command -v python3.11 || command -v python3)"
echo "  interpreter: $RELAY_PY ($("$RELAY_PY" -V 2>&1))"
make_venv "$RELAY_VENV" "$RELAY_PY"
# activate touches $PS1, which is unset in a non-interactive shell; set -u would abort
set +u; source "$RELAY_VENV/bin/activate"; set -u
python -m pip install -q --upgrade pip
pip install -q "nemo-relay==${RELAY_VER}"
python -c "import nemo_relay; print('  nemo-relay', getattr(nemo_relay,'__version__','(no __version__)'))"
deactivate

# -------------------------------------------------------------- 4. gym (3-5)
say "Gym environment  (labs 3, 4 and 5)"
GYM_PY="$(command -v python3.13 || true)"
if [ -z "$GYM_PY" ]; then
  warn "python3.13 not found. NeMo Gym ${GYM_VER} requires >= 3.13.14."
  warn "Its installation page still says 3.12 -- that page is stale. Install 3.13 and re-run."
  skip "labs 3-5"
else
  echo "  interpreter: $GYM_PY ($("$GYM_PY" -V 2>&1))"
  make_venv "$GYM_VENV" "$GYM_PY"
  set +u; source "$GYM_VENV/bin/activate"; set -u
  python -m pip install -q --upgrade pip
  pip install -q "nemo-gym==${GYM_VER}"
  gym --version || true

  # Gym builds a venv PER SERVER -- resources, agent and model -- each installing
  # the full stack. Doing that cold during lab 5 is the biggest timing risk in the
  # workshop, so warm the shared uv cache now against a shipped environment.
  say "Pre-warming the uv cache (this is the slow step, and it is why lab 5 fits)"
  gym env test --resources-server example_single_tool_call \
    || warn "cache warm-up did not complete; lab 5 will be slower on first start"
  deactivate
fi

# ------------------------------------------------------- 5. megatron bridge (6)
say "Megatron Bridge  (lab 6)"
if ! command -v docker >/dev/null 2>&1; then
  skip "docker not found -- lab 6 needs the NGC container"
elif docker image inspect "$NEMO_IMAGE" >/dev/null 2>&1; then
  echo "  image already present: $NEMO_IMAGE"
else
  warn "pulling $NEMO_IMAGE -- this is large and needs NGC credentials (docker login nvcr.io)"
  docker pull "$NEMO_IMAGE" || skip "pull failed; do this before the workshop, not during it"
fi

# ---------------------------------------------------------------- 6. summary
say "Ready"
cat <<EOF

  repo          $REPO
  relay venv    $RELAY_VENV      (labs 1-2, no credentials needed)
  gym venv      $GYM_VENV        (labs 3-5, needs a model endpoint)
  container     $NEMO_IMAGE      (lab 6)

  Before labs 3-5, write your model endpoint into env.yaml at the Gym repo root:

      policy_base_url: https://api.openai.com/v1
      policy_api_key: <key>
      policy_model_name: <model>

  Then follow RUN.md. Start with lab 1 -- it needs nothing and proves the box works:

      source $RELAY_VENV/bin/activate
      python $REPO/relay/lab1_quickstart.py

EOF

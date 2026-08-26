#!/usr/bin/env bash
# Is the model in env.yaml still alive? Run this the morning of, before anything else.
#
#   bash modelcheck.sh
#
# WHY THIS EXISTS
# On 2026-08-26 the pinned model was retired at 09:00Z, mid-session. Every lab
# from 3 to 5 then failed with a bare HTTP 500 that named nothing -- the real
# message was a 410 buried inside the model server, four layers below the CLI:
#
#   "The model 'nvidia/nemotron-mini-4b-instruct' has reached its end of life
#    on 2026-08-26T09:00:00Z and is no longer available."
#
# Diagnosing that from the client side took over an hour. This takes five
# seconds and says it in one line.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

[ -f env.yaml ] || { echo "no env.yaml here"; exit 1; }

python3 - <<'PY'
import json, sys, urllib.request

cfg = {}
for line in open("env.yaml"):
    if ":" in line and not line.strip().startswith("#"):
        k, v = line.split(":", 1)
        cfg[k.strip()] = v.split("#")[0].strip()

model = cfg.get("policy_model_name", "")
key   = cfg.get("policy_api_key", "")
base  = cfg.get("policy_base_url", "https://integrate.api.nvidia.com/v1").rstrip("/")
print(f"  model : {model}")
print(f"  base  : {base}")

if not model or not key:
    sys.exit("\n  env.yaml is missing policy_model_name or policy_api_key")

req = urllib.request.Request(
    f"{base}/chat/completions",
    data=json.dumps({"model": model,
                     "messages": [{"role": "user", "content": "Reply with B only."}],
                     "max_tokens": 5}).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})

try:
    r = json.load(urllib.request.urlopen(req, timeout=30))
    print(f"\n  \033[1;32mALIVE\033[0m — replied {r['choices'][0]['message']['content']!r}")
except Exception as e:
    code = getattr(e, "code", None)
    body = getattr(e, "read", lambda: b"")().decode()
    print(f"\n  \033[1;31mFAILED\033[0m  HTTP {code}")
    try:
        d = json.loads(body)
        print("  " + (d.get("detail") or d.get("message") or body)[:300])
    except Exception:
        print("  " + (body[:300] or repr(e)))
    if code == 410:
        print("\n  410 = the model was RETIRED. Pick another from:")
        print(f"     {base}/models   (or build.nvidia.com)")
        print("  Then update policy_model_name in env.yaml and re-run Lab 3.")
    elif code in (401, 403):
        print("\n  Your API key is rejected. Get a fresh one from build.nvidia.com.")
    sys.exit(1)
PY

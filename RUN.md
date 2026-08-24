# RUN.md — every command, and what to expect

Six labs on one **L40S** (see [BREV.md](BREV.md)). Only lab 6 uses the GPU — labs 1 and 2
need no GPU, no network and no credentials at all, and labs 3–5 send inference to your
model endpoint.

> **Verification status.**
> **Labs 1 and 2 are verified** on the box (Relay 0.7.3, Python 3.12.14) and the outputs
> below are real. Labs 3-6 are still written against the documented APIs and marked
> **⚠ likely to drift** — run them before the day and replace the placeholders as you go.
>
> All three libraries ship breaking changes on a scale of weeks. Pinned versions are in
> `setup.sh`; keep them pinned.

---

## Part 0 — Setup

```bash
bash setup.sh
```

Builds three environments — a Relay venv, a Gym venv, and the NGC container for Megatron
Bridge. Their Python versions are mutually exclusive (≥3.11, exactly 3.13.14, 3.12), which
is why they are separate.

**The slow step is the uv cache warm-up** near the end. Let it finish. Gym builds a
virtualenv per server, and doing that cold in the middle of Lab 5 is the single biggest
timing risk in the workshop.

**For labs 3–5** write your endpoint into `env.yaml` at the Gym repo root:

```yaml
policy_base_url: https://integrate.api.nvidia.com/v1
policy_api_key: nvapi-...          # free, from build.nvidia.com
policy_model_name: nvidia/nvidia-nemotron-nano-9b-v2
```

Use `--model-type inference_provider` with this endpoint, not `openai_model`.
`inference.nvidia.com` is NVIDIA-internal and will not resolve from a cloud box.

---

## Part 1 — NeMo Relay

### 1.1 Lab 1 — one scope, one tool call, one model call ✅

```bash
source ~/venv-relay/bin/activate
python relay/lab1_quickstart.py
```

**Real output** — verified on the box, Relay 0.7.3, Python 3.12.14:

```
  event=scope  name=demo-agent
  event=mark   name=initialized
  event=scope  name=search
  event=scope  name=search
  event=scope  name=demo-provider
  event=scope  name=demo-provider
  tool returned: {'echo': 'hello'}
  model returned: {'messages': [{'content': 'hi', 'role': 'user'}], 'ok': True}
  event=scope  name=demo-agent
```

**Each scope emits twice** — once on entry, once on exit. That is why `search` and
`demo-provider` each appear on two lines, and why the agent scope closes last, after
everything nested inside it has finished. Worth pointing at on screen.

**No API key, no network, no observability backend** — and there is still a structured
trace of every boundary the agent crossed.

**Flush is `await flush_async()`, not `flush()`.** Relay 0.7.3 refuses to block a running
event loop and raises a `RuntimeError` telling you so. Without the flush the process can
exit before the subscriber has printed.

---

### 1.2 Lab 2 — three middlewares: redact, reject, measure ✅

```bash
python relay/lab2_middleware.py
```

**Real output** — verified on the box, Relay 0.7.3:

```
--- a good call ------------------------------------------
  event=scope  name=demo-agent data=None
  event=scope  name=search.require_query data={'kind': 'tool_conditional_execution', 'target_name': 'search'}
  event=scope  name=search.require_query data={'allowed': True, 'rejected': False}
  event=scope  name=search data={'api_key': '***redacted***', 'query': 'hello'}
  [measure] search took 0.3 ms
  event=scope  name=search data={'hits': ['result for hello']}
  -> allowed: {'hits': ['result for hello']}

--- an empty query ---------------------------------------
  event=scope  name=search.require_query data={'kind': 'tool_conditional_execution', 'target_name': 'search'}
  event=scope  name=search.require_query data={'allowed': False, 'rejected': True, 'rejection_reason': 'query must not be empty'}
  event=mark   name=search data={'rejected': True, 'rejection_reason': 'query must not be empty'}
  -> blocked: RuntimeError: guardrail rejected: query must not be empty
  event=scope  name=demo-agent data=None
```

**The single best thing on screen: count the `search` scopes.** The good call has two —
start carrying the redacted args, end carrying the result. The blocked call has **none**,
only a `mark`. That is visual proof the tool function was never entered, rather than being
logged and allowed anyway.

**Every middleware is visible in the trace.** The guardrail gets its own scope
(`search.require_query`) whose end event carries the verdict — `allowed`, `rejected`, and
the `rejection_reason` — and that reason propagates all the way out to the caller as a
`RuntimeError`. Nothing is hidden.

**Redaction is observability-only.** The emitted event shows
`api_key: '***redacted***'`, while the tool body asserts it received
`sk-fake-not-a-real-key`. Both are true at once, and that is the point.

---

### Three API traps this lab found, all fixed in the file

| Trap | What happens | Correct form |
|---|---|---|
| `subscribers.flush()` | `RuntimeError` — cannot block a running event loop | `await subscribers.flush_async()` |
| Callback signature | `TypeError: takes 1 positional argument but 2 were given` | `callback(tool_name, args)` |
| Conditional return | **Silent** — `return False` ALLOWS the call, because False is not None | `return None` to allow, a message to block |
| Outcome import | `ImportError` from `nemo_relay.intercepts` | `from nemo_relay import ToolExecutionInterceptOutcome` |

**The third one is the dangerous one** and worth saying out loud: returning a boolean does
not error. The guardrail registers, runs, and permits everything. It is the same failure
shape as the NeMo Gym silent-drop trap in Lab 5 — the API looks satisfied and the behaviour
is quietly inverted. **Test the failure path, not just the happy path.**

---

## Part 2 — NeMo Gym

### 2.1 Lab 3 — a rollout and a reward ✅

**The endpoint.** `env.yaml` at the repo root:

```yaml
policy_base_url: https://integrate.api.nvidia.com/v1
policy_api_key: nvapi-...          # free, from build.nvidia.com
policy_model_name: nvidia/nvidia-nemotron-nano-9b-v2
```

**Two terminals.** Note `inference_provider`, not `openai_model` — the NVIDIA endpoint is
OpenAI-*compatible*, not OpenAI, and Gym has a generic server for exactly that.

```bash
source ~/venv-gym/bin/activate
mkdir -p results

# terminal 1 - blocks in the foreground
gym env start --resources-server mcqa --model-type inference_provider

# terminal 2
gym eval run --no-serve --agent mcqa_simple_agent \
    --input resources_servers/mcqa/data/example.jsonl \
    --output results/mcqa_rollouts.jsonl --limit 5 --num-repeats 1
```

**Real output** — verified on the box, NeMo Gym 0.5.0:

```
Collecting rollouts: 100%|██████| 5/5 [01:08<00:00, 13.69s/it]
Key metrics for mcqa_simple_agent:
{
    "mean/reward": 1.0,
    "pass@1[avg-of-1]/accuracy": 100.0,
    "pass@1/no_answer": 0.0,
    "majority@1/accuracy": 100.0,
    "pass@1/accuracy": 100.0
}
Fully materialized inputs: results/mcqa_rollouts_materialized_inputs.jsonl
Rollouts: results/mcqa_rollouts.jsonl
Aggregate metrics: results/mcqa_rollouts_aggregate_metrics.json
```

**Timing:** ~14s per task, 68s for five. Slower than it looks on paper — worth knowing so
you keep talking rather than watching a progress bar.

> **⚠ A perfect score is a weak demo, and it breaks Lab 4.**
> `mean/reward: 1.0` leaves nothing to point at, and Lab 4 works by changing the grading
> rule and watching the number move — which cannot happen from a ceiling. Before the day,
> get this off 1.0 by one of:
>
> - **a harder environment** — `gpqa_diamond` or `math_with_autograder` instead of `mcqa`
> - **a smaller model** — `nvidia/nemotron-mini-4b-instruct` will miss some
> - **more tasks** — `--limit` higher, so at least one goes wrong
>
> A score of 0.6-0.8 tells a far better story than 1.0, because the interesting question is
> always *which ones failed and why*.

**Three files land in `results/`.** The rollouts file is the one that matters — Lab 4
re-reads it, and training consumes it.

**`gym eval run` does not create its output directory.** Without `mkdir -p results` it dies
with a bare `FileNotFoundError` on the materialized-inputs path, which names nothing useful.

---

### 2.2 Lab 4 — re-score without re-running

```bash
gym eval reverify --rollouts results/mcqa_rollouts.jsonl \
    ++mcqa.resources_servers.mcqa.grading_mode=lenient_boxed
```

**Expect** — the same five attempts, a different `mean/reward`, and **no model call**.

**Say:** reward is not a property of the model. It is code you own, and you can change your
mind about it for free. Rollouts are expensive; grading them is not.

**If it refuses to run:** reverify is gated on the server declaring itself safe to replay
statelessly. That is a feature — check before forcing it.

---

### 2.3 Lab 5 — build an environment for your own task

The centrepiece. Twenty-five minutes, nine steps. A finished copy lives in
`gym/support_triage/` — copy it in if anything stalls.

**Step 1 — scaffold** (~2 min)

```bash
gym env init --resources-server support_triage
```

Six files appear. Walk them: `app.py`, `configs/`, `data/`, `tests/`, `requirements.txt`,
`README.md`. Note the scaffold's `verify()` already returns `reward=1.0` — the environment
works before you write anything.

**Step 2 — the verifier** (~5 min)

Paste `gym/support_triage/app.py`. Read `verify()` aloud: pull the assistant's text out,
parse the JSON, compare two fields, return `hits / 2.0`.

**Step 3 — the config** (~2 min)

Set `domain: agent`. **Delete the generated `train` and `validation` dataset blocks** — they
point at files the scaffold never created, and they require a licence field.

**Step 4 — the data** (~2 min)

Five rows, `responses_create_params` plus `verifier_metadata`. All synthetic.

**Step 5 — validate** (~1 min)

```bash
gym env validate --config resources_servers/support_triage/configs/support_triage.yaml
```

**Expect:** `✓ Config is valid.` in well under a second, with no model server running.
Note it does *not* check that your JSONL exists — it is a config check, not a data check.

**Step 6 — test** (~4 min)

```bash
gym env test --resources-server support_triage
```

**Expect:** seven parametrised cases passing — both fields right scores 1.0, one field 0.5,
prose 0.0.

**Step 7 — break it on purpose** (~3 min)

Delete `verifier_metadata` from `SupportTriageVerifyRequest` and re-run the tests.

**Expect:** the ground truth never arrives, every comparison fails, **every reward is 0.0 —
with no error anywhere.** This is the most common authoring bug in NeMo Gym and it fails
silently. `test_ground_truth_survives_parsing` is the regression test for exactly this.

**Step 8 — fix it** (~2 min) — put the two lines back. Tests pass again.

**Step 9 — a real reward** (~4 min)

```bash
gym env start --resources-server support_triage --model-type openai_model   # terminal 1
gym eval run --no-serve --agent support_triage_simple_agent \
    --input resources_servers/support_triage/data/example.jsonl \
    --output results/support_triage_rollouts.jsonl                          # terminal 2
```

**Expect:** a `mean/reward` between 0 and 1 across five tickets.

**If you are behind at step 7,** skip the deliberate break and go straight to the working
version. It is the best teaching moment in the deck and the only cuttable part of Lab 5.

---

## Part 3 — Megatron Bridge

### 3.1 Lab 6 — HF in, Megatron out, HF back

```bash
docker run --rm -it --gpus all -v $(pwd):/workdir -w /workdir \
    --entrypoint bash nvcr.io/nvidia/nemo:<TAG>
```

**Smoke test first** — if this fails, the round trip was never going to work:

```bash
python megatron_bridge/list_architectures.py
```

**Expect:** a list of convertible architectures. No torchrun, no download, no distributed
init.

**Then the round trip:**

```bash
python megatron_bridge/roundtrip.py --model meta-llama/Llama-3.2-1B
```

**Expect:** import, parallelism configured, model materialised, export written, then the
summary banner.

**Say:** train with Megatron's parallelism, serve with anyone's inference engine. The
conversion is per-parameter and parallelism-aware, so it never needs both complete models
in memory on one GPU.

**Highest-risk lab of the six.** If it stalls, talk over the code on the slide and move on.
Do not debug live.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `gym: command not found` | wrong venv | `source venv-gym/bin/activate` |
| Gym install fails on Python version | 0.5.0 needs ≥3.13.14; the docs page says 3.12 and is stale | install 3.13 and rebuild the venv |
| Lab 5 `gym env start` takes minutes | per-server venvs building cold | re-run the cache warm-up in `setup.sh` |
| Every reward is 0.0 | ground truth dropped — the silent-drop trap | declare `verifier_metadata` on the request subclass |
| `gym env init` exits immediately | the directory already exists | `rm -rf` it, but note that also removes its warm venv |
| Relay registration raises `TypeError` | callback signatures moved between releases | check the middleware guide for the pinned version |
| Lab 6 cannot find `megatron.bridge` | running outside the container | there is no pip install; use the NGC image |
| `docker pull` unauthorised | no NGC credentials | `docker login nvcr.io` |

---

## The rehearsal rule

Run all six **end to end on the actual box, twice**, before the day. Write the real output
into this file and change the ⚠ markers to ✅ as you go — the same discipline as the
fine-tuning labs, and the reason those ran clean.

Capture a fallback for everything: a finished `support_triage/`, saved rollouts for Lab 4,
and a screen recording of Lab 6.

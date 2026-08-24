# RUN.md — every command, and what to expect

Six labs. Everything runs on one A100; labs 1 and 2 need no GPU and no credentials at all.

> **Verification status — read this before the workshop.**
> Nothing in this file has been executed. It is written against the documented APIs of
> NeMo Relay 0.7.3, NeMo Gym 0.5.0 and Megatron Bridge 0.6.0, but no run has confirmed it.
> Treat every "expect" below as the intended shape, and **do a full dry run on the box
> before the day**. Where a step is more likely than usual to need a tweak, it is marked
> **⚠ likely to drift**.
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
policy_base_url: https://api.openai.com/v1
policy_api_key: <your key>
policy_model_name: <a model id>
```

---

## Part 1 — NeMo Relay

### 1.1 Lab 1 — one scope, one tool call, one model call

```bash
source /home/ubuntu/workspace/venv-relay/bin/activate
python relay/lab1_quickstart.py
```

**Expect** — event lines for the scope, tool and model lifecycles, the `initialized` mark,
then the two results:

```
  event=scope  name=demo-agent
  event=mark   name=initialized
  event=scope  name=search
  event=scope  name=demo-provider
  ...
  tool returned: {'echo': 'hello'}
  model returned: {'messages': [...], 'ok': True}
```

**If you see no events:** the flush at the end is doing real work — subscriber delivery is
asynchronous. Do not remove it.

**Say:** no API key, no network, no observability backend, and there is still a structured
trace of every boundary crossed.

---

### 1.2 Lab 2 — three middlewares ⚠ likely to drift

```bash
python relay/lab2_middleware.py
```

**Expect** — a good call that executes and is timed, with `api_key` masked in the emitted
event; then an empty query that never reaches the tool function:

```
--- a good call ------------------------------------------
  event=scope  name=search data={'query': 'hello', 'api_key': '***redacted***'}
  [measure] search took 0.3 ms
  -> allowed: {'hits': ['result for hello']}

--- an empty query ---------------------------------------
  -> blocked: ...
```

**Run the good call first.** The rejection only lands as a contrast.

**The distinction to labour:** the sanitiser changed what was *emitted*; the real tool still
received the real key (there is an `assert` in the tool proving it). The conditional
guardrail changed what *executed*.

**⚠ If a registration raises `TypeError`,** the callback signatures have moved. Check the
middleware guide for the installed version — everything else in the file is stable.

---

## Part 2 — NeMo Gym

### 2.1 Lab 3 — a rollout and a reward

```bash
source /home/ubuntu/workspace/venv-gym/bin/activate

# terminal 1 — three servers, blocks in the foreground
gym env start --resources-server mcqa --model-type openai_model

# terminal 2
gym eval run --no-serve --agent mcqa_simple_agent \
    --input resources_servers/mcqa/data/example.jsonl \
    --output results/mcqa_rollouts.jsonl --limit 5 --num-repeats 1
```

**Expect** — three servers named on startup, then a progress bar and aggregate metrics:

```
Collecting rollouts: 100%|██████| 5/5
Key metrics for mcqa_simple_agent:
{ "mean/reward": 0.8, "pass@1/accuracy": 80.0 }
```

Three files land in `results/`: materialised inputs, rollouts, aggregate metrics.
**The rollouts file is the one that matters** — Lab 4 re-reads it, and training consumes it.

**Two terminals.** `gym env start` does not return.

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

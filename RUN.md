# RUN.md — every command, what to expect, and what to say

Six labs on one **L40S** (see [BREV.md](BREV.md)). Only lab 6 uses the GPU — labs 1 and 2
need no GPU, no network and no credentials at all, and labs 3–5 send inference to your
model endpoint.

## How to read the talk track

> **🎤 SAY BEFORE** — read this out, then run the command.
>
> **🎤 SAY WHILE IT RUNS** — only on steps slow enough to need it.
>
> **🎤 SAY AFTER** — read this with the output on screen, pointing at the thing it names.
>
> **❓ IF THEY ASK** — the questions each line tends to invite, answered where you will need
> them rather than in an appendix at the back.

**These are a safety net, not a cage.** Read them aloud once while rehearsing and you will
find your own words on the day. They exist so that if you lose your thread mid-lab, there is
a sentence on the page you can simply read.

**Steps marked 🔇 NO AUDIENCE** run before anyone is in the room.

> **Verification status: all six labs are verified on the box.** Relay 0.7.3 on Python
> 3.12.14 · NeMo Gym 0.5.0 on Python 3.13.15 · Megatron Bridge 0.1.0rc4 in
> `nvcr.io/nvidia/nemo:25.09` on an L40S, driver 565.57.01. **Every output below is real,
> copied from an actual run.** Nothing here is written from documentation alone.
>
> All three libraries ship breaking changes on a scale of weeks, and the NeMo container lags
> Megatron Bridge by five minor versions. Pinned versions are in `setup.sh`; keep them
> pinned, and re-run this file before the day.

---

## Starting over

```bash
bash reset.sh              # show what goes and what stays, ask, then do it
bash reset.sh --dry-run    # show only
```

Puts the labs back to their pre-Lab-1 state. **Deletes** `results/`, `outputs/`,
`hf_exports/`, the whole Lab 5 `resources_servers/support_triage/` environment, stale
`__pycache__`, and `/tmp/ray`. **Keeps** `env.yaml`, both main venvs, the mcqa/agent/model
per-server venvs, the uv cache and the 37 GB NeMo container.

**Lab 5's environment has to go entirely** — `gym env init` refuses to run when the directory
already exists, so leaving it means step 1 fails. It is safe to delete because the canonical
copy lives in `gym/support_triage/` and step 2 copies it back.

**The uv cache is the thing never to delete.** It is what makes a per-server venv rebuild take
seconds rather than minutes — even `--deep`, which does drop the venvs, leaves it alone.

It finishes with a readiness check: credentials present, venvs present, `results/` empty,
`support_triage` gone, port 11000 free.

---

## ⚠ Run this first, every single time

```bash
bash modelcheck.sh
```

**Five seconds. It asks the endpoint whether the model in `env.yaml` still exists.**

On 26 August 2026 the pinned model was retired **at 09:00Z, mid-session**. Labs 3 to 5 then
failed with a bare HTTP 500 that named nothing — the real message was a `410 Gone` buried
inside the model server, four layers below the CLI:

> *"The model `nvidia/nemotron-mini-4b-instruct` has reached its end of life on
> 2026-08-26T09:00:00Z and is no longer available."*

Diagnosing that from the client side took over an hour. `modelcheck.sh` says it in one line.

| It prints | Meaning |
|---|---|
| `ALIVE — replied 'B'` | good, carry on |
| `FAILED HTTP 410` | **the model was retired.** Pick another and update `env.yaml` |
| `FAILED HTTP 401/403` | your API key is rejected — get a fresh one from build.nvidia.com |

---

## Part 0 — Setup  🔇 NO AUDIENCE

Days before, on your own. Nothing here is scripted because nobody is watching.

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
policy_model_name: nvidia/nemotron-mini-4b-instruct
```

Use `--model-type inference_provider` with this endpoint, not `openai_model`.
`inference.nvidia.com` is NVIDIA-internal and will not resolve from a cloud box.

The 4B is deliberate — it is eleven times faster than the 9B here *and* it scores 0.2 instead
of 1.0, which is what makes Lab 4 possible. See 2.1.

---

## Part 1 — NeMo Relay  🎤 LIVE

**Labs 1 and 2 need no API key, no network and no GPU.** Say that out loud — it is the
lowest-friction thing in the whole session and it means everyone in the room can run it
tonight.

---

### 1.1 Lab 1 — one scope, one tool call, one model call ✅

> **🎤 SAY BEFORE**
>
> "Let us start with the first of those three questions — *what did my agent actually do?*
>
> I have a tiny agent here. It calls one tool, it calls one model, and that is it. What I want
> you to watch is not what it returns — it is what gets **recorded** while it happens.
>
> And I want to flag one thing before I press return: **this needs no API key, no network
> connection, and no GPU.** There is no observability backend running. Nothing is being
> shipped anywhere."

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

> **🎤 SAY AFTER** *(pair the lines up with your finger as you talk)*
>
> "At first glance that looks repetitive. It is not — **every scope emits twice.** Once on the
> way in, once on the way out.
>
> So read it in pairs. `search` opens and closes. `demo-provider` opens and closes. And
> `demo-agent` — look where it is — opens on the very first line and closes on the very last,
> **after everything nested inside it has finished.**
>
> Those pairs are a tree. That is the whole point. A log file gives you a flat list of things
> that happened and leaves you to work out which model call caused which tool call. This
> gives you the structure directly, because the structure is what actually happened.
>
> And I said it before I ran it, so let me say it again now you have seen the output: **no API
> key, no network, no backend.** That is a complete structured trace of every boundary this
> agent crossed, and it cost nothing to get."

> **❓ IF THEY ASK**
>
> **"Isn't this just OpenTelemetry?"**
> "For the observation half — honestly, yes, the ideas overlap heavily, and if all you need is
> to see what happened then your existing tracing is probably fine. I would not sell you
> something you already have. The difference shows up in the next lab, which is that this can
> also **refuse**. OTel watches. This can intervene, and the intervention lands in the same
> trace."
>
> **"What is a `mark` versus a `scope`?"**
> "A scope has a start and an end and can contain other things. A mark is a single point in
> time — something happened, no duration. You will see marks matter in the next lab."
>
> **"Does this slow the agent down?"**
> "You are looking at microseconds against tool calls and model calls that take hundreds of
> milliseconds. It is not where your latency is."
>
> **"Where does this go in production?"**
> "Wherever you point it. There is a subscriber interface — here it is printing to stdout
> because that is the clearest thing to show you. In production it goes to whatever you
> already use."

**Flush is `await flush_async()`, not `flush()`.** Relay 0.7.3 refuses to block a running
event loop and raises a `RuntimeError` telling you so. Without the flush the process can
exit before the subscriber has printed.

---

### 1.2 Lab 2 — three middlewares: redact, reject, measure ✅

> **🎤 SAY BEFORE**
>
> "Seeing what happened is useful. But most of you do not have a visibility problem you can
> solve with a dashboard — you have a *control* problem. Somebody wants to know what stops the
> agent doing something it should not.
>
> So I have registered three pieces of middleware on that tool. One **redacts** an API key out
> of what gets recorded. One is a **guardrail** that rejects a call with an empty query. One
> just **measures** how long the call took.
>
> I am going to make two calls. A good one, and a bad one. **And I want you to count
> something for me. Count the scopes whose name is exactly `search`** — ignore the ones with
> a dot in them, those are the middleware reporting on itself."

> **⚠ Be precise about what you are asking them to count.** The word `search` appears **five**
> times in the good call, because the guardrail's own scope is named `search.require_query`.
> Only **two** of those are scopes named `search`. If you say "count the searches" the room
> counts five, your punchline lands wrong, and you spend a minute recovering. Say *"named
> exactly `search`"*, or point at the two lines as you say it.

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

**How to read the trace, line by line:**

| Line | What it is |
|---|---|
| `scope demo-agent` *(first)* | the agent scope **opens** — it stays open across *both* calls |
| `scope search.require_query` ×2 | your guardrail, opening then closing. The closing payload carries the **verdict** |
| `scope search` *(args)* | the **tool scope opens**, carrying the arguments as recorded — redacted |
| `[measure] search took 0.3 ms` | **not a Relay event** — your intercept's own `print()`, from inside the wrapper |
| `scope search` *(hits)* | the **tool scope closes**, carrying the return value |
| `mark search` *(blocked call)* | a point in time, no duration. Refused |
| `scope demo-agent` *(last)* | the agent scope **closes**, after everything nested inside it |

**The `mark` printing after the `-> blocked` line is not a bug.** The exception reaches your
code synchronously; the subscriber prints asynchronously. The ordering between those two is
not guaranteed and does not matter.

> **🎤 SAY AFTER** *(this is the strongest thirty seconds in Part 1 — do not rush it)*
>
> "So — how many scopes named `search` in the good call? **Two.** One opening, carrying the
> arguments. One closing, carrying the result.
>
> And in the blocked call? **None.** Zero. There is a `mark` saying it was rejected, and then
> nothing.
>
> That absence is the whole demo. **The tool function was never entered.** Not called and
> ignored. Not called and the result thrown away. Never entered. And you are not taking my
> word for that — you are reading it off the trace, because a scope that never opened cannot
> appear.
>
> That is the difference between a guardrail and a warning label."
>
> *(then, pointing at `search.require_query`)*
>
> "Second thing. **The guardrail itself is in the trace.** It gets its own scope, and its
> closing event carries the verdict — allowed, rejected, and the reason why. And that reason
> travels all the way back out to the caller as an exception. Nothing is hidden and nothing is
> silently swallowed.
>
> Third, and this one matters for anyone with a compliance function. Look at the arguments on
> the recorded event: **`api_key: '***redacted***'`.** Now — the tool itself received the real
> key. It had to; it needed it to work. **Both of those are true at once.** The redaction is
> about what gets *observed*, not about what the tool receives. That distinction is usually
> the difference between being allowed to turn tracing on and not."

> **❓ IF THEY ASK**
>
> **"Could a developer just not register the guardrail?"**
> "Yes — this is a library, not a sandbox. It gives you a place to put the control and makes
> the control auditable. It does not stop someone with commit access from removing it. That is
> a code-review problem, and I would rather say so than oversell it."
>
> **"What is the ordering? Can a guardrail run after the call?"**
> "No, and that is deliberate. The pipeline order is fixed: guardrails, then intercepts, then
> the recorded start, then the real call, then the recorded end. You cannot accidentally
> register a guardrail after the thing it is supposed to guard."
>
> **"What happens if my guardrail itself throws?"**
> "It fails closed — the call does not happen. That is the right default for something whose
> job is refusing."
>
> **"Can we use this with an agent framework we already have?"**
> "It is a Python library that wraps your tool functions, so mostly yes. The integration
> question is where your tool calls actually get dispatched."

> **⚠️ WORTH SAYING IF THE ROOM IS TECHNICAL.** The guardrail expects **`None` to allow** and a
> message to block. It is not a boolean — `return False` **allows** every call, because
> `False is not None`. No error, no warning. This is one of only two genuinely silent failures
> in these six labs, and volunteering it costs you nothing:
>
> "One trap, since you will hit it. This returns `None` to allow. If you return `False`
> thinking that means block — it allows. Silently. **Test your guardrail's failure path, not
> just its happy path.**"

---

### Three API traps this lab found, all fixed in the file

| Trap | What happens | Correct form |
|---|---|---|
| `subscribers.flush()` | `RuntimeError` — cannot block a running event loop | `await subscribers.flush_async()` |
| Callback signature | `TypeError: takes 1 positional argument but 2 were given` | `callback(tool_name, args)` |
| Conditional return | **Silent** — `return False` ALLOWS the call, because False is not None | `return None` to allow, a message to block |
| Outcome import | `ImportError` from `nemo_relay.intercepts` | `from nemo_relay import ToolExecutionInterceptOutcome` |

**The third one is the dangerous one** and worth saying out loud: returning a boolean does
not error. The guardrail registers, runs, and permits everything. This is one of only two
genuinely silent failures in these six labs — the other is a sloppy custom regex in Lab 4.
Lab 5's dropped-ground-truth trap, by contrast, raises loudly, just in a terminal you are not
looking at. **Test the failure path, not just the happy path.**

---

## Part 2 — NeMo Gym  🎤 LIVE

**Forty-eight of the hundred and twenty minutes are here, and twenty-five of those are
Lab 5.** That is deliberate — this is the part that changes what people do on Monday.

---

### 2.1 Lab 3 — a rollout and a reward ✅

> **🎤 SAY BEFORE**
>
> "Second question. *Is my agent getting better or worse than it was last week?*
>
> To answer that you need a number, and you do not have one. So let us make one.
>
> I have an environment here — five multiple-choice questions with known answers, and a
> grader. I am going to have a model attempt all five and score itself.
>
> Two terminals, because the environment runs as servers and the evaluation runs beside it."

```yaml
policy_base_url: https://integrate.api.nvidia.com/v1
policy_api_key: nvapi-...          # free, from build.nvidia.com
policy_model_name: nvidia/nemotron-mini-4b-instruct
```

**Use the 4B, not the 9B.** Both were run on the box. The choice is not close:

| model | reward | wall clock | sets up Lab 4 |
|---|---|---|---|
| `nvidia-nemotron-nano-9b-v2` | 1.0 | 68s (13.69 s/it) | no — nothing can move from a ceiling |
| `nemotron-mini-4b-instruct` | **0.2** | **6s (1.17 s/it)** | yes |

Eleven times faster *and* it leaves somewhere to go. A perfect score is a weak demo: the
interesting question is always which ones failed and why.

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
    --output results/mcqa_rollouts.jsonl --num-repeats 1
```

**Real output** — verified on the box, NeMo Gym 0.5.0:

```
Collecting rollouts: 100%|██████| 5/5 [00:05<00:00,  1.17s/it]
Key metrics for mcqa_simple_agent:
{
    "mean/reward": 0.2,
    "pass@1[avg-of-1]/accuracy": 20.0,
    "pass@1[avg-of-1]/no_answer": 80.0,
    "pass@1/no_answer": 80.0,
    "majority@1/accuracy": 20.0,
    "pass@1/accuracy": 20.0
}
Fully materialized inputs: results/mcqa_rollouts_materialized_inputs.jsonl
Rollouts: results/mcqa_rollouts.jsonl
Aggregate metrics: results/mcqa_rollouts_aggregate_metrics.json
```

**Then read the rows, not just the mean:**

```bash
python3 rewards.py --rows results/mcqa_rollouts.jsonl
```

**Real output** — verified on the box:

```
   ROW   REWARD   EXPECTED    EXTRACTED
  ------------------------------------------
     0     0.00   E           None
     1     0.00   B           None
     2     1.00   C           C             hit
     3     0.00   E           None
     4     0.00   I           None
  ------------------------------------------
  5 rows, mean reward 0.200, 1 scored above zero
```

**`extracted: None` on four of five rows is the whole setup for Lab 4.** Not a wrong letter
— *nothing*. The grader could not find an answer at all. This is more convincing on screen
than the aggregate, because the room can see the four blanks.

> **⚠ Sanity-check before you trust anything downstream.** If the mean is **0.0** with
> `no_answer 100%`, stop. Your rollouts are empty, and every grading mode in Lab 4 will
> faithfully report zero for a file with nothing in it — five identical zeros that look like
> a finding and are not.
>
> **This cost us half an hour on the box.** The culprit was a stale `mcqa_rollouts.jsonl`
> left behind by an earlier failed run. Check the endpoint directly first — five seconds, no
> Ray, no Gym:
>
> ```bash
> KEY=$(grep policy_api_key env.yaml | awk '{print $2}')
> MODEL=$(grep policy_model_name env.yaml | awk '{print $2}')
> curl -s https://integrate.api.nvidia.com/v1/chat/completions \
>   -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
>   -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single letter B and nothing else.\"}],\"max_tokens\":10}"
> ```
>
> Returns `B` → the endpoint is fine and the rollouts file is at fault. Clear the derived
> files and re-run Lab 3:
>
> ```bash
> ls -la results/                              # is mcqa_rollouts.jsonl older than today?
> rm -f results/rv_* results/mcqa_rollouts*
> ```

> **🎤 SAY AFTER** *(do not rush to the next lab — this is the setup for everything)*
>
> "There it is. **Zero point two.** Six seconds, five tasks, and now we have a number.
>
> That is already worth something — it is reproducible, I can run it again next week, and I
> can watch it move.
>
> But do not stop at the headline. **Look at the second metric.** Accuracy twenty per cent.
> `no_answer` — **eighty per cent.**
>
> Read what that actually says. Four of the five did not produce a wrong answer. They produced
> **nothing the grader could read at all.** There is not a single wrong-but-parseable answer
> in the set.
>
> So — the model scored 0.2. What would you do next?"
>
> *(let them answer — someone will say train it, or use a bigger model)*
>
> "That is the instinct, and it is what almost everyone says. Hold that thought. **A low score
> has at least two completely different causes and this number cannot tell you which one you
> have.** The next lab takes about a minute and it answers that question."

> **❓ IF THEY ASK**
>
> **"Why such a small model?"**
> "Deliberately. The nine-billion model scores a perfect 1.0 on this in sixty-eight seconds.
> This one scores 0.2 in six. It is eleven times faster **and** it leaves the next lab
> somewhere to go — you cannot demonstrate anything from a ceiling."
>
> **"Is this running on your GPU?"**
> "No — and this is worth knowing. Labs three through five send inference to a hosted
> endpoint. **You can build and validate an entire evaluation environment without owning a
> GPU.** The GPU only becomes necessary when you start training, and I will show you exactly
> where that line is."
>
> **"What is `inference_provider`?"**
> "Gym ships one server for OpenAI proper and a generic one for anything OpenAI-*compatible*.
> This endpoint is compatible, not OpenAI. It is a small thing that will cost you twenty
> minutes the first time you point this at a customer's self-hosted model."
>
> **"Why three servers for five questions?"**
> "Because they change at different rates and different people own them. The environment is
> yours. The agent loop and the model are swappable. Swapping the model must not mean
> rewriting your grader."

**Three files land in `results/`.** The rollouts file is the one that matters — Lab 4
re-reads it, and training consumes it.

**`gym eval run` does not create its output directory.** Without `mkdir -p results` it dies
with a bare `FileNotFoundError` on the materialized-inputs path, which names nothing useful.

---

### 2.2 Lab 4 — re-score without re-running ✅

Lab 3 ended at **0.2**, with 80% of rows scored `no_answer`. Nothing here calls a model
again. Every number below comes from the five rollouts already on disk.

> **🎤 SAY BEFORE**
>
> "So. The score was 0.2 and the instinct in the room was to train the model.
>
> Before we spend a single GPU-hour on that, let us check something. **I am not going to call
> the model again.** Not once. Everything from here uses the five attempts already sitting on
> disk from the last lab.
>
> The environment I am using ships four different grading modes. Four opinions, from the
> people who wrote it, about how to read an answer out of a model's response. Let us just try
> all of them."

**Pass the selectors again.** Reverify is not a continuation of the run — it is a fresh
invocation that happens to read the run's output, and it re-resolves the servers by name:

```bash
gym eval reverify --resources-server mcqa --model-type inference_provider \
  --rollouts results/mcqa_rollouts.jsonl \
  --inputs  results/mcqa_rollouts_materialized_inputs.jsonl \
  --output  results/rv_lenient_boxed.jsonl --overwrite \
  ++mcqa.resources_servers.mcqa.grading_mode=lenient_boxed
```

Omit `--resources-server` and you get `No server instances are configured, so there is
nothing to run` — which never hints that re-passing the selector is the fix.

#### Where the results go, and how to read them

Every run writes **two files** next to whatever you passed to `--output`:

| File | What is in it |
|---|---|
| `results/rv_<name>.jsonl` | one line per rollout, re-scored — `reward`, `expected_answer`, `extracted_answer` |
| `results/rv_<name>_aggregate_metrics.json` | the `Key metrics` block, as JSON |

**The printed metrics are easy to miss.** They land in the middle of ~20 seconds of Ray
startup chatter and a spray of GCS shutdown warnings, so live they are gone off the top of
the terminal before you have finished the sentence. Read them back instead:

```bash
python3 rewards.py            # compare every run in results/
python3 rewards.py --raw      # dump every metric key, when the table shows dashes
```

Once you reach step 2 and the custom-regex run exists, the per-row view is the one that
makes the close land:

```bash
python3 rewards.py --rows results/rv_custom_regex.jsonl
```

**Only runs you have actually done will appear.** Part-way through the sweep you will see
two or three rows, not five — that is the script reading the directory, not an error.

> **If every column shows a dash**, the metric names in your files are not the ones the
> script matches. `python3 rewards.py --raw` prints every numeric key it found, which is
> enough to fix it in one step. This happened on first use: the keys turned out to be
> agent-prefixed, and an exact-match lookup found nothing.

```
  RUN                      REWARD   ACCURACY   NO_ANSWER
  ------------------------------------------------------
  rollouts                  0.200      20.0%       80.0%
  lenient_boxed             0.200      20.0%       80.0%
  lenient_answer_colon      0.000       0.0%      100.0%
  custom_regex              0.800      80.0%        0.0%
```

**That table is the lab.** Put it on screen at the end rather than asking the room to
remember four numbers you read out over twenty minutes.

The per-row view is what makes the close land — it shows row 4 recovering as **C** against
an expected **I**, flagged `MISS`.

#### Step 1 — tour the shipped grading modes

**First, where the modes are actually declared.** The `mcqa` environment is not in this
repo — it ships inside the installed package, which is why `sed` on
`resources_servers/mcqa/app.py` finds nothing:

```bash
SP=/home/ubuntu/venv-gym/lib/python3.13/site-packages/resources_servers/mcqa
sed -n '/class MCQAResourcesServerConfig/,/^class /p' $SP/app.py
```

```python
class MCQAResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    grading_mode: Optional[
        Literal[
            "strict_single_letter_boxed",
            "lenient_boxed",
            "lenient_answer_colon",
            "lenient_answer_colon_md",
        ]
    ] = None
```

**That is worth putting on screen.** Those four strings are the entire set of opinions the
vendor has about how to read an answer — and in a moment you will find that none of them
help. `REVERIFY_MODE = STATELESS` on the line above is what permits replay at all.

The whole file is worth a look while you are in there — `verify()` is at the bottom and it
is short:

```bash
sed -n '/async def verify/,/^def /p' $SP/app.py
```

#### Now run the sweep — all four modes

The single command above shows the *shape*. This runs the whole tour in one go, which is
what you actually want on the day:

```bash
for m in strict_single_letter_boxed lenient_boxed lenient_answer_colon lenient_answer_colon_md; do
  printf '  %-30s' "$m"
  gym eval reverify --resources-server mcqa --model-type inference_provider \
    --rollouts results/mcqa_rollouts.jsonl \
    --inputs  results/mcqa_rollouts_materialized_inputs.jsonl \
    --output  results/rv_$m.jsonl --overwrite \
    ++mcqa.resources_servers.mcqa.grading_mode=$m > results/rv_$m.log 2>&1 \
    && echo "ok" || echo "FAILED — see results/rv_$m.log"
done

python3 rewards.py
```

**Output goes to a log per mode, on purpose.** Each reverify spins up its own three-server
Ray cluster — about twenty seconds of startup for under a second of grading, then a spray of
GCS shutdown warnings. Watching that four times teaches nothing, and the numbers scroll away.
`rewards.py` reads all four back at the end. **Budget about ninety seconds for the loop.**

If a mode prints `FAILED`, its log has the reason — usually a missing selector.

**Real numbers, verified:**

| `grading_mode` | reward | no_answer |
|---|---|---|
| `strict_single_letter_boxed` *(default)* | 0.2 | 80% |
| `lenient_boxed` | 0.2 | 80% |
| `lenient_answer_colon` | **0.0** | 100% |
| `lenient_answer_colon_md` | 0.2 | 80% |

**Running `strict_single_letter_boxed` explicitly is a control worth having** — it is already
the default, so it should reproduce the Lab 3 baseline exactly. If it does not, something is
wrong with the replay rather than with the modes, and the rest of the table means nothing.

#### See what each mode read, row by row

The aggregate says a mode scored 0.2. This says **which rows it managed to read**:

```bash
python3 rewards.py --matrix
```

**Real output** — verified on the box:

```
  TASK                            T0    T1    T2    T3    T4    REWARD
  EXPECTED                         C     E     B     I     E
  --------------------------------------------------------------------
  rollouts                         .     .     B     .     .     0.200
  strict_single_letter_boxed       .     .     B     .     .     0.200
  lenient_boxed                    .     .     B     .     .     0.200
  lenient_answer_colon             .     .     .     .     .     0.000
  lenient_answer_colon_md          .     .     B     .     .     0.200
  custom_regex                     C     E     B    C*     E     0.800

   .  nothing extracted — the grader could not find an answer
   X  extracted that letter, and it matched
   X* extracted that letter, and it was WRONG
```

> **Columns are tasks, not line numbers.** They are matched on `_ng_task_index`, because
> **Gym re-sorts its output** — it prints *"Sorting results to ensure consistent ordering"* —
> and reverify does not necessarily write rows back in the order it read them. Lining these
> files up by position puts a different task in every column and scrambles the expected
> answers. The first version of this tool did exactly that, and produced a grid where the one
> correct answer appeared at index 0, 2 and 3 in three files describing the same five tasks.
>
> This is the same trap as Lab 5's *"do not zip rollouts against inputs by index"*. It is
> easy to fall into twice.

**This is the single best artefact in Lab 4** — put it on screen for the close instead of the
aggregate table. Three things are visible at once that no summary conveys:

**Four columns of dots.** Rows 0, 1, 3 and 4 were unreadable to every shipped mode. Not
wrong — *unread*.

**`lenient_answer_colon` lost the only letter on the board.** Its row is dots all the way
across, including R2, which every other mode read. A mode called *lenient* went backwards.

**The last row is the payoff.** The custom regex turns four dots into letters — and R4 becomes
`C*`, a star, meaning it extracted an answer and the answer was wrong. **A dot that becomes a
letter is a row the grader was failing to read. A letter with a star is a row the model got
wrong. Those need completely different fixes**, and until this table existed you could not
tell them apart.

> **🎤 SAY AFTER THE SWEEP** *(walk down the four numbers)*
>
> "Four grading modes. Strict — 0.2. Lenient boxed — 0.2. Lenient answer-colon — **zero.**
> Lenient answer-colon-markdown — 0.2.
>
> Not one of them beat the baseline. **And one of them made it worse.**
>
> Sit with that for a second, because the name is doing something misleading. These are not
> four levels of tolerance where each one is a bit more forgiving than the last. **They are
> four different extractors, and picking one replaces the previous one.** The mode called
> *lenient* threw away the only point we had, because the one row that was working answered
> `Answer: \boxed{C}` — and that mode reads the text after `Answer:` instead of opening the
> box.
>
> So: the knob shipped, I turned it through every setting the vendor gave me, and the number
> never moved up. That is usually the moment you go and ask for a bigger model.
>
> **Let us look at what the model actually said instead.**"

> **❓ IF THEY ASK**
>
> **"Why did lenient_boxed not help?"**
> "Because it still requires a `\boxed{}` to exist — it only loosens what is allowed *inside*
> the box. Four of our five rows have no box anywhere, so it bails out before any leniency is
> reached. Worth knowing rather than guessing: the leniency is about the contents, not about
> whether the wrapper is there."

#### Step 2 — write your own extractor

Read one failing row aloud. Show the prompt:

> *"The last line of your response should be in the following format: 'Answer: \boxed{...}'"*

Then show what the model actually said: *"In summary, the correct answer is E"*. Expected
answer: **E**. Scored **0.0**.

> **🎤 SAY** *(point at the prompt, then at the response, then at the zero)*
>
> "The prompt told it, explicitly, to answer in the format `Answer: backslash-boxed-E`.
>
> The model said **'In summary, the correct answer is E.'**
>
> The expected answer was **E**.
>
> It scored **zero**.
>
> Now — is that model wrong? It is not wrong. It got the question right. It just did not
> format the answer the way the grader wanted it.
>
> So what was that grader actually measuring? **It was measuring instruction-following, not
> knowledge.** And it was scoring a correct-but-unformatted answer exactly the same as a wrong
> one.
>
> Nobody sat down and decided that. It fell out of a regular expression."

**The grader was not measuring knowledge. It was measuring instruction-following** — and
scoring a correct-but-unformatted answer identically to a wrong one. Nobody chose that. It
fell out of a regex.

> **🎤 SAY BEFORE THE PATCH**
>
> "So let us write our own. **Not fork the library, not patch anything** — the environment
> ships an extension point for exactly this, and it is checked *before* the built-in modes.
>
> Five patterns. Tried in order. Taking the **rightmost** match, because these models reason
> first and commit to an answer last.
>
> Same five rollouts. Still no model calls."

`verify()` checks `template_metadata["output_regex"]` **before** `grading_mode` — a shipped,
per-row extension point that takes a string or a list of patterns, tried in order, rightmost
match wins. Exactly the "reasoning first, answer last" shape these responses have. Patch it
onto both files; reverify reads request fields from the inputs and the response from the
rollouts:

```bash
python - <<'PY'
import json
REGEX = [
    r"correct answer is[:\s]*\(?\[?([A-J])\)?\]?",
    r"answer is[:\s]*\(?\[?([A-J])\)?\]?",
    r"statement\s+\(?([A-J])\)?\s+(?:is\s+)?the\s+correct",
    r"choice\s+\(?([A-J])\)?\s+is\s+the\s+(?:best|correct)",
    r"\banswer\s*[:\-]\s*\(?\[?([A-J])\)?\]?",
]
pairs = [("results/mcqa_rollouts.jsonl", "results/mcqa_rollouts_regex.jsonl"),
         ("results/mcqa_rollouts_materialized_inputs.jsonl", "results/mcqa_inputs_regex.jsonl")]
for src, dst in pairs:
    n = 0
    with open(src) as f, open(dst, "w") as o:
        for line in f:
            r = json.loads(line)
            r["template_metadata"] = {"output_regex": REGEX}
            o.write(json.dumps(r) + "\n"); n += 1
    print(f"{dst}: {n} rows")
PY
```

```bash
gym eval reverify --resources-server mcqa --model-type inference_provider \
  --rollouts results/mcqa_rollouts_regex.jsonl \
  --inputs  results/mcqa_inputs_regex.jsonl \
  --output  results/rv_custom_regex.jsonl --overwrite
```

No `grading_mode` override — `template_metadata` outranks it.

**Real output** — verified on the box:

```
Key metrics for mcqa_simple_agent:
{
    "mean/reward": 0.8,
    "pass@1[avg-of-1]/accuracy": 80.0,
    "pass@1[avg-of-1]/no_answer": 0.0,
    "pass@1/no_answer": 0.0,
    "majority@1/accuracy": 80.0,
    "pass@1/accuracy": 80.0
}
```

**0.2 → 0.8. Zero model calls. `no_answer` 80% → 0%.**

> **🎤 SAY AFTER**
>
> "**Zero point two, to zero point eight.** And I want to be precise about what just happened,
> because it is easy to miss.
>
> **I did not call the model.** Not once. Those are the identical five attempts, on disk,
> unchanged, from six minutes ago. The only thing that changed is five lines of regular
> expression describing how to read an answer.
>
> And look at the second metric again — `no_answer` went from eighty per cent to **zero**.
> Every single response now produces something the grader can read.
>
> **Rollouts are expensive. Grading them is free.** You collect once, and then you can argue
> about what 'correct' means as many times as the argument needs — with a customer, using
> their own data, in seconds rather than in another eval run."

#### Step 3 — the row that stayed wrong

Do not skip this. Per-row, the recovery looks like:

| row | expected | recovered | |
|---|---|---|---|
| 0 | B | B | hit |
| 1 | E | E | hit |
| 2 | C | C | hit |
| 3 | E | E | hit |
| 4 | I | **C** | miss — confidently wrong |

> **🎤 SAY — THE CLOSE** *(this is the payoff of the entire Gym section; slow down)*
>
> "Four rows recovered. One did not. Look at row four.
>
> Expected answer: **I**. The model said **C**. So it stays at zero — and it should.
>
> But notice what that means. **Row four was never a hard question the model failed.** It
> answered confidently, and it was wrong. And the strict grader we started with scored that
> **exactly the same as the four rows that were right.**
>
> So let me put the whole lab in one sentence. **Four of those five zeros belonged to the
> grader. Exactly one belonged to the model.** One number could not tell you which was which —
> and the fix for each is completely different.
>
> If you had trained on this, you would have spent a week and a GPU budget teaching a model
> that already knew the answers to format them differently. **You cannot train your way out of
> a bad regex.**
>
> And the flip side, which is the thing I would actually like you to take away: **the reward
> function is code you own.** It is not a property of the model, it is not handed down by the
> benchmark. You can fix it for free. And — as the mode called *lenient* showed us — you can
> break it just as easily."

> **❓ IF THEY ASK**
>
> **"Isn't writing your own grader just moving the goalposts?"**
> "It absolutely can be, and that is a fair challenge. The discipline is to write the grader
> **before** you look at the scores, and to version it like any other code so the change is
> visible in review. What you saw me do — look at failures, then loosen the grader — is
> exactly the thing to be careful about. I did it on stage because the failure mode was
> obviously formatting, not correctness. If I could not tell those apart, I should not have
> touched it."
>
> **"How would we catch this in our own evals?"**
> "Read the rows, not the mean. Every time. The aggregate told me 0.2 and told me nothing
> useful. Five minutes reading five actual responses told me everything."
>
> **"Does this mean benchmark scores are unreliable?"**
> "It means a benchmark score is a statement about a model **and** a grader, and people quote
> it as though it is only about the model. That is worth remembering next time someone shows
> you a leaderboard."
>
> **"Could a sloppy regex make the score look better than it is?"**
> "Yes — and that is the genuine hazard here. The custom-regex path trusts your pattern over
> the list of valid options, so a careless pattern can manufacture answers out of prose and
> report a healthy-looking number. That one *is* silent. Review your grader like you review
> production code."

#### Two traps in this lab

**`_parse_answer_with_custom_regex` trusts your regex over `allowed_letters`** and will
return a captured letter even when it is not a valid option for that question. A sloppy
pattern manufactures answers out of prose and reports a healthy-looking score — and unlike
Lab 5's dropped field, **this one really does stay quiet**: no error, no warning, just a
number that looks better than it is. Alongside the Relay boolean guardrail, it is one of only
two genuinely silent failures in these six labs.

**Reverify spins up the full three-server Ray cluster** — roughly 20 seconds of startup for
under a second of grading, then a spray of `Failed to connect to GCS ... TimedOut` warnings
on the way down. They look like errors and are not. Tell the room to ignore the red text,
and start talking before you press return. If the warnings persist between runs, a cluster
from an earlier lab is still alive:

```bash
bash gymclean.sh              # shows every target, asks, then kills
bash gymclean.sh --dry-run    # show targets, kill nothing
```

> **⚠️ Do not use a bare `pkill -9 -f raylet`.** `pkill -f` matches the *whole command
> line*, so it also kills a `tail -f raylet.log`, an editor with a Ray file open, and — on
> at least one host — PID 1, whose init carried the pattern in its argv. I did exactly this
> while writing `gymclean.sh` and killed the machine I was testing on.
>
> `gymclean.sh` filters on the **executable name** (`ps -o comm=`), never PID 1, never its
> own process group, and prints every target with its full command line before signalling
> anything. Verified against decoys: `tail -f /var/log/raylet.log` and an editor holding
> `notes-about-gcs_server.md` are both left alone.

**Reverify is gated** on the server declaring `ReverifyMode.STATELESS`. `mcqa` does, so this
all works. `--force` exists for servers that do not and prefixes its output `unsafe_` — the
gate is a feature, so check why before overriding it.

---

### 2.3 Lab 5 — build an environment for your own task ✅

The centrepiece. Twenty-five minutes, nine steps. A finished copy lives in
`gym/support_triage/` — copy it in if anything stalls. **Every step below was run on the
box; the outputs are real.**

> **🎤 SAY BEFORE THE WHOLE LAB**
>
> "Everything so far has been someone else's task. Multiple-choice questions with known
> answers — useful for showing you the machinery, useless for your actual job.
>
> So for the next twenty-five minutes we are going to build one from scratch, for a task that
> looks like something you might genuinely have. **A support ticket arrives as free text. The
> model has to return a severity and a team, as JSON.**
>
> I want you to watch how much of this is code I write versus code that is handed to me,
> because that ratio is the thing worth taking away."

---

**Step 1 — scaffold** (~2 min)

> **What a scaffold is, and why the lab starts here.** `gym env init` is a code generator —
> the same idea as `npm init` or `django-admin startproject`. You give it a name, it writes a
> working skeleton.
>
> | File | What it is |
> |---|---|
> | `app.py` | The server — config class, request/response models, and `verify()` |
> | `configs/<name>.yaml` | How it gets discovered, and which agent and model it wires to |
> | `data/` | Where your task rows go. Empty except a `.gitignore` |
> | `tests/test_app.py` | pytest against your verifier — no model, no GPU |
> | `requirements.txt` | Dependencies for this server's own venv |
> | `README.md` | A placeholder |
>
> **Why generate it live instead of handing them a finished directory?** Because the claim —
> *an environment is a directory that gets discovered, no registry, no manifest, no packaging
> step* — is otherwise just an assertion. Hand them a folder and they take your word for it.
> Let them watch six files appear and then watch `gym env start` find them by name, and they
> have seen it. Same reason you run the tests rather than saying they pass.
>
> **And why we overwrite it in step 2:** writing a real verifier from scratch does not fit in
> twenty-five minutes. The sequence is *generate the skeleton so they see how little there is
> → drop in the finished version so they see what it becomes.* Our `app.py` is that same
> skeleton with the one hole filled.

```bash
gym env init --resources-server support_triage
find resources_servers/support_triage -type f \
  -not -path '*/.venv/*' -not -path '*/__pycache__/*' -not -path '*/.pytest_cache/*' | sort
```

> **⚠ Those exclusions matter on any run after the first.** Once `gym env test` has run,
> `resources_servers/support_triage/.venv/` holds **174 packages**, and a bare `find` returns
> thousands of lines instead of six — which looks broken on stage and buries the point.
>
> **Do not delete that venv to tidy the listing.** It is the warm cache, and rebuilding it
> mid-lab is the single biggest timing risk in this workshop. Seeing `.venv/` there on a
> rehearsal run is a good sign, not a mess.

**Real output** — six files:

```
resources_servers/support_triage/README.md
resources_servers/support_triage/app.py
resources_servers/support_triage/configs/support_triage.yaml
resources_servers/support_triage/data/.gitignore
resources_servers/support_triage/requirements.txt
resources_servers/support_triage/tests/test_app.py
```

> **🎤 SAY AFTER**
>
> "**Six files.** That is an environment. Not a registry entry, not a plugin manifest, not a
> packaging step — a directory that gets discovered because it is there.
>
> And here is the bit that surprises people: **it already works.** The generated `verify()`
> returns a reward of 1.0 for everything. It is useless, but it is a functioning environment
> before I have written a line.
>
> Notice what is *missing*, though. Look at `data/` — there is a `.gitignore` in there and
> nothing else. **No example data.** Hold that thought for about four minutes; it matters."

**Put the generated `verify()` on screen — it is two lines, and that is the whole point:**

```bash
sed -n '/async def verify/,/^if __name__/p' resources_servers/support_triage/app.py
```

**Real output** — verified on the box, this is the entire generated verifier:

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    return BaseVerifyResponse(**body.model_dump(), reward=1.0)
```

> **🎤 SAY**
>
> "There it is. **Two lines.** Everything scores one point zero, unconditionally.
>
> So the environment is complete and functioning before I have written anything — it is just
> **lying to me.** Every answer is perfect, including the wrong ones.
>
> And that is a better starting point than an empty file, because the plumbing is already
> proven. The servers come up, the rollouts collect, the metrics aggregate. **The only thing
> missing is the judgement** — and the judgement is the part that is yours."

> **⚠ Do this before step 2 overwrites `app.py`.** Once the finished verifier is copied in,
> the scaffold's version is gone. To see it again afterwards, make a throwaway:
> `gym env init --resources-server scaffold_peek`, look, then
> `rm -rf resources_servers/scaffold_peek`.

**⚠ Keep the scaffold's `requirements.txt`.** It generates `nemo-gym[dev]`. Shipped
environments like `mcqa` instead carry `-e nemo-gym[dev] @ ../../`, and copying that is a
trap: those live *inside* site-packages, where `../../` is the Gym source tree. In your own
repo `../../` is your repo root, and the editable install has nothing to point at. Append,
never replace:

```bash
printf '\npytest\npytest-asyncio\n' >> resources_servers/support_triage/requirements.txt
```

Without those two, `gym env test` prints "no tests ran" and step 6 quietly proves nothing.

**Step 2 — the verifier** (~5 min)

**The finished verifier is already in this repo** at `gym/support_triage/app.py`. Copy it
over the scaffold's version:

```bash
cp gym/support_triage/app.py resources_servers/support_triage/app.py
```

Then put `verify()` on screen and read it aloud:

```bash
sed -n '/^SEVERITIES/,/^TEAMS/p;/async def verify/,/parsed=parsed)/p' \
    resources_servers/support_triage/app.py
```

**This is what appears** — the whole of the judgement, verbatim from the repo:

```python
SEVERITIES = {"P0", "P1", "P2"}
TEAMS = {"billing", "auth", "infra"}

async def verify(self, body: SupportTriageVerifyRequest) -> SupportTriageVerifyResponse:
    text = self._assistant_text(body.response)

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Not JSON at all. No partial credit for prose.
        return SupportTriageVerifyResponse(**body.model_dump(), reward=0.0, parsed=None)

    truth = body.verifier_metadata
    hits = 0
    if parsed.get("severity") in SEVERITIES and parsed.get("severity") == truth.get("severity"):
        hits += 1
    if parsed.get("team") in TEAMS and parsed.get("team") == truth.get("team"):
        hits += 1

    # Partial credit, on purpose. Pass/fail gives a training loop almost
    # nothing to work with; "one of two fields right" is a gradient.
    return SupportTriageVerifyResponse(
        **body.model_dump(), reward=hits / 2.0, parsed=parsed)
```

**If you prefer to build it live rather than copy it**, type it out from this listing — it is
short enough. Copying is the safer choice on the day; typing is better if the room is small
and engaged.

> **🎤 SAY WITH `verify()` ON SCREEN**
>
> "This is the only part I actually write. About forty lines, and the function that matters is
> about fifteen.
>
> Read it with me. **Pull the assistant's text out. Parse it as JSON. Compare two fields
> against the ground truth. Return a number.** That is an environment.
>
> One decision in there worth pausing on: `reward = hits / 2.0`. **Partial credit.** If it
> gets the severity right and the team wrong, that scores a half, not a zero.
>
> That is one line of difference and it matters more than it looks. A pass-or-fail signal
> gives a training loop almost nothing to climb — everything that is not perfect looks
> identically bad. **'One of two fields right' is a gradient.** It is the difference between a
> metric you report and a signal you can improve against.
>
> The other decision: I check that the team is one of my four legal values *before* I compare
> it to the answer. That looks like belt-and-braces. It is not, and I will show you why in
> about fifteen minutes."

> **❓ IF THEY ASK**
>
> **"Is that really all of it?"**
> "That is really all of it. There is boilerplate around it — the request and response models —
> but the logic is what you can see. If you can describe what 'good' means for your task in
> forty lines of Python, you can build one of these."
>
> **"What if scoring needs a database lookup, or an API call?"**
> "Then do it — it is a normal Python function in a normal FastAPI server. People do call out
> to sandboxes, test runners and real systems from here. Just be aware that if your verifier
> has side effects, the replay feature you saw in Lab 4 no longer applies to you, and the
> server has to declare that."
>
> **"Can the verifier use another model as a judge?"**
> "Yes, and people do. It costs you money per evaluation and it introduces a second thing that
> can be wrong. I would exhaust rules first — Lab 4 was a whole demonstration of how much
> damage a bad grader does, and an LLM judge is a grader you can inspect much less easily."

**Step 3 — the config** (~2 min)

```bash
cp gym/support_triage/configs/support_triage.yaml \
   resources_servers/support_triage/configs/support_triage.yaml
cat resources_servers/support_triage/configs/support_triage.yaml
```

**This is what appears** — the whole config, verbatim from the repo:

```yaml
support_triage:
  resources_servers:
    support_triage:
      entrypoint: app.py
      domain: agent
      verified: false
      description: Single-step support-ticket triage, scored on severity and team
      value: Route incoming tickets without a human first-pass

support_triage_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: support_triage
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: example
        type: example
        jsonl_fpath: resources_servers/support_triage/data/example.jsonl
```

**Two changes from what the scaffold generated.** `domain: agent`, and the generated `train`
and `validation` dataset blocks are **deleted** — they point at files the scaffold never
created, and they require a licence field you have no reason to fill in.

**The top-level key must equal the server name.** Every shipped environment uses one string
for both (`mcqa:` → `resources_servers: mcqa:`), and `--resources-server <name>` resolves
against the top-level key. Naming them differently — `support_triage_resources_server:` over
`support_triage:` — means the server can never start. This bit us in draft.

**Step 4 — the data** (~2 min)

```bash
cp gym/support_triage/data/example.jsonl \
   resources_servers/support_triage/data/example.jsonl
head -1 resources_servers/support_triage/data/example.jsonl | python3 -m json.tool
```

**This is what appears** — one row, and all five have this shape:

```json
{
    "responses_create_params": {
        "input": [
            {
                "role": "system",
                "content": "Classify the support ticket. Reply with ONLY a JSON object and nothing else: {\"severity\": \"P0\"|\"P1\"|\"P2\", \"team\": \"billing\"|\"auth\"|\"infra\"}. P0 is a live outage, P1 is degraded or blocking for a group, P2 is everything else."
            },
            {
                "role": "user",
                "content": "Checkout has been returning 500s for every customer for the last 20 minutes."
            }
        ]
    },
    "verifier_metadata": {
        "severity": "P0",
        "team": "infra"
    }
}
```

**Point at the two halves.** `responses_create_params` is what the model sees.
`verifier_metadata` is the correct answer, which it never sees — that is the field the
scaffold silently drops if you do not declare it, and step 7 is about exactly that.

> **🎤 SAY WITH THE DATA ON SCREEN**
>
> "Five tickets. Each row is the prompt I send, and — riding alongside it — **the correct
> answer**, which the model never sees. That is the whole shape of an evaluation dataset.
>
> And these are invented. Made-up tickets, made-up teams. **Do not build your first one of
> these on real customer records**, including your own company's. You will want to paste
> examples into a slide eventually, and the moment you do, synthetic is the difference between
> a demo and an incident."

**Step 5 — validate** (~1 min)

```bash
gym env validate --config resources_servers/support_triage/configs/support_triage.yaml
```

**Real output:** `✓ Config is valid.` — instant, with no server running.

> **🎤 SAY AFTER**
>
> "Valid, instantly, with no model server running and nothing loaded. That is the inner loop —
> you can iterate on an environment all day without spending a cent on inference.
>
> But **read what it actually checked.** It checked the config. It did **not** check that my
> data file exists — and remember, four minutes ago we saw that the scaffold never created
> one.
>
> So this is green, and my environment has no data in it. **A passing config check is not a
> working environment.** Useful thing to know before you trust one in CI."

**Step 6 — test** (~4 min)

```bash
cp gym/support_triage/tests/test_app.py \
   resources_servers/support_triage/tests/test_app.py
gym env test --resources-server support_triage
```

**That copy is not optional.** Without it you are running the scaffold's generated test, not
the eight-case suite, and the output below will not match. The suite is what makes step 7
work — it contains the regression test that catches the dropped ground truth.

First run builds the per-server venv — **174 packages, a few seconds**. Warm runs are
instant (`Checked 6 packages in 11ms`).

**Real output:**

```
collected 8 items
tests/test_app.py ........                                       [100%]
8 passed in 1.03s
```

**⚠ The response fixture is where this goes wrong.** `NeMoGymResponse` subclasses OpenAI's
`Response`, so **eight fields are required** — `id`, `created_at`, `model`, `object`,
`output`, `parallel_tool_calls`, `tool_choice`, `tools` — even though a verifier only ever
reads `output`. Faking it with a `SimpleNamespace` fails all eight tests inside the fixture,
before your `verify()` is called even once:

```
ValidationError: 1 validation error for SupportTriageVerifyRequest
response
  Input should be a valid dictionary or instance of NeMoGymResponse
```

Copy the shape from a shipped environment's tests rather than reverse-engineering it:
`site-packages/resources_servers/mcqa/tests/test_app.py`.

> **🎤 SAY AFTER STEP 6**
>
> "Eight tests, one second, no model. **That is your inner loop** — you can develop a reward
> function all day for free, and only pay when you actually collect rollouts.
>
> Which brings me to the most useful three minutes in this lab. I am going to break it on
> purpose."

**Step 7 — break it on purpose** (~3 min)

> **🎤 SAY BEFORE**
>
> "There is exactly one line in my verifier that declares the ground truth field — the correct
> answer that rides along with each ticket.
>
> I am going to delete it. Just that one line. **What do you expect to happen?**"

Comment out the one declared field and re-run the tests:

```bash
cp resources_servers/support_triage/app.py /tmp/app.py.bak
sed -i 's/^    verifier_metadata: dict\[str, Any\]/    # verifier_metadata: dict[str, Any]/' \
    resources_servers/support_triage/app.py
gym env test --resources-server support_triage 2>&1 | tail -12
```

**Real output — and it is not what the docs led me to expect:**

```
AttributeError: 'SupportTriageVerifyRequest' object has no attribute 'verifier_metadata'
6 failed, 2 passed in 1.17s
```

Pydantic really does drop undeclared fields — `BaseVerifyRequest.model_config` is `{}`, so
plain `extra="ignore"`. But reading the dropped field then raises `AttributeError`. **This
does not fail silently.** Run it against a live eval and the run dies on row 0:

```
aiohttp.client_exceptions.ClientResponseError: 500, Internal Server Error, url='.../run'
```

> **🎤 SAY AFTER** *(point at the pass/fail count first, then at the traceback)*
>
> "So — six failed. Which is what you would expect, and honestly it is a relief; it means this
> does not fail silently, and I had assumed it would.
>
> **But look at the other number. Two passed.**
>
> Which two? The one where the model replies in prose, and the one where it returns nothing.
> **The two tests that assert the reward should be zero.**
>
> And of course they pass — the verifier bails out early on unparseable output, before it ever
> goes looking for ground truth. So that code path is completely destroyed, and those two
> tests are perfectly happy.
>
> Think about what that means for a test suite you write in a hurry. **If your tests mostly
> check that bad input scores badly — which is a very natural thing to write — you can ship an
> environment whose entire ground-truth path is broken, and your suite goes green.**"
>
> *(now the traceback, if you ran the live eval version)*
>
> "And one more, which is the most practically useful thing in this lab. When this happens
> against a real run, the error you see is a `500`. Nine frames of network library. **The
> words `verifier_metadata` appear nowhere in it.**
>
> The actual error is in the *other terminal* — the server process. So: **when a Gym
> environment returns a 500, stop reading the traceback in front of you and go and look at the
> server's.** That habit alone will save you an afternoon."

> **❓ IF THEY ASK**
>
> **"Why does Pydantic drop the field instead of erroring?"**
> "Default behaviour — extra fields on the incoming data that the model does not declare are
> ignored rather than rejected. It is a reasonable default for an API boundary and a
> dangerous one here, because your data quietly loses a field it was carrying."
>
> **"How would we catch this in CI?"**
> "Exactly the test I have at the bottom of that file — one that reaches the comparison and
> asserts a *non-zero* reward. If your suite has no test that can only pass when ground truth
> arrives, you have no coverage of the thing that matters."
>
> **"Is this documented?"**
> "It is warned about, and I still hit it. Which is roughly the point — a warning in a document
> is not the same as a test in your repo."

`test_ground_truth_survives_parsing` is the regression test that turns this into a failure
you cannot miss.

**Step 8 — fix it** (~2 min)

```bash
cp /tmp/app.py.bak resources_servers/support_triage/app.py
gym env test --resources-server support_triage 2>&1 | tail -3
```

Back to `8 passed`. **Restart the server in terminal 1** — it is holding the broken module.

**Step 9 — a real reward** (~4 min)

> **🎤 SAY BEFORE**
>
> "Right — it is fixed, the tests pass, and so far this environment has never seen a model.
> Let us point a real one at it and get a number for five support tickets."

```bash
gym env start --resources-server support_triage --model-type inference_provider  # terminal 1
gym eval run --no-serve --agent support_triage_simple_agent \
    --input resources_servers/support_triage/data/example.jsonl \
    --output results/triage_rollouts.jsonl --num-repeats 1                       # terminal 2
```

**Real output:**

```
Collecting rollouts: 100%|██████| 5/5 [00:00<00:00, 13.51it/s]
Key metrics for support_triage_simple_agent:
{
    "mean/reward": 0.6,
    "mean/input_tokens": 98.4,
    "mean/output_tokens": 14.2,
    "mean/total_tokens": 112.6
}
```

**0.6 across five tickets, in under a second.** Thirty times faster than the mcqa rollouts in
Lab 3 — the model is emitting 14 tokens of JSON instead of paragraphs of reasoning.

> **🎤 SAY AFTER**
>
> "**Zero point six**, across five tickets, in under a second. Twenty-five minutes ago this
> environment did not exist.
>
> And notice it is not a round number — it is not three out of five. **That is the partial
> credit doing its job.** Some of those tickets got one field right and one wrong, and the
> environment says so instead of throwing them in with the total failures.
>
> But do not stop at 0.6. **Let us read the rows.**"

**Then open the rollouts and go through them one at a time.** This is the payoff:

```bash
python3 rewards.py --triage results/triage_rollouts.jsonl
```

```
  TICKET                                              TRUTH           MODEL SAID       REWARD
  -----------------------------------------------------------------------------------------
  Checkout has been returning 500s for every custome  P0 / infra      P1 / auth           0.0
  SSO login loops back to the sign-in page for the w  P1 / auth       P1 / marketing      0.5
  Nightly batch job has not run for three days and n  P1 / infra      P2 / infra          0.5
  Could someone rotate the API key for the staging t  P2 / auth       P2 / auth           1.0
  The March invoice shows the wrong VAT rate on line  P2 / billing    P2 / billing        1.0
```

> **Why this needs a join rather than a `for` loop over the rollouts.** The rollouts file
> carries `reward` and `parsed` — **but not the ticket text and not the ground truth.**
> `verifier_metadata` is not echoed back into it. So a plain loop gives you two of the four
> columns and you cannot tell which ticket you are looking at.
>
> `--triage` joins the rollouts to `data/example.jsonl` **on the ticket text**, not on line
> position, because the two files are not in the same order. Point it somewhere else with
> `--data`.

**Real result:**

| ticket | ground truth | the model said | reward |
|---|---|---|---|
| Checkout returning 500s for every customer, 20 minutes | **P0** / infra | P1 / auth | **0.0** |
| SSO login loops for the whole marketing team | P1 / **auth** | P1 / **marketing** | 0.5 |
| Nightly batch job has not run for three days | **P1** / infra | P2 / infra | 0.5 |
| Rotate the API key for the staging tenant | P2 / auth | P2 / auth | 1.0 |
| March invoice shows the wrong VAT rate | P2 / billing | P2 / billing | 1.0 |

> **🎤 SAY — THE CLOSE OF LAB 5** *(the best forty seconds in the lab; take them)*
>
> "Two things in this table, and both are better than the number above it.
>
> **First — look at row two. It invented a team.**
>
> It answered `team: marketing`. Marketing is not one of my four values. It is not an option
> I offered. And where did it get it? Read the ticket: *'SSO login loops for the whole
> **marketing** team.'* It lifted the word straight out of the input.
>
> That is what a language model does when a field looks like a slot — **it fills it from
> context.** Not maliciously, not randomly. It is a very reasonable-looking wrong answer.
>
> And remember that check I flagged twenty minutes ago, the one that looked like
> belt-and-braces? That is the only reason this scores as a miss instead of being quietly
> compared as a string. **Validate against your own enum, not just against the ground
> truth.**
>
> **Second — look at which one it got completely wrong.**
>
> Four tickets scored something. The single zero is *'checkout returning 500s for every
> customer.'* The live outage. **The highest-stakes row in the set.**
>
> So this model is competent on invoices and API key rotations, and worst on the one ticket
> that pages a human at three in the morning. That is exactly backwards from what you would
> want — and **an aggregate of 0.6 hides it completely.**
>
> Which is the same lesson as Lab 4, arriving from a different direction. **Read the rows,
> not the mean.**"

> **❓ IF THEY ASK**
>
> **"Would a bigger model fix the outage row?"**
> "Probably, partly — and now you can *measure* whether it does, for about a cent, which is
> the whole point of having built this. That is a much better position than arguing about it."
>
> **"Five tickets is not a lot."**
> "It is not, and I would not make a decision on five. It is enough to show you the
> machinery. For a real evaluation you want hundreds, and the good news is that adding them
> is a data problem, not an engineering one — the environment does not change."
>
> **"How would we use this against our own tickets?"**
> "Swap the data file and the two field names in the verifier. Genuinely. **The hard part is
> not the code — it is agreeing internally what the correct answer is**, and that is a
> conversation with your support leads, not an engineering task."
>
> **"Can this run in CI on every model upgrade?"**
> "That is exactly the use I would push you toward. It is a directory in your repo, it runs
> from the command line, and it exits with a number. A regression test for your agent."

**Two mechanical notes for whoever repeats this.** `verifier_metadata` is *not* echoed into
the rollouts file — ground truth comes from your input JSONL. And the output is re-sorted
("Sorting results to ensure consistent ordering"), so **do not zip rollouts against inputs
by index**; match on content or carry an id.

**If you are behind at step 7,** skip the deliberate break and go straight to the working
version. It is the best teaching moment in the lab and the only cuttable part.

---

## Part 3 — Megatron Bridge  🎤 LIVE

**The only lab that needs a GPU, a 37 GB image and NGC credentials.** Highest risk of the
six. If it stalls, talk over the code on the slide and move on — do not debug live.

---

### 3.1 Lab 6 — HF in, Megatron out, HF back ✅

> **🎤 SAY BEFORE**
>
> "Third question, and the last one. *How do I make it better at my task?*
>
> You have a number now, so you could train against it. Which raises a question people ask me
> constantly, usually about six months too late: **if I train with NVIDIA's stack, am I locked
> into NVIDIA's stack?**
>
> So let me just show you. I am going to take a model off Hugging Face, convert it into
> Megatron's format — the sharded, parallelism-aware layout you train large models in — and
> then convert it straight back out again.
>
> First, though, one thing I want you to see before we start."

```bash
docker run --rm -it --gpus all -v $(pwd):/workdir -w /workdir \
    --entrypoint bash nvcr.io/nvidia/nemo:25.09
```

**Check the bridge version first.** This is the headline of Part 3, not a preliminary:

```bash
python3 -c "import importlib.metadata as m; print(m.version('megatron-bridge'))"
```

**Real output:** `0.1.0rc4`.

> **Why not `pip show megatron-bridge | head -3`?** `head` closes the pipe as soon as it has
> its three lines, `pip` is still writing, and Python prints a `BrokenPipeError` traceback at
> shutdown. Harmless — and it puts red text on screen at exactly the moment you are making a
> point about version drift. `sed -n '1,3p'` is safe because it reads to the end; so is a
> plain `grep`. **`grep -m3` is not** — it exits early, same as `head`.

> **🎤 SAY AFTER** *(do this first, not as a footnote — it buys credibility for everything else)*
>
> "**Zero point one point zero, release candidate four.**
>
> The published package is on 0.6.0. The container — which is the *supported* way to install
> this — is running a release candidate from five minor versions ago.
>
> I am showing you that on purpose. Of the three tools today, **this is the least settled
> one**, and you would find that out in week two anyway. Better you hear it from me now.
>
> The practical advice that falls out of it: **pin the image, not the package.** There is no
> supported pip install here, so the container *is* your version. Treat the tag as the thing
> you control."

> **❓ IF THEY ASK**
>
> **"Should we be using this in production then?"**
> "For the round trip I am about to show you — it works, and it is exact, and I will prove
> that rather than assert it. For building a platform on top of, I would pin hard, test on
> every image bump, and keep an eye on the release notes. That is not unusual advice for
> something moving this fast; it is just advice people skip."

The API happened to survive the gap: `from_hf_pretrained`, `to_megatron_provider`,
`provide_distributed_model` and `save_hf_pretrained` all exist in 0.1.0rc4. Do not assume
that holds for the next image.

**Ignore the noise.** Every invocation prints this, and it is the container's own version
skew, not anything you did. Warn the room before it appears on the projector:

```
Skipping import of cpp extensions due to incompatible torch version
2.8.0a0+...nv25.06 for torchao version 0.14.1
```

**Smoke test next** — if this fails, the round trip was never going to work:

```bash
python megatron_bridge/list_architectures.py
```

**Real output:**

```
Megatron Bridge can convert 6 architectures:
  DeepseekV2ForCausalLM
  DeepseekV3ForCausalLM
  LlamaForCausalLM
  Qwen2ForCausalLM
  Qwen3ForCausalLM
  Qwen3MoeForCausalLM
```

> **🎤 SAY AFTER**
>
> "**Six.** Not 'any model' — six architecture families.
>
> Worth saying plainly: **this is a bridge for specific model families, not a general
> converter.** If your model is on that list you are in good shape. If it is not, this is not
> the tool, and you would rather know that now than after you have planned around it."

**Then the round trip:**

```bash
python megatron_bridge/roundtrip.py --model Qwen/Qwen2.5-0.5B
```

**The default is ungated on purpose.** `meta-llama/*` needs an accepted licence and an
`HF_TOKEN` inside the container — a second credential and a second failure mode, for a lab
that demonstrates nothing architecture-specific. `Qwen2ForCausalLM` is on the supported list
above, and the weights are 988 MB.

> **🎤 SAY BEFORE**
>
> "Right — half a billion parameters, off Hugging Face, into Megatron's format, and straight
> back out again. Watch the counters."

**Real output**, trimmed:

```
=== importing Qwen/Qwen2.5-0.5B ===
=== configuring parallelism before instantiation ===
=== materialising the Megatron model (TP=1, PP=1) ===
Model parallel not initialized, initializing...
model.safetensors: 100%|████████| 988M/988M [00:01<00:00, 664MB/s]
Loading from Qwen/Qwen2.5-0.5B ━━━━━ 100% (170/170) Qwen2Bridge
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 494032768
=== exporting back to Hugging Face at ./hf_exports/roundtrip ===
Converting to HuggingFace ━━━━━ 100% (170/170) Qwen2Bridge
Success: All tensors from the original checkpoint were written.
```

> **🎤 SAY AFTER** *(three things, in this order)*
>
> "Three things to point at, and then one more that matters more than all of them.
>
> **First — `Model parallel not initialized, initializing`.** It brought up its own process
> group. No `torchrun`, no launcher, no rank plumbing. For the single-GPU case it just works,
> and that is why this fits in five minutes instead of a morning.
>
> **Second — look at that count. `170 of 170`, and it appears twice.** Once importing, once
> exporting. The conversion walks parameter by parameter in both directions. That is what lets
> it convert a model far larger than the card, because it never needs both complete copies in
> memory at the same time.
>
> **Third — four hundred and ninety-four million parameters, out to Megatron and back, in
> seconds.**
>
> And at the bottom: *'Success: all tensors from the original checkpoint were written.'
>
> Now. **That is the tool telling me the tool worked.** I would like something better than
> that."

#### Prove it rather than trusting the banner

> **🎤 SAY BEFORE**
>
> "So let us not take its word for it. I am going to load the original checkpoint and the one
> we just exported, and compare them **tensor by tensor.**"

```bash
python - <<'PY'
import glob, torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

orig = snapshot_download("Qwen/Qwen2.5-0.5B", allow_patterns=["*.safetensors"])
a = load_file(glob.glob(orig + "/*.safetensors")[0])
b = load_file("hf_exports/roundtrip/model.safetensors")
print(f"tensors: original={len(a)}  exported={len(b)}")
worst, ident = 0.0, 0
for k in a:
    if k not in b: continue
    if torch.equal(a[k], b[k]): ident += 1; continue
    worst = max(worst, (a[k].float() - b[k].float()).abs().max().item())
print(f"bit-identical: {ident}/{len(a)}   largest deviation: {worst:g}")
PY
```

**Real output:**

```
tensors: original=290  exported=290
missing from export: none
bit-identical: 290/290
largest deviation: 0
```

> **🎤 SAY AFTER** *(this is the close of the whole session — land it)*
>
> "**Two hundred and ninety tensors. Two hundred and ninety identical. Largest deviation:
> zero.**
>
> Not 'close enough in sixteen-bit floating point'. Not 'within tolerance'. **The same bits.**
>
> That is the answer to the lock-in question, and it is a measured answer rather than a
> promise. Your weights go into Megatron's format for training, and they come back out as an
> ordinary Hugging Face checkpoint that anything can load. **Train with NVIDIA's parallelism,
> serve with anyone's inference engine.** The weights were never trapped in either format.
>
> And notice what I actually did there. The tool told me it had succeeded — and I checked
> anyway. **A tool reporting its own success is not evidence.**
>
> Which is, I realise, the third time today I have said a version of the same thing. Lab 4:
> read the rows, not the mean. Lab 5: read the rows, not the mean. And now: diff the tensors,
> do not trust the banner.
>
> If you take one habit out of this room rather than one product, take that one."

> **❓ IF THEY ASK**
>
> **"Why 290 tensors but 170 conversion tasks?"**
> "Good catch. The bridge groups parameters — fused query-key-value projections and the like —
> so one conversion task can carry several Hugging Face tensors. Both numbers are correct."
>
> **"Would it still be bit-identical after actually training?"**
> "No, and it should not be — training changes the weights, that is the point. What this shows
> is that the *conversion* is not what changes them. You are measuring the pipe, not the water."
>
> **"What about optimiser state? Can we resume training after a round trip?"**
> "That is a different and harder question, and I would not want to answer it from the stage
> without checking. What you have seen verified here is the weights."
>
> **"Does this work for models bigger than one GPU?"**
> "That is the case it is built for — the conversion is parallelism-aware and works parameter
> by parameter, which is exactly why it does not need both complete models resident. I have
> shown you a half-billion-parameter model because it fits in a five-minute demo."

**Still the highest-risk lab of the six** — it is the only one needing a GPU, a 37 GB image
and NGC credentials. If it stalls on the day, talk over the code on the slide and move on.
Do not debug live.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `gym: command not found` | wrong venv | `source venv-gym/bin/activate` |
| Gym install fails on Python version | 0.5.0 needs ≥3.13.14; the docs page says 3.12 and is stale | install 3.13 and rebuild the venv |
| Lab 5 `gym env start` takes minutes | per-server venvs building cold | re-run the cache warm-up in `setup.sh` |
| `gym eval run` dies with a bare `ClientResponseError: 500` | the real traceback is in the **server** terminal, not this one | read terminal 1; usually a field dropped from the request subclass |
| `AttributeError` on a field you put in your JSONL | undeclared fields are dropped (`extra="ignore"`) | declare it on the request subclass — e.g. `verifier_metadata` |
| Tests pass but rewards are all 0.0 | your passing tests only assert `reward == 0.0`, and `verify()` returns early before reading ground truth | add a test that reaches the comparison |
| `gym env test` says "no tests ran" | the per-server venv has no pytest | append `pytest` and `pytest-asyncio` to that server's `requirements.txt` |
| Test fixtures fail with `Input should be a ... NeMoGymResponse` | a `SimpleNamespace` or bare dict is not accepted | build a real `NeMoGymResponse`; eight fields are required |
| `--resources-server <name>` cannot resolve | the config's top-level key differs from the server name | make them the same string |
| Every grading mode reports **0.0 / no_answer 100%** | the rollouts file is empty or stale — you are grading nothing | curl the endpoint (Lab 3); then `rm -f results/rv_* results/mcqa_rollouts*` and re-run Lab 3 |
| `rewards.py` shows a run you did not just do | derived `rv_*` files from an earlier session are still there | `rm -f results/rv_*` before a fresh sweep |
| `[1]+ Killed  gym env start ...` appears | you ran `gymclean.sh` **while the server was running** — it cannot tell healthy from orphaned | clean first, start second, never the other way round |
| Every lab 3-5 fails with a bare `500` and the servers started fine | **the model may have been retired** — the 410 is buried in the model server, invisible to the CLI | `bash modelcheck.sh` |
| `gym env start` dies with `RuntimeError: Head server finished unexpectedly!` | a previous head still holds **127.0.0.1:11000**, its fixed port — look for `address already in use` a few lines above | `bash gymclean.sh` (checks the port), or `fuser -k 11000/tcp` then wait 2s |
| Every eval 500s no matter what you change | same cause — the head never bound, so the agent server has nothing to talk to | free port 11000 first, then restart the servers |
| Terminal floods with `Failed to get cluster ID from GCS server: TimedOut` | orphaned Ray workers whose head died — they retry forever **and hold worker slots** | `bash gymclean.sh` |
| Warnings continue after `gymclean.sh` says clean | that terminal owns a dead process group; the text is on its tty, not being generated | close that window |
| `gym env start` hangs or fails for no reason | orphans from a previous run still holding slots | `bash gymclean.sh` first |
| `gym env init` exits immediately | the directory already exists | `rm -rf` it, but note that also removes its warm venv |
| `find` on a lab dir returns thousands of lines | the per-server `.venv` is built — 174 packages | exclude `.venv`, `__pycache__` and `.pytest_cache`; **do not delete the venv** |
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

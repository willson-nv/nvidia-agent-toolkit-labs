# nvidia-agent-toolkit-labs

Six labs for a two-hour workshop on three open-source pieces of the NVIDIA Agent Toolkit:
**NeMo Relay**, **NeMo Gym** and **Megatron Bridge**.

The spine is *instrument it → score it → train on the answer*.

| Lab | What | Where | Needs |
|---|---|---|---|
| 1 | One scope, one tool call, one model call | `relay/lab1_quickstart.py` | **nothing at all** |
| 2 | Three middlewares: redact, reject, measure | `relay/lab2_middleware.py` | **nothing at all** |
| 3 | A rollout and a reward | `gym/` (shipped `mcqa`) | a model endpoint |
| 4 | Re-score without re-running | `gym/` | **no model calls** |
| 5 | **Build your own environment** | `gym/support_triage/` | a model endpoint |
| 6 | HF ↔ Megatron round trip | `megatron_bridge/` | GPU + NGC credentials |

## Where it runs

One **L40S on Brev** — see **[BREV.md](BREV.md)** for launch and setup.

Only Lab 6 uses the GPU. Labs 1 and 2 need no GPU, no network and no credentials at all;
labs 3-5 send inference to a model endpoint. Disk matters more than VRAM here, because of
the NeMo container.

## Quick start

```bash
bash setup.sh
source /home/ubuntu/workspace/venv-relay/bin/activate
python relay/lab1_quickstart.py      # needs nothing; proves the box works
```

Then follow **[RUN.md](RUN.md)**, which has every command with the output to expect
underneath it.

## Three environments, on purpose

| | Python | Why separate |
|---|---|---|
| Relay | ≥ 3.11 | — |
| Gym | **exactly 3.13.14+** | its install docs still say 3.12 and are stale |
| Megatron Bridge | 3.12, in the NGC container | there is no documented `pip install` |

These requirements are mutually exclusive. `setup.sh` builds them separately; do not try to
unify them.

## Versions

Pinned in `setup.sh`. All three libraries ship breaking changes on a scale of weeks.

| | Version | Released |
|---|---|---|
| NeMo Relay | 0.7.3 | 14 Aug 2026 |
| NeMo Gym | 0.5.0 | 7 Aug 2026 |
| Megatron Bridge | 0.6.0 | 19 Aug 2026 |

## Status

**Not yet executed.** Written against the documented APIs; no run has confirmed it. See the
banner at the top of RUN.md, and do a full dry run before presenting.

## Data

Everything in `gym/support_triage/data/` is synthetic — invented tickets, no real customers,
people or systems. The fake credential in `relay/lab2_middleware.py` is a literal
`sk-fake-...` string.

Apache-2.0 tooling; labs licensed to match.

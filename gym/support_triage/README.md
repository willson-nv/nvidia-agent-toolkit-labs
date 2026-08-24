# support_triage

A single-step NeMo Gym environment. A support ticket arrives as free text; the model
answers with JSON naming a `severity` and a `team`. The verifier awards half a point
per correct field.

Built live in Lab 5. This copy is the finished reference — and the fallback if the
live build stalls.

| | |
|---|---|
| Protocol | seed_session / verify (so it uses the shipped `simple_agent`) |
| Tools | none |
| Reward | `hits / 2.0` — 0.0, 0.5 or 1.0 |
| Data | 5 example rows, entirely synthetic |

```bash
gym env validate --config configs/support_triage.yaml   # no model needed
gym env test --resources-server support_triage          # no model needed
gym env start --resources-server support_triage --model-type openai_model
```

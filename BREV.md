# Running these labs on NVIDIA Brev — L40S

Target: **one L40S**. Do this the week before, not the morning of.

---

## 0. What an L40S actually needs to do here

Worth knowing before you size anything, because it is counterintuitive:

| Labs | GPU used |
|---|---|
| 1–2 · Relay | **none.** No GPU, no network, no credentials |
| 3–5 · Gym | **none.** Inference happens at your model endpoint |
| 6 · Megatron Bridge | yes — but a 1B round trip is ~2 GB in bf16 |

**Five of the six labs never touch the GPU.** L40S is comfortable for the sixth with room to
spare, and `sm_89` (Ada) is fully supported — FP8 needs compute capability 8.9+, which is
exactly what L40S is.

**So the specs that actually matter are disk and boot time, not VRAM.** The NeMo container
for Lab 6 is many gigabytes. Brev's own default filter asks for 500 GB of disk, which is the
right instinct; do not undercut it to save money.

---

## 1. Install and log in (once, on your laptop)

```bash
brew install brevdev/homebrew-brev/brev
brev --version
brev login
```

Credentials and SSH keys land in `~/.brev/`.

---

## 2. Find an L40S you can actually get

```bash
brev search --gpu-name L40S --min-disk 500 --stoppable --sort price --wide
```

You get instance types across providers with `$/HR`, boot time and a `FEATURES` column.
**Take one marked `S`** — stoppable, so you can pause between rehearsal and the workshop.

Preview without spending anything:

```bash
brev create ft-agent-labs --gpu-name L40S --min-disk 500 --dry-run
```

---

## 3. Launch it with the setup script attached

```bash
brev create agent-labs \
  --gpu-name L40S --min-disk 500 --stoppable \
  --startup-script @setup.sh
```

Or name a type directly, with fallbacks in case the first is out of capacity:

```bash
brev create agent-labs --type <type-A>,<type-B>,<type-C> -s @setup.sh
```

Or pipe the whole thing together:

```bash
brev search --gpu-name L40S --min-disk 500 | brev create agent-labs | brev exec @setup.sh
```

Budget **20–30 minutes** for the first run. Almost all of it is two downloads: the Gym
per-server virtualenvs, and the NeMo container.

Prefer to watch it happen? Launch bare and run it yourself:

```bash
brev create agent-labs --gpu-name L40S --min-disk 500
brev shell agent-labs
git clone https://github.com/willson-nv/nvidia-agent-toolkit-labs.git
cd nvidia-agent-toolkit-labs && bash setup.sh
```

---

## 4. Connect

```bash
brev shell agent-labs        # or: ssh agent-labs
```

If SSH fails, or you created the instance in the web console:

```bash
brev refresh                 # re-syncs ~/.brev/ssh_config with current IPs
brev list
```

Sanity check, in order of how much they prove:

```bash
nvidia-smi                                   # L40S visible, driver fine
source /home/ubuntu/workspace/venv-relay/bin/activate
python relay/lab1_quickstart.py              # needs nothing — proves the box works
```

---

## 5. The traps

### Only `/home/ubuntu/workspace` survives a stop

| Location | Survives stop | Survives delete |
|---|---|---|
| `/home/ubuntu/workspace` | yes | no |
| `~/.cache`, `/tmp` | **no** | no |
| Docker images | yes | no |
| installed system packages | yes | no |

`setup.sh` puts both virtualenvs and the repo under `workspace/` for exactly this reason.
Keep using the login shell rather than exporting your own paths.

### Stopping is not risk-free

When you stop, Brev returns the GPU to the provider. On restart it tries for the same type
in the same region; if capacity has gone, your data is unreachable until it returns.

**Push to git before every stop.** For a pause longer than a day, consider `brev delete` and
a clean relaunch — `setup.sh` rebuilds in under half an hour.

### Lab 6 needs NGC credentials

```bash
docker login nvcr.io
docker pull nvcr.io/nvidia/nemo:<TAG>
```

**Verify the tag before the workshop.** `setup.sh` carries a placeholder that I could not
confirm exists. This is the single most likely thing to fail on the day, and it fails slowly.

---

## 6. The model endpoint for labs 3–5

Two options. The labs work either way.

**Hosted key — recommended for the live run.** Fewest moving parts. Write it into `env.yaml`
at the Gym repo root:

```yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: <key>
policy_model_name: <model id>
```

`env.yaml` is gitignored. Keep it that way.

**Local vLLM on the L40S — no key, no spend.** 48 GB serves a small instruct model
comfortably, and it makes the training slide honest: hosted endpoints cannot drive RL
because they do not return logprobs, so anyone going further needs exactly this.

It also costs you a model download and a serving process to babysit. **Use the hosted key
live and mention vLLM as the production path**, unless you specifically want a
fully self-contained demo.

---

## 7. Workshop-day sequence

```bash
brev start agent-labs        # restart the stopped instance
brev refresh                 # the IP will have changed
brev shell agent-labs
nvidia-smi                                          # before you talk to anyone
source /home/ubuntu/workspace/venv-relay/bin/activate
python relay/lab1_quickstart.py                     # 5-second smoke test
docker image inspect nvcr.io/nvidia/nemo:<TAG> >/dev/null && echo "lab 6 ready"
```

Start it **before the room fills**. If L40S capacity vanished overnight you want to know
with an hour in hand, not while sharing your screen.

Have two terminals open and sized for the room — labs 3 and 5 both need them.

---

## 8. Cost note

You are paying for a GPU that five of six labs do not use. That is fine — it buys you one
box, one setup script and one thing to debug. But if budget is ever the constraint, the
honest answer is that **labs 1 through 5 would run on a laptop**, and only Lab 6 needs the
card at all.

---

## Reference

- [Brev quickstart](https://docs.nvidia.com/brev/getting-started/quickstart)
- [Instance creation — startup scripts, fallback chains, piping](https://docs.nvidia.com/brev/cli/instance-creation)
- [GPU search and filtering](https://docs.nvidia.com/brev/cli/search-discovery)
- [Data persistence and lifecycle](https://docs.nvidia.com/brev/concepts/gpu-instances)

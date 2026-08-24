#!/usr/bin/env python3
"""Lab 6 — Hugging Face in, Megatron out, Hugging Face back.

Runs inside the NeMo NGC container, on a GPU. There is no documented pip install
for Megatron Bridge; the container is the supported path.

    docker run --rm -it --gpus all -v $(pwd):/workdir -w /workdir \
        --entrypoint bash nvcr.io/nvidia/nemo:<TAG>
    python roundtrip.py

The default is deliberately an UNGATED checkpoint. meta-llama/* repos require an
accepted licence and an HF_TOKEN inside the container -- a second credential and
a second thing to fail in front of a room, in exchange for nothing: the round
trip demonstrates nothing architecture-specific.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                    help="any supported HF causal-LM checkpoint; keep it small and ungated")
    ap.add_argument("--out", default="./hf_exports/roundtrip")
    ap.add_argument("--tp", type=int, default=1, help="tensor model parallel size")
    ap.add_argument("--pp", type=int, default=1, help="pipeline model parallel size")
    a = ap.parse_args()

    from megatron.bridge import AutoBridge

    print(f"\n=== importing {a.model} ===")
    bridge = AutoBridge.from_hf_pretrained(a.model, trust_remote_code=True)

    print("=== configuring parallelism before instantiation ===")
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = a.tp
    provider.pipeline_model_parallel_size = a.pp
    provider.finalize()

    print(f"=== materialising the Megatron model (TP={a.tp}, PP={a.pp}) ===")
    model = provider.provide_distributed_model(wrap_with_ddp=False)

    print(f"=== exporting back to Hugging Face at {a.out} ===")
    bridge.save_hf_pretrained(model, a.out)

    print("\n" + "=" * 56)
    print("  Round trip complete.")
    print(f"  {a.model}  ->  Megatron Core  ->  {a.out}")
    print("  The weights were never trapped in either format.")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()

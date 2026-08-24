#!/usr/bin/env python3
"""The smallest thing that proves Megatron Bridge is installed and importable.

No torchrun, no distributed init, no checkpoint download. Run this first inside
the container -- if it fails, roundtrip.py was never going to work.

    python list_architectures.py
"""
from megatron.bridge import AutoBridge

models = AutoBridge.list_supported_models()
print(f"\nMegatron Bridge can convert {len(models)} architectures:\n")
for name in sorted(models):
    print(f"  {name}")
print()

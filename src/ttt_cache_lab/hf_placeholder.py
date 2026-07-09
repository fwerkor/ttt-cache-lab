"""Placeholder for future HuggingFace integration.

The intended integration points are:

1. run prefill with `use_cache=True` and capture `past_key_values`;
2. apply a controlled LoRA or projection update;
3. rebuild selected cache layers under the new parameter version;
4. compare logits and task outputs against full recomputation.

This file deliberately does not import torch/transformers so the base package and
CI remain lightweight. Install `ttt-cache-lab[hf]` before implementing the real
backend.
"""

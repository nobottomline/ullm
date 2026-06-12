#!/usr/bin/env python3
"""Generate a reference Gated-DeltaNet (Qwen3.5 linear attention) forward pass
for the Rust unit test, using the *real* transformers implementation as ground
truth. A tiny seeded config keeps the dump small while exercising every code
path (GQA expand, causal conv1d, the delta-rule recurrence, the gated norm).

    python3 tools/gen_deltanet_ref.py > crates/ullm-model/src/testdata/deltanet_ref.json
"""

import json
import sys

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

torch.manual_seed(0)

cfg = Qwen3_5TextConfig(
    hidden_size=16,
    linear_num_value_heads=4,
    linear_num_key_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
    rms_norm_eps=1e-6,
    hidden_act="silu",
)
mod = Qwen3_5GatedDeltaNet(cfg, layer_idx=0).eval().to(torch.float32)
# Randomize all parameters deterministically (default init leaves some at ones/zeros).
with torch.no_grad():
    for p in mod.parameters():
        p.copy_(torch.randn_like(p) * 0.2)

seq = 6
x = torch.randn(1, seq, cfg.hidden_size, dtype=torch.float32) * 0.5
with torch.no_grad():
    y = mod(x)

flat = lambda t: t.detach().reshape(-1).tolist()
out = {
    "config": {
        "hidden_size": cfg.hidden_size,
        "num_v_heads": cfg.linear_num_value_heads,
        "num_k_heads": cfg.linear_num_key_heads,
        "head_k_dim": cfg.linear_key_head_dim,
        "head_v_dim": cfg.linear_value_head_dim,
        "conv_kernel": cfg.linear_conv_kernel_dim,
        "eps": cfg.rms_norm_eps,
        "seq": seq,
    },
    "weights": {n: flat(p) for n, p in mod.named_parameters()},
    "input": flat(x),
    "output": flat(y),
}
json.dump(out, sys.stdout)
sys.stderr.write(f"ok: input {list(x.shape)} -> output {list(y.shape)}\n")

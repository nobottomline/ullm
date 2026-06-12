# Benchmarks

Honest, reproducible single-stream decode throughput. Numbers are **token
generation** speed (decode), not prompt prefill, measured by `ullm run`'s
`[perf]` line (generated tokens ÷ generation wall-time, greedy sampling).

## Setup

- **Hardware:** Apple M4 Max, 128 GB unified memory
- **uLLM:** release build (`cargo build --release`), `ullm run … --gpu`
- **Reference:** llama.cpp (Homebrew `llama-bench`, Metal backend, same model file)
- Decode of 64–128 tokens, greedy; first run discarded for load/warm-up

These are single-request, single-Mac decode numbers. They will vary with the
prompt length, model file, and thermal state — reproduce locally with the
commands below rather than trusting the table blindly.

## Results

| Model | Quant | Size | uLLM CPU | uLLM GPU | llama.cpp (Metal) |
|-------|-------|------|---------:|---------:|------------------:|
| gemma-3-4b | Q6_K | 3.0 GiB | 2.7 t/s | **80.5 t/s** | 110 t/s |
| Llama-3.2-1B | Q4_K_M | ~0.8 GiB | — | **263 t/s** | — |
| Qwen2.5-1.5B | Q4_K_M | ~1.0 GiB | — | **190 t/s** | — |
| Qwen3-4B-Instruct | BF16 (HF) | ~8 GiB | ~1 t/s | **26.6 t/s** | n/a¹ |
| Qwen3-Coder-30B-A3B | MLX 4-bit (MoE) | ~16 GiB | 0.9 t/s | **63.6 t/s** | (mlx_lm 127)² |

¹ BF16 SafeTensors / MLX is read directly (no GGUF conversion); llama.cpp can't
load it without converting/quantizing first.

² Apple's own `mlx_lm` on the same file, for reference (`mlx_lm.generate`).

The 30B is a 128-expert top-8 mixture-of-experts: the whole token stays in one
Metal command buffer (router top-k runs on the GPU), and the MLX 4-bit weights
are kept resident and dequantized in-kernel. Optimization path (decode t/s):
0.9 (CPU) → 22.7 (one command buffer + GPU top-k) → 32.5 (batched experts) →
63.6 (word-vectorized 4-bit dequant: 8 nibbles + one scale/bias load per u32).

## Grammar masking (structured output)

Constrained decoding adds a per-token step: mask every token the grammar can't
accept. Three implementations, each `mask + apply to logits`, measured with
`ullm grammar-bench` on the Qwen3-4B tokenizer (vocab 151 669), JSON grammar,
Apple M4 Max:

- **naive** — simulate every token's bytes through the grammar separately, O(vocab).
- **token trie** — one walk over a byte trie of the vocabulary, interning grammar
  states and memoizing `(state, byte)` transitions within the call.
- **persistent DFA** (`GrammarDfa`, the one used in generation) — the trie walk
  plus a per-state mask cache reused across decode steps: the first time a state
  is seen it pays the walk (cold), every later token in that state is free (warm).

| Grammar state | tokens allowed | naive | per-call trie | DFA cold | **DFA warm** |
|---------------|---------------:|------:|--------------:|---------:|-------------:|
| structural (`root`) | 1 062 | ~108 ms | ~0.17 ms | ~0.18 ms | **~34 µs** |
| inside an open string | 150 331 | ~316 ms | ~8 ms | ~5.7 ms | **~34 µs** |

A recurring grammar state (a string interior, or a structural point hit every
object) costs the trie walk *once*; after that the per-token overhead is a flat
~34 µs — just writing the cached mask onto the logits — i.e. ~0.2 % of a 60 tok/s
decode step. The grammar guarantee is effectively free per token.

## Prefill (prompt processing → time-to-first-token)

Prefill runs the whole prompt through one forward pass with a **batched matmul**
that reads each weight ONCE for all prompt positions, instead of a matvec per
token. On the GPU the batched kernels (`matmul_bf16` / `matmul_mlx4` /
`matmul_q4k` / `matmul_q6k`) write K/V straight into the cache and project only
the last token's logits. Numerically identical to the token-by-token path —
verify with `ullm prefill-check <model> --gpu` (max |Δlogit| ≤ 6e-5, argmax
agrees). Measured GPU prefill of a 508-token prompt, Apple M-series:

| Model | Quant | token-by-token | batched | speedup |
|-------|-------|---------------:|--------:|--------:|
| Qwen3-4B-Instruct (dense) | BF16 | 17.1 s | 4.2 s | **4.07×** |
| Qwen2.5-1.5B (dense) | Q4_K_M | 3.45 s | 1.96 s | **1.76×** |
| gemma-3-4b (dense) | Q6_K | 6.6 s | 4.3 s | **1.53×** |

BF16 wins most: per-token decode is memory-bound, so reading each weight once
across the batch is a near-4× cut in weight traffic. The k-quants win less — the
per-token matvec is already heavily tuned, so the batched kernel only pulls ahead
once its read-once + dequant-once amortization (sub-block-per-lane, weights
dequantized into registers and reused across the column tile) beats that. Short
prompts (< 64 tokens) stay on the per-token path, where prefill is already a few
ms. MoE models also stay per-token (experts are routed per token). The CPU path
batches too (Llama family), via the same read-once `matmul_q`.

Next lever for the k-quants: a tiled `mul_mm` kernel (simdgroup-matrix 8×8,
dequantizing a weight tile into threadgroup memory) — the technique llama.cpp
uses to reach ~1000+ t/s prefill.

## Correctness

Every GPU forward is validated against the CPU reference:

```
ullm gpu-check <model> [prompt]
```

It reports the argmax token and `max |Δlogit|` for CPU vs GPU. All shipped
architectures (Llama, Qwen2, Qwen3, Gemma-3) match at rel err ~3e-6.

## Reproduce

```sh
cargo build --release

# uLLM GPU / CPU (drop --gpu for CPU)
./target/release/ullm run model.gguf "Write a short story." --max-tokens 128 --gpu

# Hugging Face / SafeTensors directory (BF16) on GPU
./target/release/ullm run /path/to/Qwen3-4B-Instruct-2507 "The capital of France is" --max-tokens 64 --gpu

# GPU vs CPU correctness
./target/release/ullm gpu-check model.gguf

# Prefill: batched vs token-by-token (correctness + speedup), on the GPU
./target/release/ullm prefill-check model.gguf "<a long prompt>" --gpu

# llama.cpp reference (same file)
llama-bench -m model.gguf -n 128 -p 0 -ngl 99
```

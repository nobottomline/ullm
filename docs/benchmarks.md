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
accept. The naive mask simulates each token separately (O(vocab)); the token
trie walks the vocabulary once, interning grammar states and memoizing
`(state, byte)` transitions. Measured with `ullm grammar-bench`, Qwen3-4B
tokenizer (vocab 151 669), JSON grammar, Apple M4 Max:

| Grammar state | tokens allowed | naive | token trie | speedup |
|---------------|---------------:|------:|-----------:|--------:|
| structural (at `root`) | 1 062 | ~95 ms | **~0.18 ms** | ~550× |
| inside an open string | 150 331 | ~260 ms | **~7 ms** | ~35× |

Structural states (the bulk of a typical JSON/tool-call generation) cost well
under a millisecond; the worst case — an unconstrained string interior where
almost every token is legal — is the floor, bounded by having to mark each
allowed token.

## Prefill (prompt processing)

The CPU path processes the whole prompt in one batched forward — each weight is
dequantized once and reused across every prompt position (`matmul_q`) instead of
once per token. Numerically identical to the token-by-token path (verify with
`ullm prefill-check <model>`, max |Δlogit| = 0). Measured on a 277-token prompt,
Apple M4 Max:

| Model | Quant | token-by-token | batched | speedup |
|-------|-------|---------------:|--------:|--------:|
| Qwen2.5-1.5B (dense) | Q4_K_M | 38.6 s | 14.7 s | **2.63×** |
| Qwen3-4B-Instruct (dense) | BF16 | 68.3 s | 41.2 s | 1.66× |
| Qwen3-Coder-30B-A3B (MoE) | MLX 4-bit | 187.4 s | 119.0 s | 1.58× |

Dense quantized models win most (expensive dequant amortized across the batch).
The MoE win is smaller because experts are still routed per token; batching the
expert dispatch and a Metal GEMM prefill are the next steps.

**gemma-3-4b Q6_K is at ~73% of llama.cpp** on the same file. The optimization
path that got there (decode t/s): 2.7 (CPU) → 5.3 (naive 1-thread-per-row GPU)
→ 28.7 (simdgroup-per-row) → 53.8 (k-quant block split across all 32 lanes) →
75.4 (multi-row: NR0 rows per simdgroup, activation reused from registers) →
80.5 (NR0=2). The multi-row kernels are ported from ggml-metal's
`kernel_mul_mv_q6_K` / `q4_K`.

The smaller Q4_K models already exceed typical llama.cpp single-stream numbers
for their size; the remaining gemma gap is per-op dispatch overhead and matvec
bandwidth (~250 vs ~355 GB/s).

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

# llama.cpp reference (same file)
llama-bench -m model.gguf -n 128 -p 0 -ngl 99
```

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
| Qwen3-Coder-30B-A3B | MLX 4-bit (MoE) | ~16 GiB | 0.9 t/s | **32.5 t/s** | n/a¹ |

¹ BF16 SafeTensors / MLX is read directly (no GGUF conversion); llama.cpp can't
load it without converting/quantizing first.

The 30B is a 128-expert top-8 mixture-of-experts: the whole token stays in one
Metal command buffer (router top-k runs on the GPU), and the MLX 4-bit weights
are kept resident and dequantized in-kernel.

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

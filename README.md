# uLLM

**Universal local LLM inference engine — written in Rust.**

One engine for every model format and every device: fast, lightweight, quick to
start. Apple Silicon first; everywhere next.

> **Status: Phase 1, single-Mac.** Runs real models end-to-end on the Metal GPU
> — GGUF, SafeTensors (Hugging Face), and Apple MLX — including a 30B
> mixture-of-experts. Single-stream today; continuous batching and other
> backends are on the [roadmap](docs/roadmap.md).

---

## What it does today

- **Loads three formats through one runtime** — GGUF (llama.cpp), SafeTensors
  (Hugging Face), and Apple MLX (4-bit) — via a container-agnostic `WeightSource`.
- **Architectures:** Llama 2/3, Qwen2, Qwen3, **Qwen3-MoE**, Gemma-3.
- **Full GPU forward on Metal** — weights, activations and KV cache stay
  resident; the whole token is one command buffer (matvec, RMSNorm, RoPE,
  attention, SwiGLU/GeGLU, MoE router + experts), with dequant-in-kernel for
  Q4_K / Q6_K / MLX-4bit / BF16 / F16.
- **OpenAI-compatible server** — `/v1/chat/completions` (streaming + not), with
  per-model chat templates (ChatML / Gemma / Llama-3) detected automatically.
- **Quantization:** F16/BF16, Q8_0, Q4_0/1, Q5_0/1, Q4_K/Q5_K/Q6_K, MLX 4-bit.

Every GPU forward is validated against the CPU reference (`ullm gpu-check`), and
the MLX path is validated token-for-token against `mlx_lm`.

## Benchmarks

Single-stream decode on an Apple M4 Max (full numbers + how to reproduce in
[`docs/benchmarks.md`](docs/benchmarks.md)):

| Model | Format | uLLM GPU |
|-------|--------|---------:|
| Llama-3.2-1B | GGUF Q4_K_M | 263 tok/s |
| Qwen2.5-1.5B | GGUF Q4_K_M | 190 tok/s |
| gemma-3-4b | GGUF Q6_K | 80 tok/s |
| Qwen3-4B | HF BF16 | 27 tok/s |
| Qwen3-Coder-30B-A3B | MLX 4-bit (MoE) | 23 tok/s |

## Quickstart

```sh
cargo build --release

# Run a model (GGUF file, or a Hugging Face / MLX directory). Drop --gpu for CPU.
./target/release/ullm run model.gguf "The capital of France is" --gpu
./target/release/ullm run ./Qwen3-Coder-30B-A3B-MLX-4bit "Write a quicksort." --gpu

# OpenAI-compatible server
./target/release/ullm serve model.gguf --gpu          # http://127.0.0.1:8080

# Inspect a model, tokenize text, validate the GPU vs CPU forward
./target/release/ullm inspect model.gguf
./target/release/ullm gpu-check model.gguf
./target/release/ullm doctor
```

## Why uLLM

Local inference today forces a choice between two poles:

- **Server engines** (vLLM, SGLang, TensorRT-LLM): great multi-user throughput,
  but NVIDIA/CUDA-only, heavy, and absent on Apple Silicon.
- **Local engines** (llama.cpp, MLX, Ollama): portable and quick, but weaker at
  server-grade serving and fragmented across backends and formats.

uLLM aims to be one engine that is excellent on a single laptop *and* scales to
serious serving — loading whatever weights you already have, without
re-platforming when you cross that line.

## Principles

- **No fear of complexity** — hand-written Metal kernels where they win.
- **Universal** — GGUF, SafeTensors, and MLX load into one runtime; Metal now,
  Vulkan/CUDA next.
- **Startup-obsessed** — cold start and time-to-first-token are first-class.
- **Honest** — reproducible benchmarks, validated against reference engines.

## Layout

```
crates/
  ullm-core/         types + container-agnostic IR (WeightSource, dequant)
  ullm-gguf/         GGUF loader
  ullm-safetensors/  SafeTensors / Hugging Face + MLX loader
  ullm-tokenizer/    SentencePiece + byte-level BPE + tokenizer.json
  ullm-model/        CPU runtime, architectures, sampling, MLX/MoE
  ullm-metal/        Metal GPU backend (full forward + kernels)
  ullm-server/       OpenAI-compatible HTTP server
  ullm-cli/          the `ullm` binary
docs/                roadmap, architecture, benchmarks, decisions (ADRs)
```

## License

[Apache-2.0](LICENSE).

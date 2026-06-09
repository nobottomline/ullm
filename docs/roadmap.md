# Roadmap

Status legend: ☐ todo · ◐ in progress · ☑ done

## Phase 0 — Foundations (current)

A clean, compiling Rust workspace and the skeletons everything else builds on.

- ☑ Workspace, license, CI, docs
- ☑ `ullm-core`: error type, `DType`, hardware detection, IR skeleton
- ☑ `ullm-cli`: `ullm doctor`
- ☐ GGUF loader → `TensorBag` + `ModelSpec` + tokenizer metadata
- ☐ Tokenizer (GGUF vocab + HF `tokenizers`)
- ☐ CPU reference backend (correctness oracle, not speed)
- ☐ Single forward pass of one small model (e.g. Qwen3-0.6B) — greedy decode

**Exit:** load a real GGUF model and generate correct text on CPU.

## Phase 1 — Apple Silicon, fast

- ☐ Metal backend (matmul, attention/prefill, dequant kernels)
- ☐ KV cache + sampler
- ☐ OpenAI-compatible server, streaming
- ☐ Startup-time + tokens/s benchmarks vs llama.cpp / MLX

**Exit:** competitive single-Mac inference with best-in-class cold start.

## Phase 2 — Serving & scale

- ☐ Continuous batching + paged KV cache
- ☐ Prefix caching across turns
- ☐ Multi-device scale-out (pipeline / expert parallel)

## Phase 3 — Broaden

- ☐ SafeTensors + PyTorch loaders
- ☐ Vulkan and CUDA backends
- ☐ Data-driven (block-composed) model definitions

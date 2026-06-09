# Roadmap

Status legend: ☐ todo · ◐ in progress · ☑ done

## Phase 0 — Foundations (current)

A clean, compiling Rust workspace and the skeletons everything else builds on.

- ☑ Workspace, license, CI, docs
- ☑ `ullm-core`: error type, `DType`, hardware detection, IR skeleton
- ☑ `ullm-cli`: `ullm doctor`
- ☑ GGUF loader → `TensorBag` + `ModelSpec` (mmap, k-quant sizing, `ullm inspect`)
- ☑ Tokenizer — SentencePiece/SPM from GGUF (`tokenizer.ggml.*`), byte fallback, `ullm tokenize`
- ☑ CPU reference backend — F32 matmul, RMSNorm, RoPE, GQA attention, SwiGLU
- ☑ End-to-end forward pass + greedy decode (`ullm run`)

**Exit reached** — generates coherent text on stories260K (a real GGUF model).

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

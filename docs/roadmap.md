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

- ☑ Quantized weight loading on CPU (F16/BF16, Q8_0, Q4_0/1, Q5_0/1, Q4_K/Q5_K/Q6_K) — runs real Q4_K_M models
- ◐ Metal backend — validated f32 + **Q4_K/Q6_K dequant-in-kernel** GEMV, unified-memory buffers (`ullm metal-check`); forward integration + benchmark next
- ☐ KV cache + sampler
- ☑ OpenAI-compatible server — `/v1/chat/completions` (SSE streaming + non-streaming) + `/v1/models` (`ullm serve`)
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

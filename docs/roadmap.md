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
- ☑ Metal backend — **full GPU forward** (weights/activations/KV resident, one command buffer per token), simdgroup k-quant matvec. `ullm run --gpu`, validated vs CPU by `ullm gpu-check` (rel ~3e-6 on all archs)
- ☑ KV cache + sampling (greedy / temperature / top-k / top-p)
- ☑ OpenAI-compatible server — `/v1/chat/completions` (SSE streaming + non-streaming) + `/v1/models` (`ullm serve`), GGUF + HF/SafeTensors, `--gpu`, per-model chat templates (ChatML / Gemma / Llama-3)
- ☑ tokens/s benchmarks ([docs/benchmarks.md](benchmarks.md), M4 Max decode): gemma-3-4b Q6_K **80.5 t/s** (CPU 2.7; llama.cpp Metal 110 → 73%), Qwen2.5-1.5B Q4_K **190**, Llama-3.2-1B Q4_K **263**, Qwen3-4B BF16 **26.6**. Startup-time benchmarks still to add

**Exit:** competitive single-Mac inference with best-in-class cold start.

## Phase 2 — Serving & scale

- ☐ Continuous batching + paged KV cache
- ☐ Prefix caching across turns
- ☐ Multi-device scale-out (pipeline / expert parallel)

## Phase 3 — Broaden

- ☑ Byte-level BPE tokenizer — runs the Llama-3 / GPT-2 / Qwen family (verified on Llama-3.2-1B)
- ☑ Qwen2 architecture (Q/K/V attention bias) — verified on Qwen2.5-1.5B-Instruct
- ☑ Gemma 3 architecture (scaled embeddings, Q/K-norm, sandwich norms, GeGLU, NeoX RoPE) — verified on gemma-3-4b Q6_K vs llama.cpp
- ☑ Qwen3 architecture (per-head Q/K-norm, NeoX RoPE, tied embeddings) — runs from SafeTensors
- ◐ SafeTensors / Hugging Face loader — `WeightSource` trait unifies GGUF + SafeTensors; loads single-file and sharded BF16/F16/F32 models + `tokenizer.json`; runs Qwen3-4B end-to-end. PyTorch `.bin` loader still TODO
- ☑ Apple MLX loader (4-bit group quant) + Qwen3-MoE (top-k router, stacked experts) — runs Qwen3-Coder-30B-A3B-MLX, validated token-for-token vs mlx_lm. GPU MoE (router top-k + expert dispatch in one command buffer): **22.7 tok/s** on M4 Max (vs 0.9 CPU)
- ☐ Vulkan and CUDA backends
- ☐ Data-driven (block-composed) model definitions

# uLLM

**Universal LLM inference engine — written in Rust.**

One engine for every model and every device: fast, lightweight, and quick to
start. Apple Silicon first; everywhere next.

> **Status: pre-alpha.** Phase 0 (foundations) — not yet usable for inference.
> See [`docs/roadmap.md`](docs/roadmap.md).

---

## Why uLLM

Local inference today forces a choice between two poles:

- **Server engines** (vLLM, SGLang, TensorRT-LLM): great multi-user throughput,
  but NVIDIA/CUDA-only, heavy, slow to start, and absent on Apple Silicon.
- **Local engines** (llama.cpp, MLX, Ollama): portable and quick, but weak at
  server-grade serving, slow to adopt new architectures, and fragmented across
  backends.

uLLM aims to be one engine that is excellent on a single laptop *and* scales to
serious serving — without re-platforming when you cross that line.

## Principles

- **No fear of complexity** — hand-written kernels and low-level work where it wins.
- **Universal** — GGUF, SafeTensors, and PyTorch weights load into one IR; Metal
  first, then Vulkan and CUDA.
- **Startup-obsessed** — cold start and time-to-first-token are first-class metrics.
- **Honest** — reproducible benchmarks, no silent truncation, no magic.

## Architecture

A Rust core behind a stable backend interface. See
[`docs/architecture/00-overview.md`](docs/architecture/00-overview.md).

## Build

```sh
cargo build
./target/debug/ullm doctor
```

`ullm doctor` reports the hardware uLLM detected on your machine.

## Layout

```
crates/
  ullm-core/   core types + container-agnostic IR
  ullm-cli/    the `ullm` binary
docs/          strategy, architecture, roadmap, decisions (ADRs)
```

## License

[Apache-2.0](LICENSE).

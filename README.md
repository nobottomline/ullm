<p align="center">
  <img src="assets/banner.svg" alt="uLLM — the local inference engine where the model obeys" width="840">
</p>

<p align="center">
  <a href="https://github.com/nobottomline/ullm/actions/workflows/ci.yml"><img src="https://github.com/nobottomline/ullm/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="rust-toolchain.toml"><img src="https://img.shields.io/badge/rust-2024-orange.svg" alt="Rust 2024"></a>
</p>

**The local inference engine where the model obeys.** Bring any model you
already have — GGUF, Hugging Face, or Apple MLX — and get output *guaranteed* to
match a JSON Schema, a grammar, or a regex: valid JSON every time, tool calls
that are always well-formed, no retries, no JSON-repair. Pure Rust,
Apple-Silicon-first, embeddable.

> **Status:** single-Mac, structured output complete. Runs real models on the
> Metal GPU — including a 30B mixture-of-experts — and the guarantee holds on
> every format, on CPU and GPU. See the [roadmap](docs/roadmap.md).

## Install

Apple Silicon Mac (macOS 14+):

```sh
# Prebuilt binary — grab the latest release tarball, unpack, run:
#   https://github.com/nobottomline/ullm/releases/latest
tar -xzf ullm-*-aarch64-apple-darwin.tar.gz && ./ullm-*/ullm doctor

# ...or from source (needs Rust):
cargo build --release      # binary at ./target/release/ullm
# or: cargo install --path crates/ullm-cli
```

Homebrew (`brew install nobottomline/ullm/ullm`) is on the way.

## Quickstart

```sh
# Generate from a GGUF file, or a Hugging Face / MLX directory. Drop --gpu for CPU.
ullm run model.gguf "The capital of France is" --gpu

# Or chat interactively — multi-turn, with conversation memory:
ullm chat model.gguf --gpu

# Structured output that cannot come out malformed:
ullm run model.gguf "Extract: John is 30."          --json
ullm run model.gguf "Review: great blender, 5 stars" --schema grammars/review.schema.json
ullm run model.gguf "Date two days after 2024-01-13:" --regex '[0-9]{4}-[0-9]{2}-[0-9]{2}'

# OpenAI-compatible server with Structured Outputs + tool calling:
ullm serve model.gguf --gpu     # http://127.0.0.1:8080
curl localhost:8080/v1/chat/completions -d '{
  "messages": [{"role":"user","content":"Extract: Acme blender, 5 stars."}],
  "response_format": {"type":"json_schema","json_schema":{"schema":
    {"type":"object","properties":{"product":{"type":"string"},"rating":{"type":"integer"}},
     "required":["product","rating"]}}}}'   # content is guaranteed to match the schema
```

`ullm --help` also has `inspect`, `tokenize`, `doctor`, and `gpu-check`.

## What it does

- **Guaranteed structure** — GBNF grammar / JSON Schema (`$ref`, recursion,
  `enum`, `pattern`/`format`) / regex, enforced at the logit level so a token
  that would break the contract is impossible to sample. The per-token cost is
  cached down to ~tens of µs.
- **OpenAI-compatible** — `/v1/chat/completions` (streaming), `response_format`,
  and `tools` + `tool_choice` returning valid `tool_calls`. A drop-in local
  OpenAI for agents.
- **Any weights, one runtime** — GGUF, SafeTensors, and Apple MLX (4-bit) load
  with no conversion; Llama 2/3, Qwen2/3, Qwen3-MoE, Gemma-3.
- **Full Metal GPU forward** — weights, activations and KV cache stay resident,
  one command buffer per token, dequant-in-kernel; validated against the CPU
  reference (`ullm gpu-check`) and, for MLX, token-for-token against `mlx_lm`.

## Benchmarks

Single-stream decode, Apple M4 Max ([numbers + how to reproduce](docs/benchmarks.md)):

| Model | Format | tok/s |
|-------|--------|------:|
| Llama-3.2-1B | GGUF Q4_K_M | 263 |
| Qwen2.5-1.5B | GGUF Q4_K_M | 190 |
| gemma-3-4b | GGUF Q6_K | 80.5 |
| Qwen3-4B | HF BF16 | 26.6 |
| Qwen3-Coder-30B-A3B | MLX 4-bit (MoE) | 63.6 |

## Layout

```
crates/
  ullm-core/         types + container-agnostic IR (WeightSource, dequant)
  ullm-gguf/         GGUF loader
  ullm-safetensors/  SafeTensors / Hugging Face + MLX loader
  ullm-tokenizer/    SentencePiece + byte-level BPE + tokenizer.json
  ullm-grammar/      grammar / JSON-Schema / regex constraint engine
  ullm-model/        CPU runtime, architectures, sampling, MLX/MoE
  ullm-metal/        Metal GPU backend (full forward + kernels)
  ullm-server/       OpenAI-compatible HTTP server
  ullm-cli/          the `ullm` binary
```

## Docs

- [Why uLLM exists](docs/strategy/positioning.md) — the corner we own, and what we're explicitly not
- [Architecture](docs/architecture/00-overview.md) · [Roadmap](docs/roadmap.md) · [Benchmarks](docs/benchmarks.md) · [Decisions (ADRs)](docs/adr)

## License

[Apache-2.0](LICENSE).

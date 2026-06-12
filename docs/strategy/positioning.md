# Positioning — why uLLM exists

## The trap we refuse

"One engine that's great on a laptop *and* scales to serious serving" is the
pitch of every inference project. It is also a losing one for a young engine:

- **llama.cpp** already owns portable local inference — years of kernels, every
  quant, a huge community. We will not out-breadth it.
- **vLLM / SGLang / TensorRT-LLM** own datacenter throughput on NVIDIA. We are
  not going to be a better datacenter engine.
- **MLX / mlx_lm** own Python-native Apple Silicon.

"The same thing, but in Rust and with a slightly faster cold start" is not a
reason for anyone to switch. If that is all we are, we should not exist.

## The corner we own

The thing that genuinely hurts in 2026 — and that the incumbents do clunkily,
heavily, or not at all — is **making a local model obey a contract**. Agents and
apps don't want prose; they want a tool call that is well-formed *every time*, a
JSON object that *always* parses, an answer drawn from a fixed set.

> **uLLM is the local inference engine for agents and apps that need the model
> to obey** — bring any model you already have, and get output that is
> *guaranteed* to match a schema, a grammar, or a regex, from one small
> embeddable Rust binary.

Three pillars, each chosen because the incumbents structurally can't or won't
match the *combination*:

1. **Guaranteed structure, not hoped-for.** A JSON Schema / regex / GBNF grammar
   constrains decoding at the logit level: tokens that would break the contract
   are masked to `-inf` and become impossible to sample. Every response parses;
   every tool call is well-formed. No "please respond in JSON," no retry loops,
   no JSON-repair post-processing. llama.cpp's GBNF is a C afterthought; vLLM's
   outlines/xgrammar are heavy and CUDA/server-bound. We make it the *core
   primitive*, fast and first-class.

2. **Bring the weights you already have.** GGUF, Hugging Face SafeTensors, and
   Apple MLX (4-bit) load through one runtime — no conversion, no re-download.
   The same `--grammar` runs on all of them. This is our existing moat: very few
   engines run all three, and none pair it with guaranteed structure.

3. **Embeddable, pure Rust, Apple-Silicon-first.** `cargo add ullm`, one binary,
   no Python and no C toolchain to ship local AI inside a desktop or server app.
   MLX-quantized models run *without Python* — rare. This is the deployment
   story Tauri / desktop / Rust-service developers actually want.

## Who it's for

Developers building **agents, tools, and apps on local models** who are tired of
unreliable output and Python/C glue:

- A Rust/Tauri desktop app that needs on-device JSON extraction that never fails.
- An agent loop where every tool call must be valid the first time.
- A local service that classifies into a fixed label set and must never
  hallucinate a 12th label.
- Anyone shipping Apple MLX models without bundling Python.

## What we are explicitly NOT

- Not chasing vLLM on datacenter-GPU throughput.
- Not chasing llama.cpp on exhaustive format/quant breadth.
- Not a research framework. It runs models and guarantees their output; it does
  not train them.

We pick the **local + reliable + embeddable** corner and own it. Everything on
the roadmap is judged against one question: *does it make a local model more
trustworthy to build on?* If not, it waits.

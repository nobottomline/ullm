# uLLM

**The local inference engine for agents — where the model has to obey.**

Bring any model you already have — GGUF, Hugging Face, or Apple MLX — and get
output that is *guaranteed* to match a schema, a grammar, or a regex. Valid JSON
every time, a tool call that is always well-formed, an answer from a fixed set —
with no retries and no JSON-repair. Pure Rust, Apple-Silicon-first, embeddable.

> **Status: Phase 1, single-Mac.** Runs real models end-to-end on the Metal GPU
> — GGUF, SafeTensors (Hugging Face), and Apple MLX — including a 30B
> mixture-of-experts, with grammar-constrained decoding on every format.
> See [why uLLM exists](docs/strategy/positioning.md) and the
> [roadmap](docs/roadmap.md).

---

## What it does today

- **Guaranteed structured output** — a GBNF grammar, a JSON Schema, or a regex
  constrains decoding at the logit level, so tokens that would break the contract
  are impossible to sample. `--json` always yields parseable JSON; `--schema`
  (incl. `$ref` / recursion, `enum`, `pattern`/`format`) yields schema-conforming
  JSON; `--regex` matches a pattern; `--grammar` runs an arbitrary GBNF grammar.
  Works on **every format** (GGUF / HF / MLX) and on **CPU and GPU**, with the
  per-token cost cached down to ~tens of µs.
- **OpenAI-compatible Structured Outputs & tool calling** — the server honors
  `response_format` (`json_object` / `json_schema`), `tools` + `tool_choice`
  (returning `tool_calls` whose arguments are *guaranteed* to match the function
  schema), and a raw `grammar` extension. A drop-in local OpenAI for agents.
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

# Guaranteed structured output — the response is always valid JSON...
./target/release/ullm run model.gguf "Extract name and age: John is 30." --json --gpu
# ...conforming to a JSON Schema (right keys, types, enums — provably valid)...
./target/release/ullm run model.gguf "Review: great blender, 5 stars." --schema grammars/review.schema.json --gpu
# ...matching a regular expression (a date, an ID, a phone number)...
./target/release/ullm run model.gguf "The date two days after 2024-01-13 is" --regex '[0-9]{4}-[0-9]{2}-[0-9]{2}'
# ...or constrained to your own grammar (e.g. a fixed label set).
echo 'root ::= "positive" | "negative" | "neutral"' > sentiment.gbnf
./target/release/ullm run model.gguf "Sentiment of 'I loved it'. Answer:" --grammar sentiment.gbnf

# OpenAI-compatible server — with Structured Outputs (a drop-in local OpenAI)
./target/release/ullm serve model.gguf --gpu          # http://127.0.0.1:8080
curl localhost:8080/v1/chat/completions -d '{
  "messages": [{"role":"user","content":"Extract: Acme blender, 5 stars."}],
  "response_format": {"type":"json_schema","json_schema":{"schema":{
    "type":"object",
    "properties":{"product":{"type":"string"},"rating":{"type":"integer"}},
    "required":["product","rating"]}}}
}'   # -> assistant content is guaranteed to match the schema

# Inspect a model, tokenize text, validate the GPU vs CPU forward
./target/release/ullm inspect model.gguf
./target/release/ullm gpu-check model.gguf
./target/release/ullm doctor
```

## Why uLLM

"One engine that's great on a laptop *and* scales to serving" is everyone's
pitch — and a losing one for a young project. We won't out-breadth llama.cpp or
out-throughput vLLM, and "the same thing but in Rust" is no reason to switch.

So we pick a corner the incumbents do clunkily or not at all and own it:
**making a local model obey a contract.** Agents and apps don't want prose; they
want a tool call that is well-formed *every time* and a JSON object that *always*
parses. uLLM constrains decoding so invalid output is impossible — on any model
format you already have, from one small embeddable Rust binary. MLX-quantized
models run with no Python.

The full argument — who it's for and what we explicitly are not — is in
[`docs/strategy/positioning.md`](docs/strategy/positioning.md).

## Principles

- **The model obeys** — structured output is guaranteed at the logit level, not
  prompted-for and hoped-for.
- **Bring your own weights** — GGUF, SafeTensors, and MLX load into one runtime;
  the same grammar works on all of them.
- **Embeddable** — pure Rust, one binary, no Python or C toolchain; Metal now,
  Vulkan/CUDA next.
- **Honest** — reproducible benchmarks, validated against reference engines.

## Layout

```
crates/
  ullm-core/         types + container-agnostic IR (WeightSource, dequant)
  ullm-gguf/         GGUF loader
  ullm-safetensors/  SafeTensors / Hugging Face + MLX loader
  ullm-tokenizer/    SentencePiece + byte-level BPE + tokenizer.json
  ullm-grammar/      GBNF grammar engine (guaranteed structured output)
  ullm-model/        CPU runtime, architectures, sampling, MLX/MoE
  ullm-metal/        Metal GPU backend (full forward + kernels)
  ullm-server/       OpenAI-compatible HTTP server
  ullm-cli/          the `ullm` binary
grammars/            ready-to-use GBNF grammars (json, sentiment, ...)
docs/                roadmap, architecture, benchmarks, strategy, decisions
```

## License

[Apache-2.0](LICENSE).

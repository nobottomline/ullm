# Changelog

Notable changes to uLLM. Format: [Keep a Changelog](https://keepachangelog.com);
versioning: [SemVer](https://semver.org) (pre-1.0, so minor versions may break).

## [0.2.0] — 2026-06-13

Much faster prompt processing, far broader model coverage — including the new
Qwen3.5 / Qwen3-Next hybrid (linear-attention + MoE) architecture.

### Added

- **Qwen3.5 / Qwen3-Next hybrid models** — the Gated-DeltaNet linear-attention
  (state-space) block that replaces softmax attention on most of their layers,
  plus output-gated full attention, partial rotary, `(1 + weight)` RMSNorm, and a
  sparse MoE FFN with a shared expert. Ported and validated layer-by-layer
  against the `transformers` reference; runs the dense (27B) and MoE (35B-A3B)
  hybrids on CPU.
- **Generic Hugging Face loading** — a `model_type` family registry, nested
  `text_config` (multimodal text decoders such as Qwen3-VL), automatic decoder
  tensor-prefix detection, and clear errors for unsupported families and
  non-text modalities. Llama, Mistral, Qwen2, Qwen3 and Qwen3 multimodal text
  decoders all load.
- **`no_repeat_ngram`** sampling — a hard block on verbatim loops (default 3 in
  the CLI), on top of the repetition penalty.

### Changed

- **GPU batched prefill** — the whole prompt now runs in one pass, reading each
  weight once via batched Metal matmuls, with the norms/RoPE batched and a
  flash-attention kernel (single-pass online softmax). Time-to-first-token on a
  508-token prompt drops ~**8.5×** for BF16 and ~**2.3–2.4×** for the k-quants;
  `ullm prefill-check --gpu` verifies it matches the per-token path.

## [0.1.0] — 2026-06-12

First public release. A local inference engine whose differentiator is
*guaranteed* structured output.

### Added

- **Guaranteed structured output** — GBNF grammars, a JSON Schema compiler
  (`$ref` + recursion, `enum`, `const`, `anyOf`/`oneOf`, `pattern`/`format`),
  regex, and a built-in JSON mode, enforced at the logit level so invalid tokens
  can't be sampled. A token trie + a persistent per-state DFA mask cache keep the
  per-token cost to ~tens of µs.
- **OpenAI-compatible server** — `/v1/chat/completions` (streaming),
  `response_format` (`json_object` / `json_schema`), and `tools` + `tool_choice`
  returning streamed `tool_calls` whose arguments match the function schema.
- **One runtime, many formats** — GGUF, Hugging Face SafeTensors, and Apple MLX
  (4-bit) behind a `WeightSource` trait; Llama 2/3, Qwen2/3, Qwen3-MoE, Gemma-3.
- **Full Metal GPU forward** — resident weights/activations/KV, one command
  buffer per token, dequant-in-kernel; validated against the CPU reference and,
  for MLX, token-for-token against `mlx_lm`.
- **CLI** — `run` (streamed, applies the chat template, `--json`/`--schema`/
  `--regex`/`--grammar`), `chat` (interactive multi-turn), `serve`, `doctor`,
  `inspect`, `tokenize`, `gpu-check`, `prefill-check`, `grammar-bench`.
- Batched CPU prefill; repetition penalty; per-model chat-template detection.

### Known limitations

- GPU prefill is token-by-token, so time-to-first-token grows with prompt length
  — batched GPU prefill is the next milestone.
- One request at a time; context capped at 8192; the GPU path is
  Apple-Silicon / Metal only.

[0.1.0]: https://github.com/nobottomline/ullm/releases/tag/v0.1.0

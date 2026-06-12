# Changelog

Notable changes to uLLM. Format: [Keep a Changelog](https://keepachangelog.com);
versioning: [SemVer](https://semver.org) (pre-1.0, so minor versions may break).

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

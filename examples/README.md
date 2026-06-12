# Examples

Small, runnable examples that show the thing uLLM is for: **output you can rely
on**. Each needs a model — any GGUF file, or a Hugging Face / MLX directory.

## Over the OpenAI API (Python)

uLLM's server is a drop-in local OpenAI, so these use the official `openai`
client — point `base_url` back at OpenAI and the same code runs unchanged.

```sh
ullm serve /path/to/model.gguf --gpu     # in another terminal
pip install openai
python examples/structured_extraction.py  # JSON guaranteed to match a schema
python examples/tool_call.py              # a tool call guaranteed to be valid
```

- **`structured_extraction.py`** — `response_format` with a JSON Schema: the
  reply always parses and matches the schema. No retries, no JSON-repair.
- **`tool_call.py`** — `tools` + `tool_choice`: the model's tool call always has
  the right function name and schema-valid arguments — the building block of an
  agent that can't emit a malformed call.

## Embedded (Rust)

No server, no Python — use the engine as a library:

```sh
cargo run --release -p ullm-cli --example embed -- /path/to/model.gguf
# -> {"name":"John","age":30}
```

See [`../crates/ullm-cli/examples/embed.rs`](../crates/ullm-cli/examples/embed.rs):
~30 lines that load a model, compile a JSON Schema to a grammar, and generate
JSON the decoder physically cannot make invalid.

## Build something bigger?

These are usage samples. A full app (a desktop GUI, an agent product) belongs in
its own repo, built on `ullm serve` (the OpenAI API) or on the crates directly —
not in this one. uLLM is the engine; the apps are yours.

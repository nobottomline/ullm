# Architecture Overview

uLLM is a Rust-core inference engine. The guiding idea: **decouple the file
format from the model architecture from the hardware backend**, so each can
evolve independently — and add one thing the incumbents lack, a constraint layer
that makes structured output *guaranteed* rather than hoped-for.

In the diagram, `✓` is implemented today and `◦` is planned (see the
[roadmap](../roadmap.md)).

```
        OpenAI-compatible API  ·  CLI (`ullm`)  ·  Rust SDK
                              |
                    Serving layer (Rust)              ✓ single request
        request scheduler · batching · KV-cache policy ◦ continuous batching
                              |
        Constraint layer (ullm-grammar)               ✓ guaranteed structure
        GBNF · JSON Schema · regex → token-trie + DFA mask
                              |
                     Runtime / executor               ✓ sampler · KV cache
        per-architecture forward · sampler · KV cache ◦ block-composed graphs
                              |
        Container-agnostic IR:  TensorBag + ModelSpec + tokenizer
              ^             ^              ^
           GGUF ✓      SafeTensors ✓    PyTorch ◦        (loaders; + MLX ✓)
                              |
            Backend interface (stable, pluggable)
              |            |            |
            Metal ✓     Vulkan ◦      CUDA ◦            (compute)
```

## Layers

1. **Loaders** parse a container (GGUF, SafeTensors, Apple MLX; PyTorch planned)
   into the same `TensorBag` (named, lazily-mapped tensors) + a normalized model
   config + a canonical tokenizer, behind the `WeightSource` trait. The rest of
   the engine never branches on where the weights came from.
2. **Runtime** runs the forward pass (attention variants, RoPE, norms, MoE
   router, activations) selected from the model config, plus the KV cache and the
   sampler. Today each architecture has an explicit forward; a future step folds
   them into config-driven, block-composed graphs.
3. **Constraint layer** (`ullm-grammar`) is uLLM's differentiator: a GBNF
   grammar — or a JSON Schema or regex compiled to one — drives a byte-level
   automaton that masks the logits each step so only grammar-valid tokens can be
   sampled. A token trie + a persistent per-state mask cache make the per-token
   cost negligible. This is what powers `--json` / `--schema` / `--regex`,
   `response_format`, and guaranteed-valid tool calls.
4. **Backend interface** is a single stable trait; each backend (Metal first) is
   a pluggable module. Quantized weights are dequantized at the kernel boundary.
5. **Serving layer** exposes everything through an OpenAI-compatible API (chat
   completions, streaming, structured outputs, tools). One request at a time
   today; continuous batching is a later phase.

## Beachhead: Apple Silicon

We start on Apple Silicon (Metal + unified memory) because it is underserved by
the server engines and is where startup-time and memory advantages are largest.
The design is cross-platform from day one; Vulkan and CUDA follow the Metal
backend.

See [`../roadmap.md`](../roadmap.md) for sequencing and
[`../strategy/positioning.md`](../strategy/positioning.md) for why the constraint
layer is the product, not a feature.

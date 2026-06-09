# Architecture Overview

uLLM is a Rust-core inference engine. The guiding idea: **decouple the file
format from the model architecture from the hardware backend**, so each can
evolve independently.

```
        OpenAI-compatible API  ·  CLI (`ullm`)  ·  Rust SDK
                              |
                    Serving layer (Rust)
        request scheduler · batching · KV-cache policy
                              |
                     Runtime / executor
        block-composed model graphs · sampler · KV cache
                              |
        Container-agnostic IR:  TensorBag + ModelSpec + tokenizer
              ^             ^              ^
           GGUF        SafeTensors      PyTorch         (loaders)
                              |
            Backend interface (stable, pluggable)
              |            |            |
            Metal        Vulkan        CUDA            (compute)
```

## Layers

1. **Loaders** parse a container (GGUF, SafeTensors, PyTorch) into the same
   `TensorBag` (named, lazily-mapped tensors) + `ModelSpec` (normalized
   hyperparameters) + a canonical tokenizer. The rest of the engine never
   branches on where the weights came from.
2. **Runtime** executes a model defined as a composition of typed blocks
   (attention variants, RoPE, norm, MoE router, activation) parameterized by
   `ModelSpec`. A new architecture should be configuration, not new per-backend
   code, wherever possible.
3. **Backend interface** is a single stable trait; each backend (Metal first) is
   a pluggable module. Quantized weights are dequantized at the kernel boundary.
4. **Serving layer** adds the scheduler, batching, and KV-cache policy, exposed
   through an OpenAI-compatible API.

## Beachhead: Apple Silicon

We start on Apple Silicon (Metal + unified memory) because it is underserved by
the server engines and is where startup-time and memory advantages are largest.
The design is cross-platform from day one; Vulkan and CUDA follow the Metal
backend.

See [`../roadmap.md`](../roadmap.md) for sequencing.

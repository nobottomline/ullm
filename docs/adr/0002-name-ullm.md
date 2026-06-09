# ADR 0002 — Name: uLLM

Date: 2026-06-09
Status: accepted

## Context

We want a short, technical name in the style of `vllm`, `llama.cpp`, `mlx`. The
first choice, `xLLM`, was rejected: it collides with an existing high-performance
inference engine (`jd-opensource/xllm`, ~1.3k★) and was already taken on
crates.io and PyPI.

## Decision

**uLLM** — "universal LLM". Brand it `uLLM`; all technical identifiers are
lowercase `ullm` (crate, binary, repo `nobottomline/ullm`), matching the vLLM
convention.

Availability verified 2026-06-09: crate name `ullm` is free on crates.io; no
competing project of note on GitHub.

## Consequences

- `u` = universal carries the product's positioning.
- PyPI/npm `ullm` are taken; acceptable for a Rust-first project (client
  packages can be namespaced later).

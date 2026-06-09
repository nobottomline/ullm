# ADR 0001 — Core language: Rust

Date: 2026-06-09
Status: accepted

## Context

We are building a high-performance inference engine that needs a small
footprint, fast cold start, and a safe concurrent serving/scheduler layer, while
still authoring hand-written GPU kernels (Metal/CUDA) for hot paths.

GPU kernels are written in CUDA C++ / Metal Shading Language regardless of host
language, so the host-language choice governs the executor, scheduler, serving,
and IR layers — not the kernels themselves.

## Decision

Rust for the core (executor, scheduler, serving, IR). Hand-written Metal/CUDA
kernels via FFI. This mirrors the proven candle / mistral.rs approach.

## Consequences

- Memory safety and fearless concurrency where serving bugs concentrate.
- Tiny binaries and fast startup; clean FFI to kernels.
- We still write or bind low-level kernels for the hottest paths.

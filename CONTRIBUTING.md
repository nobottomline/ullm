# Contributing

Thanks for looking. uLLM is early and solo-maintained, so the bar is simple:
keep it **correct, honest, and lean**.

## Dev loop

```sh
cargo build
cargo test --workspace
cargo fmt --all                                   # must be clean
cargo clippy --workspace --all-targets -- -D warnings   # must be clean
```

CI enforces fmt + clippy + build + test on macOS. If you touch a Metal kernel or
the forward pass, confirm `ullm gpu-check <model>` still reports an argmax match
against the CPU reference.

## Pull requests

- One focused change per PR; explain the **why** in the description.
- Update `docs/benchmarks.md` if you change behaviour or numbers.
- Anything that constrains or shapes output should ship with a test — the
  `ullm-grammar` crate has good examples to copy.

## Scope

uLLM is the local engine for **guaranteed structured output**
(see [`docs/strategy/positioning.md`](docs/strategy/positioning.md)). Changes
that sharpen that — or the engine it runs on — are welcome. A desktop GUI or a
training stack are out of scope; build those on top via the OpenAI-compatible
server.

## License

By contributing you agree your work is licensed under [Apache-2.0](LICENSE).

# Prism Engine

Palettized LUT inference runtime with Metal GPU acceleration (Apple Silicon) and CPU fallback. Cross-platform Linux build with `--features backend-cpu` (tracking).

One format (`.cimage`), supporting Metal GPU on Apple Silicon. CPU backend in development. OpenAI-compatible API.

## Quick Start

```bash
# Pull a model from HuggingFace (downloads + compiles to .cimage)
cargo run --release --bin prism --features full-apple -- pull Qwen/Qwen2.5-0.5B-Instruct

# Run the server
cargo run --release --bin prism --features full-apple -- run qwen2.5-0.5b-instruct

# Try it
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}'
```

## CLI

| Command | Description |
|---------|-------------|
| `prism pull <repo>` | Download + compile from HuggingFace |
| `prism run <model>` | Start OpenAI-compatible server |
| `prism list` | List compiled models |
| `prism compile <source>` | Compile a model name, local dir, or `.gguf` file |

## Features

Cargo feature flags (compose as needed):

- `server` — OpenAI-compatible HTTP server (`/v1/chat/completions`)
- `ane` — Apple Neural Engine backend (macOS only)
- `metal-dispatch` — Metal GPU GEMV acceleration for the root LUT engine (macOS only; requires the Xcode toolchain)
- `gguf-compile` — enable `prism compile <file.gguf>` support
- `prism-backend` — full compute-core execution path (Metal + ANE + Accelerate)
- `full` — `server` + `ane`
- `full-apple` — `prism-backend` + `ane` + `server` (the complete Apple-Silicon runtime)

## Platform Support

| Platform | Status |
|----------|--------|
| macOS (Apple Silicon) | **Supported** — Metal GPU + ANE via `full-apple` |
| Linux / other (CPU) | **In progress** — portable CPU path; builds without the Metal toolchain |
| NVIDIA / AMD / Intel GPU | **Planned** — see the cross-platform hardening roadmap |

> Off macOS the Metal kernels cannot be compiled, so the build emits a
> placeholder kernel library to keep the crate linkable. Set `PRISM_MOCK_BUILD=1`
> to force this on macOS for a fast, non-GPU dev build — never ship a mock build.
## Model Format

Models are stored in `~/.prism/models/<name>/`:

```
model.cimage       Compiled palettized weights
config.json        HuggingFace model config
tokenizer.json     HuggingFace tokenizer
```

## License

AGPL-3.0-only. See [LICENSE](LICENSE).

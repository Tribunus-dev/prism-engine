# Prism Engine

Palettized LUT inference runtime with Metal GPU acceleration.

One format (`.cimage`), three backends: Metal GPU, macOS ANE, CPU. OpenAI-compatible API.

## Quick Start

```bash
# Pull a model from HuggingFace (downloads + compiles to .cimage)
cargo run --release --bin prism --features full -- pull Qwen/Qwen2.5-0.5B-Instruct

# Run the server
cargo run --release --bin prism --features full -- run qwen2.5-0.5b-instruct

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
| `prism compile <name>` | Recompile without re-downloading |

## Features

- `metal-dispatch` — Metal GPU GEMV acceleration (macOS only)
- `server` — OpenAI-compatible HTTP server
- `full` — both metal-dispatch and server

## Model Format

Models are stored in `~/.prism/models/<name>/`:

```
model.cimage       Compiled palettized weights
config.json        HuggingFace model config
tokenizer.json     HuggingFace tokenizer
```

## License

AGPL-3.0-only. See [LICENSE](LICENSE).

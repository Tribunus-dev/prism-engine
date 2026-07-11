# Prism Engine

Prism Engine is the native compute runtime behind Tribunus. It combines a local
model runtime and `.cimage` compiler with a transactional compute kernel for
governing model artifacts, devices, residency, sessions, work, compilation, and
execution evidence.

The project is optimized first for Apple Silicon. Its production-oriented path
uses Metal, Accelerate, and optional Core ML integration; a portable CPU path is
being hardened for Linux. Prism is under active development and its public API,
model format, and operational contracts may still change.

## What exists today

| Surface | Current state |
|---|---|
| Local inference | The `prism` CLI can pull, compile, list, and run supported models, including an interactive chat mode and an OpenAI-compatible HTTP endpoint. |
| Compute images | Prism compiles model inputs into `.cimage`, a versioned artifact format designed to carry weights, layouts, execution plans, and validation evidence. |
| Apple Silicon execution | Metal dispatch is the primary accelerated path. The compute core also contains ANE/Core ML integration and heterogeneous device planning at differing maturity levels. |
| Constitutional ECS | Artifact ingestion is replay-verified, device discovery is canonical, and model, session, work, compilation, multimodal, distributed, and ingress domains have transactional shadow paths. |
| Recovery and evidence | The kernel includes durable-before-ack event storage, receipts, replay registration, restart recovery, stale-outcome rejection, and rebuildable projections. |
| Integration | Rust libraries, a C ABI, a Swift bridge, Node-API bindings, CLI binaries, and HTTP server surfaces live in this workspace. |
| Cross-platform execution | Linux CPU builds are continuously checked, but the portable runtime is still being completed. AMD ROCm, Intel, NVIDIA, and Tenstorrent support are development surfaces rather than supported production backends. |

The important architectural distinction is that a backend result is not, by
itself, authoritative. Prism’s compute kernel is being built so that state
changes pass through validated transactions and durable receipts, while caches,
queues, hardware handles, and analytical projections remain explicitly derived
or ephemeral.

## Repository map

| Path | Role |
|---|---|
| `compute-core/` | The Tribunus compute kernel: ECS authority, compilation, backends, scheduling, evidence, replay, server, and hardware integration. |
| `src/` | The focused Prism LUT runtime, model graph, tokenizer, CLI, and local server. |
| `prism-ffi/` | C-compatible integration surface. |
| `prism-bridge/` | Swift-facing bridge. |
| `prism-napi/` | Node-API bindings. |
| `evidence-schema/` | Shared evidence and receipt schemas. |
| `kernels/` | Metal kernels, performance contracts, and implementation notes. |
| `models/` | Model-family integrations and experimental generation crates. |

For the governing architecture and its migration status, see
[`compute-core/ARCHITECTURE.md`](compute-core/ARCHITECTURE.md). For the compute
image ABI, see [`docs/cimage-layout-abi-v1.md`](docs/cimage-layout-abi-v1.md).

## Quick start

The most complete end-user path currently targets an Apple Silicon Mac with the
Rust and Xcode command-line toolchains installed.

```bash
# Download a supported Hugging Face model and compile it to .cimage.
cargo run --release --bin prism --features full-apple -- \
  pull Qwen/Qwen2.5-0.5B-Instruct

# Start the local OpenAI-compatible server.
cargo run --release --bin prism --features full-apple -- \
  run qwen2.5-0.5b-instruct

# Send a chat completion.
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
```

Compiled models are stored under `${PRISM_HOME:-$HOME/.prism}/models/<name>/`.
Each model directory contains its `model.cimage` alongside the configuration and
tokenizer artifacts needed to load it.

The CLI also exposes `prism list`, `prism compile <source>`, and
`prism run <model> --chat`. Run `cargo run --bin prism -- --help` for the current
command surface.

## Build profiles

| Feature | Intended use |
|---|---|
| `full-apple` | Root CLI/server with the full Apple Silicon compute path. |
| `prism-backend` | Compute-core Metal, server, image, and Core ML plumbing. |
| `metal-dispatch` | Metal acceleration for the focused root LUT runtime. |
| `mlx-backend` | Experimental MLX-backed research paths. |
| `backend-cpu` | Portable CPU compilation and ongoing Linux hardening. |
| `server-dashboard` | Server plus dashboard adapters and projections. |

Feature ownership differs between the root crate and `tribunus-compute-core`.
Treat the two `Cargo.toml` files as the source of truth when embedding a specific
surface. Off macOS, Metal kernels are replaced with a linkable placeholder.
`PRISM_MOCK_BUILD=1` forces that development mode on macOS; artifacts produced
that way must not be shipped as accelerated builds.

## Project status

Prism Engine is pre-1.0 research and systems software. Continuous integration
checks the Apple Silicon `prism-backend` library and a narrower Linux CPU build.
That does not imply that every declared binary, backend, model family, or
experimental feature is production-ready. Supported behavior is the behavior
covered by the selected build, test, hardware, and receipt gates.

The authority migration is also intentionally incremental. Several domains run
in shadow mode while legacy registries are removed. The status in
[`CAMPAIGN.md`](CAMPAIGN.md) records that cutover work; “shadow” means the new
transactional representation exists, not that it is already the sole runtime
authority.

## License

Prism Engine is licensed under AGPL-3.0-only. See [`LICENSE`](LICENSE).

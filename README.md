# Prism Engine

Prism Engine is a native compiler and runtime for inspectable, heterogeneous AI
inference. It turns model weights and execution graphs into target-aware compute
images (`.cimage`) and runs them through an ECS-native runtime with explicit
placement, residency, scheduling, validation, and execution evidence.

Prism targets the machine that is actually available: Apple Silicon and Metal,
portable CPU execution, AMD ROCm GPUs such as MI300X, and AMD XDNA/XDNA2
heterogeneous accelerator plans. The compiler can preserve mixed precision,
progressively quantize or ternarize tensors, search KV-cache layouts, and route
work across CPU, GPU, and NPU execution islands. Prism is pre-1.0 systems
software, so APIs, artifact formats, and backend contracts remain subject to
change.

## What exists today

| Surface | Current state |
|---|---|
| Local inference | The `prism` CLI can pull, compile, list, and run supported models, including an interactive chat mode and an OpenAI-compatible HTTP endpoint. |
| Compute images | Prism compiles model inputs into `.cimage`, a versioned artifact format designed to carry weights, layouts, execution plans, and validation evidence. |
| Heterogeneous execution | Native runtime paths cover CPU, Metal, ROCm/HIP, and AMD XDNA planning, with backend capabilities and cross-device handoffs represented explicitly. |
| Progressive representation | Quantization, ternarization, mixed-precision fallback, calibration, and evolutionary search are compiler/runtime concerns rather than a single fixed LUT format. |
| Constitutional ECS | Artifact ingestion is replay-verified, device discovery is canonical, and model, session, work, compilation, multimodal, distributed, and ingress domains have transactional shadow paths. |
| Recovery and evidence | The kernel includes durable-before-ack event storage, receipts, replay registration, restart recovery, stale-outcome rejection, and rebuildable projections. |
| Integration | Rust libraries, a C ABI, a Swift bridge, Node-API bindings, CLI binaries, and HTTP server surfaces live in this workspace. |
| Hardware validation | Apple Silicon and MI300X validation paths are active. XDNA execution is compile-verified and resource/legalization tested, while Ryzen AI hardware validation remains dependent on access to an XDNA-capable system. |

The important architectural distinction is that a backend result is not, by
itself, authoritative. Prism’s compute kernel is being built so that state
changes pass through validated transactions and durable receipts, while caches,
queues, hardware handles, and analytical projections remain explicitly derived
or ephemeral.

## Repository map

| Path | Role |
|---|---|
| `crates/prism-spatial-ir/` | Native spatial/dataflow IR, XDNA/XDNA2 legality, resource planning, and execution plans. |
| `crates/prism-amd-npu-runtime/` | Native AMD NPU artifacts, tile/FIFO/DMA sequencing, Linux device boundaries, and runtime routing. |
| `crates/prism-rocm-runtime/` | MI300X ROCm/HIP target support, calibration, ternary execution, and GPU compilation surfaces. |
| `crates/prism-ecs-*/` | ECS-native compilation, model representation, quantization/search, scheduling, runtime, and evidence domains. |
| `src/` | Prism CLI, local server, model graph, and compatibility runtime surfaces. |
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

Prism Engine is licensed under the GNU Affero General Public License, version 3
(AGPLv3). See [`LICENSE`](LICENSE). Commercial licenses are available for
organizations that need an alternative to the AGPLv3 obligations, including
closed-source distribution or proprietary hosted deployments. To discuss a
commercial license, email [julian@tribunus.dev](mailto:julian@tribunus.dev).

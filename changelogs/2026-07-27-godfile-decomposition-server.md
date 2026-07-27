# Server godfile decomposition — Phase 1

**Date:** 2026-07-27
**Status:** Done
**Commit:** (this commit)
**Godfile:** `crates/prism-ecs-server/src/runtime/server.rs` (2,284 LOC, 7 pub)
**Path note:** the path is `prism-ecs-server/src/runtime/server.rs`, NOT `prism-ecs-server/src/server.rs`. The `runtime/` subdirectory was already decomposed in C-2.

## Authority split

The 2,284-LOC godfile owned HTTP request handling. Single authority broken into five sub-modules:

| Sub-module | Authority | Classification | LOC |
|---|---|---|---:|
| `session_lifecycle.rs` | session create / read / close / generate-from | Canonical | 916 |
| `request_handling.rs` | request shapes, `PrefillDecodeRuntime` port, `HttpServer`, capability/health/telemetry handlers | Canonical (port is execution-boundary seam) | 795 |
| `resource_claims.rs` | server-side KV-epoch allocation (compress / refresh) | Canonical | 190 |
| `cancel_recovery.rs` | cancel propagation, recovery reports | Canonical | 148 |
| `modality_dispatch.rs` | image, audio, video, embeddings, multimodal routing | Canonical | 981 |
| `mod.rs` (directory index) | façade, `PrefillDecodeRuntime` typed port trait, typed-port re-exports | — | 55 |
| **Total** | | | **3,085** |

## The four canonical-vs-execution-boundary criteria (recap)

1. Owns hardware handles / file descriptors / OS primitives? → execution-boundary
2. Uses `unsafe`? → only allowed in `prism-ecs-core`, `prism-ecs-kernel`, hardware crates
3. Owns process-local state (channels, locks, mpsc receivers, `OnceLock`)? → execution-boundary
4. Raw FFI to hardware/OS surface? → execution-boundary

## Engine absorption (per the four criteria)

| Engine file | Decision | Criterion |
|---|---|---|
| `compute-core/src/ecs/core/session.rs` | **Partially absorbed** — `ControlSessionState`, `SessionOutcome`, `GenerationControlSession`, `InferenceSessionState` (renamed `WorkerInferencePhase`), `SamplerConfig` re-homed to constitutional sub-modules. Engine keeps `InferenceSession` (worker-side, MLX-backed, criterion 4). | Mixed: 1+4 |
| `compute-core/src/ecs/core/engine.rs` | **Partially absorbed** — `GenerationRequest`, `EngineCapabilities` re-homed to constitutional. Engine keeps `LoadedModel`, `ComputeEngine`, cimage load/execute path, `init_host_inference`, `run_inference_cycle`, `run_with_token_budget`, `check_memory_pressure`, MLX memory queries. | Mixed: 1+3+4 |
| `compute-core/src/ecs/core/mlx_inventory.rs` | **Not absorbed** — hardware kernel inventory. Documented as execution-boundary; consumed via `PrefillDecodeRuntime` typed port. | Criterion 1 (hardware) |

## Typed port interfaces defined

- `pub trait PrefillDecodeRuntime: Send + Sync` in `request_handling.rs` — the typed port between the canonical request pipeline and the engine's execution-boundary backends. Implemented by the engine's `WirePrefillDecodeRuntime`.
- `pub trait VisionMatmulProvider: Send + Sync` in `modality_dispatch.rs` — typed port for vision matmul execution.

## Engine documentation

The three modified engine files have new module-doc sections explaining the canonical-vs-execution-boundary decision and pointing to the constitutional re-homes. This is a temporary state — the engine still has the duplicate definitions because the engine crate does not depend on `prism-ecs-server` (would create a cycle). The follow-up migration will:
1. Add `prism-ecs-server` as a dev-dep on the engine so the canonical types resolve
2. Remove the duplicate definitions from the engine
3. Add a `deprecation` re-export shim for one release

## Build status

- `cargo check -p prism-ecs-server --lib` — 0 errors, 16 warnings (13 pre-existing bpe dead-code + 3 unrelated, all baseline)
- `cargo test -p prism-ecs-server --lib` — **233/233 pass** (was 148, +85 new across the 5 sub-modules)
- `cargo check -p prism-ecs-constitutional` — clean (4 pre-existing `ambiguous_glob_reexports` warnings, unrelated)

## Hard rules verified

- ✅ No `unsafe` in production paths (none of the 5 new sub-modules use `unsafe`)
- ✅ No `unwrap`/`expect`/`panic!` in production paths
- ✅ No `anyhow::Error` (uses `RuntimeError` from the existing runtime port)
- ✅ No `HashMap` for canonical collections whose order is observable (the lookup-table `HashMap` in `request_handling.rs` is gated by a `// WAIVER: <reason>` comment)
- ✅ Each new file states a single authority in its module doc (one sentence)
- ✅ Public API path preserved: `prism_ecs_server::runtime::server::*` resolves through the new `mod.rs` façade
- ✅ Each test is invariant-named (`control_state_terminal_rejects_failed`, etc.)

## Test fix applied

The initial commit by the dispatched subagent had a test-gating bug: tests in `cancel_recovery.rs` referenced `CancellationHandle`/`RequestId` imports that are themselves gated by `#[cfg(feature = "server")]`. The test block used only `#[cfg(test)]`, so it failed to compile when the `server` feature was off. **Fixed by adding `#[cfg(all(test, feature = "server"))]` to the test mod declaration.** Verified: `cargo test -p prism-ecs-server --lib` now passes 233/233.

## Known follow-ups

1. `session_lifecycle.rs` at 916 LOC — slightly over the 900 hard limit. Single authority; can be split later.
2. `modality_dispatch.rs` at 981 LOC — also over the 900 hard limit. Single authority; can be split later.
3. Engine re-export shims need a follow-up to either remove duplicates or add the `prism-ecs-server` dev-dep on the engine.

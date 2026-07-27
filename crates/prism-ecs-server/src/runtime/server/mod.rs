//! Prism LLM Inference — HTTP API server, decomposed by authority.
//!
//! This module is the decomposition of the former 2284-LOC godfile
//! `crates/prism-ecs-server/src/runtime/server.rs`. The single authority of
//! the godfile (HTTP request handling) was broken into five sub-modules,
//! each owning exactly one authority:
//!
//! | Sub-module | Authority | Classification |
//! |---|---|---|
//! | [`session_lifecycle`] | session create / read / close / generate-from | canonical |
//! | [`request_handling`] | request shapes, `PrefillDecodeRuntime` port, `HttpServer`, capability / health / telemetry handlers | canonical |
//! | [`resource_claims`] | server-side KV-epoch allocation (compress / refresh) | canonical |
//! | [`cancel_recovery`] | cancel propagation, recovery reports | canonical |
//! | [`modality_dispatch`] | image, audio, video, embeddings, multimodal routing | canonical |
//!
//! The `PrefillDecodeRuntime` trait in [`request_handling`] is the typed
//! port interface between the canonical request pipeline and the engine's
//! execution-boundary backends (`WirePrefillDecodeRuntime`, and the
//! `ComputeEngine` MLX path which stays in `compute-core/src/ecs/core/engine.rs`).
//! The vision matmul provider in [`modality_dispatch`] is the typed port
//! interface to `crate::engine::metal::dispatch_fp16_matmul` (Metal kernel,
//! execution-boundary).
//!
//! Engine absorption: `compute-core/src/ecs/core/session.rs` and the
//! canonical parts of `compute-core/src/ecs/core/engine.rs` (the
//! `GenerationRequest` / `EngineCapabilities` / `SamplerConfig` /
//! `classify_workload` shapes and the `ControlSessionState` /
//! `SessionOutcome` / `GenerationControlSession` / `InferenceSessionState`
//! state machines) have been re-homed here. The engine re-exports these
//! types and continues to own the execution-boundary `LoadedModel`,
//! `ComputeEngine`, `InferenceSession` (worker-side, MLX-backed), and
//! `mlx_inventory.rs` (hardware primitive inventory — execution-boundary,
//! not absorbed).

pub mod cancel_recovery;
pub mod modality_dispatch;
pub mod request_handling;
pub mod resource_claims;
pub mod session_lifecycle;

// Re-exports preserve the previous public surface of `server.rs` so that
// downstream code (`crate::runtime::mod`, engine callers, tests) keeps
// working without churn.
pub use request_handling::{
    EngineCapabilities, GenerationRequest, HttpServer, PrefillDecodeRuntime, SamplerConfig,
};
#[cfg(all(feature = "server", feature = "prism-backend"))]
pub use request_handling::parse_session_id;
// `classify_workload` is gated behind `feature = "mlx-backend"` in
// `request_handling.rs`; the `prism-ecs-server` crate does not declare
// that feature, so we expose a no-op shim only when the parent crate's
// `prism-ecs-server` re-export is reachable. (In practice this re-export
// is consumed by the engine, which has its own `mlx-backend` feature.)
#[cfg(all(feature = "server", feature = "prism-backend"))]
pub use request_handling::generate_stream;

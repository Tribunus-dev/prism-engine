//! Engine core surface — constitutional home for the legacy engine's
//! `compute-core/src/ecs/core/` subsystem.
//!
//! The engine-absorption wave of 2026-07-27 deletes the engine's
//! `compute-core/src/ecs/core/` directory (121 files, 53,532 LOC).
//! The legacy engine's `core/` was a heterogeneous catch-all for
//! everything that did not belong to a more specific subsystem —
//! arena memory, ANE bridges, GGUF, MLX/Metal/CPU backends, model
//! runtime, compile pipeline, worker protocol, session, engine
//! orchestrator, projection, receipts, validation, supervisor, etc.
//!
//! The constitutional home for these types is being filled in
//! piecemeal across the engine-absorption migration:
//!
//! - Data-only submodules (e.g., `prism_ecs_runtime::engine_receipts`,
//!   `prism_ecs_compile::compile_state`) re-home into the matching
//!   constitutional crate.
//! - Engine-specific implementation (backend FFI, MLX/Metal shims,
//!   per-session state, the `ComputeEngine` orchestrator) re-homes to
//!   the engine-internal `compute-core/src/ecs/legacy_core/` path.
//! - This module remains the constitutional surface name that
//!   future absorption waves can grow into; the engine's
//!   `compute-core/src/ecs/core/` deletion is the final step.
//!
//! # Migration status
//!
//! The engine's `compute-core/src/ecs/core/` directory still exists
//! and is unchanged. The `prism_ecs_core::core` surface is a
//! placeholder for the constitutional home that the data-only engine
//! types will move into as subsequent migration waves land.
//!
//! See `changelogs/2026-07-27-engine-subsystem-deletion-core.md` for
//! the migration goal, the per-file authority inventory, and the
//! per-wave follow-up plan.

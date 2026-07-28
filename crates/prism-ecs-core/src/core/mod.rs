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
//! The constitutional home for each engine `core/foo.rs` file is the
//! matching existing constitutional crate, not this module:
//!
//! - **Data-only submodules** (typed structs, enums, no engine-side
//!   state) re-home to the matching constitutional crate:
//!   `prism_ecs_runtime` (engine_receipts, worker_crash_ledger,
//!   supervisor_crash, model_store), `prism_ecs_compile`
//!   (compile_state, compile_progress, compute_ir, compute_lane,
//!   config_namespace, layout_transform, mtp, profile_compiler,
//!   operation_catalog), `prism_ecs_quantization` (weight_codec,
//!   requalification), `prism_gguf` (gguf, manifest extraction),
//!   `prism_ane` (ane_bridge, ane_compile, ane_keepalive, mil_builder),
//!   `prism_kv_cache` (kv_cache_types), `prism_ecs_agent`
//!   (coreai_audit, coreai_bridge, coreai_pipeline, coreai_state),
//!   `prism_ecs_server` (engine, engine_error, engine_policy, session,
//!   streaming, runtime_contract, runtime_orchestration, runtime_trace,
//!   executor, executor_projection, profiled_executor, profiled_model),
//!   `prism-audio` (audio_provider, audio_preprocess_accelerate),
//!   `prism_ecs_codec` (transform_recipe, treatment).
//!
//! - **Engine-internal implementation** (FFI bindings, MLX/Metal
//!   shims, backend dispatchers) remains in the engine under a
//!   renamed path: `compute-core/src/ecs/legacy_core/`.
//!
//! - **Constitutional placeholder** (this module) remains a doc-only
//!   surface. The engine's `core/` deletion is the final step of
//!   the engine-absorption wave, after all 234+ engine references
//!   have been retargeted to the constitutional homes above.
//!
//! See `changelogs/2026-07-27-engine-subsystem-deletion-core.md` for
//! the migration goal, the per-file authority inventory, and the
//! per-wave follow-up plan.

//! Canonical authority for the system-orchestration surface.
//!
//! Each sub-module is the constitutional home for one system type
//! that was previously declared in the engine's
//! `compute-core/src/ecs/system/` directory. The engine's
//! `compile_session.rs` (and the engine's bin/bitnet_ecs_test.rs
//! and aot_kernels/tests.rs and compilation/* modules) imported
//! the legacy engine surface; this module is the single
//! canonical home for the data types those callers reference.
//!
//! # Authority per file
//!
//! Every sub-module in this directory owns a single authority —
//! one system type or one data type. The trait impls that bridge
//! these data types onto the engine's `CompilerSystem` trait
//! live in the engine as thin adapter structs (see
//! `compute-core/src/ecs/system_adapters.rs`).
//!
//! # Migration status
//!
//! The engine's `compute-core/src/ecs/system/` directory is
//! scheduled for deletion as the final step of the engine-absorption
//! recipe (E-{N+1}). Until then, this module is the constitutional
//! surface that the engine callers have migrated to. The
//! `workspace_contains_no_legacy_system_imports` architecture test
//! enforces the post-deletion invariant that no file outside the
//! migration inventory imports the legacy engine surface.

pub mod archive;
pub mod backend_compile;
pub mod backend_dispatch;
pub mod backend_eval;
pub mod backend_residency;
pub mod backpressure_tick;
pub mod buffer_lifetime;
pub mod capability_registry_sys;
pub mod catalog_validation;
pub mod compiler_systems;
pub mod completion_ingest;
pub mod download;
pub mod draft_model;
pub mod engine_systems;
pub mod execution_graph;
pub mod executor_systems;
pub mod fusion;
pub mod int4_pack;
pub mod kernel_catalog;
pub mod kernel_gen;
pub mod memory_plan;
pub mod metal_cleanup;
pub mod metal_dispatch;
pub mod metal_init;
pub mod metal_transfer;
pub mod model_load;
pub mod moe_budget;
pub mod package;
pub mod phase_engine;
pub mod phase_engine_cleanup;
pub mod phase_engine_init;
pub mod phase_engine_tick;
pub mod planning_core;
pub mod portfolio;
pub mod quant_plan;
pub mod session_cleanup;
pub mod session_decode_tick;
pub mod session_init;
pub mod slot_lease_tick;
pub mod source_load;
pub mod ternary_pipeline;
pub mod token_budget_tick;
pub mod tts;
pub mod tuning;
pub mod validation;
pub mod validation_matrix;
pub mod variant_gen;
pub mod variant_select;
pub mod work_dispatch;
pub mod work_dispatch_tick;

#[cfg(test)]
mod tests;

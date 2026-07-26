//! Re-export shim — the canonical MIL builder and high-level ANE program
//! constructors now live in `prism_ane::mil_builder` and
//! `prism_ane::mil_layer_programs`.
//!
//! This file used to be the engine's parallel implementation of
//! `MilBuilder` (2,226 LOC). It has been absorbed into the constitutional
//! `prism-ane` crate. The engine's callers continue to import
//! `crate::ecs::mil_builder::MilBuilder` (and related types); this shim
//! preserves the API while the engine migrates to use the canonical
//! path directly.
//!
//! # Migration
//!
//! New engine code should use `prism_ane::mil_builder::MilBuilder` and
//! `prism_ane::mil_layer_programs::{build_full_ane_layer_program,
//! build_batched_matmul_program}` directly. This shim will be removed
//! in a follow-up phase once the engine's internal callers have been
//! updated.
//!
//! # Absorption summary
//!
//! Per Phase 4-C of the compute-core absorption plan, the engine's
//! `MilBuilder` was merged with the smaller `prism-ane` version. The
//! engine had unique methods that were absorbed:
//!
//! - `topk`, `batch_size`, `silu`, `softmax`, `matmul_transpose_y`,
//!   `concat`, `conv`, `reshape`, `transpose`, `const_i32`,
//!   `reserve_names` — added to `prism_ane::mil_builder::MilBuilder`.
//! - `gather(params, indices, axis)` — REPLACED the prism-ane stub
//!   (2-arg with hardcoded `[1, 1]` shape) with the engine's
//!   full axis-aware implementation.
//! - `int32_arg`, `ints32_arg`, `ints_attr`, `ints32_attr`,
//!   `multi_named_arg` helpers — added.
//!
//! The two high-level program constructors that lived in the engine
//! were moved to `prism_ane::mil_layer_programs`:
//!
//! - `build_full_ane_layer_program` — fused transformer layer with
//!   integrated KV compaction. Re-implemented with a typed
//!   `MilBuildError::ProgramEncodeFailed` return path instead of the
//!   engine's `eprintln!` + empty `Vec<u8>` error swallowing.
//! - `build_batched_matmul_program` — batch-fused matmul.
//!
//! Bug fixes absorbed:
//!
//! - `gather` engine helper produced `gemma4.llama.embedding_length`
//!   instead of `gemma4.embedding_length`. Fixed in the canonical
//!   version.
//! - The engine's `build_full_ane_layer_program` referenced SSA names
//!   (`"lut"`, `"gather_0"`, `"matmul_7"`) that did not match the
//!   builder's own counter. Fixed in the canonical version with the
//!   correct names derived from the actual counter state.
//!
//! MIGRATION: prefer `prism_ane::mil_builder::MilBuilder` for new code.

#![cfg(any(feature = "mlx-backend", feature = "prism-backend"))]

// Re-export the canonical MilBuilder and related types so that existing
// engine callers (`crate::ecs::mil_builder::MilBuilder`, etc.) continue
// to resolve.
pub use prism_ane::mil_builder::{
    CoreMlUnaryOpType, MilBuildError, MilBuilder,
};

// Re-export the two high-level ANE program constructors.
pub use prism_ane::mil_layer_programs::{
    build_batched_matmul_program, build_full_ane_layer_program,
};

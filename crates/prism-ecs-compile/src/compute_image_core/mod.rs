//! Constitutional surface for the `compute_image/` core subsystem.
//!
//! This module owns the canonical authority for the engine's
//! `compute-core/src/ecs/compute_image/` core surface data-only
//! files. The engine-side implementation lives at
//! `compute-core/src/ecs/legacy_compute_image_core/` for engine-
//! coupled code that depends on Metal/Accelerate/Core ML, MLX, and
//! other engine-internal types.
//!
//! # Authority
//!
//! This module is the canonical home for data-only ComputeImage
//! types: fusion receipts, phase graph metadata, KV cache plan
//! types, slot state, residency plan types, CImage manifest data
//! shapes, and the cross-platform adapter/telemetry stubs. Engine-
//! coupled implementations (Metal epilogues, ANE MIL builders, Core
//! ML pipeline code, MLX runtime, Apple shared arena, CImage loader
//! with Metal buffer allocation) remain engine-side.
//!
//! # Re-exports
//!
//! Submodules expose their public types at the top of this module
//! for ergonomic engine-side use:
//! `prism_ecs_compile::compute_image_core::X`.

#![cfg_attr(
    all(
        not(feature = "prism-backend"),
        not(feature = "mlx-backend"),
        not(feature = "metal-dispatch")
    ),
    allow(unused_imports)
)]

pub mod error;
pub use error::{Error, Result};

// ── Data-only / std-only ComputeImage surface ─────────────────────

/// Diagnostics stubs.
pub mod diag;
/// HuggingFace Hub download shim.
pub mod hf;
/// Compile-time quantization transform shim.
pub mod quant;
/// Source loading shim.
pub mod source;
/// Execution shape taxonomy.
pub mod execution_shape;
/// Phase graph data types (PhaseId, edge semantics, declared
/// fallback decompositions).
pub mod phase_graph;
/// Phase graph → sibling module binding.
pub mod phase_graph_binding;
/// Phase graph construction helpers.
pub mod phase_graph_builder;
/// Phase graph validation helpers.
pub mod phase_graph_validation;
/// Phase DAG data types (EmittedPhaseGraph, EmittedArenaPlan, …).
pub mod phase_dag;
/// Phase DAG test fixtures.
pub mod phase_dag_test;
/// Phase fallback registry data types.
pub mod phase_fallback;
/// Phase program version identifier.
pub mod phase_program_version;
/// Slot state and failure-reason taxonomy.
pub mod slot_types;
/// Layout constants for the Tensix backend.
pub mod layout_tensix;
/// Fusion ABI data types (sealed Metal fusion artifact, launch
/// contract, artifact hash).
pub mod fusion_abi;
/// Fusion execution receipts.
pub mod fusion_receipts;
/// Fusion sealing data types (sealed artifact metadata).
pub mod fusion_sealing;
/// Hardware assessment receipt data types.
pub mod hw_assessment;
/// Hardware benchmark suite result data types.
pub mod hw_bench_suite;
/// KV cache plan and codec taxonomy.
pub mod kv_plan;
/// KV interleave ABI constants and data types.
pub mod kv_interleave;
/// Adapter (cross-platform stub).
pub mod adapter;
/// Apple CImage manifest data types.
pub mod apple_cimage_manifest;
/// Tensix compute image data types.
pub mod tensix;
/// Tree attention (cfg-gated Metal pipeline surface).
pub mod tree_attention;
/// VM manager stub.
pub mod vm_manager;
/// Speculative routing data types.
pub mod speculative_routing;
/// Receipts data types.
pub mod receipts;
/// Fusion Tensix placer.
pub mod fusion_tensix;

// ── Engine-coupled (not re-exported; lives at legacy_compute_image_core) ──
//
// The following 21 top-level files are engine-coupled and remain at
// `compute-core/src/ecs/legacy_compute_image_core/`:
//   alpha_types, ane_compile, ane_prefill, apple_shared_arena,
//   cimage_loader, compaction, compatibility, fallback_plan,
//   kernel_provider, metal_codegen_model_test, metal_epilogue,
//   paged_cache, phase_dag (engine-coupled variant), plan, segment,
//   subgraph_mil, subgraph_mil_phase2, fusion_plan (engine-coupled
//   variant), and the engine-coupled inlines of kv_interleave /
//   tree_attention.

//! Constitutional `compute_image_runtime` surface — runtime + ancillary
//! data types and pure algorithms absorbed from the engine's
//! `compute-core/src/ecs/compute_image/{residency,...,verification}/`
//! directory on 2026-07-27.
//!
//! ## Authority
//!
//! This module owns the **runtime + ancillary** surface of the engine's
//! `compute_image` subsystem: residency plans, content store, executable
//! descriptors, variant selection, program/phase IR, kernel selection,
//! megakernel, multimodal bindings, model-family bindings, scheduling
//! helpers, and verification receipts. The compile-time **core** of
//! `compute_image` (manifest, cimage_packer, compile pipeline,
//! orchestrator) lives in its own constitutional homes
//! (`prism_ecs_compile::cimage_manifest`, `::cimage_packer`,
//! `::compile_pipeline`, `::runtime`) and is handled by separate
//! migration agents.
//!
//! ## Sub-modules (single authority per file)
//!
//! | Sub-module | Authority |
//! |---|---|
//! | [`residency`] | Compiled residency plans, residency classes, memory contracts. |
//! | [`content_store`] | Content-addressed store types, integrity, mmap, layout. |
//! | [`executable`] | Executable descriptors (admission, profile, schema, seal, variant). |
//! | [`program`] | Phase program IR (lane graph, serialization, validation). |
//! | [`variants`] | Shape variants (definition, coverage, selection, compatibility). |
//! | [`kernel_selection`] | Runtime kernel selection (compatibility, evidence, selection). |
//! | [`megakernel`] | Megakernel fusion (pipeline, kernels, gather, KV). |
//! | [`heterogeneous`] | Heterogeneous execution image types and builders. |
//! | [`multimodal`] | Multimodal binding, descriptor, adapter, projection. |
//! | [`model_family`] | Per-model-family bindings (Gemma4, Qwen2.5-Omni, etc.). |
//! | [`scheduler`] | Scheduling helpers (batch scheduler, load monitor). |
//! | [`verification`] | Verification receipts (numerical, resource-fit, phase graph, residency). |
//!
//! ## Migration status
//!
//! Absorbed 68 files (~14K LOC) from the engine's
//! `compute-core/src/ecs/compute_image/{residency,...,verification}/`
//! on 2026-07-27. Data-only files live here; engine-coupled
//! implementations (those depending on Metal/ANE/MLX runtime plumbing,
//! `cimage_loader`, `kv_interleave`, `metal_backend`, etc.) stay
//! engine-side at `compute-core/src/ecs/legacy_compute_image_runtime/`.

pub mod content_store;
pub mod executable;
pub mod heterogeneous;
pub mod kernel_selection;
pub mod megakernel;
pub mod model_family;
pub mod multimodal;
pub mod program;
pub mod residency;
pub mod scheduler;
pub mod shared;
pub mod variants;
pub mod verification;

pub use shared::{ContentHash, ExecutionShapeClass};

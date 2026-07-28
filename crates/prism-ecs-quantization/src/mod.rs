#![allow(clippy::approx_constant)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::unnecessary_to_owned)]
#![allow(clippy::explicit_counter_loop)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::result_large_err)]
#![allow(clippy::same_item_push)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(unexpected_cfgs)]
#![allow(clippy::while_let_loop)]
#![allow(clippy::wildcard_in_or_patterns)]
//! Quantization admission module.
//!
//! Defines the NF4 tile640 representation family and a fail-closed admission
//! pipeline that proves each tensor's chosen representation preserves
//! weight-space and operator-space behavior before the artifact is sealed.
//!
//! ## Module structure
//!
//! - `contract` — representation formats, reconstruction contracts, validation
//!   profiles, and admission pipeline types.
//! - `validation` — weight-space (RMSE, NRMSE, zero-collapse) and two-layer
//!   operator-space validation (stress bank + optional activation bank).
//! - `admission` — candidate generation, packing, reconstruction, and the
//!   `quantize_tensor` pipeline with dual-layer validation and evidence tracking.
//! - `calibration` — `StressSuite` (deterministic, always built) and
//!   `CalibrationSuite` (prerendered, optional for production qualification).
//! - `ternarization` — ternarization engine: candidate types, scale
//!   optimization, residual codecs, candidate gates, and packaging.

pub mod admission;
pub mod ane_orchestration;
pub mod calibration;
pub mod contract;
/// Quantization algorithm codec families — AWQ, GPTQ, SmoothQuant.
pub mod families;
/// Generalized substitution pipeline — tries ranked codec candidates against
/// evidence gates and uses the most aggressive one that passes.
pub mod substitution;
pub mod sweep;
/// Ternarization engine — candidate types, scale optimization, residual
/// codecs, gates, and physical packaging for ternary representation.
pub mod ternarization;
/// BitNet b1.58 2B4T native ternary weight importer — re-implementation
/// of the engine's `compute-core/src/ecs/bitnet/` subsystem. The engine's
/// parallel `bitnet` module has been deleted; engine callers now read
/// from `prism_ecs_quantization::bitnet`.
pub mod bitnet;
/// Ternary base-weight assimilation — opt-in mutations behind a research-only gate.
pub mod ternary_assimilation;
/// Ternary substitution pass — replaces primary codecs with ternary on eligible
/// tensor classes when evidence gates are satisfied.
pub mod ternary_substitution;
pub mod validation;

/// Substitution pass — ranked trial of tile640 codec candidates.
pub mod substitution_pass;

// Pre-existing quantization submodules (preserved from original mod.rs).
pub mod bonsai_cimage;
/// Metal GPU dispatch for Bonsai ternary GEMV kernel.
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
pub mod bonsai_metal_dispatch;
/// Bonsai 2-bit → 1.58-bit Tile640 ternary conversion pipeline.
pub mod bonsai_ternary;
/// Canonical model source — format-independent model abstraction.
pub mod canonical_model_source;
pub mod cimage;
pub mod compile_config;
pub mod compiler;
pub mod embed_cluster;
/// Execution plan types — local copy for crate-internal use.
pub mod execution_plan;
/// Generic GGUF tensor reading — format agnostic.
/// GGUF tensor provider — implements `TensorProvider` for GGUF files.
#[cfg(feature = "gguf-compile")]
pub mod gguf_provider;
/// Generic GGUF tensor reading — format agnostic.
pub mod gguf_reader;
pub mod kv_search;
/// MLX model adapter — format detection and config parsing.
pub mod mlx_adapter;
/// NF4 tile640 weight format — local copy for crate-internal use.
pub mod nf4tile640;
/// ONNX model adapter — minimal protobuf parser + TensorProvider
pub mod onnx_adapter;
pub mod oq;
pub mod palette;
/// Per-class precision policy and M1 memory budget admission.
pub mod precision_policy;
/// Per-tensor quantization result types — the structured plan that
/// downstream emission and the constitutional ECS read instead of
/// recomputing per-tensor policy.
pub mod quantization_plan;
/// SafeTensor provider — implements `TensorProvider` for safetensors directories.
pub mod safetensors_provider;
/// Internal semantic tensor-family classification and layout candidate planning.
pub mod tensor_layout;
pub mod turboquant_kv;

pub use admission::quantize_tensor;
pub use calibration::*;
pub use contract::*;
pub use validation::*;

pub use substitution::SubstitutionContext;

/// Per-tensor quantization result types. Re-exported so that downstream
/// callers (the constitutional ECS, the CLI, the dashboard) can
/// reference the plan by its canonical name without depending on the
/// module path.
pub use quantization_plan::{QuantizationResult, QuantizedTensorSelection};

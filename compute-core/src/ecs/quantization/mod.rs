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
pub mod calibration;
pub mod contract;
/// Generalized substitution pipeline — tries ranked codec candidates against
/// evidence gates and uses the most aggressive one that passes.
pub mod substitution;
pub mod sweep;
/// Ternarization engine — candidate types, scale optimization, residual
/// codecs, gates, and physical packaging for ternary representation.
pub mod ternarization;
/// Ternary base-weight assimilation — opt-in mutations behind a research-only gate.
pub mod ternary_assimilation;
/// Ternary substitution pass — replaces primary codecs with ternary on eligible
/// tensor classes when evidence gates are satisfied.
pub mod ternary_substitution;
pub mod validation;

/// Substitution pass — ranked trial of tile640 codec candidates.
pub mod substitution_pass;

// Pre-existing quantization submodules (preserved from original mod.rs).
pub mod cimage;
pub mod embed_cluster;
pub mod oq;
pub mod palette;
/// Per-class precision policy and M1 memory budget admission.
pub mod precision_policy;
pub mod turboquant_kv;

pub use admission::quantize_tensor;
pub use calibration::*;
pub use contract::*;
pub use validation::*;

pub use substitution::SubstitutionContext;

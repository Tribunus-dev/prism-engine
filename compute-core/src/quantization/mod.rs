//! Quantization admission module.
//!
//! Defines the NF4 tile640 representation family and a fail-closed admission
//! pipeline that proves each tensor's chosen representation preserves
//! weight-space and operator-space behavior before the artifact is sealed.
//!
//! ## Module structure
///!
///! - `contract` — representation formats, reconstruction contracts, validation
///!   profiles, and admission pipeline types.
///! - `validation` — weight-space (RMSE, NRMSE, zero-collapse) and two-layer
///!   operator-space validation (stress bank + optional activation bank).
///! - `admission` — candidate generation, packing, reconstruction, and the
///!   `quantize_tensor` pipeline with dual-layer validation and evidence tracking.
///! - `calibration` — `StressSuite` (deterministic, always built) and
///!   `CalibrationSuite` (prerendered, optional for production qualification).
pub mod admission;
pub mod calibration;
pub mod contract;
pub mod sweep;
pub mod validation;

// Pre-existing quantization submodules (preserved from original mod.rs).
pub mod cimage;
pub mod embed_cluster;
pub mod oq;
pub mod palette;
pub mod turboquant_kv;

pub use admission::quantize_tensor;
pub use calibration::*;
pub use contract::*;
pub use validation::*;

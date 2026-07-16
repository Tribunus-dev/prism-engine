//! Stress-bank and activation-bank generators for operator-space validation.
//!
//! Provides:
//! - `StressBank` / `StressSuite` — deterministic synthetic vectors that
//!   exercise codec pathologies. Always built, mandatory for admission.
//! - `ActivationBank` / `CalibrationSuite` — model-native prerendered
//!   activation vectors from reference model execution. Optional; required
//!   for `ProductionQualified` admission.
//!
//! Sub-modules:
//! - `calibrator` — Calibrator trait and generic calibration types.
//! - `awq_calibrator` — AWQ activation magnitude profiling.
//! - `gptq_calibrator` — GPTQ Hessian accumulation and quantization.
//! - `ternary` — Ternary-specific scale calibration.
//! - `suite` — Stress/activation bank generation.

mod suite;
pub use suite::*;
pub use suite::{
    stratified_sample, StratifiedSample, DEFAULT_SAMPLE_SEED, STRATIFY_NUM_STRATA_HOLDOUT,
    STRATIFY_NUM_STRATA_PROBE, STRATIFY_NUM_STRATA_PROMO,
};

pub mod calibrator;
pub mod awq_calibrator;
pub mod gptq_calibrator;
pub mod ternary;

pub use calibrator::*;
pub use awq_calibrator::*;
pub use gptq_calibrator::*;
pub use ternary::*;

//! Stress-bank and activation-bank generators for operator-space validation.
//!
//! Provides:
//! - `StressBank` / `StressSuite` — deterministic synthetic vectors that
//!   exercise codec pathologies. Always built, mandatory for admission.
//! - `ActivationBank` / `CalibrationSuite` — model-native prerendered
//!   activation vectors from reference model execution. Optional; required
//!   for `ProductionQualified` admission.

mod suite;
pub use suite::*;

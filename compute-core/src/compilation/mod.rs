//! Compilation pipeline — phase IR, admission gate, ANE calibration lane,
//! ring-buffered staging, and GPU k-means infrastructure.
//!
//! Types delegate to the runtime residency contract in
//! `crate::backend::residency` for cross-backend transfer decisions.

pub mod phase_ir;
pub mod admission;
pub mod ane_lane;
pub mod staging;
pub mod evidence_probe;
pub mod tri_lane;
pub mod qualification_gate;
pub mod profitability;

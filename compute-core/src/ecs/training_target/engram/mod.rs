//! Engram training — produces EngramArtifact from calibration data.
//!
//! An engram captures residual patterns in ternary weights after base
//! quantization. Training optimizes these patterns against a held-out
//! calibration set, producing an EngramArtifact that the inference
//! pipeline applies at the configured insertion point.

pub mod config;
pub mod receipt;
pub mod trainer;

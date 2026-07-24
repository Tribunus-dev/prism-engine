//! Ternarization engine — tensor/group candidate types, scale optimization,
//! residual codecs, candidate gates, and physical packaging.
//!
//! This module implements the ternarization pipeline (plan Section 6):
//!
//! - `candidate` — `TernarizationCandidate` type with ternary weights,
//!   per-group scales, and residual policy enums.
//! - `optimizer` — `ScaleOptimizer` that finds per-group scale factors
//!   minimizing reconstruction RMSE.
//! - `residual` — `ResidualCodec` for dense and sparse residual encoding
//!   and application.
//! - `gates` — `CandidateGates` and structural/reconstruction/combined
//!   gate functions for candidate admission.
//! - `packaging` — `TernaryPackage` with physical layout and byte-level
//!   pack/unpack for artifact assembly.

pub mod candidate;
pub mod gates;
pub mod optimizer;
pub mod packaging;
pub mod promotion;
pub mod residual;

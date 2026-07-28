//! Compile-time verification receipts for `SealedComputeImageExecutable`.
//!
//! This module owns the canonical receipt types used by the seal-proof
//! and artifact-selection layers. Each receipt attests to one
//! verification dimension of a compiled executable.

pub mod bundle;
pub mod numerical;
pub mod phase_graph;
pub mod residency;
pub mod resource_fit;

//! Compile-time verification receipts for SealedComputeImageExecutable.
//!
//! These modules define and re-export the canonical receipt types used by the
//! seal-proof and artifact-selection layers.

pub mod bundle;
pub mod numerical;
pub mod phase_graph;
pub mod residency;
pub mod resource_fit;

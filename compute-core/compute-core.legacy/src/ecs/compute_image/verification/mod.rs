//! Compile-time verification receipts for SealedComputeImageExecutable.
//!
//! These modules define and re-export the canonical receipt types used by the
//! seal-proof and artifact-selection layers.

pub mod bundle;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod numerical;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod phase_graph;
pub mod residency;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod resource_fit;

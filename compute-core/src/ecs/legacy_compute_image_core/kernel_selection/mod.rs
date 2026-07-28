//! Kernel variant selection — compile-time artifact selection, candidate
//! benchmark evidence, and selection receipts for SealedComputeImageExecutable.
//!
//! At compile time the compiler benchmarks candidate kernel implementations
//! against the target profile's hardware contract.  The best-performing
//! candidate is selected per operation/shape-class pair, and a
//! [`KernelSelectionReceipt`] records the selection policy version, candidate
//! artifacts, resource-fit and numerical qualification outcomes, and the
//! chosen winner.  These receipts become part of the
//! [`CompileTimeReceiptBundle`] embedded in the sealed executable.
//!
//! [`KernelCandidateEvidence`] captures per-candidate benchmark results
//! (median/min latency, resource-fit pass/fail, numerical pass/fail) that
//! feed the selection policy.
//!
//! [`KernelConfiguration`] records the tiling parameters and pipeline id
//! for a candidate kernel variant.
//!
<<<<<<<< HEAD:compute-core/src/ecs/compute_image/legacy_compute_image_runtime/kernel_selection/mod.rs
//! [`CompileTimeReceiptBundle`]: crate::ecs::compute_image::legacy_compute_image_runtime::executable::schema::CompileTimeReceiptBundle
|||||||| e64c7d94:compute-core/src/ecs/compute_image/kernel_selection/mod.rs
//! [`CompileTimeReceiptBundle`]: crate::ecs::compute_image::executable::schema::CompileTimeReceiptBundle
========
//! [`CompileTimeReceiptBundle`]: crate::ecs::legacy_compute_image_core::executable::schema::CompileTimeReceiptBundle
>>>>>>>> migrate/ci-core:compute-core/src/ecs/legacy_compute_image_core/kernel_selection/mod.rs

#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod selection;

pub mod evidence;
pub use evidence::*;

#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod compatibility;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use compatibility::*;

#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use selection::*;

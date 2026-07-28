//! Executable descriptors — pure data types and pure algorithms for
//! `SealedComputeImageExecutable` and related artifact types.
//!
//! This module owns the data-only types absorbed from the engine's
//! `compute-core/src/ecs/compute_image/executable/` directory on
//! 2026-07-27. Engine-coupled implementations (those that depend on
//! Metal/MLX/Core ML execution plumbing) stay at
//! `compute-core/src/ecs/legacy_compute_image_runtime/executable/`.

pub mod admission;
pub mod profile;
pub mod provenance;
pub mod receipt;
pub mod seal;
pub mod variant;

pub use admission::ExecutableAdmissionError;
pub use profile::{
    DefaultVariantSelection, ExecutableTargetProfile, HardwareTargetContract,
    RuntimeTargetContract,
};
pub use provenance::CompilerProvenance;
pub use receipt::ExecutableCompilationReceipt;
pub use seal::{ExecutableSeal, ExecutableSignature};
pub use variant::{ShapeProfile, ShapeSpecializedProgram, ShapeSpecializedVariantId};

//! AOT kernel variant generation and dispatch for Apple Silicon.
//!
//! Provides an ahead-of-time kernel variant catalog system:
//! - Profile database of known Apple Silicon hardware (M1–M5, all tiers)
//! - Template expansion for parametric Metal kernel generation
//! - AOT compiler wrapper for build-time variant compilation
//! - Kernel catalog for embedding multiple variants in a CImage
//! - Runtime variant selector for picking the best embedded kernel
//! - Quality × performance scoring with held-out validation

pub mod catalog;
pub mod compiler;
pub mod device_match;
pub mod generator;
pub mod parameters;
pub mod profile_db;
pub mod profile_id;
pub mod receipts;
pub mod selector;
pub mod template;
#[cfg(test)]
pub mod tests;
pub mod validate;

pub use catalog::{CImageKernelCatalog, KernelMetallibPayloadRef, KernelVariantEntry};
pub use compiler::{AotMetalCompiler, CompileError, CompiledKernelVariant};
pub use device_match::{
    match_amd_device_to_profile, match_device_to_profile, RuntimeAmdDeviceProfile,
    RuntimeMetalDeviceProfile,
};
pub use generator::{AotTargetMatrix, KernelVariantGenerator};
pub use parameters::{DType, KernelFamily, KernelParameters};
pub use profile_db::{
    AmdGpuProfile, AmdProfileDb, AneProfile, AppleSiliconProfile, AppleSiliconProfileDb,
    GpuProfile, KernelMicrobenchReceipt, MeasuredKernelProfile, MemoryProfile, MetalGpuFamily,
    ProfileSourceReceipt, StaticMetalCaps,
};
pub use profile_id::{AmdGpuProfileId, AppleSiliconProfileId, ProfileEvidenceStatus};
pub use receipts::{
    HeldOutShapeResult, KernelCompileReceipt, KernelPerformanceReceipt, KernelValidationReceipt,
    QualityPerformanceScore,
};
pub use selector::{KernelVariantSelector, MatchType, VariantSelection};
pub use template::{KernelTemplateExpander, MetalKernelTemplate, TemplateError};
pub use validate::{CatalogValidator, HeldOutValidator, ValidationCheck, ValidationReport};

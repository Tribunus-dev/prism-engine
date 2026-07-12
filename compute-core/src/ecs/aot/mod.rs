//! AOT (ahead-of-time) compiler — kernel variants and public compilation API.
//!
//! This module is being consolidated into the unified PrismCompiler API.
//! See `prism_compiler` for the single public compilation entry point.
//! The remaining submodules provide AOT kernel variant support.

// PrismCompiler — the single public compilation API.
// Gated behind prism-backend for the full compile path.
pub mod gguf_frontend;
#[cfg(feature = "prism-backend")]
pub mod prism_compiler;

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
pub mod validate;

pub use catalog::{CImageKernelCatalog, KernelMetallibPayloadRef, KernelVariantEntry};
pub use compiler::{CompileError, CompiledKernelVariant};
pub use device_match::{
    match_device_to_profile, RuntimeAmdDeviceProfile, RuntimeMetalDeviceProfile,
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
pub use template::{MetalKernelTemplate, TemplateError};
pub use validate::{CatalogValidator, HeldOutValidator, ValidationCheck, ValidationReport};

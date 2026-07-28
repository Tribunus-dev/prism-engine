//! Executable admission — typed refusal for runtime compatibility
//! checks against a [`super::ExecutableTargetProfile`].

use serde::{Deserialize, Serialize};

/// Reasons an executable may be refused admission by a runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutableAdmissionError {
    /// The seal is invalid (signature, hash, or structure failure).
    InvalidSeal,
    /// The format version is not supported by this runtime.
    UnsupportedFormatVersion,
    /// The executable has no target profile matching this runtime.
    MissingTargetProfile,
    /// The target hardware contract does not match the runtime's hardware.
    IncompatibleHardwareProfile,
    /// The runtime contract does not match the executable's requirements.
    IncompatibleRuntimeProfile,
    /// A required feature is missing. The contained string names the feature.
    MissingRequiredFeature(String),
    /// The artifact hash did not match the seal.
    ArtifactHashMismatch,
    /// A content object hash did not match the seal.
    ContentObjectHashMismatch,
    /// The executable has no program variant for the requested shape.
    MissingProgramVariant,
    /// The arena plan cannot be satisfied by the available memory.
    ArenaPlanUnsatisfied,
    /// The residency plan cannot be satisfied by the available memory.
    ResidencyPlanUnsatisfied,
    /// The KV cache plan cannot be satisfied by the available memory.
    KvPlanUnsatisfied,
    /// A Core ML artifact required for the executable is unavailable.
    CoreAiArtifactUnavailable,
    /// A Metal pipeline required for the executable is unavailable.
    MetalPipelineUnavailable,
    /// An Accelerate artifact required for the executable is unavailable.
    AccelerateArtifactUnavailable,
    /// A state domain required for the executable is unavailable.
    StateDomainUnavailable,
}

//! Runtime kernel selection — pure data types and pure algorithms for
//! kernel variant selection, candidate benchmark evidence, and proof
//! seals.

pub mod evidence;
pub mod proof_seal;
pub mod selection;

pub use evidence::{
    CandidateBenchmarkEvidence, CompiledCandidateEvidence, MeasurementEnvironment,
    SelectionConfidence,
};
pub use proof_seal::{ProfileProofSeal, ProfileProofSealBundle};
pub use selection::{
    KernelCandidateEvidence, KernelConfiguration, KernelSelectionReceipt, KernelVariantId,
    PreselectedKernelVariant,
};

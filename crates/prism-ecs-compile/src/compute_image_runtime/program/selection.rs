//! Program artifact selection — pure data types and pure algorithms for
//! selecting the right program variant for an incoming request.

use serde::{Deserialize, Serialize};

use super::phase_program::ProgramId;

/// A selected program artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramArtifactSelection {
    /// Program identifier.
    pub program_id: ProgramId,
    /// Reason for the selection.
    pub reason: String,
    /// Declared fallback chain.
    pub fallback_chain: DeclaredFallbackChain,
}

/// Declared fallback chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeclaredFallbackChain {
    /// Steps in the fallback chain.
    pub steps: Vec<FallbackStep>,
}

/// A single fallback step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackStep {
    /// Program identifier to try.
    pub program_id: ProgramId,
    /// Trigger that causes this fallback to be used.
    pub trigger: String,
}

/// Reasons a variant selection may be refused.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariantSelectionRefusal {
    /// No variant supports the requested shape class.
    NoCompatibleVariant,
    /// All variants failed the selection qualification.
    AllVariantsIneligible,
    /// The variant list is empty.
    EmptyVariantList,
}

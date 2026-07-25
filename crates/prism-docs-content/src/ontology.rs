//! Ontology — the typed enums that classify entities.
//!
//! This module owns the canonical authority for the site's vocabulary
//! enums. Every other module re-exports from here.

use serde::{Deserialize, Serialize};

/// How confident we are that a claim matches reality.
///
/// This is the typed analog of the previous JS `KNOWLEDGE_STATES` and
/// `BELIEF_STATES` constants. The values are deliberately equal to the
/// old JS string values so existing content migrates losslessly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KnowledgeState {
    Unknown,
    Hypothesized,
    Derived,
    Observed,
    Verified,
    Measured,
    Historical,
}

impl KnowledgeState {
    pub const ALL: &'static [KnowledgeState] = &[
        KnowledgeState::Unknown,
        KnowledgeState::Hypothesized,
        KnowledgeState::Derived,
        KnowledgeState::Observed,
        KnowledgeState::Verified,
        KnowledgeState::Measured,
        KnowledgeState::Historical,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            KnowledgeState::Unknown => "unknown",
            KnowledgeState::Hypothesized => "hypothesized",
            KnowledgeState::Derived => "derived",
            KnowledgeState::Observed => "observed",
            KnowledgeState::Verified => "verified",
            KnowledgeState::Measured => "measured",
            KnowledgeState::Historical => "historical",
        }
    }
}

/// The lifecycle state of an entity — possible, active, sealed, etc.
///
/// Mirrors the previous JS `EXISTENCE_STATES` constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExistenceState {
    Possible,
    Active,
    Sealed,
    Executing,
    Complete,
}

impl ExistenceState {
    pub const ALL: &'static [ExistenceState] = &[
        ExistenceState::Possible,
        ExistenceState::Active,
        ExistenceState::Sealed,
        ExistenceState::Executing,
        ExistenceState::Complete,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            ExistenceState::Possible => "possible",
            ExistenceState::Active => "active",
            ExistenceState::Sealed => "sealed",
            ExistenceState::Executing => "executing",
            ExistenceState::Complete => "complete",
        }
    }
}

/// The class of a claim — illustrative, architectural, repository,
/// compile-verified, or measured.
///
/// Mirrors the previous JS `CLAIM_CLASSES` constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ClaimClass {
    Illustrative,
    Architectural,
    Repository,
    CompileVerified,
    Measured,
}

impl ClaimClass {
    pub const ALL: &'static [ClaimClass] = &[
        ClaimClass::Illustrative,
        ClaimClass::Architectural,
        ClaimClass::Repository,
        ClaimClass::CompileVerified,
        ClaimClass::Measured,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            ClaimClass::Illustrative => "illustrative",
            ClaimClass::Architectural => "architectural",
            ClaimClass::Repository => "repository",
            ClaimClass::CompileVerified => "compile-verified",
            ClaimClass::Measured => "measured",
        }
    }

    /// The constitutional rule: a `Measured` claim must carry at least
    /// one source reference and a measurement constraint.
    pub fn requires_source(self) -> bool {
        matches!(self, ClaimClass::Measured)
    }
}

/// The reader's mode — what they're here to do.
///
/// Mirrors the previous JS `OBSERVER_MODES` constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ObserverMode {
    Observer,
    Builder,
    Researcher,
    CompilerEngineer,
    RuntimeEngineer,
    InfrastructureEngineer,
}

impl ObserverMode {
    pub const ALL: &'static [ObserverMode] = &[
        ObserverMode::Observer,
        ObserverMode::Builder,
        ObserverMode::Researcher,
        ObserverMode::CompilerEngineer,
        ObserverMode::RuntimeEngineer,
        ObserverMode::InfrastructureEngineer,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            ObserverMode::Observer => "observer",
            ObserverMode::Builder => "builder",
            ObserverMode::Researcher => "researcher",
            ObserverMode::CompilerEngineer => "compiler-engineer",
            ObserverMode::RuntimeEngineer => "runtime-engineer",
            ObserverMode::InfrastructureEngineer => "infrastructure-engineer",
        }
    }
}

/// The lens through which the site reads at the moment — focus,
/// dispersion, exploration, etc.
///
/// Mirrors the previous JS `OPTICAL_STATES` constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpticalState {
    Observation,
    Neutral,
    Focus,
    Dispersion,
    Exploration,
    Commit,
    Evidence,
}

impl OpticalState {
    pub const ALL: &'static [OpticalState] = &[
        OpticalState::Observation,
        OpticalState::Neutral,
        OpticalState::Focus,
        OpticalState::Dispersion,
        OpticalState::Exploration,
        OpticalState::Commit,
        OpticalState::Evidence,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            OpticalState::Observation => "observation",
            OpticalState::Neutral => "neutral",
            OpticalState::Focus => "focus",
            OpticalState::Dispersion => "dispersion",
            OpticalState::Exploration => "exploration",
            OpticalState::Commit => "commit",
            OpticalState::Evidence => "evidence",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knowledge_state_round_trip() {
        for state in KnowledgeState::ALL {
            let s = serde_json::to_string(state).unwrap();
            let back: KnowledgeState = serde_json::from_str(&s).unwrap();
            assert_eq!(*state, back);
        }
    }

    #[test]
    fn existence_state_round_trip() {
        for state in ExistenceState::ALL {
            let s = serde_json::to_string(state).unwrap();
            let back: ExistenceState = serde_json::from_str(&s).unwrap();
            assert_eq!(*state, back);
        }
    }

    #[test]
    fn claim_class_measured_requires_source() {
        assert!(ClaimClass::Measured.requires_source());
        for class in ClaimClass::ALL.iter().filter(|c| **c != ClaimClass::Measured) {
            assert!(!class.requires_source());
        }
    }
}

//! EvaluationRole — distinguishes execution purposes for
//! provenance, admission, and auditing.
//!
//! This module owns the canonical authority for the role a backend
//! plays in a single evaluation run. The role is what the admission
//! gate inspects to reject evidence where the candidate backend is
//! also its only oracle.

use serde::{Deserialize, Serialize};

/// Distinguishes execution purposes for provenance, admission, and auditing.
///
/// Admission must reject evidence where the candidate backend is also its
/// only oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvaluationRole {
    /// CPU/oracle reference execution — trusted independent comparison.
    Oracle,
    /// Candidate backend execution under test.
    Candidate,
    /// Intermediate transform validation (e.g. planar checkpoint).
    PlanarTransform,
    /// Cross-check between two independent backends.
    CrossCheck,
    /// Replay of a previously recorded evaluation.
    Replay,
}

impl EvaluationRole {
    /// Returns true if this role is allowed to serve as an oracle.
    ///
    /// Only `Oracle` and `Replay` can satisfy the
    /// `require_independent_oracle` admission policy.
    pub fn is_oracle_eligible(self) -> bool {
        matches!(self, Self::Oracle | Self::Replay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_variants_are_distinct() {
        let roles = [
            EvaluationRole::Oracle,
            EvaluationRole::Candidate,
            EvaluationRole::PlanarTransform,
            EvaluationRole::CrossCheck,
            EvaluationRole::Replay,
        ];
        for (i, a) in roles.iter().enumerate() {
            for (j, b) in roles.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b, "roles at index {i} and {j} must differ");
                }
            }
        }
    }

    #[test]
    fn oracle_eligibility() {
        assert!(EvaluationRole::Oracle.is_oracle_eligible());
        assert!(EvaluationRole::Replay.is_oracle_eligible());
        assert!(!EvaluationRole::Candidate.is_oracle_eligible());
        assert!(!EvaluationRole::PlanarTransform.is_oracle_eligible());
        assert!(!EvaluationRole::CrossCheck.is_oracle_eligible());
    }

    #[test]
    fn role_serializes() {
        let role = EvaluationRole::Oracle;
        let json = serde_json::to_string(&role).expect("serialize");
        let restored: EvaluationRole = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, role);
    }
}

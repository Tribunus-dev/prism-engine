//! Evaluation role — distinguishes execution purposes for provenance and
//! admission governance.

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

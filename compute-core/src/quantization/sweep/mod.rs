//! QuantSweep — parametric quantization lab for codec family comparison.
//!
//! Sweeps codec parameters across representative tensors and emits comparable
//! receipts for analysis and policy selection.

pub mod candidate;
pub mod families;
pub mod runner;
pub mod spec;
#[cfg(test)]
pub mod tests;

// Re-export commonly used types at the sweep module level.
pub use candidate::*;
pub use families::{FamilyCandidate, ParamError, QuantError, QuantFamily, SweepScratch};
pub use crate::quantization::contract::SourceMatrixLayout;
pub use spec::*;

/// Admission status of a candidate after weight-space validation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SweepCandidateStatus {
    /// Candidate has not yet been evaluated.
    Pending,
    /// All metrics within target thresholds.
    Passed,
    /// Weight metric exceeds target but within investigation ceiling.
    InvestigationBand { warning: String },
    /// Metric exceeds ceiling; rejected.
    Rejected { reason: String },
}

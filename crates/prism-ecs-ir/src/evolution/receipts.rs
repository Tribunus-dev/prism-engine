//! Evolution-stage validation and performance receipts.
//!
//! Owns the canonical authority for the two receipt types the
//! constitutional evolution pipeline emits after candidate evaluation:
//!
//! - [`NumericalReceipt`] — records the numeric-validation outcome
//!   (max absolute / relative error vs a reference) for one candidate.
//! - [`PerformanceReceipt`] — records the latency / throughput
//!   measurement for one candidate.
//!
//! Both receipts carry a `provenance: Vec<String>` chain listing the
//! compiled-artifact digests that participated in producing the
//! receipt. The receipt is engine-agnostic: it stores a digest list
//! rather than engine-specific kernel-abi types, so it is portable
//! across any backend that can produce a content-addressed digest.
//!
//! # Migration provenance
//!
//! These types are the constitutional re-home of the engine's
//! `compute-core/src/ecs/evolution/foundation::{NumericalReceipt,
//! PerformanceReceipt}` types. The engine's types carried a
//! `Vec<crate::ecs::canonical::kernel_abi::ArtifactProvenance>`
//! provenance chain. The constitutional types carry a simpler
//! `Vec<String>` of compiled-artifact digests. Engine callers map
//! their `ArtifactProvenance::compiled_byte_digest` to the
//! constitutional string list. The digest list is sufficient for
//! replay and propagation, which is the constitutional invariant:
//! a receipt must be content-addressable and replayable.
//!
//! # Module authority
//!
//! This module owns exactly one authority: the receipt shapes for
//! evolution-stage validation and measurement. It does not own
//! candidate identity (see `foundation`), scoring (see
//! `foundation::FitnessScore`), or the search loop (see `joint`).

use serde::{Deserialize, Serialize};

use crate::evolution::foundation::CandidateId;

/// Numerical validation receipt — compares candidate output to reference.
///
/// Produced by the evaluation system after a candidate has been
/// executed and its output compared against a reference trace.
/// A candidate is admitted to the next generation only when
/// `passed == true` and both error bounds stay below the chosen
/// `threshold`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NumericalReceipt {
    /// The candidate the receipt was produced for.
    pub candidate_id: CandidateId,
    /// Whether the candidate passed numerical validation.
    pub passed: bool,
    /// Worst observed absolute error vs the reference.
    pub max_absolute_error: f64,
    /// Worst observed relative error vs the reference.
    pub max_relative_error: f64,
    /// The threshold (typically the candidate's tolerance band)
    /// that the absolute and relative errors were compared against.
    pub threshold: f64,
    /// Provenance chain: compiled-artifact digests linked to this
    /// receipt. The list is content-addressed — every digest
    /// uniquely identifies the compiled artifact that produced part
    /// of the candidate's output.
    pub provenance: Vec<String>,
}

/// Performance measurement receipt.
///
/// Produced by the evaluation system after a candidate has been
/// timed on the target backend. Latency numbers are in nanoseconds;
/// `memory_traffic_bytes` is the total bytes the candidate moved
/// through the memory subsystem during a measured run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PerformanceReceipt {
    /// The candidate the receipt was produced for.
    pub candidate_id: CandidateId,
    /// 50th-percentile latency in nanoseconds.
    pub latency_p50_ns: u64,
    /// 95th-percentile latency in nanoseconds.
    pub latency_p95_ns: u64,
    /// Time spent encoding / dispatching the candidate in nanoseconds.
    pub encode_time_ns: u64,
    /// Time spent on device–host synchronization in nanoseconds.
    pub sync_time_ns: u64,
    /// Total memory traffic in bytes across all measured runs.
    pub memory_traffic_bytes: u64,
    /// Optional energy measurement in microjoules. `None` means
    /// the backend does not expose an energy counter.
    pub energy_uj: Option<u64>,
    /// Number of measured repetitions the latency figures were
    /// derived from.
    pub repetitions: usize,
    /// Provenance chain: compiled-artifact digests linked to this
    /// receipt. See [`NumericalReceipt::provenance`] for the
    /// contract.
    pub provenance: Vec<String>,
}

impl NumericalReceipt {
    /// Construct a passing numerical receipt with the given
    /// error bounds and an empty provenance chain.
    pub fn passing(candidate_id: CandidateId, threshold: f64) -> Self {
        Self {
            candidate_id,
            passed: true,
            max_absolute_error: 0.0,
            max_relative_error: 0.0,
            threshold,
            provenance: Vec::new(),
        }
    }

    /// Construct a failing numerical receipt with the given
    /// observed error bounds and an empty provenance chain.
    pub fn failing(
        candidate_id: CandidateId,
        max_absolute_error: f64,
        max_relative_error: f64,
        threshold: f64,
    ) -> Self {
        Self {
            candidate_id,
            passed: false,
            max_absolute_error,
            max_relative_error,
            threshold,
            provenance: Vec::new(),
        }
    }
}

impl PerformanceReceipt {
    /// Construct a performance receipt with the given latency
    /// figures and an empty provenance chain.
    pub fn new(
        candidate_id: CandidateId,
        latency_p50_ns: u64,
        latency_p95_ns: u64,
        repetitions: usize,
    ) -> Self {
        Self {
            candidate_id,
            latency_p50_ns,
            latency_p95_ns,
            encode_time_ns: 0,
            sync_time_ns: 0,
            memory_traffic_bytes: 0,
            energy_uj: None,
            repetitions,
            provenance: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numerical_receipt_passing_starts_with_empty_provenance() {
        let r = NumericalReceipt::passing(CandidateId("cand-1".into()), 0.05);
        assert!(r.passed);
        assert_eq!(r.max_absolute_error, 0.0);
        assert_eq!(r.max_relative_error, 0.0);
        assert_eq!(r.threshold, 0.05);
        assert!(r.provenance.is_empty());
    }

    #[test]
    fn numerical_receipt_failing_preserves_error_bounds() {
        let r = NumericalReceipt::failing(
            CandidateId("cand-2".into()),
            0.123,
            0.456,
            0.05,
        );
        assert!(!r.passed);
        assert_eq!(r.max_absolute_error, 0.123);
        assert_eq!(r.max_relative_error, 0.456);
        assert_eq!(r.threshold, 0.05);
        assert!(r.provenance.is_empty());
    }

    #[test]
    fn performance_receipt_new_initializes_zero_sync_and_traffic() {
        let r = PerformanceReceipt::new(CandidateId("cand-3".into()), 100, 120, 5);
        assert_eq!(r.latency_p50_ns, 100);
        assert_eq!(r.latency_p95_ns, 120);
        assert_eq!(r.repetitions, 5);
        assert_eq!(r.encode_time_ns, 0);
        assert_eq!(r.sync_time_ns, 0);
        assert_eq!(r.memory_traffic_bytes, 0);
        assert!(r.energy_uj.is_none());
        assert!(r.provenance.is_empty());
    }

    #[test]
    fn receipts_serde_roundtrip_preserves_provenance() {
        let mut r = NumericalReceipt::passing(CandidateId("cand-4".into()), 0.01);
        r.provenance
            .push("deadbeef".to_string());
        let json = serde_json::to_string(&r).unwrap();
        let back: NumericalReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
        assert_eq!(back.provenance, vec!["deadbeef".to_string()]);
    }
}

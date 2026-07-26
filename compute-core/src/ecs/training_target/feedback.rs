//! Training feedback — compares evidence against training targets and produces
//! a structured feedback report with per-target failure modes and suggested loss terms.
//!
//! The [`TrainingFeedbackBuilder`] is the main entry point: it takes a set of
//! training targets (each with gates/thresholds) and a ledger of observed
//! evidence, then returns a [`TrainingFeedbackReport`] with per-target items,
//! an aggregate status, and summary statistics.
//!
//! # Gate comparison
//!
//! | Gate threshold              | Observed field              | Failure mode              | Suggested loss                |
//! |-----------------------------|-----------------------------|---------------------------|-------------------------------|
//! | `max_weight_nrmse`          | `observed_nrmse`            | `WeightNrmseTooHigh`      | `ReduceWeightReconstructionError` |
//! | `max_zero_collapse_ratio`   | `observed_zero_collapse`    | `ZeroCollapseTooHigh`     | `ReduceZeroCollapse`          |
//! | `max_operator_nrmse`        | `observed_nrmse`            | `OperatorNrmseTooHigh`    | `ReduceActivationWeightedError`  |
//! | `min_operator_cosine`       | `observed_cosine`           | `OperatorCosineTooLow`    | `PreserveHiddenDirection`     |
//! | `max_operator_abs_error`    | `observed_max_abs`          | `OperatorAbsTailTooHigh`  | `ReduceOperatorTailError`     |
//! | `min_byte_savings_ratio`    | `observed_byte_savings`     | `ByteSavingsTooLow`       | `IncreaseDraftAcceptance`     |

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::gates::{TargetedLossTerm, TrainingFailureMode, TrainingTargetStatus};
/// Backward-compat alias: `TargetStatus` was originally defined in this
/// module; now it re-exports [`TrainingTargetStatus`] from `gates`.
pub type TargetStatus = TrainingTargetStatus;

// ── Evidence entry ──────────────────────────────────────────────────────────

/// A single observed measurement from the evidence ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceEntry {
    pub tensor_key: String,
    pub tensor_class: String,
    pub operator: String,
    pub observed_nrmse: Option<f64>,
    pub observed_zero_collapse: Option<f64>,
    pub observed_cosine: Option<f64>,
    pub observed_max_abs: Option<f64>,
    pub byte_savings_ratio: Option<f64>,
}

// ── Gate thresholds ─────────────────────────────────────────────────────────

/// Per-target gate thresholds to compare against observed evidence.
///
/// Every optional field is `None` when the gate is not applicable. A `None`
/// threshold means the gate passes unconditionally on that axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateThresholds {
    /// Maximum allowable weight-space NRMSE.
    pub max_weight_nrmse: Option<f64>,
    /// Maximum allowable zero-collapse ratio.
    pub max_zero_collapse_ratio: Option<f64>,
    /// Maximum allowable operator-level NRMSE.
    pub max_operator_nrmse: Option<f64>,
    /// Minimum allowable operator-level cosine similarity.
    pub min_operator_cosine: Option<f64>,
    /// Maximum allowable operator-level absolute error tail.
    pub max_operator_abs_error: Option<f64>,
    /// Minimum byte-savings ratio vs baseline.
    pub min_byte_savings_ratio: Option<f64>,
}

impl Default for GateThresholds {
    fn default() -> Self {
        Self {
            max_weight_nrmse: None,
            max_zero_collapse_ratio: None,
            max_operator_nrmse: None,
            min_operator_cosine: None,
            max_operator_abs_error: None,
            min_byte_savings_ratio: None,
        }
    }
}

// ── Target descriptor (lightweight input) ───────────────────────────────────

/// A training target paired with its gates, used as input to
/// [`TrainingFeedbackBuilder::build`].
///
/// This lightweight representation avoids depending on the full
/// `WeightTrainingTarget` / `TrainingTargetSpec` types from `spec.rs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetWithGates {
    pub target_id: String,
    /// One or more tensor key patterns that identify matching evidence entries.
    pub tensor_key_match: Vec<String>,
    pub tensor_class: String,
    pub gates: GateThresholds,
}

// ── Feedback item ───────────────────────────────────────────────────────────

/// One gate check result for a single target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingFeedbackItem {
    pub target_id: String,
    pub tensor_key: String,
    pub tensor_class: String,
    /// Human-readable name of the gate that failed (e.g. `"max_weight_nrmse"`).
    pub failed_gate: String,
    pub failure_mode: TrainingFailureMode,
    pub observed_value: Option<f64>,
    pub required_value: Option<f64>,
    /// How severely this gate was violated.
    ///
    /// For upper-bound gates (e.g. NRMSE ≤ threshold): `observed / threshold - 1.0`.
    /// For lower-bound gates (e.g. cosine ≥ threshold): `threshold / observed - 1.0`.
    /// `0.0` means borderline; positive values are violations.
    pub severity: f64,
    pub suggested_loss: Option<TargetedLossTerm>,
}

// ── Summary ─────────────────────────────────────────────────────────────────

/// Aggregate statistics across all targets in a feedback run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingFeedbackSummary {
    pub total_targets: usize,
    pub satisfied: usize,
    pub failed: usize,
    pub warnings: usize,
    /// Gate name → fraction of targets that passed that gate (0.0 – 1.0).
    pub gate_results: HashMap<String, f64>,
}

// ── Report ──────────────────────────────────────────────────────────────────

/// The complete feedback report produced by [`TrainingFeedbackBuilder::build`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingFeedbackReport {
    pub report_version: u32,
    pub spec_digest: String,
    pub checkpoint_digest: String,
    pub evidence_ledger_digest: String,
    pub status: TrainingTargetStatus,
    pub items: Vec<TrainingFeedbackItem>,
    pub summary: TrainingFeedbackSummary,
}

// ── Builder ─────────────────────────────────────────────────────────────────

/// Compares evidence against training targets and produces a
/// [`TrainingFeedbackReport`].
///
/// # Gate comparison rules
///
/// For each target:
///
/// 1. Collect all evidence entries whose `tensor_key` matches one of the
///    target's `tensor_key_match` patterns AND whose `tensor_class` matches.
/// 2. Merge matching entries by taking the first non-`None` value for each
///    observed field.
/// 3. For each non-`None` gate threshold, compare against the corresponding
///    observed value:
///    - `max_weight_nrmse` vs `observed_nrmse` → `WeightNrmseTooHigh`
///    - `max_zero_collapse_ratio` vs `observed_zero_collapse` → `ZeroCollapseTooHigh`
///    - `max_operator_nrmse` vs `observed_nrmse` → `OperatorNrmseTooHigh`
///    - `min_operator_cosine` vs `observed_cosine` → `OperatorCosineTooLow`
///    - `max_operator_abs_error` vs `observed_max_abs` → `OperatorAbsTailTooHigh`
///    - `min_byte_savings_ratio` vs `byte_savings_ratio` → `ByteSavingsTooLow`
/// 4. If a gate fails, produce a [`TrainingFeedbackItem`] with the appropriate
///    failure mode and suggested loss term.
///
/// # Example
///
/// ```ignore
/// let report = TrainingFeedbackBuilder::build(
///     &[target_a, target_b],
///     &evidence_map,
///     "spec-v1",
///     "ckpt-abc",
///     "ev-ledger-123",
/// );
/// ```
pub struct TrainingFeedbackBuilder;

impl TrainingFeedbackBuilder {
    /// Build a feedback report by comparing every target's gates against matching
    /// evidence entries.
    pub fn build(
        targets: &[TargetWithGates],
        evidence: &HashMap<String, Vec<EvidenceEntry>>,
        spec_digest: &str,
        checkpoint_digest: &str,
        evidence_ledger_digest: &str,
    ) -> TrainingFeedbackReport {
        let mut all_items: Vec<TrainingFeedbackItem> = Vec::new();
        let mut status_counts: HashMap<TrainingTargetStatus, usize> = HashMap::new();

        for target in targets {
            // ── 1. Collect matching evidence entries ──────────────────
            let matching: Vec<&EvidenceEntry> = evidence
                .iter()
                .flat_map(|(_, entries)| entries.iter())
                .filter(|entry| {
                    target
                        .tensor_key_match
                        .iter()
                        .any(|k| k == &entry.tensor_key)
                        && entry.tensor_class == target.tensor_class
                })
                .collect();

            if matching.is_empty() {
                // No evidence at all for this target.
                *status_counts
                    .entry(TrainingTargetStatus::EvidenceIncomplete)
                    .or_insert(0) += 1;
                continue;
            }

            // ── 2. Merge into a single observed view ──────────────────
            let merged = merge_evidence(&matching);

            // ── 3. Check every gate ───────────────────────────────────
            let mut failures: Vec<TrainingFeedbackItem> = Vec::new();
            let gate_count = applicable_gate_count(&target.gates);

            check_weight_nrmse(target, &merged, &mut failures);
            check_zero_collapse(target, &merged, &mut failures);
            check_operator_nrmse(target, &merged, &mut failures);
            check_operator_cosine(target, &merged, &mut failures);
            check_operator_abs_error(target, &merged, &mut failures);
            check_byte_savings(target, &merged, &mut failures);

            // ── 4. Determine per-target status ────────────────────────
            let target_status = if failures.is_empty() {
                TrainingTargetStatus::Satisfied
            } else if gate_count > 0 && failures.len() < gate_count {
                TrainingTargetStatus::PartiallySatisfied
            } else {
                TrainingTargetStatus::Failed
            };

            *status_counts.entry(target_status).or_insert(0) += 1;
            all_items.extend(failures);
        }

        // ── Aggregate status across all targets ──────────────────────────
        let aggregate_status = aggregate_status(&status_counts);

        // ── Gate result fractions ────────────────────────────────────────
        let gate_results = compute_gate_results(targets, evidence);

        let total_targets = targets.len();
        let satisfied = status_counts
            .get(&TrainingTargetStatus::Satisfied)
            .copied()
            .unwrap_or(0);
        let failed = status_counts
            .get(&TrainingTargetStatus::Failed)
            .copied()
            .unwrap_or(0);
        let warnings = status_counts
            .get(&TrainingTargetStatus::EvidenceIncomplete)
            .copied()
            .unwrap_or(0);

        TrainingFeedbackReport {
            report_version: 1,
            spec_digest: spec_digest.to_string(),
            checkpoint_digest: checkpoint_digest.to_string(),
            evidence_ledger_digest: evidence_ledger_digest.to_string(),
            status: aggregate_status,
            items: all_items,
            summary: TrainingFeedbackSummary {
                total_targets,
                satisfied,
                failed,
                warnings,
                gate_results,
            },
        }
    }
}

// ── Individual gate checks ──────────────────────────────────────────────────

fn check_weight_nrmse(
    target: &TargetWithGates,
    merged: &EvidenceEntry,
    failures: &mut Vec<TrainingFeedbackItem>,
) {
    let threshold = match target.gates.max_weight_nrmse {
        Some(v) => v,
        None => return,
    };
    let observed = match merged.observed_nrmse {
        Some(v) => v,
        None => return,
    };
    if observed > threshold {
        let severity = observed / threshold - 1.0;
        failures.push(TrainingFeedbackItem {
            target_id: target.target_id.clone(),
            tensor_key: merged.tensor_key.clone(),
            tensor_class: merged.tensor_class.clone(),
            failed_gate: "max_weight_nrmse".into(),
            failure_mode: TrainingFailureMode::WeightNrmseTooHigh,
            observed_value: Some(observed),
            required_value: Some(threshold),
            severity,
            suggested_loss: Some(TargetedLossTerm::ReduceWeightReconstructionError),
        });
    }
}

fn check_zero_collapse(
    target: &TargetWithGates,
    merged: &EvidenceEntry,
    failures: &mut Vec<TrainingFeedbackItem>,
) {
    let threshold = match target.gates.max_zero_collapse_ratio {
        Some(v) => v,
        None => return,
    };
    let observed = match merged.observed_zero_collapse {
        Some(v) => v,
        None => return,
    };
    if observed > threshold {
        let severity = observed / threshold - 1.0;
        failures.push(TrainingFeedbackItem {
            target_id: target.target_id.clone(),
            tensor_key: merged.tensor_key.clone(),
            tensor_class: merged.tensor_class.clone(),
            failed_gate: "max_zero_collapse_ratio".into(),
            failure_mode: TrainingFailureMode::ZeroCollapseTooHigh,
            observed_value: Some(observed),
            required_value: Some(threshold),
            severity,
            suggested_loss: Some(TargetedLossTerm::ReduceZeroCollapse),
        });
    }
}

fn check_operator_nrmse(
    target: &TargetWithGates,
    merged: &EvidenceEntry,
    failures: &mut Vec<TrainingFeedbackItem>,
) {
    let threshold = match target.gates.max_operator_nrmse {
        Some(v) => v,
        None => return,
    };
    let observed = match merged.observed_nrmse {
        Some(v) => v,
        None => return,
    };
    if observed > threshold {
        let severity = observed / threshold - 1.0;
        failures.push(TrainingFeedbackItem {
            target_id: target.target_id.clone(),
            tensor_key: merged.tensor_key.clone(),
            tensor_class: merged.tensor_class.clone(),
            failed_gate: "max_operator_nrmse".into(),
            failure_mode: TrainingFailureMode::OperatorNrmseTooHigh,
            observed_value: Some(observed),
            required_value: Some(threshold),
            severity,
            suggested_loss: Some(TargetedLossTerm::ReduceActivationWeightedError),
        });
    }
}

fn check_operator_cosine(
    target: &TargetWithGates,
    merged: &EvidenceEntry,
    failures: &mut Vec<TrainingFeedbackItem>,
) {
    let threshold = match target.gates.min_operator_cosine {
        Some(v) => v,
        None => return,
    };
    let observed = match merged.observed_cosine {
        Some(v) => v,
        None => return,
    };
    if observed < threshold {
        // Lower-bound gate: severity = threshold / observed - 1.0
        let severity = if observed > 0.0 {
            threshold / observed - 1.0
        } else {
            f64::MAX
        };
        failures.push(TrainingFeedbackItem {
            target_id: target.target_id.clone(),
            tensor_key: merged.tensor_key.clone(),
            tensor_class: merged.tensor_class.clone(),
            failed_gate: "min_operator_cosine".into(),
            failure_mode: TrainingFailureMode::OperatorCosineTooLow,
            observed_value: Some(observed),
            required_value: Some(threshold),
            severity,
            suggested_loss: Some(TargetedLossTerm::PreserveHiddenDirection),
        });
    }
}

fn check_operator_abs_error(
    target: &TargetWithGates,
    merged: &EvidenceEntry,
    failures: &mut Vec<TrainingFeedbackItem>,
) {
    let threshold = match target.gates.max_operator_abs_error {
        Some(v) => v,
        None => return,
    };
    let observed = match merged.observed_max_abs {
        Some(v) => v,
        None => return,
    };
    if observed > threshold {
        let severity = observed / threshold - 1.0;
        failures.push(TrainingFeedbackItem {
            target_id: target.target_id.clone(),
            tensor_key: merged.tensor_key.clone(),
            tensor_class: merged.tensor_class.clone(),
            failed_gate: "max_operator_abs_error".into(),
            failure_mode: TrainingFailureMode::OperatorAbsTailTooHigh,
            observed_value: Some(observed),
            required_value: Some(threshold),
            severity,
            suggested_loss: Some(TargetedLossTerm::ReduceOperatorTailError),
        });
    }
}

fn check_byte_savings(
    target: &TargetWithGates,
    merged: &EvidenceEntry,
    failures: &mut Vec<TrainingFeedbackItem>,
) {
    let threshold = match target.gates.min_byte_savings_ratio {
        Some(v) => v,
        None => return,
    };
    let observed = match merged.byte_savings_ratio {
        Some(v) => v,
        None => return,
    };
    if observed < threshold {
        let severity = if observed > 0.0 {
            threshold / observed - 1.0
        } else {
            f64::MAX
        };
        failures.push(TrainingFeedbackItem {
            target_id: target.target_id.clone(),
            tensor_key: merged.tensor_key.clone(),
            tensor_class: merged.tensor_class.clone(),
            failed_gate: "min_byte_savings_ratio".into(),
            failure_mode: TrainingFailureMode::ByteSavingsTooLow,
            observed_value: Some(observed),
            required_value: Some(threshold),
            severity,
            suggested_loss: Some(TargetedLossTerm::IncreaseDraftAcceptance),
        });
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Merge multiple evidence entries by taking the first non-`None` / non-empty
/// value for each field.
fn merge_evidence(entries: &[&EvidenceEntry]) -> EvidenceEntry {
    let mut merged = EvidenceEntry {
        tensor_key: String::new(),
        tensor_class: String::new(),
        operator: String::new(),
        observed_nrmse: None,
        observed_zero_collapse: None,
        observed_cosine: None,
        observed_max_abs: None,
        byte_savings_ratio: None,
    };

    for entry in entries {
        if merged.tensor_key.is_empty() && !entry.tensor_key.is_empty() {
            merged.tensor_key.clone_from(&entry.tensor_key);
        }
        if merged.tensor_class.is_empty() && !entry.tensor_class.is_empty() {
            merged.tensor_class.clone_from(&entry.tensor_class);
        }
        if merged.operator.is_empty() && !entry.operator.is_empty() {
            merged.operator.clone_from(&entry.operator);
        }
        if merged.observed_nrmse.is_none() {
            merged.observed_nrmse = entry.observed_nrmse;
        }
        if merged.observed_zero_collapse.is_none() {
            merged.observed_zero_collapse = entry.observed_zero_collapse;
        }
        if merged.observed_cosine.is_none() {
            merged.observed_cosine = entry.observed_cosine;
        }
        if merged.observed_max_abs.is_none() {
            merged.observed_max_abs = entry.observed_max_abs;
        }
        if merged.byte_savings_ratio.is_none() {
            merged.byte_savings_ratio = entry.byte_savings_ratio;
        }
    }

    merged
}

/// Count how many gates have an active (non-`None`) threshold.
fn applicable_gate_count(gates: &GateThresholds) -> usize {
    let mut count = 0usize;
    if gates.max_weight_nrmse.is_some() {
        count += 1;
    }
    if gates.max_zero_collapse_ratio.is_some() {
        count += 1;
    }
    if gates.max_operator_nrmse.is_some() {
        count += 1;
    }
    if gates.min_operator_cosine.is_some() {
        count += 1;
    }
    if gates.max_operator_abs_error.is_some() {
        count += 1;
    }
    if gates.min_byte_savings_ratio.is_some() {
        count += 1;
    }
    count
}

/// Determine the aggregate status from per-target status counts (pessimistic
/// worst-first ordering).
fn aggregate_status(counts: &HashMap<TrainingTargetStatus, usize>) -> TrainingTargetStatus {
    if counts.contains_key(&TrainingTargetStatus::Failed) {
        TrainingTargetStatus::Failed
    } else if counts.contains_key(&TrainingTargetStatus::PartiallySatisfied) {
        TrainingTargetStatus::PartiallySatisfied
    } else if counts.contains_key(&TrainingTargetStatus::EvidenceIncomplete) {
        TrainingTargetStatus::EvidenceIncomplete
    } else if counts.contains_key(&TrainingTargetStatus::Satisfied) {
        TrainingTargetStatus::Satisfied
    } else {
        TrainingTargetStatus::Draft
    }
}

/// Compute the fraction of targets that pass each named gate.
///
/// For each gate name (e.g. `"max_weight_nrmse"`), count how many targets have
/// the threshold set AND have matching evidence that passes it.
fn compute_gate_results(
    targets: &[TargetWithGates],
    evidence: &HashMap<String, Vec<EvidenceEntry>>,
) -> HashMap<String, f64> {
    // (pass_count, total_with_gate_and_evidence)
    let mut results: HashMap<String, (usize, usize)> = HashMap::new();

    for target in targets {
        let matching: Vec<&EvidenceEntry> = evidence
            .iter()
            .flat_map(|(_, entries)| entries.iter())
            .filter(|entry| {
                target
                    .tensor_key_match
                    .iter()
                    .any(|k| k == &entry.tensor_key)
                    && entry.tensor_class == target.tensor_class
            })
            .collect();

        if matching.is_empty() {
            continue;
        }
        let merged = merge_evidence(&matching);

        // max_weight_nrmse
        if let Some(thresh) = target.gates.max_weight_nrmse {
            let pass = merged.observed_nrmse.map_or(false, |v| v <= thresh);
            let entry = results.entry("max_weight_nrmse".into()).or_insert((0, 0));
            if pass {
                entry.0 += 1;
            }
            entry.1 += 1;
        }

        // max_zero_collapse_ratio
        if let Some(thresh) = target.gates.max_zero_collapse_ratio {
            let pass = merged.observed_zero_collapse.map_or(false, |v| v <= thresh);
            let entry = results
                .entry("max_zero_collapse_ratio".into())
                .or_insert((0, 0));
            if pass {
                entry.0 += 1;
            }
            entry.1 += 1;
        }

        // max_operator_nrmse
        if let Some(thresh) = target.gates.max_operator_nrmse {
            let pass = merged.observed_nrmse.map_or(false, |v| v <= thresh);
            let entry = results.entry("max_operator_nrmse".into()).or_insert((0, 0));
            if pass {
                entry.0 += 1;
            }
            entry.1 += 1;
        }

        // min_operator_cosine
        if let Some(thresh) = target.gates.min_operator_cosine {
            let pass = merged.observed_cosine.map_or(false, |v| v >= thresh);
            let entry = results
                .entry("min_operator_cosine".into())
                .or_insert((0, 0));
            if pass {
                entry.0 += 1;
            }
            entry.1 += 1;
        }

        // max_operator_abs_error
        if let Some(thresh) = target.gates.max_operator_abs_error {
            let pass = merged.observed_max_abs.map_or(false, |v| v <= thresh);
            let entry = results
                .entry("max_operator_abs_error".into())
                .or_insert((0, 0));
            if pass {
                entry.0 += 1;
            }
            entry.1 += 1;
        }

        // min_byte_savings_ratio
        if let Some(thresh) = target.gates.min_byte_savings_ratio {
            let pass = merged.byte_savings_ratio.map_or(false, |v| v >= thresh);
            let entry = results
                .entry("min_byte_savings_ratio".into())
                .or_insert((0, 0));
            if pass {
                entry.0 += 1;
            }
            entry.1 += 1;
        }
    }

    results
        .into_iter()
        .map(|(name, (pass, total))| {
            let fraction = if total > 0 {
                pass as f64 / total as f64
            } else {
                0.0
            };
            (name, fraction)
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_evidence_entry(
        tensor_key: &str,
        tensor_class: &str,
        nrmse: Option<f64>,
        zero_collapse: Option<f64>,
        cosine: Option<f64>,
        max_abs: Option<f64>,
        byte_savings: Option<f64>,
    ) -> EvidenceEntry {
        EvidenceEntry {
            tensor_key: tensor_key.to_string(),
            tensor_class: tensor_class.to_string(),
            operator: "test_op".to_string(),
            observed_nrmse: nrmse,
            observed_zero_collapse: zero_collapse,
            observed_cosine: cosine,
            observed_max_abs: max_abs,
            byte_savings_ratio: byte_savings,
        }
    }

    // ── feedback_zero_collapse ──────────────────────────────────────────

    /// A zero-collapse gate fails → item with `ReduceZeroCollapse` suggestion.
    #[test]
    fn feedback_zero_collapse() {
        let target = TargetWithGates {
            target_id: "t1".into(),
            tensor_key_match: vec!["layer1.weight".into()],
            tensor_class: "weight".into(),
            gates: GateThresholds {
                max_zero_collapse_ratio: Some(0.05),
                ..Default::default()
            },
        };

        let mut evidence: HashMap<String, Vec<EvidenceEntry>> = HashMap::new();
        evidence.insert(
            "layer1.weight".into(),
            vec![make_evidence_entry(
                "layer1.weight",
                "weight",
                None,
                Some(0.15), // exceeds 0.05
                None,
                None,
                None,
            )],
        );

        let report =
            TrainingFeedbackBuilder::build(&[target], &evidence, "spec-v1", "ckpt-abc", "ev-123");

        assert_eq!(report.items.len(), 1, "should have one failure item");
        let item = &report.items[0];
        assert_eq!(item.failed_gate, "max_zero_collapse_ratio");
        assert_eq!(item.failure_mode, TrainingFailureMode::ZeroCollapseTooHigh);
        assert_eq!(
            item.suggested_loss,
            Some(TargetedLossTerm::ReduceZeroCollapse),
            "zero-collapse failure should suggest ReduceZeroCollapse loss term"
        );
        assert_eq!(item.observed_value, Some(0.15));
        assert_eq!(item.required_value, Some(0.05));
        // severity = 0.15 / 0.05 - 1.0 = 2.0
        assert!(
            (item.severity - 2.0).abs() < 1e-9,
            "severity should be 2.0, got {}",
            item.severity
        );
        assert_eq!(
            report.status,
            TrainingTargetStatus::Failed,
            "one target failed → overall Failed"
        );
    }

    // ── feedback_satisfied_target ───────────────────────────────────────

    /// All gates pass → target is `Satisfied`, no failure items.
    #[test]
    fn feedback_satisfied_target() {
        let target = TargetWithGates {
            target_id: "t2".into(),
            tensor_key_match: vec!["layer2.weight".into()],
            tensor_class: "weight".into(),
            gates: GateThresholds {
                max_weight_nrmse: Some(0.10),
                max_zero_collapse_ratio: Some(0.05),
                max_operator_nrmse: Some(0.08),
                min_operator_cosine: Some(0.95),
                max_operator_abs_error: Some(0.50),
                min_byte_savings_ratio: Some(0.30),
            },
        };

        let mut evidence: HashMap<String, Vec<EvidenceEntry>> = HashMap::new();
        evidence.insert(
            "layer2.weight".into(),
            vec![make_evidence_entry(
                "layer2.weight",
                "weight",
                Some(0.05), // < 0.10 ✓
                Some(0.02), // < 0.05 ✓
                Some(0.98), // > 0.95 ✓
                Some(0.30), // < 0.50 ✓
                Some(0.45), // > 0.30 ✓
            )],
        );

        let report =
            TrainingFeedbackBuilder::build(&[target], &evidence, "spec-v1", "ckpt-abc", "ev-123");

        assert!(report.items.is_empty(), "all gates pass → no failure items");
        assert_eq!(
            report.status,
            TrainingTargetStatus::Satisfied,
            "all gates pass → overall Satisfied"
        );
        assert_eq!(report.summary.total_targets, 1);
        assert_eq!(report.summary.satisfied, 1);
        assert_eq!(report.summary.failed, 0);
    }

    // ── feedback_multiple_targets ───────────────────────────────────────

    /// Three targets: one passes, one fails weight_nrmse, one has no matching
    /// evidence. Verifies aggregate status and summary counts.
    #[test]
    fn feedback_multiple_targets() {
        let t_good = TargetWithGates {
            target_id: "t_good".into(),
            tensor_key_match: vec!["good.weight".into()],
            tensor_class: "weight".into(),
            gates: GateThresholds {
                max_weight_nrmse: Some(0.10),
                ..Default::default()
            },
        };
        let t_bad = TargetWithGates {
            target_id: "t_bad".into(),
            tensor_key_match: vec!["bad.weight".into()],
            tensor_class: "weight".into(),
            gates: GateThresholds {
                max_weight_nrmse: Some(0.05),
                ..Default::default()
            },
        };
        let t_missing = TargetWithGates {
            target_id: "t_missing".into(),
            tensor_key_match: vec!["missing.weight".into()],
            tensor_class: "weight".into(),
            gates: GateThresholds {
                max_zero_collapse_ratio: Some(0.05),
                ..Default::default()
            },
        };

        let mut evidence: HashMap<String, Vec<EvidenceEntry>> = HashMap::new();
        evidence.insert(
            "good.weight".into(),
            vec![make_evidence_entry(
                "good.weight",
                "weight",
                Some(0.03),
                None,
                None,
                None,
                None,
            )],
        );
        evidence.insert(
            "bad.weight".into(),
            vec![make_evidence_entry(
                "bad.weight",
                "weight",
                Some(0.12),
                None,
                None,
                None,
                None,
            )],
        );

        let report = TrainingFeedbackBuilder::build(
            &[t_good, t_bad, t_missing],
            &evidence,
            "spec-v1",
            "ckpt-abc",
            "ev-123",
        );

        // Only t_bad should produce a failure item.
        assert_eq!(
            report.items.len(),
            1,
            "only t_bad should have a failure item"
        );
        assert_eq!(report.items[0].target_id, "t_bad");
        assert_eq!(
            report.items[0].failure_mode,
            TrainingFailureMode::WeightNrmseTooHigh,
            "t_bad fails weight_nrmse"
        );

        // Summary counts.
        assert_eq!(report.summary.total_targets, 3);
        assert_eq!(report.summary.satisfied, 1, "t_good satisfied");
        assert_eq!(report.summary.failed, 1, "t_bad failed");
        assert_eq!(
            report.summary.warnings, 1,
            "t_missing has no evidence → warning"
        );

        // Aggregate: at least one Failed → overall Failed.
        assert_eq!(
            report.status,
            TrainingTargetStatus::Failed,
            "at least one target Failed → overall Failed"
        );

        // gate_results should have max_weight_nrmse with 1/2 = 0.5 passing.
        let w_nrmse_result = report.summary.gate_results.get("max_weight_nrmse");
        assert!(
            w_nrmse_result.is_some(),
            "gate_results should contain max_weight_nrmse"
        );
        assert!(
            (w_nrmse_result.unwrap() - 0.5).abs() < 1e-9,
            "max_weight_nrmse: 1/2 targets pass = 0.5"
        );
    }
}

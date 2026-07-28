//! `pipeline::ane::legality` — ANE backend legality rules.
//!
//! This file owns the canonical authority for the ANE legality model
//! based on the 20 empirically-discovered ANE restrictions that the
//! engine inherited from Orion. The [`AneRule`] trait is the
//! extension point; the [`AneLegality`] evaluator aggregates rule
//! verdicts into an [`AneLegalityReceipt`].

use std::time::Instant;

use prism_ecs_backend::routing::{EvidenceDigest, OperationId, TensorId};
use prism_ecs_backend::DType;

use super::super::pass::PassIdentity;
use super::super::scheduled::ScheduledRegion;

// ── Rule identity ─────────────────────────────────────────────────────────

/// Identity of a single ANE rule.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuleIdentity {
    /// Stable rule id.
    pub id: String,
    /// Rule version.
    pub version: String,
    /// Source provenance.
    pub provenance: String,
    /// Content digest of the rule implementation.
    pub implementation_digest: EvidenceDigest,
    /// Evidence qualification state.
    pub evidence_state: RuleEvidenceState,
}

/// Evidence state for a rule — prevents Orion's observed behavior
/// from becoming an unquestioned Tribunus hardware invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuleEvidenceState {
    /// Rule was imported but not verified.
    ImportedUnverified,
    /// Rule has been reproduced.
    Reproduced,
    /// Rule has been contradicted by evidence.
    Contradicted,
    /// Rule has been superseded by a newer rule.
    Superseded,
}

// ── Rule category ─────────────────────────────────────────────────────────

/// Coarse category for ANE rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleCategory {
    /// MIL graph structure.
    MilGraph,
    /// Per-op lowering.
    OperationLowering,
    /// Tensor shape and dtype.
    TensorShapeDtype,
    /// IOSurface allocation.
    IoSurfaceAllocation,
    /// Input/output ordering.
    InputOutputOrdering,
    /// Weight artifact constraints.
    WeightArtifact,
    /// Compilation-time resource limits.
    CompilationResource,
    /// Runtime numerical hazards.
    RuntimeNumericalHazard,
}

// ── Legality status ───────────────────────────────────────────────────────

/// Legality verdict for a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegalityStatus {
    /// Region is fully legal.
    Legal,
    /// Region is legal after a required rewrite.
    LegalAfterRewrite,
    /// Region is illegal.
    Illegal,
    /// Region has not been qualified.
    Unqualified,
}

// ── Rule evaluation ───────────────────────────────────────────────────────

/// Result of evaluating a single rule against a region.
#[derive(Debug, Clone)]
pub struct RuleEvaluation {
    /// Rule identity.
    pub rule: RuleIdentity,
    /// Rule category.
    pub category: RuleCategory,
    /// Whether the rule was satisfied.
    pub satisfied: bool,
    /// Human-readable description.
    pub description: String,
    /// Affected operation ids.
    pub affected_ops: Vec<OperationId>,
    /// Affected tensor ids.
    pub affected_tensors: Vec<TensorId>,
}

// ── Legality violation ────────────────────────────────────────────────────

/// A single ANE legality violation.
#[derive(Debug, Clone)]
pub struct AneLegalityViolation {
    /// Rule identity.
    pub rule: RuleIdentity,
    /// Rule category.
    pub category: RuleCategory,
    /// Operations involved.
    pub operations: Vec<OperationId>,
    /// Tensors involved.
    pub tensors: Vec<TensorId>,
    /// Human-readable message.
    pub message: String,
    /// Whether the violation is fatal.
    pub fatal: bool,
}

// ── Required rewrite ──────────────────────────────────────────────────────

/// A rewrite required to bring a region into legality.
#[derive(Debug, Clone)]
pub struct RequiredRewrite {
    /// Rewrite id.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// Operations affected.
    pub affected_operations: Vec<OperationId>,
    /// Tensors affected.
    pub affected_tensors: Vec<TensorId>,
    /// Output contract after rewrite.
    pub output_contract: OutputContract,
    /// Numerical tolerance.
    pub tolerance: f64,
    /// Compiler pass identity.
    pub pass: PassIdentity,
    /// Rule that this rewrite resolves.
    pub resolves_violation: RuleIdentity,
}

/// Output contract for a rewrite.
#[derive(Debug, Clone)]
pub struct OutputContract {
    /// Number of output elements.
    pub element_count: u64,
    /// Output byte size.
    pub byte_size: u64,
    /// Output shape.
    pub shape: Vec<u64>,
    /// Output dtype.
    pub dtype: DType,
}

// ── ANE legality receipt ──────────────────────────────────────────────────

/// Full ANE legality receipt for one region.
#[derive(Debug, Clone)]
pub struct AneLegalityReceipt {
    /// Identity of the rule set.
    pub rule_set: RuleSetIdentity,
    /// Region digest.
    pub region_digest: EvidenceDigest,
    /// Machine profile digest.
    pub machine_profile_digest: EvidenceDigest,
    /// Final status.
    pub status: LegalityStatus,
    /// Rules that passed.
    pub satisfied_rules: Vec<RuleEvaluation>,
    /// Rules that failed.
    pub violations: Vec<AneLegalityViolation>,
    /// Required rewrites.
    pub required_rewrites: Vec<RequiredRewrite>,
    /// Digest of this receipt.
    pub receipt_digest: EvidenceDigest,
    /// Evaluation duration in nanoseconds.
    pub evaluation_ns: u64,
}

/// Identity of a rule set.
#[derive(Debug, Clone)]
pub struct RuleSetIdentity {
    /// Rule set name.
    pub name: String,
    /// Rule set version.
    pub version: String,
    /// Number of rules in the set.
    pub rule_count: u32,
    /// Source provenance.
    pub provenance: String,
}

// ── ANE legality evaluator ────────────────────────────────────────────────

/// Aggregates rule verdicts and produces an [`AneLegalityReceipt`].
pub struct AneLegality {
    rules: Vec<Box<dyn AneRule>>,
    machine_profile_digest: EvidenceDigest,
    rule_set: RuleSetIdentity,
}

impl AneLegality {
    /// Create a new evaluator with the given machine profile digest.
    pub fn new(machine_profile_digest: EvidenceDigest) -> Self {
        Self {
            rules: Vec::new(),
            machine_profile_digest,
            rule_set: RuleSetIdentity {
                name: "ane-legality-v1".into(),
                version: "1.0.0".into(),
                rule_count: 0,
                provenance: "Orion pass_ane_validate.c + Apple MIL spec".into(),
            },
        }
    }

    /// Add a rule to the evaluator.
    pub fn add_rule(&mut self, rule: Box<dyn AneRule>) {
        self.rules.push(rule);
        self.rule_set.rule_count = self.rules.len() as u32;
    }

    /// Evaluate a region against the rule set.
    pub fn evaluate_region(&self, region: &ScheduledRegion) -> AneLegalityReceipt {
        let start = Instant::now();
        let mut satisfied = Vec::new();
        let mut violations = Vec::new();
        let mut required_rewrites = Vec::new();

        for rule in &self.rules {
            let eval = rule.evaluate(region);
            if !eval.satisfied {
                violations.push(AneLegalityViolation {
                    rule: eval.rule.clone(),
                    category: eval.category,
                    operations: eval.affected_ops.clone(),
                    tensors: eval.affected_tensors.clone(),
                    message: eval.description.clone(),
                    fatal: rule.is_fatal(),
                });
                if let Some(rw) = rule.suggested_rewrite(region, &eval) {
                    required_rewrites.push(rw);
                }
            }
            satisfied.push(eval);
        }

        let status = if self.rules.is_empty() {
            LegalityStatus::Unqualified
        } else if violations.iter().any(|v| v.fatal) {
            LegalityStatus::Illegal
        } else if !violations.is_empty() {
            LegalityStatus::LegalAfterRewrite
        } else {
            LegalityStatus::Legal
        };

        let evaluation_ns = start.elapsed().as_nanos() as u64;
        let region_digest = region_digest_from_region(region);
        let receipt_digest = compute_receipt_digest(&region_digest, &status, &violations);

        AneLegalityReceipt {
            rule_set: self.rule_set.clone(),
            region_digest,
            machine_profile_digest: self.machine_profile_digest.clone(),
            status,
            satisfied_rules: satisfied,
            violations,
            required_rewrites,
            receipt_digest,
            evaluation_ns,
        }
    }
}

// ── ANE rule trait ────────────────────────────────────────────────────────

/// Trait implemented by every ANE legality rule.
pub trait AneRule {
    /// Rule identity.
    fn identity(&self) -> RuleIdentity;
    /// Rule category.
    fn category(&self) -> RuleCategory;
    /// Evaluate the rule against a region.
    fn evaluate(&self, region: &ScheduledRegion) -> RuleEvaluation;
    /// Whether violations of this rule are fatal.
    fn is_fatal(&self) -> bool;
    /// Optional rewrite that resolves the violation.
    fn suggested_rewrite(
        &self,
        _region: &ScheduledRegion,
        _violation: &RuleEvaluation,
    ) -> Option<RequiredRewrite> {
        None
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn region_digest_from_region(region: &ScheduledRegion) -> EvidenceDigest {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(format!("{}", region.region_id.0).as_bytes());
    h.update(format!("{}", region.operations.len()).as_bytes());
    EvidenceDigest(format!("{:x}", h.finalize()))
}

fn compute_receipt_digest(
    region_digest: &EvidenceDigest,
    status: &LegalityStatus,
    violations: &[AneLegalityViolation],
) -> EvidenceDigest {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(region_digest.0.as_bytes());
    h.update(format!("{:?}", status).as_bytes());
    for v in violations {
        h.update(v.rule.id.as_bytes());
        h.update(v.message.as_bytes());
    }
    EvidenceDigest(format!("{:x}", h.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::scheduled::RegionId;
    use prism_ecs_backend::routing::BackendId;

    fn empty_region() -> ScheduledRegion {
        ScheduledRegion {
            region_id: RegionId(1),
            name: "test".into(),
            operations: vec![],
            selected_backend: BackendId(4),
            physical_tensors: vec![],
            inputs: vec![],
            outputs: vec![],
            dependencies: vec![],
            fusions: vec![],
            state_effects: vec![],
            temp_memory_bytes: 0,
            fusion_regions: vec![],
            is_fence: false,
        }
    }

    #[test]
    fn empty_rule_set_is_unqualified() {
        let legality = AneLegality::new(EvidenceDigest("test".into()));
        let receipt = legality.evaluate_region(&empty_region());
        assert_eq!(receipt.status, LegalityStatus::Unqualified);
        assert!(!receipt.receipt_digest.0.is_empty());
    }
}

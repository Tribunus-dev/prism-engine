//! Mixed precision planning — selects per-tensor precision from profile evidence.
//!
//! The planner takes a base codec candidate and sweep receipts, attributes
//! error per tensor group, and promotes the highest-error groups to a rescue
//! codec (INT8/FP16). The result is a `PrecisionPlan` that can be integrated
//! into a `ModelExecutionPlan`.
//!
//! ## Algorithm
//!
//! 1. Accept a base codec candidate and per-unit error receipts.
//! 2. Sort units by error contribution descending.
//! 3. Promote top-N units to a rescue codec.
//! 4. Recompute effective byte cost.
//! 5. Return a `PrecisionPlan` or a rejection if the plan regresses.

use serde::{Deserialize, Serialize};

use super::CodecFamily;

// ── MixedDispatchPolicy ────────────────────────────────────────────────────

/// How to dispatch mixed-precision operations at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MixedDispatchPolicy {
    /// Raise every promoted group to the rescue codec individually.
    PerGroupPromotion,
    /// Raise all groups in a fused block to the rescue codec.
    BlockPromotion,
    /// Raise all groups in the entire layer to the rescue codec.
    LayerPromotion,
}

// ── PrecisionOverrideEntry ─────────────────────────────────────────────────

/// One override entry: map a tensor unit to a rescue codec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionOverrideEntry {
    /// Identifier for the tensor group being overridden.
    pub unit_id: String,
    /// The rescue codec applied to this unit.
    pub rescue_codec: CodecFamily,
    /// Byte cost with the rescue codec applied.
    pub effective_bytes: u64,
    /// Residual error contribution after promotion.
    pub residual_error: f64,
}

// ── PrecisionOverrideTable ─────────────────────────────────────────────────

/// Ordered table of precision override entries, highest error first.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionOverrideTable {
    /// Entries sorted by decreasing error contribution.
    pub entries: Vec<PrecisionOverrideEntry>,
    /// Total effective byte cost after all overrides.
    pub total_effective_bytes: u64,
    /// Residual aggregate error after promotion.
    pub total_residual_error: f64,
}

// ── PrecisionSidecar ───────────────────────────────────────────────────────

/// A sidecar structure that travels with an execution plan to describe
/// mixed-precision overrides applied during planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionSidecar {
    /// The base codec used for non-promoted groups.
    pub base_codec: CodecFamily,
    /// The rescue codec promoted groups were raised to.
    pub rescue_codec: CodecFamily,
    /// Number of units promoted.
    pub promoted_count: usize,
    /// Fraction of units promoted.
    pub promoted_fraction: f64,
    /// Byte savings vs. full-rescue plan.
    pub byte_savings_vs_full_rescue: u64,
    /// Effective quality retention estimate.
    pub quality_retention: f64,
}

// ── PrecisionPlan ──────────────────────────────────────────────────────────

/// Result of a mixed-precision planning pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrecisionPlan {
    /// The plan is feasible and includes explicit overrides.
    ///
    /// Contains the override table and sidecar for downstream integration
    /// into kernel specialization and plan construction.
    Accepted {
        /// Per-unit override table.
        overrides: PrecisionOverrideTable,
        /// Sidecar metadata.
        sidecar: PrecisionSidecar,
    },
    /// The mixed-precision plan was rejected.
    ///
    /// Reasons include: no units to promote, no evidence available, or
    /// the plan would regress the baseline.
    Rejected {
        /// Human-readable reason for rejection.
        reason: String,
    },
}

// ── MixedPrecisionLayout ───────────────────────────────────────────────────

/// Describes how a tensor's layout is chosen under mixed precision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionLayout {
    /// The base tile family.
    pub tile_family: String,
    /// Base codec for non-promoted groups.
    pub base_codec: CodecFamily,
    /// Rescue codec for promoted groups.
    pub rescue_codec: CodecFamily,
    /// Dispatch policy.
    pub dispatch_policy: MixedDispatchPolicy,
    /// Per-group override table (empty if no promotions).
    pub overrides: PrecisionOverrideTable,
}

// ── MixedPrecisionReceipt ──────────────────────────────────────────────────

/// Receipt emitted by a mixed-precision planning pass.
///
/// Captures the planning outcome for audit trails, drift detection, and
/// downstream compilation decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionReceipt {
    /// Identifier for the planning pass (e.g. `"mp-pass-001"`).
    pub pass_id: String,
    /// The base codec considered.
    pub base_codec: CodecFamily,
    /// The rescue codec selected.
    pub rescue_codec: CodecFamily,
    /// Number of tensor units evaluated.
    pub total_units: usize,
    /// Number of units promoted to the rescue codec.
    pub promoted_units: usize,
    /// Byte cost of the base-only plan.
    pub base_byte_cost: u64,
    /// Byte cost after mixed-precision overrides.
    pub mixed_byte_cost: u64,
    /// Byte cost if all units used the rescue codec.
    pub full_rescue_byte_cost: u64,
    /// Savings fraction vs. full rescue.
    pub savings_fraction: f64,
    /// Aggregate error contribution before promotion.
    pub aggregate_pre_error: f64,
    /// Aggregate residual error after promotion.
    pub aggregate_post_error: f64,
    /// Whether the plan was accepted or rejected.
    pub accepted: bool,
    /// Human-readable reason if rejected.
    pub rejection_reason: Option<String>,
}

// ── SweepReceiptError ──────────────────────────────────────────────────────

/// Per-unit error attribution from a sweep receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepReceiptError {
    /// Identifier for the tensor unit.
    pub unit_id: String,
    /// Error contribution (e.g. weight NRMSE, operator NRMSE).
    pub error_contribution: f64,
    /// Byte cost of this unit in the base codec.
    pub byte_cost: u64,
    /// Byte cost if promoted to the rescue codec.
    pub rescue_byte_cost: u64,
}

// ── MixedPrecisionPlanner ──────────────────────────────────────────────────

/// Plans mixed-precision tensor assignments from profile sweep evidence.
///
/// # Algorithm
///
/// 1. Accept a base codec and a list of per-unit error attributions.
/// 2. Sort units by `error_contribution` descending.
/// 3. Promote the top N units to a rescue codec (INT8 or FP16).
/// 4. Compute effective byte cost after promotion.
/// 5. Return `PrecisionPlan::Accepted` with the override table, or
///    `PrecisionPlan::Rejected` if no promotion is beneficial.
pub struct MixedPrecisionPlanner;

impl MixedPrecisionPlanner {
    /// Run the mixed-precision planning algorithm.
    ///
    /// # Parameters
    ///
    /// * `base_codec` — The base codec used everywhere (e.g. NF4).
    /// * `rescue_codec` — The higher-precision codec to promote error-heavy
    ///   units to (e.g. INT8 or FP16).
    /// * `errors` — Per-unit error attributions from sweep receipts.
    /// * `promotion_budget` — Maximum fraction of units to promote
    ///   (e.g. `0.2` for 20%).
    /// * `pass_id` — Identifier for this planning pass.
    ///
    /// # Returns
    ///
    /// A `PrecisionPlan` describing either an accepted override table or a
    /// rejection with the reason.
    pub fn plan(
        base_codec: CodecFamily,
        rescue_codec: CodecFamily,
        errors: &[SweepReceiptError],
        promotion_budget: f64,
        _pass_id: &str,
    ) -> PrecisionPlan {
        if errors.is_empty() {
            return PrecisionPlan::Rejected {
                reason: "No sweep receipt errors provided".into(),
            };
        }

        // 1. Sort by error contribution descending.
        let mut sorted: Vec<&SweepReceiptError> = errors.iter().collect();
        sorted.sort_by(|a, b| {
            b.error_contribution
                .partial_cmp(&a.error_contribution)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 2. Determine how many units to promote.
        let promote_count = ((sorted.len() as f64) * promotion_budget).ceil() as usize;
        let promote_count = promote_count.min(sorted.len()).max(1);

        // 3. Compute aggregate pre-promotion error.
        let _base_byte_cost: u64 = errors.iter().map(|e| e.byte_cost).sum();
        let full_rescue_byte_cost: u64 = errors.iter().map(|e| e.rescue_byte_cost).sum();
        let aggregate_pre_error: f64 = errors.iter().map(|e| e.error_contribution).sum();

        // 4. Promote top-N, compute post-promotion cost.
        let mut total_post_cost: u64 = 0;
        let mut total_residual_error: f64 = 0.0;
        let mut overrides = Vec::with_capacity(promote_count);

        for (i, unit) in sorted.iter().enumerate() {
            let promoted = i < promote_count;
            let cost = if promoted {
                unit.rescue_byte_cost
            } else {
                unit.byte_cost
            };
            let residual = if promoted {
                0.0 // error eliminated by promotion
            } else {
                unit.error_contribution
            };
            total_post_cost += cost;
            total_residual_error += residual;

            if promoted {
                overrides.push(PrecisionOverrideEntry {
                    unit_id: unit.unit_id.clone(),
                    rescue_codec,
                    effective_bytes: unit.rescue_byte_cost,
                    residual_error: residual,
                });
            }
        }

        // 5. Check that the plan isn't regressive: if post-cost >= full rescue
        //    cost and pre_error was negligible, reject.
        if total_post_cost >= full_rescue_byte_cost && aggregate_pre_error < 1e-9 {
            return PrecisionPlan::Rejected {
                reason: "Mixed precision plan has no benefit over full rescue".into(),
            };
        }

        let promoted_fraction = promote_count as f64 / sorted.len() as f64;

        PrecisionPlan::Accepted {
            overrides: PrecisionOverrideTable {
                entries: overrides,
                total_effective_bytes: total_post_cost,
                total_residual_error,
            },
            sidecar: PrecisionSidecar {
                base_codec,
                rescue_codec,
                promoted_count: promote_count,
                promoted_fraction,
                byte_savings_vs_full_rescue: full_rescue_byte_cost.saturating_sub(total_post_cost),
                quality_retention: 1.0 - (total_residual_error / aggregate_pre_error.max(1e-12)),
            },
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a SweepReceiptError with given error contribution.
    fn unit(id: &str, error: f64, cost: u64, rescue_cost: u64) -> SweepReceiptError {
        SweepReceiptError {
            unit_id: id.to_string(),
            error_contribution: error,
            byte_cost: cost,
            rescue_byte_cost: rescue_cost,
        }
    }

    #[test]
    fn planner_nf4_base_int8_rescue() {
        // NF4 base, INT8 rescue. 10 units, promote 20% → 2 units.
        let errors = vec![
            unit("layer0_qkv", 0.050, 100, 200),
            unit("layer0_attn_out", 0.030, 90, 180),
            unit("layer0_mlp_gate", 0.010, 120, 240),
            unit("layer0_mlp_up", 0.008, 110, 220),
            unit("layer0_mlp_down", 0.007, 130, 260),
            unit("layer1_qkv", 0.045, 100, 200),
            unit("layer1_attn_out", 0.025, 90, 180),
            unit("layer1_mlp_gate", 0.009, 120, 240),
            unit("layer1_mlp_up", 0.006, 110, 220),
            unit("layer1_mlp_down", 0.005, 130, 260),
        ];

        let result = MixedPrecisionPlanner::plan(
            CodecFamily::Nf4,
            CodecFamily::Int8,
            &errors,
            0.2, // 20% promotion budget
            "test-pass-001",
        );

        match &result {
            PrecisionPlan::Accepted { overrides, sidecar } => {
                // 20% of 10 → 2 units promoted
                assert_eq!(sidecar.promoted_count, 2);
                assert!((sidecar.promoted_fraction - 0.2).abs() < 1e-9);
                assert_eq!(sidecar.base_codec, CodecFamily::Nf4);
                assert_eq!(sidecar.rescue_codec, CodecFamily::Int8);

                // Top 2 errors: layer0_qkv (0.050) and layer1_qkv (0.045)
                assert_eq!(overrides.entries.len(), 2);
                assert_eq!(overrides.entries[0].unit_id, "layer0_qkv");
                assert_eq!(overrides.entries[1].unit_id, "layer1_qkv");

                // Total base cost: sum of all byte_cost
                let _expected_base: u64 = errors.iter().map(|e| e.byte_cost).sum();
                assert!(sidecar.byte_savings_vs_full_rescue > 0);

                // Aggregate post error: sum of all non-promoted errors
                let expected_post: f64 = errors
                    .iter()
                    .map(|e| e.error_contribution)
                    .sum::<f64>()
                    - (0.050 + 0.045);
                assert!((overrides.total_residual_error - expected_post).abs() < 1e-12);

                // Ensure accepted
                assert!(sidecar.quality_retention > 0.0);
            }
            other => panic!("Expected Accepted plan, got {:?}", other),
        }
    }

    #[test]
    fn planner_rejects_empty_errors() {
        let result = MixedPrecisionPlanner::plan(
            CodecFamily::Nf4,
            CodecFamily::Int8,
            &[],
            0.2,
            "empty-test",
        );
        match result {
            PrecisionPlan::Rejected { reason } => {
                assert!(reason.contains("No sweep receipt errors"));
            }
            other => panic!("Expected Rejected, got {:?}", other),
        }
    }

    #[test]
    fn planner_rejects_regressive() {
        // Near-zero errors → no benefit from promotion.
        let errors = vec![
            unit("layer0", 1e-15, 100, 200),
            unit("layer1", 1e-15, 100, 200),
        ];
        let result = MixedPrecisionPlanner::plan(
            CodecFamily::Nf4,
            CodecFamily::Int8,
            &errors,
            0.5,
            "regressive-test",
        );
        match result {
            PrecisionPlan::Rejected { reason } => {
                assert!(reason.contains("no benefit"));
            }
            other => panic!("Expected Rejected, got {:?}", other),
        }
    }
}

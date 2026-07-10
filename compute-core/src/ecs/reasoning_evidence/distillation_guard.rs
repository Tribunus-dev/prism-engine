//! Distillation guardrails — signal decomposition receipts, threshold
//! configuration, and promotion eligibility checks.

use serde::{Deserialize, Serialize};

use super::epistemic::EpistemicBehaviorReceipt;

/// Receipt capturing the distillation signal decomposition for a teacher →
/// student transfer. Describes which signal pathways were available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationSignalDecompositionReceipt {
    pub teacher_id: String,
    pub student_id: String,
    pub dataset_id: String,
    pub has_reference_conditioned_teacher: bool,
    pub has_reference_only_teacher: bool,
    pub uses_inference_transferable_residual: bool,
    pub uses_pmi_target_distribution: bool,
    pub promotion_eligible: bool,
}

impl DistillationSignalDecompositionReceipt {
    /// Returns true when the decomposition matches the Purified OPSD pattern:
    /// reference-conditioned teacher + PMI target distribution → residual PMI training.
    pub fn is_purified_opsd(&self) -> bool {
        self.has_reference_conditioned_teacher && self.uses_pmi_target_distribution
    }
}

/// Per-dimension epistemic thresholds used by the guardrail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicThresholds {
    pub min_uncertainty_marker_rate: f64,
    pub min_self_correction_rate: f64,
    pub min_reasoning_length: f64,
    pub max_in_domain_gain_for_ood_loss: f64,
}

impl Default for EpistemicThresholds {
    fn default() -> Self {
        Self {
            min_uncertainty_marker_rate: 0.05,
            min_self_correction_rate: 0.02,
            min_reasoning_length: 10.0,
            max_in_domain_gain_for_ood_loss: 0.03,
        }
    }
}

/// Guardrail configuration that evaluates promotion eligibility from epistemic
/// behavior receipts and optional distillation decomposition receipts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationGuardrail {
    pub guard_id: String,
    pub epistemic_thresholds: EpistemicThresholds,
    pub ood_threshold: f64,
    pub trace_collapse_threshold: f64,
}

impl Default for DistillationGuardrail {
    fn default() -> Self {
        Self {
            guard_id: "default_v1".into(),
            epistemic_thresholds: EpistemicThresholds::default(),
            ood_threshold: 0.85,
            trace_collapse_threshold: 5.0,
        }
    }
}

impl DistillationGuardrail {
    /// Evaluate promotion eligibility against a set of promotion rules.
    ///
    /// # Promotion rules
    /// 1. If OOD accuracy drops beyond `ood_threshold` → not eligible.
    /// 2. If epistemic marker rate collapses below minimum thresholds → not eligible.
    /// 3. If average reasoning trace length collapses without matching OOD improvement → not eligible.
    /// 4. If in-domain gains are paired with OOD degradation → not eligible.
    /// 5. If Purified OPSD metadata shows reference-conditioned teacher and PMI target → eligible (if all thresholds pass).
    pub fn check_promotion(
        &self,
        epistemic: &EpistemicBehaviorReceipt,
        distillation: &Option<DistillationSignalDecompositionReceipt>,
    ) -> PromotionCheck {
        let mut reasons: Vec<String> = Vec::new();
        let mut eligible = true;

        // Rule 1: OOD accuracy threshold
        if let Some(ood_acc) = epistemic.ood_accuracy {
            if ood_acc < self.ood_threshold {
                eligible = false;
                reasons.push(format!(
                    "OOD accuracy {:.3} below threshold {:.3}",
                    ood_acc, self.ood_threshold
                ));
            }
        }

        // Rule 2: Epistemic marker rate collapse
        if epistemic.uncertainty_marker_rate < self.epistemic_thresholds.min_uncertainty_marker_rate
        {
            eligible = false;
            reasons.push(format!(
                "Uncertainty marker rate {:.3} below minimum {:.3}",
                epistemic.uncertainty_marker_rate,
                self.epistemic_thresholds.min_uncertainty_marker_rate
            ));
        }
        if epistemic.self_correction_marker_rate
            < self.epistemic_thresholds.min_self_correction_rate
        {
            eligible = false;
            reasons.push(format!(
                "Self-correction marker rate {:.3} below minimum {:.3}",
                epistemic.self_correction_marker_rate,
                self.epistemic_thresholds.min_self_correction_rate
            ));
        }

        // Rule 3: Reasoning trace collapse
        if epistemic.average_reasoning_length < self.trace_collapse_threshold {
            // Only flag as collapse if OOD is not improving (allow short traces with strong OOD)
            if epistemic
                .ood_accuracy
                .map_or(true, |ood| ood < self.ood_threshold)
            {
                eligible = false;
                reasons.push(format!(
                    "Reasoning trace length {:.1} collapsed below threshold {:.1} without OOD improvement",
                    epistemic.average_reasoning_length, self.trace_collapse_threshold
                ));
            }
        }

        // Rule 4: In-domain gain paired with OOD loss
        if let (Some(ood), Some(ind)) = (epistemic.ood_accuracy, epistemic.in_domain_accuracy) {
            // This rule applies when we have some baseline reference; it flags
            // in-domain improvement that coincides with OOD degradation.
            // Here we check the reverse: if in_domain_accuracy > ood_accuracy by a large margin
            // and ood is below threshold, flag it.
            if ind > ood && ood < self.ood_threshold {
                let gain = ind - ood;
                if gain > self.epistemic_thresholds.max_in_domain_gain_for_ood_loss {
                    eligible = false;
                    reasons.push(format!(
                        "In-domain gain {:.3} paired with OOD loss ({:.3}) exceeds max allowed {:.3}",
                        gain,
                        ood,
                        self.epistemic_thresholds.max_in_domain_gain_for_ood_loss
                    ));
                }
            }
        }

        // Rule 5: Purified OPSD — makes eligible if thresholds pass
        if let Some(d) = distillation {
            if d.is_purified_opsd() && eligible {
                reasons.push(
                    "Purified OPSD: reference-conditioned teacher + PMI target -> residual PMI training"
                        .into(),
                );
            }
        }

        // If the epistemic receipt already carries degradation flags, record them.
        for flag in &epistemic.degradation_flags {
            reasons.push(format!("Degradation flag: {flag:?}"));
        }

        PromotionCheck {
            promotion_eligible: eligible,
            reasons,
        }
    }
}

/// Result of a promotion eligibility check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionCheck {
    pub promotion_eligible: bool,
    pub reasons: Vec<String>,
}

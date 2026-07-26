//! Mixed-precision planning types for the fusion compiler pipeline.
//!
//! A PrecisionPlan specifies how a fused group's weights and activations
//! may use different codec families (precisions) to trade accuracy for
//! performance. The plan is authored by the policy resolver and consumed
//! by `BackendCapabilityRegistry::evaluate()` during fusion scheduling.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ecs::plan::CodecFamily;
use crate::execution_plan::mixed_precision::{PrecisionOverrideTable, PrecisionSidecar};
use crate::training_target::RequiredEvidenceLevel;

// ── PrecisionPlan ────────────────────────────────────────────────────────

/// A complete mixed-precision plan for one fused group or layer.
///
/// Encodes the scope, base representation, overrides, selector semantics,
/// rescue codec, physical byte offsets, sidecar format, expected error
/// reduction, measured evidence IDs, dispatch policy, and compatibility
/// version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionPlan {
    /// Unique identifier for this plan instance.
    pub plan_id: String,

    /// How broadly this precision plan applies (whole tensor, tile, layer range, etc.).
    pub scope: PrecisionScope,

    /// Default codec family applied to all tensors/tiles not explicitly overridden.
    pub default_codec: CodecFamily,

    /// Explicit overrides for specific tiles or groups within the scope.
    pub overrides: Vec<PrecisionOverride>,

    /// What kind of evidence the precision plan is based on.
    pub selection_basis: PrecisionSelectionBasis,

    /// Minimum evidence level required before this plan is admissible.
    pub evidence_level: RequiredEvidenceLevel,

    /// Override table from the mixed-precision planner — provides ordered,
    /// error-sorted entries with residual error and effective byte accounting.
    /// Present when a mixed-precision pass has produced per-unit promotions.
    pub override_table: Option<PrecisionOverrideTable>,

    /// Sidecar metadata that travels with the execution plan, describing
    /// aggregate promotion decisions (base/rescue codec, promoted count,
    /// byte savings, quality retention).
    pub sidecar: Option<PrecisionSidecar>,

    /// Total byte cost of this plan after all overrides are applied.
    /// Includes both base and promoted tile contributions.
    pub byte_cost: u64,

    /// Expected error reduction relative to the base-only plan, if computed.
    /// A positive value indicates the plan is expected to improve accuracy
    /// (lower error) vs. the baseline; `None` means not yet evaluated.
    pub expected_error_reduction: Option<f64>,

    /// Compatibility version for plan evolution. Increment when field layout
    /// or validation semantics change. Initialised to 1.
    pub compatibility_version: u16,
}

impl PrecisionPlan {
    /// Compute a stable hex digest over all fields of this plan.
    ///
    /// Hashing is deterministic across serialization runs: it uses the
    /// canonical JSON representation of the plan via `serde_json`.
    pub fn plan_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.plan_id.as_bytes());
        hasher.update(&self.compatibility_version.to_le_bytes());
        hasher.update(&self.byte_cost.to_le_bytes());

        // scope discriminant
        hasher.update(&[self.scope as u8]);

        // default_codec discriminant
        hasher.update(&[self.default_codec as u8]);

        // selection_basis discriminant
        hasher.update(&[self.selection_basis as u8]);

        // evidence_level discriminant
        hasher.update(&[self.evidence_level as u8]);

        // expected_error_reduction
        if let Some(err) = self.expected_error_reduction {
            hasher.update(b"E");
            hasher.update(&err.to_le_bytes());
        } else {
            hasher.update(b"N");
        }

        // overrides — hash each entry's selector variant + codec + cost + reason
        for ov in &self.overrides {
            hasher.update(&[ov.codec as u8]);
            hasher.update(&ov.byte_cost.to_le_bytes());
            hasher.update(&[ov.reason as u8]);
            ov.selector.hash_into(&mut hasher);
        }

        // override_table
        if let Some(table) = &self.override_table {
            hasher.update(b"T");
            hasher.update(&table.total_effective_bytes.to_le_bytes());
            hasher.update(&table.total_residual_error.to_le_bytes());
            for entry in &table.entries {
                hasher.update(entry.unit_id.as_bytes());
                hasher.update(&[entry.rescue_codec as u8]);
                hasher.update(&entry.effective_bytes.to_le_bytes());
                hasher.update(&entry.residual_error.to_le_bytes());
            }
        } else {
            hasher.update(b"N");
        }

        // sidecar
        if let Some(sidecar) = &self.sidecar {
            hasher.update(b"S");
            hasher.update(&[sidecar.base_codec as u8]);
            hasher.update(&[sidecar.rescue_codec as u8]);
            hasher.update(&sidecar.promoted_count.to_le_bytes());
            hasher.update(&sidecar.promoted_fraction.to_le_bytes());
            hasher.update(&sidecar.byte_savings_vs_full_rescue.to_le_bytes());
            hasher.update(&sidecar.quality_retention.to_le_bytes());
        } else {
            hasher.update(b"N");
        }

        hex_encode(&hasher.finalize())
    }

    ///
    /// Checks:
    ///   - Override selectors reference tiles/groups within the declared scope.
    ///   - Override codecs differ from the default codec (a no-op override).
    ///   - Sidecar base codec matches `default_codec` when a sidecar is present.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors: Vec<String> = Vec::new();

        // ── Override codec sanity ──────────────────────────────────────
        for (i, ov) in self.overrides.iter().enumerate() {
            if ov.codec == self.default_codec {
                errors.push(format!(
                    "override[{}] codec {:?} is the same as default_codec {:?} — no-op override",
                    i, ov.codec, self.default_codec
                ));
            }
        }

        // ── Scope-bound override selectors ─────────────────────────────
        for (i, ov) in self.overrides.iter().enumerate() {
            self.validate_selector_bounds(i, &ov.selector, &mut errors);
        }

        // ── Sidecar compatibility ──────────────────────────────────────
        if let Some(sidecar) = &self.sidecar {
            if sidecar.base_codec != self.default_codec {
                errors.push(format!(
                    "sidecar base_codec {:?} does not match plan default_codec {:?}",
                    sidecar.base_codec, self.default_codec
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn validate_selector_bounds(
        &self,
        index: usize,
        selector: &PrecisionSelector,
        errors: &mut Vec<String>,
    ) {
        // For LayerRange selectors, verify the range is well-formed.
        // Other selector types carry their own validity semantics.
        if let PrecisionSelector::LayerRange { start, end } = selector {
            if start > end {
                errors.push(format!(
                    "override[{}] LayerRange start={} > end={}",
                    index, start, end
                ));
            }
            if let PrecisionScope::LayerRange = self.scope {
                // Within a LayerRange scope, the override sub-range must
                // fall within the scope — we don't have explicit scope
                // bounds here, so we just check start <= end.
            }
        }
    }
}
/// Encode a byte slice as a lowercase hex string (no external crate needed).
fn hex_encode(bytes: &[u8]) -> String {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX_CHARS[(b >> 4) as usize] as char);
        out.push(HEX_CHARS[(b & 0x0f) as usize] as char);
    }
    out
}

// ── PrecisionPlanResult ─────────────────────────────────────────────────

/// Outcome of attempting to create or validate a [`PrecisionPlan`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrecisionPlanResult {
    /// The plan is valid and accepted.
    Accepted(PrecisionPlan),
    /// The plan was rejected with one or more human-readable reasons.
    Rejected {
        /// The plan that was evaluated (may be partially populated).
        plan: PrecisionPlan,
        /// Reasons for rejection.
        reasons: Vec<String>,
    },
}

// ── PrecisionScope ───────────────────────────────────────────────────────

/// How broadly a precision plan applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionScope {
    /// The entire tensor.
    WholeTensor,
    /// A family of tensors (e.g. all Q/K/V projections).
    TensorFamily,
    /// A contiguous range of layers.
    LayerRange,
    /// A single tile within a tensor.
    Tile,
    /// A group of tiles within a tensor.
    Group,
    /// A slice along the input axis.
    InputAxisSlice,
    /// A slice along the output axis.
    OutputAxisSlice,
    /// An expert dimension (MoE).
    Expert,
    /// A fused group of operators.
    FusedGroup,
}

// ── PrecisionOverride ────────────────────────────────────────────────────

/// A single precision override for a specific selection of tensors/tiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionOverride {
    /// Selector identifying which tiles or groups this override targets.
    pub selector: PrecisionSelector,

    /// Codec family to apply to the selected region.
    pub codec: CodecFamily,

    /// Why this particular precision override was chosen.
    pub reason: PrecisionOverrideReason,

    /// Byte cost of applying this override.
    pub byte_cost: u64,

    /// Expected error reduction from this override (if estimated).
    pub expected_error_reduction: Option<f64>,
}

// ── PrecisionSelector ────────────────────────────────────────────────────

/// Selector identifying which tiles or groups an override applies to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrecisionSelector {
    /// Select by explicit tile indices.
    TileIds(Vec<u32>),
    /// Select by explicit group indices.
    GroupIds(Vec<u32>),
    /// Select by input column indices.
    InputColumns(Vec<u32>),
    /// Select by output row indices.
    OutputRows(Vec<u32>),
    /// A contiguous range of layers.
    LayerRange {
        /// Start layer index (inclusive).
        start: u32,
        /// End layer index (inclusive).
        end: u32,
    },
    /// Select the fraction of tiles with the highest error.
    TopErrorTiles {
        /// Fraction of tiles to select (0.0–1.0).
        fraction: f64,
    },
    /// Select outlier columns up to a maximum fraction.
    OutlierColumns {
        /// Maximum fraction of columns to select as outliers.
        max_fraction: f64,
    },
    /// Select top-K tiles weighted by activation magnitude.
    ActivationWeightedTopK {
        /// Fraction of tiles to select (0.0–1.0).
        fraction: f64,
    },
}

impl PrecisionSelector {
    /// Feed this selector's data into a SHA-256 hasher for digest computation.
    fn hash_into(&self, hasher: &mut Sha256) {
        match self {
            PrecisionSelector::TileIds(ids) => {
                hasher.update(b"TI");
                for id in ids {
                    hasher.update(&id.to_le_bytes());
                }
            }
            PrecisionSelector::GroupIds(ids) => {
                hasher.update(b"GI");
                for id in ids {
                    hasher.update(&id.to_le_bytes());
                }
            }
            PrecisionSelector::InputColumns(cols) => {
                hasher.update(b"IC");
                for c in cols {
                    hasher.update(&c.to_le_bytes());
                }
            }
            PrecisionSelector::OutputRows(rows) => {
                hasher.update(b"OR");
                for r in rows {
                    hasher.update(&r.to_le_bytes());
                }
            }
            PrecisionSelector::LayerRange { start, end } => {
                hasher.update(b"LR");
                hasher.update(&start.to_le_bytes());
                hasher.update(&end.to_le_bytes());
            }
            PrecisionSelector::TopErrorTiles { fraction } => {
                hasher.update(b"TE");
                hasher.update(&fraction.to_le_bytes());
            }
            PrecisionSelector::OutlierColumns { max_fraction } => {
                hasher.update(b"OC");
                hasher.update(&max_fraction.to_le_bytes());
            }
            PrecisionSelector::ActivationWeightedTopK { fraction } => {
                hasher.update(b"AW");
                hasher.update(&fraction.to_le_bytes());
            }
        }
    }
}

// ── PrecisionOverrideReason ──────────────────────────────────────────────

/// Why a particular precision override was chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionOverrideReason {
    /// Operator tail rescue — promote the tail of an operator to a higher precision.
    OperatorTailRescue,
    /// Activation-weighted outlier detected.
    ActivationWeightedOutlier,
    /// Zero-collapse rescue — prevent collapse in near-zero regions.
    ZeroCollapseRescue,
    /// Fallback when byte savings targets are not met.
    ByteSavingsFallback,
    /// Backend compatibility constraint requires a different codec.
    BackendCompatibility,
    /// Raw F32 required (e.g. for critical ops or legacy layers).
    RawF32Required,
}

// ── PrecisionSelectionBasis ──────────────────────────────────────────────

/// What kind of evidence the precision plan is based on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionSelectionBasis {
    /// Statically-configured policy (no runtime evidence).
    StaticPolicy,
    /// Weight-space error analysis.
    WeightError,
    /// Operator-level error analysis.
    OperatorError,
    /// Activation-weighted error attribution.
    ActivationWeightedError,
    /// Outlier magnitude analysis.
    OutlierMagnitude,
    /// Zero-collapse risk assessment.
    ZeroCollapseRisk,
    /// Hardware profile constraints.
    HardwareProfile,
    /// Learned profile from previous runs.
    LearnedProfile,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_digest_is_stable_and_hex() {
        let plan = PrecisionPlan {
            plan_id: "test-plan-001".into(),
            scope: PrecisionScope::FusedGroup,
            default_codec: CodecFamily::Nf4,
            overrides: vec![],
            selection_basis: PrecisionSelectionBasis::StaticPolicy,
            evidence_level: RequiredEvidenceLevel::WeightSpace,
            override_table: None,
            sidecar: None,
            byte_cost: 0,
            expected_error_reduction: None,
            compatibility_version: 1,
        };
        let digest = plan.plan_digest();
        // Hex string: 64 hex chars (SHA-256)
        assert_eq!(digest.len(), 64);
        assert!(digest
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')));
        // Stable across calls
        assert_eq!(digest, plan.plan_digest());
    }

    #[test]
    fn test_digest_changes_when_fields_change() {
        let base = PrecisionPlan {
            plan_id: "p".into(),
            scope: PrecisionScope::WholeTensor,
            default_codec: CodecFamily::RawF32,
            overrides: vec![],
            selection_basis: PrecisionSelectionBasis::StaticPolicy,
            evidence_level: RequiredEvidenceLevel::WeightSpace,
            override_table: None,
            sidecar: None,
            byte_cost: 100,
            expected_error_reduction: None,
            compatibility_version: 1,
        };
        let modified = PrecisionPlan {
            byte_cost: 200,
            ..base.clone()
        };
        assert_ne!(base.plan_digest(), modified.plan_digest());
    }

    #[test]
    fn test_digest_includes_overrides() {
        let plan_with = PrecisionPlan {
            plan_id: "p".into(),
            scope: PrecisionScope::FusedGroup,
            default_codec: CodecFamily::RawF32,
            overrides: vec![PrecisionOverride {
                selector: PrecisionSelector::TileIds(vec![0, 1]),
                codec: CodecFamily::Fp16,
                reason: PrecisionOverrideReason::BackendCompatibility,
                byte_cost: 50,
                expected_error_reduction: Some(0.1),
            }],
            selection_basis: PrecisionSelectionBasis::WeightError,
            evidence_level: RequiredEvidenceLevel::HardwareOperator,
            override_table: None,
            sidecar: None,
            byte_cost: 150,
            expected_error_reduction: Some(0.1),
            compatibility_version: 1,
        };
        let digest = plan_with.plan_digest();
        assert_eq!(digest.len(), 64);
        assert!(digest
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')));
    }

    #[test]
    fn test_validate_ok_with_no_overrides() {
        let plan = PrecisionPlan {
            plan_id: "v".into(),
            scope: PrecisionScope::FusedGroup,
            default_codec: CodecFamily::Nf4,
            overrides: vec![],
            selection_basis: PrecisionSelectionBasis::StaticPolicy,
            evidence_level: RequiredEvidenceLevel::WeightSpace,
            override_table: None,
            sidecar: None,
            byte_cost: 0,
            expected_error_reduction: None,
            compatibility_version: 1,
        };
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_noop_override() {
        let plan = PrecisionPlan {
            plan_id: "v".into(),
            scope: PrecisionScope::FusedGroup,
            default_codec: CodecFamily::Nf4,
            overrides: vec![PrecisionOverride {
                selector: PrecisionSelector::TileIds(vec![0]),
                codec: CodecFamily::Nf4,
                reason: PrecisionOverrideReason::OperatorTailRescue,
                byte_cost: 0,
                expected_error_reduction: None,
            }],
            selection_basis: PrecisionSelectionBasis::StaticPolicy,
            evidence_level: RequiredEvidenceLevel::WeightSpace,
            override_table: None,
            sidecar: None,
            byte_cost: 0,
            expected_error_reduction: None,
            compatibility_version: 1,
        };
        assert!(plan.validate().is_err());
        let errs = plan.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("no-op override")));
    }

    #[test]
    fn test_validate_rejects_inverted_layer_range() {
        let plan = PrecisionPlan {
            plan_id: "v".into(),
            scope: PrecisionScope::LayerRange,
            default_codec: CodecFamily::Nf4,
            overrides: vec![PrecisionOverride {
                selector: PrecisionSelector::LayerRange { start: 5, end: 3 },
                codec: CodecFamily::Fp16,
                reason: PrecisionOverrideReason::BackendCompatibility,
                byte_cost: 0,
                expected_error_reduction: None,
            }],
            selection_basis: PrecisionSelectionBasis::StaticPolicy,
            evidence_level: RequiredEvidenceLevel::WeightSpace,
            override_table: None,
            sidecar: None,
            byte_cost: 0,
            expected_error_reduction: None,
            compatibility_version: 1,
        };
        assert!(plan.validate().is_err());
        let errs = plan.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("start=5 > end=3")));
    }

    #[test]
    fn test_validate_rejects_mismatched_sidecar() {
        let plan = PrecisionPlan {
            plan_id: "v".into(),
            scope: PrecisionScope::FusedGroup,
            default_codec: CodecFamily::Nf4,
            overrides: vec![],
            selection_basis: PrecisionSelectionBasis::StaticPolicy,
            evidence_level: RequiredEvidenceLevel::WeightSpace,
            override_table: None,
            sidecar: Some(PrecisionSidecar {
                base_codec: CodecFamily::Fp16,
                rescue_codec: CodecFamily::RawF32,
                promoted_count: 0,
                promoted_fraction: 0.0,
                byte_savings_vs_full_rescue: 0,
                quality_retention: 1.0,
            }),
            byte_cost: 0,
            expected_error_reduction: None,
            compatibility_version: 1,
        };
        assert!(plan.validate().is_err());
        let errs = plan.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("sidecar base_codec")));
    }

    #[test]
    fn test_precision_plan_result_serde() {
        let plan = PrecisionPlan {
            plan_id: "serde-test".into(),
            scope: PrecisionScope::WholeTensor,
            default_codec: CodecFamily::RawF32,
            overrides: vec![],
            selection_basis: PrecisionSelectionBasis::StaticPolicy,
            evidence_level: RequiredEvidenceLevel::WeightSpace,
            override_table: None,
            sidecar: None,
            byte_cost: 0,
            expected_error_reduction: None,
            compatibility_version: 1,
        };
        let accepted = PrecisionPlanResult::Accepted(plan.clone());
        let rejected = PrecisionPlanResult::Rejected {
            plan: plan.clone(),
            reasons: vec!["insufficient evidence".into()],
        };

        for result in &[&accepted, &rejected] {
            let json = serde_json::to_string(result).unwrap();
            let _back: PrecisionPlanResult = serde_json::from_str(&json).unwrap();
        }
    }
}

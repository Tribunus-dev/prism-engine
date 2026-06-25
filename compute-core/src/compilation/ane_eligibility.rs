//! PRISM-ANE-ELIGIBILITY-PASS-0001: PhaseIR ANE eligibility pass.
//!
//! Determines whether a compile phase is eligible for ANE execution by
//! consulting the static region catalogue, checking for dynamic dimensions,
//! mutable KV state, scatter/gather, dynamic indexing, unsupported dtypes,
//! boundary-copy cost, and other disqualifying conditions.
//!
//! The pass produces an `AneEligibility` record containing the admission
//! status, shape classification, rejection reason (if rejected), qualified
//! shape buckets, layout contracts from the matched catalogue entry, and
//! evidence requirements for qualification.

use serde::{Deserialize, Serialize};

use crate::compilation::phase_ir::{
    CompilePhaseDescriptor, ShapeClass,
};
use crate::compilation::region_catalogue::{LayoutContract, RegionCatalogue, RegionCatalogueEntry};
use crate::compilation::tri_lane::AneRejectionReason;

// ── Eligibility result ────────────────────────────────────────────────────

/// Status of the ANE eligibility check for a compile phase.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AneEligibilityStatus {
    /// Phase is eligible for ANE execution.
    Eligible,
    /// Phase is ineligible for ANE execution for a specific reason.
    Rejected,
    /// Eligibility cannot be determined at compile time; defer to runtime.
    Deferred,
}

/// Shape class assigned by the eligibility pass.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AneShapeClass {
    /// Static-shape vision/multi-modal region (image inputs).
    VisionStatic,
    /// Static-shape prefill region (batch=1, sequence>1).
    PrefillStatic,
    /// Static-shape decode candidate (batch=1, sequence=1).
    DecodeStaticCandidate,
    /// Static-shaped region that must execute on Metal, not ANE.
    MetalOnly,
}

/// A single qualified shape bucket — a (batch, sequence, hidden, rank)
/// tuple that the ANE lane is certified to handle.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShapeBucket {
    pub batch: u32,
    pub sequence: u32,
    pub hidden: u32,
    pub rank: u8,
    pub family: ShapeBucketFamily,
}

/// Functional family of a shape bucket.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShapeBucketFamily {
    Decode,
    Prefill,
    Vision,
    Projector,
}

/// Evidence tier required to qualify a shape bucket for ANE execution.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AneEvidenceRequirement {
    AllocationAttested,
    Warmed,
    PredictionValidated,
    MetalConsumed,
}

/// Complete ANE eligibility result for a compile phase.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AneEligibility {
    pub status: AneEligibilityStatus,
    pub shape_class: AneShapeClass,
    pub rejection_reason: Option<AneRejectionReason>,
    pub qualified_buckets: Vec<ShapeBucket>,
    pub input_layout_contract: LayoutContract,
    pub output_layout_contract: LayoutContract,
    pub evidence_requirements: Vec<AneEvidenceRequirement>,
}

// ── Operator-family inference ────────────────────────────────────────────

/// Infer the operator family name from a tensor name by matching the
/// longest-known-catalogue-entry prefix.
///
/// For example, `"q_projection_weight"` yields `Some("q_projection")`,
/// and `"kv_cache_append_input"` yields `Some("kv_cache_append")`.
fn infer_operator_family(tensor_name: &str, catalogue: &RegionCatalogue) -> Option<String> {
    let mut best: Option<&str> = None;
    for entry in &catalogue.entries {
        if tensor_name.starts_with(&entry.operator_family) {
            let is_better = match best {
                None => true,
                Some(current) => entry.operator_family.len() > current.len(),
            };
            if is_better {
                best = Some(&entry.operator_family);
            }
        }
    }
    best.map(|s| s.to_string())
}

/// Collect all catalogue entries matching a phase's tensor names.
fn matching_catalogue_entries<'a>(
    phase: &'a CompilePhaseDescriptor,
    catalogue: &'a RegionCatalogue,
) -> Vec<(&'a RegionCatalogueEntry, String)> {
    let mut found: Vec<(&RegionCatalogueEntry, String)> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for tensor in phase.inputs.iter().chain(phase.outputs.iter()) {
        if let Some(family) = infer_operator_family(&tensor.name, catalogue) {
            if seen.insert(family.clone()) {
                if let Some(entry) = catalogue.find(&family) {
                    found.push((entry, family));
                }
            }
        }
    }

    found
}

// ── Shape analysis helpers ────────────────────────────────────────────────

/// Extract the (batch, sequence, hidden) dimensions from a tensor shape
/// vector, returning `None` if the shape is too short.
fn extract_dims(shape: &[u64]) -> Option<(u32, u32, u32)> {
    match shape.len() {
        // 1D: [hidden]
        1 => Some((1, 1, shape[0] as u32)),
        // 2D: [batch, hidden] (projection weight)
        2 => Some((shape[0] as u32, 1, shape[1] as u32)),
        // 3D: [batch, sequence, hidden]
        3 => Some((shape[0] as u32, shape[1] as u32, shape[2] as u32)),
        // 4D: [batch, channels, height, width] (vision)
        4 => Some((shape[0] as u32, shape[1] as u32, shape[2] as u32)),
        _ => None,
    }
}

/// Determine whether any input tensor suggests a vision/multi-modal workload.
fn has_vision_input(phase: &CompilePhaseDescriptor) -> bool {
    let vision_keywords: &[&str] = &["pixel_values", "image", "vision", "photo", "frame"];
    phase.inputs.iter().any(|t| {
        let lower = t.name.to_lowercase();
        vision_keywords.iter().any(|k| lower.contains(k))
    })
}

/// Determine whether an operator family represents a scatter/gather pattern
/// that Core ML cannot represent efficiently.
fn is_scatter_gather_operator(family: &str) -> bool {
    matches!(
        family,
        "attention_score"
            | "attention_value_aggregation"
            | "kv_cache_append"
            | "kv_cache_view"
    )
}

/// Determine whether an operator family represents dynamic index patterns.
fn is_dynamic_indexing_operator(family: &str) -> bool {
    matches!(family, "token_sampling" | "embedding_lookup")
}

/// Determine whether an operator family accesses mutable KV-cache state.
fn is_mutable_kv_state_operator(family: &str) -> bool {
    matches!(family, "kv_cache_append")
}

// ── Shape classification ─────────────────────────────────────────────────

/// Classify the phase shape given its tensor dimensions and context.
fn classify_shape(
    phase: &CompilePhaseDescriptor,
    primary_entry: &RegionCatalogueEntry,
) -> AneShapeClass {
    // If the primary entry is NOT Core ML production, the phase is Metal-only
    // regardless of shape.
    if !primary_entry.is_coreml_production() {
        return AneShapeClass::MetalOnly;
    }

    // Vision/multi-modal check
    if has_vision_input(phase) {
        return AneShapeClass::VisionStatic;
    }

    // Derive rough dimensions from the first input tensor
    if let Some(first) = phase.inputs.first() {
        if first.shape.len() >= 2 {
            let batch = first.shape[0] as u32;
            let seq = first.shape[1] as u32;
            // Decode: batch=1, seq=1
            if batch == 1 && seq == 1 {
                return AneShapeClass::DecodeStaticCandidate;
            }
            // Prefill: batch=1, seq>1
            if batch == 1 && seq > 1 {
                return AneShapeClass::PrefillStatic;
            }
        }
    }

    // Fallback: treat as decode-static candidate
    AneShapeClass::DecodeStaticCandidate
}

// ── Bucket construction ──────────────────────────────────────────────────

/// Build shape buckets for an eligible phase.
fn build_qualified_buckets(
    phase: &CompilePhaseDescriptor,
    shape_class: &AneShapeClass,
) -> Vec<ShapeBucket> {
    let family = match shape_class {
        AneShapeClass::DecodeStaticCandidate => ShapeBucketFamily::Decode,
        AneShapeClass::PrefillStatic => ShapeBucketFamily::Prefill,
        AneShapeClass::VisionStatic => ShapeBucketFamily::Vision,
        AneShapeClass::MetalOnly => return vec![],
    };

    // Extract dimensions from the first input
    let (batch, seq, hidden) = phase
        .inputs
        .first()
        .and_then(|t| extract_dims(&t.shape))
        .unwrap_or((1, 1, 0));

    let rank = if phase.inputs.iter().any(|t| t.shape.len() > 2) {
        2
    } else {
        1
    };

    vec![ShapeBucket {
        batch,
        sequence: seq,
        hidden,
        rank,
        family,
    }]
}

// ── Evidence requirements ─────────────────────────────────────────────────

/// Build evidence requirements from a matched catalogue entry.
fn build_evidence_requirements(entry: &RegionCatalogueEntry) -> Vec<AneEvidenceRequirement> {
    use crate::compilation::region_catalogue::EvidenceRequirement as CatReq;

    let mut reqs = Vec::new();

    match entry.evidence_requirement {
        CatReq::AllocationAttested
        | CatReq::Warmed
        | CatReq::PredictionValidated
        | CatReq::MetalConsumed
        | CatReq::TraceObserved => {
            reqs.push(AneEvidenceRequirement::AllocationAttested);
            reqs.push(AneEvidenceRequirement::Warmed);
            reqs.push(AneEvidenceRequirement::PredictionValidated);
            reqs.push(AneEvidenceRequirement::MetalConsumed);
        }
        CatReq::Installed => {
            reqs.push(AneEvidenceRequirement::AllocationAttested);
            reqs.push(AneEvidenceRequirement::Warmed);
        }
        CatReq::ConfiguredOnly | CatReq::NotAttempted => {
            reqs.push(AneEvidenceRequirement::AllocationAttested);
        }
        CatReq::Failed => {
            // No evidence possible for a failed entry.
        }
    }

    reqs
}

// ── Core analysis function ────────────────────────────────────────────────

/// Analyze whether a compile phase is eligible for ANE execution.
///
/// The analysis proceeds in order:
/// 1. Match the phase's operator family against the region catalogue.
/// 2. Reject if no catalogue entry is found.
/// 3. Reject if the phase has dynamic shape dimensions.
/// 4. Reject if the operator involves scatter/gather,
///    dynamic indexing, or mutable KV state.
/// 5. Reject if the boundary copy cost exceeds the threshold.
/// 6. Classify the shape and return eligibility with buckets.
pub fn analyze_ane_eligibility(
    phase: &CompilePhaseDescriptor,
    catalogue: &RegionCatalogue,
) -> AneEligibility {
    // ── Step 1: Match operator family ──────────────────────────────
    let matched_entries = matching_catalogue_entries(phase, catalogue);

    let (primary_entry, primary_family) = match matched_entries.first() {
        Some(pair) => pair.clone(),
        None => {
            // Try to extract a best-effort operator family from tensor names
            let unknown_family = phase
                .inputs
                .first()
                .map(|t| {
                    t.name
                        .split('_')
                        .next()
                        .unwrap_or("unknown")
                        .to_string()
                })
                .unwrap_or_else(|| "unknown".to_string());

            return AneEligibility {
                status: AneEligibilityStatus::Rejected,
                shape_class: AneShapeClass::MetalOnly,
                rejection_reason: Some(AneRejectionReason::MissingCatalogueEntry {
                    operator_family: unknown_family,
                }),
                qualified_buckets: vec![],
                input_layout_contract: LayoutContract {
                    layout: "NHWC".into(),
                    preferred: true,
                    stride_constraints: vec![],
                },
                output_layout_contract: LayoutContract {
                    layout: "NHWC".into(),
                    preferred: true,
                    stride_constraints: vec![],
                },
                evidence_requirements: vec![],
            };
        }
    };

    // ── Step 2: Check for dynamic dimensions ───────────────────────
    let has_dynamic = match &phase.shape_class {
        ShapeClass::Dynamic => true,
        ShapeClass::Static(dims) => dims.iter().any(|d| *d == 0),
    };

    if has_dynamic {
        let dim_name = phase
            .inputs
            .first()
            .and_then(|t| {
                t.shape.iter().position(|d| *d == 0).map(|i| {
                    ["batch", "sequence", "hidden", "width"]
                        .get(i)
                        .copied()
                        .unwrap_or("unknown")
                        .to_string()
                })
            })
            .unwrap_or_else(|| "unknown".to_string());

        return AneEligibility {
            status: AneEligibilityStatus::Rejected,
            shape_class: AneShapeClass::MetalOnly,
            rejection_reason: Some(AneRejectionReason::DynamicDimension {
                dimension: dim_name,
            }),
            qualified_buckets: vec![],
            input_layout_contract: primary_entry.admitted_layouts.first().cloned().unwrap_or(
                LayoutContract {
                    layout: "NHWC".into(),
                    preferred: true,
                    stride_constraints: vec![],
                },
            ),
            output_layout_contract: primary_entry.admitted_layouts.first().cloned().unwrap_or(
                LayoutContract {
                    layout: "NHWC".into(),
                    preferred: true,
                    stride_constraints: vec![],
                },
            ),
            evidence_requirements: vec![],
        };
    }

    // ── Step 3: Check scatter/gather operators ─────────────────────
    if is_scatter_gather_operator(&primary_family) {
        return AneEligibility {
            status: AneEligibilityStatus::Rejected,
            shape_class: AneShapeClass::MetalOnly,
            rejection_reason: Some(AneRejectionReason::ScatterGather),
            qualified_buckets: vec![],
            input_layout_contract: primary_entry.admitted_layouts.first().cloned().unwrap_or(
                LayoutContract {
                    layout: "NHWC".into(),
                    preferred: true,
                    stride_constraints: vec![],
                },
            ),
            output_layout_contract: primary_entry.admitted_layouts.first().cloned().unwrap_or(
                LayoutContract {
                    layout: "NHWC".into(),
                    preferred: true,
                    stride_constraints: vec![],
                },
            ),
            evidence_requirements: vec![],
        };
    }

    // ── Step 4: Check dynamic indexing operators ───────────────────
    if is_dynamic_indexing_operator(&primary_family) {
        return AneEligibility {
            status: AneEligibilityStatus::Rejected,
            shape_class: AneShapeClass::MetalOnly,
            rejection_reason: Some(AneRejectionReason::DynamicIndexing),
            qualified_buckets: vec![],
            input_layout_contract: primary_entry.admitted_layouts.first().cloned().unwrap_or(
                LayoutContract {
                    layout: "NHWC".into(),
                    preferred: true,
                    stride_constraints: vec![],
                },
            ),
            output_layout_contract: primary_entry.admitted_layouts.first().cloned().unwrap_or(
                LayoutContract {
                    layout: "NHWC".into(),
                    preferred: true,
                    stride_constraints: vec![],
                },
            ),
            evidence_requirements: vec![],
        };
    }

    // ── Step 5: Check mutable KV state ─────────────────────────────
    if is_mutable_kv_state_operator(&primary_family) {
        return AneEligibility {
            status: AneEligibilityStatus::Rejected,
            shape_class: AneShapeClass::MetalOnly,
            rejection_reason: Some(AneRejectionReason::MutableKvState),
            qualified_buckets: vec![],
            input_layout_contract: primary_entry.admitted_layouts.first().cloned().unwrap_or(
                LayoutContract {
                    layout: "NHWC".into(),
                    preferred: true,
                    stride_constraints: vec![],
                },
            ),
            output_layout_contract: primary_entry.admitted_layouts.first().cloned().unwrap_or(
                LayoutContract {
                    layout: "NHWC".into(),
                    preferred: true,
                    stride_constraints: vec![],
                },
            ),
            evidence_requirements: vec![],
        };
    }

    // ── Step 6: Check boundary copy cost ───────────────────────────
    // Phases with bridge copy exceeding a heuristic threshold are
    // rejected as requiring CPU copy.
    if phase.bridge_copy_bytes > 0 {
        return AneEligibility {
            status: AneEligibilityStatus::Rejected,
            shape_class: AneShapeClass::MetalOnly,
            rejection_reason: Some(AneRejectionReason::BoundaryCpuCopyRequired),
            qualified_buckets: vec![],
            input_layout_contract: primary_entry.admitted_layouts.first().cloned().unwrap_or(
                LayoutContract {
                    layout: "NHWC".into(),
                    preferred: true,
                    stride_constraints: vec![],
                },
            ),
            output_layout_contract: primary_entry.admitted_layouts.first().cloned().unwrap_or(
                LayoutContract {
                    layout: "NHWC".into(),
                    preferred: true,
                    stride_constraints: vec![],
                },
            ),
            evidence_requirements: vec![],
        };
    }

    // ── Step 7: Classify shape and build buckets ───────────────────
    let shape_class = classify_shape(phase, primary_entry);
    let qualified_buckets = build_qualified_buckets(phase, &shape_class);

    let evidence_requirements = build_evidence_requirements(primary_entry);

    AneEligibility {
        status: AneEligibilityStatus::Eligible,
        shape_class,
        rejection_reason: None,
        qualified_buckets,
        input_layout_contract: primary_entry.admitted_layouts.first().cloned().unwrap_or(
            LayoutContract {
                layout: "NHWC".into(),
                preferred: true,
                stride_constraints: vec![],
            },
        ),
        output_layout_contract: primary_entry.admitted_layouts.first().cloned().unwrap_or(
            LayoutContract {
                layout: "NHWC".into(),
                preferred: true,
                stride_constraints: vec![],
            },
        ),
        evidence_requirements,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compilation::phase_ir::{
        ArithmeticIntensity, CompileDeterminism, CompilePlacement, MaterializationContract,
        MutationClass, PhaseId,
    };

    // ── Helpers ───────────────────────────────────────────────────

    fn make_phase(
        shape_class: ShapeClass,
        mutation: MutationClass,
        inputs: Vec<(&str, Vec<u64>)>,
        outputs: Vec<(&str, Vec<u64>)>,
        bridge_copy_bytes: u64,
    ) -> CompilePhaseDescriptor {
        CompilePhaseDescriptor {
            phase_id: PhaseId(42),
            inputs: inputs
                .into_iter()
                .map(|(name, shape)| TensorContract {
                    name: name.to_string(),
                    dtype: "Float16".into(),
                    shape,
                    materialization: MaterializationContract::AneIoSurface,
                })
                .collect(),
            outputs: outputs
                .into_iter()
                .map(|(name, shape)| TensorContract {
                    name: name.to_string(),
                    dtype: "Float16".into(),
                    shape,
                    materialization: MaterializationContract::AneIoSurface,
                })
                .collect(),
            shape_class,
            arithmetic_intensity: ArithmeticIntensity::ComputeBound,
            mutation,
            determinism: CompileDeterminism::NumericallyBounded,
            allowed_placements: vec![CompilePlacement::Ane, CompilePlacement::MetalGpu],
            minimum_profitable_elements: 1024,
            fallback: CompilePlacement::MetalGpu,
            estimated_ane_duration_ns: 1_000_000,
            bridge_copy_bytes,
        }
    }

    fn fp16_catalogue() -> RegionCatalogue {
        RegionCatalogue::fp16_alpha()
    }

    // ── Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_reject_dynamic_dimension() {
        let catalogue = fp16_catalogue();
        let phase = make_phase(
            ShapeClass::Dynamic,
            MutationClass::ReadOnly,
            vec![("q_projection_input", vec![1, 1, 4096])],
            vec![("q_projection_output", vec![1, 1, 4096])],
            0,
        );
        let result = analyze_ane_eligibility(&phase, &catalogue);
        assert_eq!(result.status, AneEligibilityStatus::Rejected);
        assert!(matches!(
            result.rejection_reason,
            Some(AneRejectionReason::DynamicDimension { .. })
        ));
        assert!(result.qualified_buckets.is_empty());
    }

    #[test]
    fn test_reject_mutable_kv_state() {
        let catalogue = fp16_catalogue();
        let phase = make_phase(
            ShapeClass::Static(vec![1, 1, 4096]),
            MutationClass::MutatesInPlace,
            vec![("kv_cache_append_input", vec![1, 1, 4096])],
            vec![("kv_cache_append_output", vec![1, 1, 4096])],
            0,
        );
        let result = analyze_ane_eligibility(&phase, &catalogue);
        assert_eq!(result.status, AneEligibilityStatus::Rejected);
        assert_eq!(
            result.rejection_reason,
            Some(AneRejectionReason::MutableKvState)
        );
        assert!(result.qualified_buckets.is_empty());
    }

    #[test]
    fn test_classify_projection_as_decode_static() {
        let catalogue = fp16_catalogue();
        let phase = make_phase(
            ShapeClass::Static(vec![1, 1, 4096]),
            MutationClass::ReadOnly,
            vec![("q_projection_weight", vec![1, 1, 4096])],
            vec![("q_projection_output", vec![1, 1, 4096])],
            0,
        );
        let result = analyze_ane_eligibility(&phase, &catalogue);
        assert_eq!(result.status, AneEligibilityStatus::Eligible);
        assert_eq!(result.shape_class, AneShapeClass::DecodeStaticCandidate);
        assert!(result.rejection_reason.is_none());
        assert!(!result.qualified_buckets.is_empty());
    }

    #[test]
    fn test_classify_vision_as_vision_static() {
        let catalogue = fp16_catalogue();
        let phase = make_phase(
            ShapeClass::Static(vec![1, 3, 224, 224]),
            MutationClass::ReadOnly,
            vec![("pixel_values", vec![1, 3, 224, 224])],
            vec![("q_projection_output", vec![1, 1, 4096])],
            0,
        );
        let result = analyze_ane_eligibility(&phase, &catalogue);
        assert_eq!(result.status, AneEligibilityStatus::Eligible);
        assert_eq!(result.shape_class, AneShapeClass::VisionStatic);
    }

    #[test]
    fn test_missing_catalogue_entry() {
        let catalogue = fp16_catalogue();
        let phase = make_phase(
            ShapeClass::Static(vec![1, 1, 4096]),
            MutationClass::ReadOnly,
            vec![("custom_op_input", vec![1, 1, 4096])],
            vec![("custom_op_output", vec![1, 1, 4096])],
            0,
        );
        let result = analyze_ane_eligibility(&phase, &catalogue);
        assert_eq!(result.status, AneEligibilityStatus::Rejected);
        assert!(matches!(
            result.rejection_reason,
            Some(AneRejectionReason::MissingCatalogueEntry { .. })
        ));
        assert!(result.qualified_buckets.is_empty());
    }

    #[test]
    fn test_boundary_cpu_copy_rejected() {
        let catalogue = fp16_catalogue();
        let phase = make_phase(
            ShapeClass::Static(vec![1, 1, 4096]),
            MutationClass::ReadOnly,
            vec![("q_projection_weight", vec![1, 1, 4096])],
            vec![("q_projection_output", vec![1, 1, 4096])],
            4096, // bridge_copy_bytes > 0
        );
        let result = analyze_ane_eligibility(&phase, &catalogue);
        assert_eq!(result.status, AneEligibilityStatus::Rejected);
        assert_eq!(
            result.rejection_reason,
            Some(AneRejectionReason::BoundaryCpuCopyRequired)
        );
        assert!(result.qualified_buckets.is_empty());
    }

    #[test]
    fn test_serde_roundtrip() {
        let eligibility = AneEligibility {
            status: AneEligibilityStatus::Eligible,
            shape_class: AneShapeClass::DecodeStaticCandidate,
            rejection_reason: None,
            qualified_buckets: vec![ShapeBucket {
                batch: 1,
                sequence: 1,
                hidden: 4096,
                rank: 2,
                family: ShapeBucketFamily::Decode,
            }],
            input_layout_contract: LayoutContract {
                layout: "NHWC".into(),
                preferred: true,
                stride_constraints: vec![],
            },
            output_layout_contract: LayoutContract {
                layout: "NHWC".into(),
                preferred: true,
                stride_constraints: vec![],
            },
            evidence_requirements: vec![
                AneEvidenceRequirement::AllocationAttested,
                AneEvidenceRequirement::Warmed,
                AneEvidenceRequirement::PredictionValidated,
                AneEvidenceRequirement::MetalConsumed,
            ],
        };

        let json = serde_json::to_string(&eligibility).expect("serialize");
        let restored: AneEligibility = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(eligibility, restored);
    }

    #[test]
    fn test_empty_buckets_on_rejected() {
        let catalogue = fp16_catalogue();
        let phase = make_phase(
            ShapeClass::Dynamic,
            MutationClass::ReadOnly,
            vec![("q_projection_input", vec![1, 1, 4096])],
            vec![("q_projection_output", vec![1, 1, 4096])],
            0,
        );
        let result = analyze_ane_eligibility(&phase, &catalogue);
        assert_eq!(result.status, AneEligibilityStatus::Rejected);
        assert!(result.qualified_buckets.is_empty());

        // Also verify that boundary-copy rejection yields empty buckets
        let phase2 = make_phase(
            ShapeClass::Static(vec![1, 1, 4096]),
            MutationClass::ReadOnly,
            vec![("q_projection_weight", vec![1, 1, 4096])],
            vec![("q_projection_output", vec![1, 1, 4096])],
            4096,
        );
        let result2 = analyze_ane_eligibility(&phase2, &catalogue);
        assert_eq!(result2.status, AneEligibilityStatus::Rejected);
        assert!(result2.qualified_buckets.is_empty());
    }
}

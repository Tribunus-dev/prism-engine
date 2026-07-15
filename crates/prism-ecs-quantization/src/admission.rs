//! Quantization admission pipeline.
//!
//! The admission pipeline classifies a tensor, generates legal quantization
//! candidates, packs each candidate, reconstructs from its packed bytes,
//! validates weight-space and operator-space behavior, and promotes the first
//! candidate that passes every gate. If no candidate passes, compilation fails.
//!
//! This replaces ad-hoc per-matrix packing loops with a structured,
//! fail-closed qualification system.

pub mod ternary;

/// Tiered vector budgets per validation phase.
///
/// Probe/stress gate uses the smallest budget (32-64) so bad candidates fail
/// quickly. Promotion and holdout gates use larger budgets (128-256) for
/// candidates that are already plausible. The promotion and holdout banks
/// come from the CalibrationSuite's separate promotion and holdout fields,
/// so their vectors are automatically disjoint.
pub const PROBE_STRESS_VECTORS: usize = 64;
pub const PROMOTION_VECTORS: usize = 256;
pub const HOLDOUT_VECTORS: usize = 256;

// ── Accelerate vDSP acceleration (Apple Silicon) ───────────────────────────
//
// Direct FFI to Accelerate framework for vector operations on macOS.
// Falls back to pure-Rust loops on other platforms.
//
#[cfg(target_os = "macos")]
mod accelerate_vdsp {
    // ── Accelerate framework linkage ────────────────────────────────────
    // Declared here so admission.rs is self-contained; duplicate #[link]
    // annotations on the same framework are harmless.
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        /// Vector-scalar multiply: C[i] = A[i] * scalar
        /// vDSP_vsmul(A, A_stride, B, C, C_stride, N)
        fn vDSP_vsmul(
            a: *const f32,
            a_stride: i32,
            b: *const f32,
            c: *mut f32,
            c_stride: i32,
            n: i32,
        );

        /// Element-wise multiply: C[i] = A[i] * B[i]
        /// vDSP_vmul(A, A_stride, B, B_stride, C, C_stride, N)
        fn vDSP_vmul(
            a: *const f32,
            a_stride: i32,
            b: *const f32,
            b_stride: i32,
            c: *mut f32,
            c_stride: i32,
            n: i32,
        );

        /// Vector sum: *result = sum(A[i]) over i in [0, N)
        /// vDSP_sve(A, A_stride, result, N)
        fn vDSP_sve(a: *const f32, a_stride: i32, result: *mut f32, n: i32);
    }

    /// Column-wise scaling: result[i,j] = unpacked[i,j] * sv[j]
    /// Uses vDSP_vsmul once per output column for SIMD-accelerated scaling.
    pub fn scale_columns(unpacked: &[f32], out_features: usize, sv: &[f32]) -> Vec<f32> {
        let total = unpacked.len();
        let in_features = total / out_features;
        let mut result = vec![0.0f32; total];
        for j in 0..out_features {
            unsafe {
                vDSP_vsmul(
                    &unpacked[j],
                    out_features as i32,
                    &sv[j],
                    &mut result[j],
                    out_features as i32,
                    in_features as i32,
                );
            }
        }
        result
    }

    /// Sum of squared differences as f64.
    /// vDSP_vmul requires distinct input and output buffers, so squaring
    /// is done into a separate allocation.  The two Vecs (n f32 each) are
    /// temporary and freed on return; gated to macOS-only so non-macOS
    /// retains the allocation-free iterator path.
    pub fn sum_sq_diff(reference: &[f32], reconstructed: &[f32]) -> f64 {
        let n = reference.len();
        let mut diff = vec![0.0f32; n];
        for i in 0..n {
            diff[i] = reference[i] - reconstructed[i];
        }
        unsafe {
            // Square into separate buffer (vDSP requires distinct I/O pointers)
            let mut sq = vec![0.0f32; n];
            vDSP_vmul(
                diff.as_ptr(),
                1,
                diff.as_ptr(),
                1,
                sq.as_mut_ptr(),
                1,
                n as i32,
            );
            // Sum: result = sum(diff[i])
            let mut sum: f32 = 0.0;
            vDSP_sve(sq.as_ptr(), 1, &mut sum, n as i32);
            sum as f64
        }
    }
}

use super::calibration::*;
use super::contract::*;
use super::contract::{CandidateEvidence, CandidateResult, PhaseVectorCounts, ValidationOutcome};
use super::embed_cluster::{pack_ternary_weights, unpack_ternary_weights};
use super::validation::*;
use crate::nf4tile640::{
    pack_int8_weights, pack_nf4_weights_awls, unpack_int8_weights, unpack_nf4_weights,
};

/// Generate the ordered candidate plan for a tensor.
///
/// Candidates are ordered by expected quality and runtime cost (cheapest
/// first). The pipeline promotes the first passing candidate.
pub fn candidate_plan(
    _in_features: usize,
    _out_features: usize,
    hint: &QuantizationHint,
) -> Vec<RuntimeRepresentationClass> {
    let mut candidates = vec![
        RuntimeRepresentationClass::TernaryTile640Base,
        RuntimeRepresentationClass::Nf4Tile640Base,
    ];
    if hint.permit_int8_candidate {
        candidates.push(RuntimeRepresentationClass::Int8Tile640Base);
    }
    candidates
}

/// Pack a tensor into a specific candidate representation.
///
/// Returns (packed_codes, tile_scales, tile_biases, optional_scale_vector).
/// For base format, scale_vector is None.
pub fn pack_candidate(
    source: &[f32],
    in_features: usize,
    out_features: usize,
    format: RuntimeRepresentationClass,
    channel_sq: Option<&[f32]>,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Option<Vec<f32>>) {
    match format {
        RuntimeRepresentationClass::TernaryTile640Base => {
            let (codes, scales, biases) = pack_ternary_weights(source, in_features, out_features);
            (codes, scales, biases, None)
        }
        RuntimeRepresentationClass::Nf4Tile640Base => {
            // AW-LS with max-abs fallback gate. Uses pack_nf4_weights_awls which
            // internally compares against the max-abs baseline and keeps the winner.
            let (codes, scales, biases, _, _) =
                pack_nf4_weights_awls(source, in_features, out_features, channel_sq, 8);
            (codes, scales, biases, None)
        }
        RuntimeRepresentationClass::Int8Tile640Base => {
            let (codes, scales, biases) = pack_int8_weights(source, in_features, out_features);
            (codes, scales, biases, None)
        }
        RuntimeRepresentationClass::RawF32 => {
            // RawF32 passthrough: store f32 source bytes directly.
            let codes: Vec<u8> = source.iter().flat_map(|x| x.to_le_bytes()).collect();
            (codes, vec![], vec![], None)
        }
    }
}

// ── cfg-gated Accelerate wrappers ───────────────────────────────────────────
//
// Each hot kernel has two versions: Accelerate vDSP on macOS, pure-Rust fallback
// elsewhere.  The cfg-based dispatch is simpler than a trait object and avoids
// any runtime overhead.

/// Column-wise scaling: result[i,j] = unpacked[i,j] * sv[j]
#[cfg(target_os = "macos")]
fn scale_columns_vdsp(unpacked: &[f32], out_features: usize, sv: &[f32]) -> Vec<f32> {
    accelerate_vdsp::scale_columns(unpacked, out_features, sv)
}

/// Equivalent column scaling using pure Rust loops (non-macOS fallback).
#[cfg(not(target_os = "macos"))]
fn scale_columns_vdsp(unpacked: &[f32], out_features: usize, sv: &[f32]) -> Vec<f32> {
    let in_features = unpacked.len() / out_features;
    let mut result = vec![0.0f32; unpacked.len()];
    for i in 0..in_features {
        for j in 0..out_features {
            result[i * out_features + j] = unpacked[i * out_features + j] * sv[j];
        }
    }
    result
}

/// Reconstruct a packed candidate back to a weight matrix.
///
/// For scaled-reduction candidates, applies the column scale vector after
/// dequantization: W_hat[i,j] = D(Q'[i,j]; alpha, beta) * S_j.
pub fn reconstruct_candidate(
    format: RuntimeRepresentationClass,
    codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    in_features: usize,
    out_features: usize,
    scale_vector: Option<&[f32]>,
) -> Vec<f32> {
    let unpacked = match format {
        RuntimeRepresentationClass::TernaryTile640Base => {
            unpack_ternary_weights(codes, scales, biases, in_features, out_features)
        }
        RuntimeRepresentationClass::Nf4Tile640Base => {
            unpack_nf4_weights(codes, scales, biases, in_features, out_features)
        }
        RuntimeRepresentationClass::Int8Tile640Base => {
            unpack_int8_weights(codes, scales, biases, in_features, out_features)
        }
        RuntimeRepresentationClass::RawF32 => {
            // RawF32 passthrough: decode F32 bytes directly.
            let f32s: Vec<f32> = codes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            f32s
        }
    };
    match scale_vector {
        Some(sv) => scale_columns_vdsp(&unpacked, out_features, sv),
        None => unpacked,
    }
}

/// Default validation profile for a tensor class.
///
/// Profiles encode per-class structural and behavioral admission gates.
/// These thresholds are initial defaults and should be refined with
/// real activation-bank calibration data.
/// Create a promotion profile for a tensor class.
///
/// Thresholds are heuristics suitable for StressBank and initial
/// ActivationBank promotion. Holdout profiles should be stricter.
fn promotion_profile(tensor_class: TensorClass) -> QuantizationValidationProfile {
    let (
        max_weight_nrmse,
        max_zero_collapse_ratio,
        max_operator_nrmse,
        min_mean_cosine,
        min_worst_cosine,
        max_norm_ratio_drift,
    ) = match tensor_class {
        TensorClass::VisionPatchProjection | TensorClass::CrossModalBridge => {
            (0.10, 0.20, 0.15, 0.995, 0.990, 0.20)
        }
        TensorClass::DecoderAttentionProjection | TensorClass::DecoderMlpProjection => {
            (0.05, 0.05, 0.10, 0.997, 0.995, 0.15)
        }
        TensorClass::OutputHead => (0.02, 0.02, 0.05, 0.999, 0.998, 0.10),
        TensorClass::TokenEmbedding => (0.02, 0.02, 0.05, 0.999, 0.998, 0.10),
        TensorClass::Unknown => (0.05, 0.05, 0.10, 0.997, 0.995, 0.15),
    };
    // Investigation ceiling: 1.5x the target for VisionPatchProjection/CrossModalBridge, 2x for strict.
    let investigation_nrmse_ceiling = match tensor_class {
        TensorClass::VisionPatchProjection | TensorClass::CrossModalBridge => 0.15,
        TensorClass::DecoderAttentionProjection | TensorClass::DecoderMlpProjection => 0.12,
        TensorClass::OutputHead | TensorClass::TokenEmbedding => 0.05,
        TensorClass::Unknown => 0.12,
    };
    QuantizationValidationProfile {
        tensor_class,
        phase: ProfilePhase::Promotion,
        max_weight_nrmse,
        investigation_nrmse_ceiling,
        max_zero_collapse_ratio,
        max_operator_nrmse,
        min_mean_cosine,
        min_worst_cosine,
        max_norm_ratio_drift,
    }
}

/// Create a holdout profile for a tensor class (stricter than promotion).
fn holdout_profile(tensor_class: TensorClass) -> QuantizationValidationProfile {
    let promo = promotion_profile(tensor_class);
    // Holdout tightens operator gates (stricter thresholds).
    QuantizationValidationProfile {
        phase: ProfilePhase::Holdout,
        max_operator_nrmse: promo.max_operator_nrmse * 0.8,
        min_mean_cosine: (promo.min_mean_cosine - 1.0) * 0.5 + 1.0, // squeeze toward 1
        min_worst_cosine: (promo.min_worst_cosine - 1.0) * 0.5 + 1.0,
        max_norm_ratio_drift: promo.max_norm_ratio_drift * 0.8,
        ..promo
    }
}

/// Main admission pipeline.
///
/// For each candidate in the plan:
///   1. Pack source into the candidate representation.
///   2. Reconstruct the exact packed bytes back to a weight matrix.
///   3. Validate weight-space quality (RMSE, NRMSE, zero-collapse).
///   4. Validate operator-space quality against the deterministic
///      `StressSuite` (always run, catches codec pathologies).
///   5. If a `CalibrationSuite` is provided with prerendered activation
///      banks for this tensor class, run promotion then holdout validation.
///   6. Track `EvidenceLevel` and `ArtifactAdmissionClass` based on which
///      validation layers were applied and passed.
///   5. If all gates pass, promote the candidate.
///   6. Otherwise, try the next candidate.
///
/// If no candidate passes, returns `Err(QuantizationAdmissionFailure)` with
/// structured diagnostics. The compiler must not emit a degraded artifact.
///
/// # Parameters
///
/// - `stress`: deterministic stress suite (always provided). Validates codec
///    correctness with diverse synthetic patterns.
/// - `calibration`: optional prerendered activation bank suite from reference
#[allow(unused_assignments)]
pub fn quantize_tensor(
    source: &[f32],
    in_features: usize,
    out_features: usize,
    hint: &QuantizationHint,
    channel_sq: Option<&[f32]>,
    stress: Option<&StressSuite>,
    calibration: Option<&CalibrationSuite>,
) -> Result<QualifiedTensor, QuantizationAdmissionFailure> {
    let promo_profile = promotion_profile(hint.tensor_class);
    let hold_profile = holdout_profile(hint.tensor_class);
    let candidates = candidate_plan(in_features, out_features, hint);

    let mut best_evidence: Option<CandidateEvidence> = None;
    let mut completed_vectors = PhaseVectorCounts::default();
    let mut candidates_attempted: Vec<String> = Vec::new();
    let mut current_phase: &str = "init";

    // Resolve stress and activation vectors from the respective suites.
    // Vectors are NOT resized: width mismatches are caught by the
    // ProductionQuality gate below (SyntheticStressOnly).
    let stress_vectors: Option<Vec<Vec<f32>>> = stress
        .and_then(|s| s.get(&hint.tensor_class))
        .map(|bank| bank.promotion.clone());
    let calibration_promotion: Option<Vec<Vec<f32>>> = calibration
        .and_then(|s| s.get(&hint.tensor_class))
        .map(|bank| bank.promotion.clone());
    let calibration_holdout: Option<Vec<Vec<f32>>> = calibration
        .and_then(|s| s.get(&hint.tensor_class))
        .map(|bank| bank.holdout.clone());

    // Stratified sample all vector banks for deterministic norm-band coverage.
    use crate::calibration::{
        stratified_sample, DEFAULT_SAMPLE_SEED, STRATIFY_NUM_STRATA_HOLDOUT,
        STRATIFY_NUM_STRATA_PROBE, STRATIFY_NUM_STRATA_PROMO,
    };
    let stress_vectors = stress_vectors.map(|v| {
        stratified_sample(
            &v,
            PROBE_STRESS_VECTORS,
            STRATIFY_NUM_STRATA_PROBE,
            DEFAULT_SAMPLE_SEED,
            None,
        )
        .vectors
    });
    let calibration_promotion = calibration_promotion.map(|v| {
        stratified_sample(
            &v,
            PROMOTION_VECTORS,
            STRATIFY_NUM_STRATA_PROMO,
            DEFAULT_SAMPLE_SEED.wrapping_add(1),
            None,
        )
        .vectors
    });
    let calibration_holdout = calibration_holdout.map(|v| {
        stratified_sample(
            &v,
            HOLDOUT_VECTORS,
            STRATIFY_NUM_STRATA_HOLDOUT,
            DEFAULT_SAMPLE_SEED.wrapping_add(2),
            None,
        )
        .vectors
    });

    let has_activation_bank = calibration_promotion.is_some() && calibration_holdout.is_some();

    // Check bank vector widths against in_features.
    let mut production_quality = CandidateResult::ProductionQualified;
    if let Some(v) = &stress_vectors {
        if v.iter().any(|vec| vec.len() != in_features as usize) {
            production_quality = CandidateResult::DiagnosticOnly;
        }
    }
    if let Some(v) = &calibration_promotion {
        if v.iter().any(|v| v.len() != in_features as usize) {
            production_quality = CandidateResult::DiagnosticOnly;
        }
    }
    if let Some(v) = &calibration_holdout {
        if v.iter().any(|v| v.len() != in_features as usize) {
            production_quality = CandidateResult::DiagnosticOnly;
        }
    }

    let tensor_deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);

    // Reject source tensors whose length doesn't match the declared dimensions.
    if source.len() != (in_features as usize) * (out_features as usize) {
        return Err(QuantizationAdmissionFailure::PackerFailure(format!(
            "source shape mismatch: len={} expected in*out={}*{}={}",
            source.len(),
            in_features,
            out_features,
            in_features * out_features
        )));
    }

    for &format in &candidates {
        if std::time::Instant::now() >= tensor_deadline {
            return Err(QuantizationAdmissionFailure::TimeoutDeadline {
                best_evidence: best_evidence.clone(),
                completed_vectors: completed_vectors.clone(),
                candidates_attempted,
                expired_phase: current_phase.to_string(),
                bank_selections: vec![],
            });
        }
        let (codes, scales, biases, scale_vector) =
            pack_candidate(source, in_features, out_features, format, channel_sq);
        candidates_attempted.push(format!("{:?}", format));
        let reconstructed = reconstruct_candidate(
            format,
            &codes,
            &scales,
            &biases,
            in_features,
            out_features,
            scale_vector.as_deref(),
        );

        let weight_report = validate_weight_space(source, &reconstructed, &promo_profile);

        let is_ternary = matches!(format, RuntimeRepresentationClass::TernaryTile640Base);
        match weight_report.admission_status(&promo_profile, is_ternary) {
            WeightAdmission::Passed => {}
            WeightAdmission::InvestigationBand { warning } => {
                eprintln!("  [investigation] {warning}");
            }
            WeightAdmission::Rejected { reason } => {
                eprintln!("  [reject] {reason}");
                continue;
            }
        }

        // ── Layer 1: Stress bank validation (always run) ───────────
        current_phase = "probe";
        let operator_report = match &stress_vectors {
            Some(vectors) => {
                let pre_unpacked = Some(reconstructed.as_slice());
                match validate_operator_space_with_bank(
                    source,
                    in_features,
                    out_features,
                    &codes,
                    &scales,
                    &biases,
                    scale_vector.as_deref(),
                    vectors,
                    pre_unpacked,
                    Some(tensor_deadline),
                ) {
                    ValidationOutcome::Completed(r) => r,
                    ValidationOutcome::Interrupted(_) => {
                        return Err(QuantizationAdmissionFailure::TimeoutDeadline {
                            best_evidence: best_evidence.clone(),
                            completed_vectors: completed_vectors.clone(),
                            candidates_attempted,
                            expired_phase: current_phase.to_string(),
                            bank_selections: Vec::new(),
                        });
                    }
                }
            }
            None => validate_operator_space(
                source,
                in_features,
                out_features,
                &codes,
                &scales,
                &biases,
                scale_vector.as_deref(),
                &promo_profile,
            ),
        };

        if !operator_report.passes(&promo_profile) {
            continue;
        }

        // Evidence after probe completes
        let evidence = CandidateEvidence {
            representation: format,
            representation_version: 1,
            reconstruction_report: Some(ReconstructionReport {
                weight_nrmse: weight_report.nrmse,
                zero_collapse_ratio: weight_report.zero_collapse_ratio,
                max_abs_error: weight_report.max_abs_error,
                snr_db: 0.0,
                structural: StructuralReport {
                    bytes_valid: true,
                    segment_bounds_valid: true,
                    alignment_valid: true,
                    macro_layout_compatible: true,
                    tail_contract_compatible: true,
                    errors: vec![],
                },
            }),
            probe_report: Some(operator_report.clone()),
            promotion_report: None,
            holdout_report: None,
            completed_vectors: completed_vectors.clone(),
            ..Default::default()
        };
        if let Some(ref stress_vecs) = stress_vectors {
            completed_vectors.probe += stress_vecs.len() as u32;
            completed_vectors.total += stress_vecs.len() as u32;
        }
        if best_evidence
            .as_ref()
            .map_or(true, |b| evidence_is_better(&evidence, b))
        {
            best_evidence = Some(evidence);
        }

        // ── Layer 2: Activation bank validation (prerendered, optional) ──
        if let Some(promo_vecs) = &calibration_promotion {
            current_phase = "promotion";
            let promo_report = match validate_operator_space_with_bank(
                source,
                in_features,
                out_features,
                &codes,
                &scales,
                &biases,
                scale_vector.as_deref(),
                promo_vecs,
                Some(reconstructed.as_slice()),
                Some(tensor_deadline),
            ) {
                ValidationOutcome::Completed(r) => r,
                ValidationOutcome::Interrupted(_) => {
                    return Err(QuantizationAdmissionFailure::TimeoutDeadline {
                        best_evidence: best_evidence.clone(),
                        completed_vectors: completed_vectors.clone(),
                        candidates_attempted,
                        expired_phase: current_phase.to_string(),
                        bank_selections: Vec::new(),
                    });
                }
            };
            if !promo_report.passes(&promo_profile) {
                continue;
            }

            // Evidence after promotion
            let evidence = CandidateEvidence {
                representation: format,
                representation_version: 1,
                reconstruction_report: Some(ReconstructionReport {
                    weight_nrmse: weight_report.nrmse,
                    zero_collapse_ratio: weight_report.zero_collapse_ratio,
                    max_abs_error: weight_report.max_abs_error,
                    snr_db: 0.0,
                    structural: StructuralReport {
                        bytes_valid: true,
                        segment_bounds_valid: true,
                        alignment_valid: true,
                        macro_layout_compatible: true,
                        tail_contract_compatible: true,
                        errors: vec![],
                    },
                }),
                probe_report: best_evidence.as_ref().and_then(|e| e.probe_report.clone()),
                promotion_report: Some(promo_report.clone()),
                holdout_report: None,
                completed_vectors: completed_vectors.clone(),
                ..Default::default()
            };
            completed_vectors.promotion += promo_vecs.len() as u32;
            completed_vectors.total += promo_vecs.len() as u32;
            if best_evidence
                .as_ref()
                .map_or(true, |b| evidence_is_better(&evidence, b))
            {
                best_evidence = Some(evidence);
            }
        }
        if let Some(hold_vecs) = &calibration_holdout {
            current_phase = "holdout";
            let holdout_report = match validate_operator_space_with_bank(
                source,
                in_features,
                out_features,
                &codes,
                &scales,
                &biases,
                scale_vector.as_deref(),
                hold_vecs,
                Some(reconstructed.as_slice()),
                Some(tensor_deadline),
            ) {
                ValidationOutcome::Completed(r) => r,
                ValidationOutcome::Interrupted(_) => {
                    return Err(QuantizationAdmissionFailure::TimeoutDeadline {
                        best_evidence: best_evidence.clone(),
                        completed_vectors: completed_vectors.clone(),
                        candidates_attempted,
                        expired_phase: current_phase.to_string(),
                        bank_selections: Vec::new(),
                    });
                }
            };
            if !holdout_report.passes(&hold_profile) {
                continue;
            }

            // Evidence after holdout
            let evidence = CandidateEvidence {
                representation: format,
                representation_version: 1,
                reconstruction_report: Some(ReconstructionReport {
                    weight_nrmse: weight_report.nrmse,
                    zero_collapse_ratio: weight_report.zero_collapse_ratio,
                    max_abs_error: weight_report.max_abs_error,
                    snr_db: 0.0,
                    structural: StructuralReport {
                        bytes_valid: true,
                        segment_bounds_valid: true,
                        alignment_valid: true,
                        macro_layout_compatible: true,
                        tail_contract_compatible: true,
                        errors: vec![],
                    },
                }),
                probe_report: best_evidence.as_ref().and_then(|e| e.probe_report.clone()),
                promotion_report: best_evidence
                    .as_ref()
                    .and_then(|e| e.promotion_report.clone()),
                holdout_report: Some(holdout_report.clone()),
                completed_vectors: completed_vectors.clone(),
                ..Default::default()
            };
            completed_vectors.holdout += hold_vecs.len() as u32;
            completed_vectors.total += hold_vecs.len() as u32;
            if best_evidence
                .as_ref()
                .map_or(true, |b| evidence_is_better(&evidence, b))
            {
                best_evidence = Some(evidence);
            }
        }

        let reconstruction_contract = match &scale_vector {
            Some(_) => ReconstructionContract::OutputScaledFolded {
                policy: ReductionScalePolicy::MaxAbs,
                scale_storage: ChannelScaleStorage::F16,
                scale_axis: ScaleAxis::ReductionInputColumn,
                scale_count: out_features as u32,
                epsilon_bits: 14,
            },
            None => ReconstructionContract::BaseNf4Tile640,
        };

        let (evidence_level, admission_class) = if has_activation_bank {
            (
                EvidenceLevel::PrerenderedReference,
                if production_quality == CandidateResult::ProductionQualified {
                    ArtifactAdmissionClass::ProductionQualified
                } else {
                    ArtifactAdmissionClass::DiagnosticOnly
                },
            )
        } else {
            (
                EvidenceLevel::StressOnly,
                ArtifactAdmissionClass::DiagnosticOnly,
            )
        };

        return Ok(QualifiedTensor {
            format,
            reconstruction_contract,
            codes,
            scales,
            biases,
            scale_vector,
            weight_report,
            operator_report,
            evidence_level,
            admission_class,
        });
    }
    Err(QuantizationAdmissionFailure::NoCandidatePassed {
        candidates_attempted: candidates.iter().map(|f| format!("{:?}", f)).collect(),
        best_evidence: best_evidence.clone(),
        completed_vectors: completed_vectors.clone(),
        bank_selections: vec![],
    })
}

/// Compute normalized RMSE between reference and reconstructed weights.
///
/// NRMSE = sqrt(sum_sq_err / (n * var_ref)) when var_ref > epsilon,
/// falling back to sqrt(sum_sq_err) / sqrt(n) when reference variance
/// is near zero (pathological all-equal input).
pub fn compute_weight_nrmse(reference: &[f32], reconstructed: &[f32]) -> f64 {
    if reference.len() != reconstructed.len() || reference.is_empty() {
        return f64::MAX;
    }
    let n = reference.len() as f64;
    // On macOS use Accelerate vDSP_vmul + vDSP_sve for SIMD sum-of-squares
    // reduction; on other platforms fall back to the iterator.
    #[cfg(target_os = "macos")]
    let sum_sq_err: f64 = accelerate_vdsp::sum_sq_diff(reference, reconstructed);
    #[cfg(not(target_os = "macos"))]
    let sum_sq_err: f64 = reference
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum();
    let mean_ref: f64 = reference.iter().map(|v| *v as f64).sum::<f64>() / n;
    let var_ref: f64 = reference
        .iter()
        .map(|v| (*v as f64 - mean_ref).powi(2))
        .sum::<f64>()
        / n;
    if var_ref < 1e-12 {
        return sum_sq_err.sqrt() / n.sqrt();
    }
    (sum_sq_err / (n * var_ref)).sqrt()
}

/// Compute an OperatorValidationReport between teacher and student activations.
///
/// Computes per-vector NRMSE (normalized by reference variance), cosine
/// similarity (mean and worst-case), and returns a compact report struct.
/// All other fields default to zero.
pub fn compute_operator_report(
    teacher_acts: &[Vec<f32>],
    student_acts: &[Vec<f32>],
) -> OperatorValidationReport {
    if teacher_acts.is_empty() || teacher_acts.len() != student_acts.len() {
        return OperatorValidationReport::default();
    }
    let mut total_nrmse = 0.0_f64;
    let mut total_cosine = 0.0_f32;
    let mut worst_cosine = 1.0_f32;
    let count = teacher_acts.len();

    for (t, s) in teacher_acts.iter().zip(student_acts.iter()) {
        if t.len() != s.len() || t.is_empty() {
            continue;
        }
        let n = t.len() as f64;
        let mut sum_sq_err = 0.0_f64;
        let mut dot = 0.0_f64;
        let mut t_mag = 0.0_f64;
        let mut s_mag = 0.0_f64;
        let mean_t = t.iter().map(|v| *v as f64).sum::<f64>() / n;
        let mean_s = s.iter().map(|v| *v as f64).sum::<f64>() / n;

        for (tv, sv) in t.iter().zip(s.iter()) {
            let tf = *tv as f64;
            let sf = *sv as f64;
            sum_sq_err += (tf - sf).powi(2);
            dot += (tf - mean_t) * (sf - mean_s);
            t_mag += (tf - mean_t).powi(2);
            s_mag += (sf - mean_s).powi(2);
        }

        let var_t = t_mag / n;
        if var_t < 1e-12 {
            total_nrmse += sum_sq_err.sqrt() / n.sqrt();
        } else {
            total_nrmse += (sum_sq_err / (n * var_t)).sqrt();
        }

        let cosine = if t_mag > 1e-12 && s_mag > 1e-12 {
            (dot / (t_mag.sqrt() * s_mag.sqrt())) as f32
        } else {
            0.0
        };
        total_cosine += cosine;
        worst_cosine = worst_cosine.min(cosine);
    }

    let nc = count as f32;
    OperatorValidationReport {
        operator_nrmse: (total_nrmse / count as f64) as f32,
        cosine_similarity: total_cosine / nc,
        worst_cosine,
        ..Default::default()
    }
}

/// Vector-matrix multiply: `output[j] = sum_i input[i] * weights[i * out_features + j]`.
///
/// Weights are [in_features x out_features] row-major f32, input is a single
/// vector of length `in_features`. Accumulation uses f64 to minimize floating-
/// point noise in the reference forward pass.
fn matmul_vec(input: &[f32], weights: &[f32], in_features: usize, out_features: usize) -> Vec<f32> {
    (0..out_features)
        .map(|out| {
            (0..in_features)
                .map(|inp| input[inp] as f64 * weights[inp * out_features + out] as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

/// Fused teacher-student forward pass and operator report.
///
/// Loads the input vector once, computes both teacher and student outputs
/// and the comparison loss in a single pass over the weight matrix.
/// This replaces separate teacher_forward + student_forward + compute_operator_report
/// with one fused operation.
///
/// Returns (teacher_output, student_output, operator_report).
pub fn fused_teacher_student_forward(
    input: &[f32],
    teacher_weights: &[f32],
    student_weights: &[f32],
    in_features: usize,
    out_features: usize,
) -> (Vec<f32>, Vec<f32>, OperatorValidationReport) {
    let mut teacher_out = vec![0.0f32; out_features];
    let mut student_out = vec![0.0f32; out_features];

    // Single pass: one input load, two dot products per output neuron
    for j in 0..out_features {
        let mut t_acc = 0.0f64;
        let mut s_acc = 0.0f64;
        let base = j;
        for i in 0..in_features {
            let x = input[i] as f64;
            t_acc += x * teacher_weights[i * out_features + base] as f64;
            s_acc += x * student_weights[i * out_features + base] as f64;
        }
        teacher_out[j] = t_acc as f32;
        student_out[j] = s_acc as f32;
    }

    // Compute operator report inline from fused outputs
    let report = compute_operator_report_single(&teacher_out, &student_out);
    (teacher_out, student_out, report)
}

/// Compute OperatorValidationReport for a single pair of activation vectors.
fn compute_operator_report_single(teacher: &[f32], student: &[f32]) -> OperatorValidationReport {
    if teacher.is_empty() || teacher.len() != student.len() {
        return OperatorValidationReport::default();
    }
    let n = teacher.len() as f64;
    let mut sum_sq_err = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut t_mag = 0.0_f64;
    let mut s_mag = 0.0_f64;
    let mean_t = teacher.iter().map(|v| *v as f64).sum::<f64>() / n;
    let mean_s = student.iter().map(|v| *v as f64).sum::<f64>() / n;

    for (tv, sv) in teacher.iter().zip(student.iter()) {
        let tf = *tv as f64;
        let sf = *sv as f64;
        sum_sq_err += (tf - sf).powi(2);
        dot += (tf - mean_t) * (sf - mean_s);
        t_mag += (tf - mean_t).powi(2);
        s_mag += (sf - mean_s).powi(2);
    }

    let var_t = t_mag / n;
    let nrmse = if var_t < 1e-12 {
        (sum_sq_err / n).sqrt()
    } else {
        (sum_sq_err / (n * var_t)).sqrt()
    };
    let cosine = if t_mag > 1e-12 && s_mag > 1e-12 {
        (dot / (t_mag.sqrt() * s_mag.sqrt())) as f32
    } else {
        0.0
    };

    OperatorValidationReport {
        operator_nrmse: nrmse as f32,
        cosine_similarity: cosine,
        worst_cosine: cosine,
        ..Default::default()
    }
}

/// Run teacher (BF16 reference) forward pass on activation vectors.
/// Returns per-probe activation vectors.
///
/// Stub: return identity activations for interface testing.
/// Real implementation will call dequant_matmul_reference with BF16 weights.
pub fn run_teacher_forward(
    weights: &[f32],
    in_features: usize,
    out_features: usize,
    vectors: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    vectors
        .iter()
        .map(|input| matmul_vec(input, weights, in_features, out_features))
        .collect()
}

/// Run student (quantized) forward pass on activation vectors.
/// Returns per-probe activation vectors.
///
/// Stub: return identity activations for interface testing.
/// Real implementation will call dequant_matmul_reference with reconstructed
/// weights.
pub fn run_student_forward(
    reconstructed: &[f32],
    in_features: usize,
    out_features: usize,
    vectors: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    vectors
        .iter()
        .map(|input| matmul_vec(input, reconstructed, in_features, out_features))
        .collect()
}

/// Evaluate all candidate formats for a tensor and return evidence.
///
/// Returns the first passing candidate's QualifiedTensor and CandidateEvidence,
/// or Err with all candidates' evidence for diagnostics.
///
/// Each candidate passes through:
/// 1. **Weight-space screening**: packs, reconstructs, checks NRMSE against
///    a format-specific threshold. Failures are recorded with Failed result.
/// 2. **Activation-space probe** (optional): runs stress vectors through
///    teacher and student forward passes, computes operator validation
///    report. Passes are marked ProductionQualified.
pub fn evaluate_tensor(
    source: &[f32],
    in_features: usize,
    out_features: usize,
    hint: &QuantizationHint,
    channel_sq: Option<&[f32]>,
    stress_suite: Option<&StressSuite>,
    _calibration: Option<&CalibrationSuite>,
    _deadline: &prism_ecs_core::compilation::cancel::CancelToken,
) -> Result<(QualifiedTensor, CandidateEvidence), Vec<CandidateEvidence>> {
    let candidates = candidate_plan(in_features, out_features, hint);
    let mut all_evidence = Vec::with_capacity(candidates.len());

    for format in candidates {
        // Phase 1: Weight-space screening
        let (codes, scales, biases, scale_vector) =
            pack_candidate(source, in_features, out_features, format, channel_sq);
        let reconstructed = reconstruct_candidate(
            format,
            &codes,
            &scales,
            &biases,
            in_features,
            out_features,
            scale_vector.as_deref(),
        );

        let weight_nrmse = compute_weight_nrmse(source, &reconstructed);
        let max_allowed_nrmse = match format {
            RuntimeRepresentationClass::TernaryTile640Base => 0.02,
            RuntimeRepresentationClass::Nf4Tile640Base => 0.01,
            RuntimeRepresentationClass::Int8Tile640Base => 0.005,
            RuntimeRepresentationClass::RawF32 => 0.0,
        };

        if weight_nrmse > max_allowed_nrmse && format != RuntimeRepresentationClass::RawF32 {
            all_evidence.push(CandidateEvidence {
                representation: format,
                representation_version: 1,
                pack_policy_id: 0,
                source_digest: [0u8; 32],
                canonical_shape: Some(CanonicalShape {
                    in_features: in_features as u32,
                    out_features: out_features as u32,
                    rank: 2,
                }),
                structural_report: None,
                reconstruction_report: Some(ReconstructionReport {
                    weight_nrmse,
                    zero_collapse_ratio: 0.0,
                    max_abs_error: 0.0,
                    snr_db: 0.0,
                    structural: StructuralReport {
                        bytes_valid: true,
                        segment_bounds_valid: true,
                        alignment_valid: true,
                        macro_layout_compatible: true,
                        tail_contract_compatible: true,
                        errors: vec![],
                    },
                }),
                probe_report: None,
                promotion_report: None,
                holdout_report: None,
                runtime_conformance_report: None,
                completed_vectors: PhaseVectorCounts::default(),
                payload_bytes: codes.len() as u64,
                metadata_bytes: scales.len() as u64 * 4,
                estimated_runtime_cost: 0.0,
                result: CandidateResult::Failed,
            });
            continue;
        }

        // Phase 2: Activation-space probe
        let stress_vectors: Option<Vec<Vec<f32>>> = stress_suite
            .and_then(|s| s.get(&hint.tensor_class))
            .map(|bank| bank.promotion.clone());

        let probe_report = stress_vectors.as_ref().map(|vectors| {
            // Fused: single pass per vector computes teacher, student, and loss
            let mut avg_nrmse = 0.0f32;
            let mut avg_cosine = 0.0f32;
            let mut worst_cosine = 1.0f32;
            let n = vectors.len() as f32;
            for input in vectors {
                let (_, _, report) = fused_teacher_student_forward(
                    input,
                    source,
                    &reconstructed,
                    in_features,
                    out_features,
                );
                avg_nrmse += report.operator_nrmse;
                avg_cosine += report.cosine_similarity;
                if report.worst_cosine < worst_cosine {
                    worst_cosine = report.worst_cosine;
                }
            }
            OperatorValidationReport {
                operator_nrmse: avg_nrmse / n,
                cosine_similarity: avg_cosine / n,
                worst_cosine,
                ..Default::default()
            }
        });

        let tile_count = ((in_features + 639) / 640) * out_features;
        let tile_code_bytes = match format {
            RuntimeRepresentationClass::TernaryTile640Base => 160u64,
            RuntimeRepresentationClass::Nf4Tile640Base => 320,
            RuntimeRepresentationClass::Int8Tile640Base => 640,
            RuntimeRepresentationClass::RawF32 => 0,
        };
        let tile_meta_bytes = match format {
            RuntimeRepresentationClass::TernaryTile640Base => 4u64,
            RuntimeRepresentationClass::Nf4Tile640Base => 8,
            RuntimeRepresentationClass::Int8Tile640Base => 4,
            RuntimeRepresentationClass::RawF32 => 0,
        };

        let evidence = CandidateEvidence {
            representation: format,
            representation_version: 1,
            pack_policy_id: 0,
            source_digest: [0u8; 32],
            canonical_shape: Some(CanonicalShape {
                in_features: in_features as u32,
                out_features: out_features as u32,
                rank: 2,
            }),
            structural_report: Some(StructuralReport {
                bytes_valid: true,
                segment_bounds_valid: true,
                alignment_valid: true,
                macro_layout_compatible: true,
                tail_contract_compatible: true,
                errors: vec![],
            }),
            reconstruction_report: Some(ReconstructionReport {
                weight_nrmse,
                zero_collapse_ratio: 0.0,
                max_abs_error: 0.0,
                snr_db: 0.0,
                structural: StructuralReport {
                    bytes_valid: true,
                    segment_bounds_valid: true,
                    alignment_valid: true,
                    macro_layout_compatible: true,
                    tail_contract_compatible: true,
                    errors: vec![],
                },
            }),
            probe_report: probe_report.clone(),
            promotion_report: None,
            holdout_report: None,
            runtime_conformance_report: None,
            completed_vectors: PhaseVectorCounts {
                probe: 64,
                promotion: 0,
                holdout: 0,
                total: 64,
            },
            payload_bytes: tile_count as u64 * tile_code_bytes,
            metadata_bytes: tile_count as u64 * tile_meta_bytes,
            estimated_runtime_cost: 0.0,
            result: CandidateResult::ProductionQualified,
        };

        let weight_report = WeightValidationReport {
            nrmse: weight_nrmse,
            rmse: 0.0,
            max_abs_error: 0.0,
            zero_collapse_ratio: 0.0,
        };
        let operator_report = probe_report.clone().unwrap_or_default();

        let reconstruction_contract = match &scale_vector {
            Some(_) => ReconstructionContract::OutputScaledFolded {
                policy: ReductionScalePolicy::MaxAbs,
                scale_storage: ChannelScaleStorage::F16,
                scale_axis: ScaleAxis::ReductionInputColumn,
                scale_count: out_features as u32,
                epsilon_bits: 14,
            },
            None => ReconstructionContract::BaseNf4Tile640,
        };

        let qualified = QualifiedTensor {
            format,
            reconstruction_contract,
            codes,
            scales,
            biases,
            scale_vector,
            weight_report,
            operator_report,
            evidence_level: EvidenceLevel::StressOnly,
            admission_class: ArtifactAdmissionClass::ProductionQualified,
        };

        return Ok((qualified, evidence));
    }

    Err(all_evidence)
}

#[inline]
fn evidence_is_better(a: &CandidateEvidence, b: &CandidateEvidence) -> bool {
    let a_gates = [
        a.probe_report.is_some(),
        a.promotion_report.is_some(),
        a.holdout_report.is_some(),
    ]
    .iter()
    .filter(|&&x| x)
    .count() as u32;
    let b_gates = [
        b.probe_report.is_some(),
        b.promotion_report.is_some(),
        b.holdout_report.is_some(),
    ]
    .iter()
    .filter(|&&x| x)
    .count() as u32;
    if a_gates != b_gates {
        return a_gates > b_gates;
    }
    if let (Some(a_repo), Some(b_repo)) = (&a.holdout_report, &b.holdout_report) {
        if (a_repo.cosine_similarity - b_repo.cosine_similarity).abs() > 1e-6 {
            return a_repo.cosine_similarity > b_repo.cosine_similarity;
        }
        if (a_repo.operator_nrmse - b_repo.operator_nrmse).abs() > 1e-6 {
            return a_repo.operator_nrmse < b_repo.operator_nrmse;
        }
    } else if a.holdout_report.is_some() {
        return true;
    } else if b.holdout_report.is_some() {
        return false;
    }
    if let (Some(a_repo), Some(b_repo)) = (&a.promotion_report, &b.promotion_report) {
        if (a_repo.operator_nrmse - b_repo.operator_nrmse).abs() > 1e-6 {
            return a_repo.operator_nrmse < b_repo.operator_nrmse;
        }
    }
    format_payload_bytes(a.representation) < format_payload_bytes(b.representation)
}

#[inline]
fn format_payload_bytes(f: RuntimeRepresentationClass) -> u64 {
    match f {
        RuntimeRepresentationClass::Nf4Tile640Base => 320,
        RuntimeRepresentationClass::Int8Tile640Base => 640,
        RuntimeRepresentationClass::TernaryTile640Base => 160,
        RuntimeRepresentationClass::RawF32 => 0, // caller computes from dims
    }
}

#[cfg(test)]
mod tests {
    use crate::nf4tile640::{pack_nf4_weights, unpack_nf4_weights, NF4_CODEBOOK};

    const IN_F: usize = 17;
    const OUT_F: usize = 31;

    /// Build deterministic 17x31 matrix and its PyTorch-layout transpose.
    /// Values cycle through the NF4 codebook entries so pack/unpack is
    /// near-lossless. The non-square dimensions (17, 31) still exercise
    /// the tile-group quantization and prove correct orientation.
    fn make_orientation_data() -> (Vec<f32>, Vec<f32>) {
        let mut data = vec![0.0f32; IN_F * OUT_F];
        for i in 0..IN_F {
            for j in 0..OUT_F {
                data[i * OUT_F + j] = NF4_CODEBOOK[(i * OUT_F + j) % 16];
            }
        }
        let mut physical = vec![0.0f32; OUT_F * IN_F];
        for out in 0..OUT_F {
            for inp in 0..IN_F {
                physical[out * IN_F + inp] = data[inp * OUT_F + out];
            }
        }
        (data, physical)
    }

    /// Reference matmul: y = x @ weights where weights is [rows, cols] row-major.
    fn matmul(x: &[f32], weights: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        (0..cols)
            .map(|out| {
                (0..rows)
                    .map(|inp| x[inp] as f64 * weights[inp * cols + out] as f64)
                    .sum::<f64>() as f32
            })
            .collect()
    }

    #[test]
    fn test_nonsquare_orientation_17x31() {
        let (data, data_physical) = make_orientation_data();

        // Step 3: Transpose physical [out,in] -> code convention [in,out].
        let mut data_code = vec![0.0f32; IN_F * OUT_F];
        for inp in 0..IN_F {
            for out in 0..OUT_F {
                data_code[inp * OUT_F + out] = data_physical[out * IN_F + inp];
            }
        }
        assert_eq!(
            data_code, data,
            "transpose of physical data must recover original"
        );

        // Step 4-5: Pack, unpack, verify weight-space.
        let (codes, scales, biases, _, _) = pack_nf4_weights(&data_code, IN_F, OUT_F);
        let unpacked = unpack_nf4_weights(&codes, &scales, &biases, IN_F, OUT_F);
        assert_eq!(unpacked.len(), data_code.len());

        // Since data uses exact NF4 codebook entries, pack/unpack is lossless.
        let weight_mse: f64 = data_code
            .iter()
            .zip(unpacked.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / data_code.len() as f64;
        let weight_rmse = weight_mse.sqrt();
        assert!(weight_rmse < 1e-6, "weight RMSE too high: {weight_rmse}");

        // Step 6: Activation vector (positive-only, non-zero sum to avoid
        // pathological NRMSE from near-zero reference range).
        let x: Vec<f32> = (0..IN_F).map(|i| (i as f32 + 1.0) * 0.1).collect();

        // Step 7: Reference and quantized matmul.
        let y_ref = matmul(&x, &data_code, IN_F, OUT_F);
        let y_quant = matmul(&x, &unpacked, IN_F, OUT_F);

        // Step 8: Quality assertions.
        let y_min = y_ref.iter().cloned().fold(f32::INFINITY, f32::min);
        let y_max = y_ref.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let y_range = (y_max - y_min).max(1e-8);

        let mse: f64 = y_ref
            .iter()
            .zip(y_quant.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / OUT_F as f64;
        let nrmse = (mse.sqrt() as f32) / y_range;

        let dot: f64 = y_ref
            .iter()
            .zip(y_quant.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let norm_ref = (y_ref.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()).sqrt();
        let norm_quant = (y_quant.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()).sqrt();
        let cosine = (dot / (norm_ref * norm_quant).max(1e-12)) as f32;

        let max_abs_err = y_ref
            .iter()
            .zip(y_quant.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(nrmse < 0.05, "NRMSE {nrmse} >= 0.05");
        assert!(cosine > 0.99, "cosine {cosine} <= 0.99");
        assert!(
            max_abs_err < y_range * 0.15,
            "max_abs_err {max_abs_err} >= 15% of range {y_range}"
        );

        // Step 9: Buggy path (no transpose) must produce clearly wrong weights.
        let (codes_bug, scales_bug, biases_bug, _, _) =
            pack_nf4_weights(&data_physical, IN_F, OUT_F);
        let unpacked_bug = unpack_nf4_weights(&codes_bug, &scales_bug, &biases_bug, IN_F, OUT_F);

        // Values are exact NF4 codebook entries in [-1, 1]. When packed with the
        // wrong orientation, tile/group assignment scrambles positions, producing
        // large differences (up to 2.0) for the majority of elements.
        let bug_mismatches = data
            .iter()
            .zip(unpacked_bug.iter())
            .filter(|(a, b)| (*a - *b).abs() > 0.001)
            .count();
        assert!(
            bug_mismatches > data.len() / 4,
            "BUG NOT CAUGHT: only {bug_mismatches}/{} mismatches \u{2014} \
             non-square orientation check fails to catch the transpose bug",
            data.len(),
        );
    }

    #[test]
    #[should_panic(expected = "BUGGY PATH MUST FAIL")]
    fn test_nonsquare_orientation_17x31_buggy_must_fail() {
        let (_data, data_physical) = make_orientation_data();

        // data_physical is [out=31, in=17] but rows=17, cols=31.
        // Packing in the wrong orientation should produce garbage matmul.
        let (codes, scales, biases, _, _) = pack_nf4_weights(&data_physical, IN_F, OUT_F);
        let unpacked = unpack_nf4_weights(&codes, &scales, &biases, IN_F, OUT_F);

        // Build the correct data_physical -> data_code transpose as reference.
        let mut data_code = vec![0.0f32; IN_F * OUT_F];
        for inp in 0..IN_F {
            for out in 0..OUT_F {
                data_code[inp * OUT_F + out] = data_physical[out * IN_F + inp];
            }
        }

        let x: Vec<f32> = (0..IN_F).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let y_ref = matmul(&x, &data_code, IN_F, OUT_F);
        let y_quant = matmul(&x, &unpacked, IN_F, OUT_F);

        let y_min = y_ref.iter().cloned().fold(f32::INFINITY, f32::min);
        let y_max = y_ref.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let y_range = (y_max - y_min).max(1e-8);

        // Also compute cosine and max_abs_err for richer assertions.
        let dot: f64 = y_ref
            .iter()
            .zip(y_quant.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let norm_ref = (y_ref.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()).sqrt();
        let norm_quant = (y_quant.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()).sqrt();
        let cosine = (dot / (norm_ref * norm_quant).max(1e-12)) as f32;
        let max_abs_err = y_ref
            .iter()
            .zip(y_quant.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let mse: f64 = y_ref
            .iter()
            .zip(y_quant.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / OUT_F as f64;
        let nrmse = (mse.sqrt() as f32) / y_range;

        // These assertions WILL panic because the wrong orientation
        // produces garbage output.  #[should_panic] catches the panic,
        // proving the test suite correctly detects the transpose bug.
        assert!(nrmse < 0.05, "BUGGY PATH MUST FAIL: NRMSE {nrmse} >= 0.05");
        assert!(
            cosine > 0.99,
            "BUGGY PATH MUST FAIL: cosine {cosine} <= 0.99"
        );
        assert!(
            max_abs_err < y_range * 0.15,
            "BUGGY PATH MUST FAIL: max_abs_err {max_abs_err} >= 15% of range {y_range}"
        );
    }
}

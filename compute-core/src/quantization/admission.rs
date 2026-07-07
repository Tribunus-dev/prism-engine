//! Quantization admission pipeline.
//!
//! The admission pipeline classifies a tensor, generates legal quantization
//! candidates, packs each candidate, reconstructs from its packed bytes,
//! validates weight-space and operator-space behavior, and promotes the first
//! candidate that passes every gate. If no candidate passes, compilation fails.
//!
//! This replaces ad-hoc per-matrix packing loops with a structured,
//! fail-closed qualification system.

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

use super::calibration::*;
use super::contract::*;
use super::validation::*;
use crate::nf4tile640::{
    pack_int8_weights, pack_nf4_weights, pack_nf4_weights_awls, unpack_int8_weights,
    unpack_nf4_weights,
};
use super::embed_cluster::{pack_ternary_weights, unpack_ternary_weights};

/// Generate the ordered candidate plan for a tensor.
///
/// Candidates are ordered by expected quality and runtime cost (cheapest
/// first). The pipeline promotes the first passing candidate.
pub fn candidate_plan(
    _rows: usize,
    _cols: usize,
    hint: &QuantizationHint,
) -> Vec<QuantizedMatrixFormat> {
    let mut candidates = vec![
        QuantizedMatrixFormat::TernaryTile640Base,
        QuantizedMatrixFormat::Nf4Tile640Base,
    ];
    if hint.permit_scale_candidate {
        candidates.push(QuantizedMatrixFormat::TernaryTile640ScaledReductionAxis);
        candidates.push(QuantizedMatrixFormat::Nf4Tile640ScaledReductionAxis);
    }
    if hint.permit_int8_candidate {
        candidates.push(QuantizedMatrixFormat::Int8Tile640Base);
    }
    candidates
}

/// Pack a tensor into a specific candidate representation.
///
/// Returns (packed_codes, tile_scales, tile_biases, optional_scale_vector).
/// For base format, scale_vector is None.
pub fn pack_candidate(
    source: &[f32],
    rows: usize,
    cols: usize,
    format: QuantizedMatrixFormat,
    channel_sq: Option<&[f32]>,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Option<Vec<f32>>) {
    match format {
        QuantizedMatrixFormat::TernaryTile640Base
        | QuantizedMatrixFormat::TernaryTile640ScaledReductionAxis => {
            let (codes, scales, biases) = pack_ternary_weights(source, rows, cols);
            (codes, scales, biases, None)
        }
        QuantizedMatrixFormat::Nf4Tile640Base => {
            // AW-LS with activation weighting, falling back to max-abs
            // semantics inside the packer.
            let (codes, scales, biases, _, _) =
                pack_nf4_weights_awls(source, rows, cols, channel_sq, 8);
            (codes, scales, biases, None)
        }
        QuantizedMatrixFormat::Nf4Tile640ScaledReductionAxis => {
            // 1. Compute column-wise MaxAbs scale vector.
            let eps = (2.0f32).powi(-14);
            let mut col_scales = vec![0.0f32; cols];
            for j in 0..cols {
                let mut max_abs = 0.0f32;
                for i in 0..rows {
                    let v = source[i * cols + j].abs();
                    if v > max_abs {
                        max_abs = v;
                    }
                }
                col_scales[j] = if max_abs > eps { max_abs } else { 1.0 };
            }

            // 2. Normalize source by column scales.
            let normalized: Vec<f32> = source
                .iter()
                .enumerate()
                .map(|(idx, &v)| v / col_scales[idx % cols])
                .collect();

            // 3. Pack normalized matrix through standard max-abs NF4 packer.
            let (codes, scales, biases, _, _) = pack_nf4_weights(&normalized, rows, cols);
            (codes, scales, biases, Some(col_scales))
        }
        QuantizedMatrixFormat::Int8Tile640Base => {
            let (codes, scales, biases) = pack_int8_weights(source, rows, cols);
            (codes, scales, biases, None)
        }
    }
}

/// Reconstruct a packed candidate back to a weight matrix.
///
/// For scaled-reduction candidates, applies the column scale vector after
/// dequantization: W_hat[i,j] = D(Q'[i,j]; alpha, beta) * S_j.
pub fn reconstruct_candidate(
    format: QuantizedMatrixFormat,
    codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    scale_vector: Option<&[f32]>,
) -> Vec<f32> {
    let unpacked = match format {
        QuantizedMatrixFormat::TernaryTile640Base
        | QuantizedMatrixFormat::TernaryTile640ScaledReductionAxis => {
            unpack_ternary_weights(codes, scales, biases, rows, cols)
        }
        QuantizedMatrixFormat::Nf4Tile640Base
        | QuantizedMatrixFormat::Nf4Tile640ScaledReductionAxis => {
            unpack_nf4_weights(codes, scales, biases, rows, cols)
        }
        QuantizedMatrixFormat::Int8Tile640Base => {
            unpack_int8_weights(codes, scales, biases, rows, cols)
        }
    };
    match scale_vector {
        Some(sv) => {
            let mut result = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    result[i * cols + j] = unpacked[i * cols + j] * sv[j];
                }
            }
            result
        }
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
///    model execution. When provided, enables `ProductionQualified` admission.
pub fn quantize_tensor(
    source: &[f32],
    rows: usize,
    cols: usize,
    hint: &QuantizationHint,
    channel_sq: Option<&[f32]>,
    stress: Option<&StressSuite>,
    calibration: Option<&CalibrationSuite>,
) -> Result<QualifiedTensor, QuantizationAdmissionFailure> {
    let promo_profile = promotion_profile(hint.tensor_class);
    let hold_profile = holdout_profile(hint.tensor_class);
    let candidates = candidate_plan(rows, cols, hint);

    let mut last_weight_nrmse = 0.0f64;
    let mut last_zero_collapse_ratio = 0.0f64;
    let mut last_operator_rmse = 0.0f32;
    let mut last_operator_nrmse = 0.0f32;
    let mut last_cosine_similarity = 0.0f32;
    let mut last_ref_output_rms = 0.0f32;
    let mut vectors_processed: u32 = 0;
    let mut current_phase: &str = "init";

    // Resolve stress and activation vectors from the respective suites.
    // Resize vectors to match the weight's row count (stress bank dim may differ
    // from the actual matrix, e.g. o_proj has 4096 rows vs stress input_dim=3840).
    let resize_vectors = |vecs: Vec<Vec<f32>>, target_len: usize| -> Vec<Vec<f32>> {
        vecs.into_iter()
            .map(|mut v| {
                if v.len() > target_len {
                    v.truncate(target_len);
                } else if v.len() < target_len {
                    v.resize(target_len, 0.0f32);
                }
                v
            })
            .collect()
    };
    let stress_vectors: Option<Vec<Vec<f32>>> = stress
        .and_then(|s| s.get(&hint.tensor_class))
        .map(|bank| resize_vectors(bank.promotion.clone(), rows));
    let calibration_promotion: Option<Vec<Vec<f32>>> = calibration
        .and_then(|s| s.get(&hint.tensor_class))
        .map(|bank| resize_vectors(bank.promotion.clone(), rows));
    let calibration_holdout: Option<Vec<Vec<f32>>> = calibration
        .and_then(|s| s.get(&hint.tensor_class))
        .map(|bank| resize_vectors(bank.holdout.clone(), rows));

    // Cap all vector banks to MAX_VALIDATION_VECTORS to prevent pathological
    // Tiered vector budgets per validation phase.
    // Probe/stress: 64 \u2014 fast fail for bad candidates.
    // Promotion:    256 \u2014 reasonable depth for plausible candidates.
    // Holdout:      256 \u2014 from the CalibrationSuite's separate holdout bank,
    //                      so vectors are automatically disjoint from promotion.
    let stress_vectors = stress_vectors.map(|mut v| { v.truncate(PROBE_STRESS_VECTORS); v });
    let calibration_promotion = calibration_promotion.map(|mut v| { v.truncate(PROMOTION_VECTORS); v });
    let calibration_holdout = calibration_holdout.map(|mut v| { v.truncate(HOLDOUT_VECTORS); v });

    let has_activation_bank = calibration_promotion.is_some() && calibration_holdout.is_some();

    // Per-tensor wall-clock deadline. If exceeded, the current candidate times
    // out and the tensor fails closed. Default: 60 seconds per tensor.
    let tensor_deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);

    for &format in &candidates {
        if std::time::Instant::now() >= tensor_deadline {
            return Err(QuantizationAdmissionFailure::TimeoutDeadline {
                candidates_attempted: candidates.iter().map(|f| format!("{:?}", f)).collect(),
                last_weight_nrmse,
                last_zero_collapse_ratio,
                last_operator_rmse,
                last_operator_nrmse,
                last_cosine_similarity,
                last_ref_output_rms,
                vectors_processed,
                expired_phase: current_phase.to_string(),
            });
        }
        let (codes, scales, biases, scale_vector) =
            pack_candidate(source, rows, cols, format, channel_sq);

        let reconstructed = reconstruct_candidate(
            format,
            &codes,
            &scales,
            &biases,
            rows,
            cols,
            scale_vector.as_deref(),
        );

        let weight_report = validate_weight_space(source, &reconstructed, &promo_profile);
        last_weight_nrmse = weight_report.nrmse;
        last_zero_collapse_ratio = weight_report.zero_collapse_ratio;

        // Three-tier weight-space check: Passed → continue normally.
        // InvestigationBand → warn and continue to operator validation.
        // Rejected → skip this candidate.
        let is_ternary = matches!(format, QuantizedMatrixFormat::TernaryTile640Base | QuantizedMatrixFormat::TernaryTile640ScaledReductionAxis);
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
        if let Some(ref v) = stress_vectors {
            vectors_processed += v.len() as u32;
        }
        let operator_report = match &stress_vectors {
            Some(vectors) => {
                let pre_unpacked = Some(reconstructed.as_slice());
                validate_operator_space_with_bank(
                    source,
                    rows,
                    cols,
                    &codes,
                    &scales,
                    &biases,
                    scale_vector.as_deref(),
                    vectors,
                    pre_unpacked,
                )
            }
            None => validate_operator_space(
                source,
                rows,
                cols,
                &codes,
                &scales,
                &biases,
                scale_vector.as_deref(),
                &promo_profile,
            ),
        };
        last_operator_rmse = operator_report.rmse;
        last_operator_nrmse = operator_report.operator_nrmse;
        last_cosine_similarity = operator_report.cosine_similarity;
        last_ref_output_rms = operator_report.ref_output_rms;

        if !operator_report.passes(&promo_profile) {
            continue;
        }

        // ── Layer 2: Activation bank validation (prerendered, optional) ──
        if let Some(ref promo_vecs) = calibration_promotion {
            current_phase = "promotion";
            vectors_processed += promo_vecs.len() as u32;
            let promo_report = validate_operator_space_with_bank(
                source,
                rows,
                cols,
                &codes,
                &scales,
                &biases,
                scale_vector.as_deref(),
                promo_vecs,
                Some(reconstructed.as_slice()),
            );
            if !promo_report.passes(&promo_profile) {
                last_operator_rmse = promo_report.rmse;
                continue;
            }
        }
        if let Some(ref hold_vecs) = calibration_holdout {
            current_phase = "holdout";
            vectors_processed += hold_vecs.len() as u32;
            let holdout_report = validate_operator_space_with_bank(
                source,
                rows,
                cols,
                &codes,
                &scales,
                &biases,
                scale_vector.as_deref(),
                hold_vecs,
                Some(reconstructed.as_slice()),
            );
            if !holdout_report.passes(&hold_profile) {
                last_operator_rmse = holdout_report.rmse;
                continue;
            }
        }

        let reconstruction_contract = match &scale_vector {
            Some(_) => ReconstructionContract::ScaledReductionAxis {
                policy: ReductionScalePolicy::MaxAbs,
                scale_storage: ChannelScaleStorage::F16,
                scale_axis: ScaleAxis::ReductionInputColumn,
                scale_count: cols as u32,
                epsilon_bits: 14,
            },
            None => ReconstructionContract::BaseNf4Tile640,
        };

        let (evidence_level, admission_class) = if has_activation_bank {
            (
                EvidenceLevel::PrerenderedReference,
                ArtifactAdmissionClass::ProductionQualified,
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
        last_weight_nrmse,
        last_zero_collapse_ratio,
        last_operator_rmse,
        last_operator_nrmse,
        last_cosine_similarity,
        last_ref_output_rms,
    })
}

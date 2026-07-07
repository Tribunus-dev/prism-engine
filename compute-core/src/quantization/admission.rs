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
use super::contract::{CandidateEvidence, PhaseVectorCounts, ProductionQuality, ValidationOutcome};
use super::embed_cluster::{pack_ternary_weights, unpack_ternary_weights};
use super::validation::*;
use crate::nf4tile640::{
    pack_int8_weights, pack_nf4_weights, pack_nf4_weights_awls, unpack_int8_weights,
    unpack_nf4_weights,
};

/// Generate the ordered candidate plan for a tensor.
///
/// Candidates are ordered by expected quality and runtime cost (cheapest
/// first). The pipeline promotes the first passing candidate.
pub fn candidate_plan(
    _in_features: usize,
    _out_features: usize,
    hint: &QuantizationHint,
) -> Vec<QuantizedMatrixFormat> {
    let mut candidates = vec![
        QuantizedMatrixFormat::TernaryTile640Base,
        QuantizedMatrixFormat::Nf4Tile640Base,
    ];
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
    in_features: usize,
    out_features: usize,
    format: QuantizedMatrixFormat,
    channel_sq: Option<&[f32]>,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Option<Vec<f32>>) {
    match format {
        QuantizedMatrixFormat::TernaryTile640Base => {
            let (codes, scales, biases) = pack_ternary_weights(source, in_features, out_features);
            (codes, scales, biases, None)
        }
        QuantizedMatrixFormat::Nf4Tile640Base => {
            // Packing uses the active Nf4PackPolicy (MaxAbsV1, AwlsV1, or OutputScaledFoldedV1).
            // OutputScaledFoldedV1 folds per-output-channel scales into tile alpha/beta
            // at pack time \u2014 the emitted format is standard Nf4Tile640Base with no sidecar.
            let (codes, scales, biases, _, _) =
                pack_nf4_weights_awls(source, in_features, out_features, channel_sq, 8);
            (codes, scales, biases, None)
        }
        QuantizedMatrixFormat::Int8Tile640Base => {
            let (codes, scales, biases) = pack_int8_weights(source, in_features, out_features);
            (codes, scales, biases, None)
        }
        QuantizedMatrixFormat::RawF16 => {
            // RawF16 passthrough: convert f32 source to F16 bytes.
            let codes: Vec<u8> = source.iter()
                .flat_map(|x| (half::f16::from_f32(*x)).to_le_bytes())
                .collect();
            (codes, vec![], vec![], None)
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
    in_features: usize,
    out_features: usize,
    scale_vector: Option<&[f32]>,
) -> Vec<f32> {
    let unpacked = match format {
        QuantizedMatrixFormat::TernaryTile640Base => {
            unpack_ternary_weights(codes, scales, biases, in_features, out_features)
        }
        QuantizedMatrixFormat::Nf4Tile640Base => {
            unpack_nf4_weights(codes, scales, biases, in_features, out_features)
        }
        QuantizedMatrixFormat::Int8Tile640Base => {
            unpack_int8_weights(codes, scales, biases, in_features, out_features)
        }
        QuantizedMatrixFormat::RawF16 => {
            // RawF16 passthrough: decode F16 bytes directly to f32.
            let f16s: Vec<half::f16> = codes.chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
                .collect();
            f16s.iter().map(|x| x.to_f32()).collect()
        }
    };
    match scale_vector {
        Some(sv) => {
            let mut result = vec![0.0f32; in_features * out_features];
            for i in 0..in_features {
                for j in 0..out_features {
                    result[i * out_features + j] = unpacked[i * out_features + j] * sv[j];
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
    use crate::quantization::calibration::{
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
    let mut production_quality = ProductionQuality::ProductionQualified;
    if let Some(v) = &stress_vectors {
        if v.iter().any(|vec| vec.len() != in_features as usize) {
            production_quality = ProductionQuality::SyntheticStressOnly;
        }
    }
    if let Some(v) = &calibration_promotion {
        if v.iter().any(|v| v.len() != in_features as usize) {
            production_quality = ProductionQuality::SyntheticStressOnly;
        }
    }
    if let Some(v) = &calibration_holdout {
        if v.iter().any(|v| v.len() != in_features as usize) {
            production_quality = ProductionQuality::SyntheticStressOnly;
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

        let is_ternary = matches!(
            format,
            QuantizedMatrixFormat::TernaryTile640Base
        );
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
            format,
            weight_nrmse: weight_report.nrmse,
            zero_collapse_ratio: weight_report.zero_collapse_ratio,
            probe: Some(operator_report.clone()),
            promotion: None,
            holdout: None,
            completed_vectors: completed_vectors.clone(),
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
                format,
                weight_nrmse: weight_report.nrmse,
                zero_collapse_ratio: weight_report.zero_collapse_ratio,
                probe: best_evidence.as_ref().and_then(|e| e.probe.clone()),
                promotion: Some(promo_report.clone()),
                holdout: None,
                completed_vectors: completed_vectors.clone(),
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
                format,
                weight_nrmse: weight_report.nrmse,
                zero_collapse_ratio: weight_report.zero_collapse_ratio,
                probe: best_evidence.as_ref().and_then(|e| e.probe.clone()),
                promotion: best_evidence.as_ref().and_then(|e| e.promotion.clone()),
                holdout: Some(holdout_report.clone()),
                completed_vectors: completed_vectors.clone(),
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
                if production_quality == ProductionQuality::ProductionQualified {
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

#[inline]
fn evidence_is_better(a: &CandidateEvidence, b: &CandidateEvidence) -> bool {
    let a_gates = [
        a.probe.is_some(),
        a.promotion.is_some(),
        a.holdout.is_some(),
    ]
    .iter()
    .filter(|&&x| x)
    .count() as u32;
    let b_gates = [
        b.probe.is_some(),
        b.promotion.is_some(),
        b.holdout.is_some(),
    ]
    .iter()
    .filter(|&&x| x)
    .count() as u32;
    if a_gates != b_gates {
        return a_gates > b_gates;
    }
    if let (Some(a_repo), Some(b_repo)) = (&a.holdout, &b.holdout) {
        if (a_repo.cosine_similarity - b_repo.cosine_similarity).abs() > 1e-6 {
            return a_repo.cosine_similarity > b_repo.cosine_similarity;
        }
        if (a_repo.operator_nrmse - b_repo.operator_nrmse).abs() > 1e-6 {
            return a_repo.operator_nrmse < b_repo.operator_nrmse;
        }
    } else if a.holdout.is_some() {
        return true;
    } else if b.holdout.is_some() {
        return false;
    }
    if let (Some(a_repo), Some(b_repo)) = (&a.promotion, &b.promotion) {
        if (a_repo.operator_nrmse - b_repo.operator_nrmse).abs() > 1e-6 {
            return a_repo.operator_nrmse < b_repo.operator_nrmse;
        }
    }
    format_payload_bytes(a.format) < format_payload_bytes(b.format)
}

#[inline]
fn format_payload_bytes(f: QuantizedMatrixFormat) -> u64 {
    match f {
        QuantizedMatrixFormat::Nf4Tile640Base => 320,
        QuantizedMatrixFormat::Int8Tile640Base => 640,
        QuantizedMatrixFormat::TernaryTile640Base => 80,
        _ => 640,
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

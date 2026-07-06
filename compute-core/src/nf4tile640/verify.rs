use crate::nf4tile640::roles::classify_matrix_role;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ════════════════════════════════════════════════════════════════════════════
// Section 1: Core enums and structs
// ════════════════════════════════════════════════════════════════════════════

/// Level of verification to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationLevel {
    /// Only structural checks (pack/unpack parity, codec correctness).
    Structural,
    /// Structural + per-matrix quality metrics.
    Matrix,
    /// Structural + matrix + per-role aggregated policy checks.
    Role,
    /// Full: structural + matrix + role + model behavioral gates.
    Model,
}

/// Quality status for a verification check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityStatus {
    Passed,
    Failed,
    Warning,
}

/// Quality metrics for a single matrix after pack→unpack round-trip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixQualityMetrics {
    pub matrix_name: String,
    pub role: String,
    pub profile_id: u32,
    pub weight_rmse: f32,
    pub weight_nrmse: f32,
    pub max_abs_error: f32,
    pub sqnr_db: f32,
    pub effective_bpw: f32,
    pub quality_status: QualityStatus,
}

/// Aggregated quality report for a single role across all its matrices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleQualityReport {
    pub role: String,
    pub num_matrices: usize,
    pub avg_weight_rmse: f32,
    pub max_weight_rmse: f32,
    pub avg_sqnr_db: f32,
    pub min_sqnr_db: f32,
    pub passes: usize,
    pub failures: usize,
    pub quality_status: QualityStatus,
}

/// Complete receipt for an entire verification run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReceipt {
    pub verification_level: VerificationLevel,
    pub num_matrices_checked: usize,
    pub num_passed: usize,
    pub num_failed: usize,
    pub per_matrix: Vec<MatrixQualityMetrics>,
    pub per_role: Vec<RoleQualityReport>,
    pub overall_status: QualityStatus,
    pub compiler_revision: String,
    pub source_model_digest: String,
}

// ════════════════════════════════════════════════════════════════════════════
// Section 2: Structural verification
// ════════════════════════════════════════════════════════════════════════════

/// Verify structural correctness of a packed matrix.
///
/// Checks:
/// 1. Codes buffer size matches `packed_size(rows, cols)`
/// 2. All codes are nibbles in range 0..16
/// 3. Scales are all finite (not NaN, not inf)
/// 4. Biases are all finite
/// 5. Unpack produces the correct shape
/// 6. Unpacked values are finite
///
/// Returns `Ok(())` on success, or `Err` with a list of error messages.
pub fn structural_verify(
    codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    original_rows: u32,
    original_cols: u32,
) -> Result<(), Vec<String>> {
    use crate::nf4tile640::packed_size;
    use crate::nf4tile640::TILE_ELEMENTS;
    let rows = original_rows as usize;
    let cols = original_cols as usize;
    let mut errors: Vec<String> = Vec::new();

    // 1. Codes buffer size
    let expected_codes_len = packed_size(rows, cols);
    if codes.len() != expected_codes_len {
        errors.push(format!(
            "codes buffer length {} does not match expected packed size {} for shape [{}, {}]",
            codes.len(),
            expected_codes_len,
            rows,
            cols,
        ));
    }

    // 2. All codes are valid nibbles (0..16)
    for (i, &byte) in codes.iter().enumerate() {
        let lo = byte & 0x0F;
        let hi = byte >> 4;
        if lo >= 16 || hi >= 16 {
            errors.push(format!(
                "invalid code nibble at byte {}: byte={:#04x} lo={} hi={}",
                i, byte, lo, hi,
            ));
        }
    }

    // 3. Scales are all finite
    for (i, &s) in scales.iter().enumerate() {
        if !s.is_finite() {
            errors.push(format!(
                "non-finite scale at index {}: value={}",
                i, s,
            ));
        }
    }

    // 4. Biases are all finite
    for (i, &b) in biases.iter().enumerate() {
        if !b.is_finite() {
            errors.push(format!(
                "non-finite bias at index {}: value={}",
                i, b,
            ));
        }
    }

    // 5. Unpack produces correct shape
    // We can attempt unpack; the function will panic on size mismatch, so
    // we check the size ourselves here.
    let tiles_per_row = cols.div_ceil(TILE_ELEMENTS);
    let total_tiles = rows * tiles_per_row;
    let expected_scales_len = total_tiles * 5; // SCALES_F32_PER_TILE
    if scales.len() != expected_scales_len {
        errors.push(format!(
            "scales length {} does not match expected {} for shape [{}, {}]",
            scales.len(),
            expected_scales_len,
            rows,
            cols,
        ));
    }
    // Also check biases length (unpack_nf4_weights would panic on mismatch).
    if biases.len() != expected_scales_len {
        errors.push(format!(
            "biases length {} does not match expected {} for shape [{}, {}]",
            biases.len(),
            expected_scales_len,
            rows,
            cols,
        ));
    }

    // 6. If we can unpack without errors, check that output values are finite
    if errors.is_empty() {
        let unpacked = crate::nf4tile640::unpack_nf4_weights(codes, scales, biases, rows, cols);
        if unpacked.len() != rows * cols {
            errors.push(format!(
                "unpacked length {} does not match expected {}",
                unpacked.len(),
                rows * cols,
            ));
        }
        for (i, &v) in unpacked.iter().enumerate() {
            if !v.is_finite() {
                errors.push(format!(
                    "non-finite unpacked value at index {}: value={}",
                    i, v,
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 3: Matrix-level quality metrics
// ════════════════════════════════════════════════════════════════════════════

/// Compute quality metrics for a single matrix by comparing original vs packed+unpacked.
///
/// The `quality_status` field in the result reflects only structural correctness
/// (codes valid, finite values). Role-level quality policy thresholds are applied
/// by the caller via [`apply_quality_policy`].
pub fn compute_matrix_metrics(
    matrix_name: &str,
    original: &[f32],
    codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    profile_id: u32,
) -> MatrixQualityMetrics {
    // Classify role from the matrix name.
    let role = classify_matrix_role(matrix_name);

    // Run structural verification.
    let structural_result = structural_verify(
        codes,
        scales,
        biases,
        rows as u32,
        cols as u32,
    );

    // If structural checks fail, return early with Failed status.
    // We cannot safely unpack because unpack_nf4_weights uses assert_eq!
    // on buffer sizes and would panic on mismatch.
    if structural_result.is_err() {
        let role_str = role.to_string();
        return MatrixQualityMetrics {
            matrix_name: matrix_name.to_string(),
            role: role_str,
            profile_id,
            weight_rmse: f32::NAN,
            weight_nrmse: f32::NAN,
            max_abs_error: f32::NAN,
            sqnr_db: f32::NAN,
            effective_bpw: 4.5,
            quality_status: QualityStatus::Failed,
        };
    }

    // Unpack (structural checks passed, so buffer sizes are correct).
    use crate::nf4tile640::unpack_nf4_weights;
    let reconstructed = unpack_nf4_weights(codes, scales, biases, rows, cols);


    // Compute per-element error.
    let n = original.len().min(reconstructed.len());
    let mut sum_sq_error = 0.0f64;
    let mut max_abs_err: f32 = 0.0;
    let mut sum_sq_original = 0.0f64;
    let mut sum_sq_error_for_sqnr = 0.0f64;

    let mut o_min = f32::MAX;
    let mut o_max = f32::MIN;

    for i in 0..n {
        let o = original[i];
        let r = reconstructed[i];
        let err = o - r;
        let abs_err = err.abs();
        let sq_err = (err as f64) * (err as f64);

        sum_sq_error += sq_err;
        sum_sq_original += (o as f64) * (o as f64);
        sum_sq_error_for_sqnr += sq_err;

        if abs_err > max_abs_err {
            max_abs_err = abs_err;
        }
        if o < o_min {
            o_min = o;
        }
        if o > o_max {
            o_max = o;
        }
    }

    let mean_sq_error = sum_sq_error / n as f64;
    let rmse = mean_sq_error.sqrt() as f32;

    // NRMSE = RMSE / (max - min)
    let range = o_max - o_min;
    let nrmse = if range > 1e-30 { rmse / range } else { 0.0 };

    // SQNR = 10 * log10(var(original) / var(error))
    let _mean_original = sum_sq_original / n as f64;
    // Var = mean(x²) - mean(x)². For zero-centered signal, use mean(x²) directly.
    // To avoid division-by-zero, we compute signal_var = mean(x²) since we don't
    // have the actual mean of original values readily available here.
    // Actually, let's do it properly: var(error) = mean(error²) - mean(error)²
    // But for SQNR, the standard formula uses mean squared error, not variance.
    // SQNR = 10 * log10(E[original²] / E[error²]) when signals are zero-mean.
    // More generally, we use the ratio of power:
    let error_power = sum_sq_error_for_sqnr / n as f64;
    let signal_power = sum_sq_original / n as f64;
    let sqnr = if error_power > 1e-30 {
        10.0 * (signal_power / error_power).log10() as f32
    } else {
        100.0 // essentially infinite SQNR
    };

    // Effective BPW: NF4 uses 4 bits per weight + sidecar overhead.
    // Sidecar: each tile has 5 f32 scales + 5 f32 biases = 40 bytes overhead
    // for 640 weights. 40 * 8 / 640 = 0.5 bits per weight overhead.
    // Total = 4.0 + 0.5 = 4.5 bits per weight.
    let effective_bpw = 4.5;

    // Quality status is Passed if structural checks passed.
    let quality_status = if structural_result.is_ok() {
        QualityStatus::Passed
    } else {
        QualityStatus::Failed
    };

    let role_str = role.to_string();

    MatrixQualityMetrics {
        matrix_name: matrix_name.to_string(),
        role: role_str,
        profile_id,
        weight_rmse: rmse,
        weight_nrmse: nrmse,
        max_abs_error: max_abs_err,
        sqnr_db: sqnr,
        effective_bpw,
        quality_status,
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 4: Role-level verification
// ════════════════════════════════════════════════════════════════════════════

/// Classify tensors and verify per-role aggregated quality.
///
/// Groups `matrices` by their `role` field and computes aggregate statistics:
/// average/max RMSE, average/min SQNR, pass/fail counts.
pub fn classify_matrices_and_verify_role(
    matrices: &[MatrixQualityMetrics],
) -> Vec<RoleQualityReport> {
    // Group by role.
    let mut role_map: HashMap<&str, Vec<&MatrixQualityMetrics>> = HashMap::new();
    for m in matrices {
        role_map.entry(m.role.as_str()).or_default().push(m);
    }

    let mut reports: Vec<RoleQualityReport> = Vec::with_capacity(role_map.len());
    for (role_name, group) in role_map {
        let num_matrices = group.len();
        let mut sum_rmse = 0.0f64;
        let mut max_rmse = f32::MIN;
        let mut sum_sqnr = 0.0f64;
        let mut min_sqnr = f32::MAX;
        let mut passes = 0;
        let mut failures = 0;

        for m in &group {
            let r = m.weight_rmse as f64;
            sum_rmse += r;
            if m.weight_rmse > max_rmse {
                max_rmse = m.weight_rmse;
            }

            let s = m.sqnr_db as f64;
            sum_sqnr += s;
            if m.sqnr_db < min_sqnr {
                min_sqnr = m.sqnr_db;
            }

            match m.quality_status {
                QualityStatus::Passed | QualityStatus::Warning => passes += 1,
                QualityStatus::Failed => failures += 1,
            }
        }

        let avg_rmse = (sum_rmse / num_matrices as f64) as f32;
        let avg_sqnr = (sum_sqnr / num_matrices as f64) as f32;

        // Count warnings separately for overall status.
        let warnings = group.iter().filter(|m| m.quality_status == QualityStatus::Warning).count();

        let quality_status = if failures > 0 {
            QualityStatus::Failed
        } else if warnings > 0 {
            QualityStatus::Warning
        } else {
            QualityStatus::Passed
        };

        // Adjust passes to only count true passes (not warnings).
        let true_passes = passes - warnings;

        reports.push(RoleQualityReport {
            role: role_name.to_string(),
            num_matrices,
            avg_weight_rmse: avg_rmse,
            max_weight_rmse: max_rmse,
            avg_sqnr_db: avg_sqnr,
            min_sqnr_db: min_sqnr,
            passes: true_passes,
            failures,
            quality_status,
        });
    }

    reports
}

// ════════════════════════════════════════════════════════════════════════════
// Section 5: Full verification run
// ════════════════════════════════════════════════════════════════════════════

/// Run structural + matrix-level verification for all matrices and produce a receipt.
///
/// Each tuple in `matrices` is:
/// `(name, original, codes, scales, biases, rows, cols, profile_id)`
pub fn full_structure_verify_run(
    matrices: &[(&str, &[f32], &[u8], &[f32], &[f32], usize, usize, u32)],
    compiler_revision: &str,
    source_model_digest: &str,
) -> VerificationReceipt {
    let per_matrix: Vec<MatrixQualityMetrics> = matrices
        .iter()
        .map(|(name, original, codes, scales, biases, rows, cols, profile_id)| {
            compute_matrix_metrics(
                name,
                original,
                codes,
                scales,
                biases,
                *rows,
                *cols,
                *profile_id,
            )
        })
        .collect();

    let per_role = classify_matrices_and_verify_role(&per_matrix);

    let num_matrices_checked = per_matrix.len();
    let num_passed = per_matrix
        .iter()
        .filter(|m| m.quality_status == QualityStatus::Passed)
        .count();
    let num_failed = per_matrix
        .iter()
        .filter(|m| m.quality_status == QualityStatus::Failed)
        .count();

    let overall_status = if num_failed == 0 {
        QualityStatus::Passed
    } else {
        QualityStatus::Failed
    };

    VerificationReceipt {
        verification_level: VerificationLevel::Structural,
        num_matrices_checked,
        num_passed,
        num_failed,
        per_matrix,
        per_role,
        overall_status,
        compiler_revision: compiler_revision.to_string(),
        source_model_digest: source_model_digest.to_string(),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 6: Quality policy application
// ════════════════════════════════════════════════════════════════════════════

/// Apply a quality policy to matrix metrics, returning updated metrics with
/// quality_status reflecting role-based thresholds.
///
/// Policies:
/// - `"strict"`: RMSE must be < 0.01, max_abs_error < 0.5, SQNR > 20 dB for Passed
/// - `"default"`: RMSE < 0.05, max_abs_error < 1.0, SQNR > 15 dB
/// - `"experimental"`: Only structural checks (codes valid, finite values)
///
/// For boundary roles (Embedding, LmHead, MultimodalProjection), the strict
/// policy is always used regardless of the `policy` argument.
pub fn apply_quality_policy(
    metrics: &[MatrixQualityMetrics],
    policy: &str,
) -> Vec<MatrixQualityMetrics> {
    metrics
        .iter()
        .map(|m| {
            // Determine the effective policy for this matrix's role.
            let is_boundary = matches!(
                m.role.as_str(),
                "embedding" | "lm_head" | "multimodal_projection"
            );

            // Determine thresholds.
            let (rmse_threshold, max_abs_threshold, sqnr_threshold) = if is_boundary {
                (0.01f32, 0.5f32, 20.0f32) // always strict for boundary
            } else {
                match policy {
                    "strict" => (0.01, 0.5, 20.0),
                    "default" => (0.05, 1.0, 15.0),
                    "experimental" => (f32::MAX, f32::MAX, f32::MIN),
                    _ => (0.05, 1.0, 15.0), // unknown → default
                }
            };

            // If structural check failed, keep Failed.
            if m.quality_status == QualityStatus::Failed {
                return m.clone();
            }

            // Apply policy thresholds.
            let structural_ok = m.quality_status == QualityStatus::Passed;
            let within_rmse = m.weight_rmse < rmse_threshold;
            let within_abs = m.max_abs_error < max_abs_threshold;
            let within_sqnr = m.sqnr_db > sqnr_threshold;

            let passed = structural_ok && within_rmse && within_abs && within_sqnr;

            let updated_status = if passed {
                QualityStatus::Passed
            } else {
                QualityStatus::Failed
            };

            MatrixQualityMetrics {
                quality_status: updated_status,
                ..m.clone()
            }
        })
        .collect()
}

// ════════════════════════════════════════════════════════════════════════════
// Section 7: Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nf4tile640::{pack_nf4_tile, pack_nf4_weights, TILE_ELEMENTS};
    use crate::nf4tile640::profile::PROFILE_ID_CANONICAL_NF4_V1;

    /// Helper: generate a random-ish test tile.
    fn make_test_tile() -> [f32; TILE_ELEMENTS] {
        let mut tile = [0.0f32; TILE_ELEMENTS];
        for i in 0..TILE_ELEMENTS {
            // Mix of values spanning the NF4 codebook range.
            tile[i] = ((i as f32) / (TILE_ELEMENTS as f32) - 0.5) * 2.0;
        }
        tile
    }

    // ────────────────────────────────────────────────────────────────────
    // structural_verify
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_structural_verify_passes_valid_data() {
        let weights = make_test_tile();
        let (codes, scales, biases) = pack_nf4_tile(&weights);
        let result = structural_verify(&codes, &scales, &biases, 1, TILE_ELEMENTS as u32);
        assert!(result.is_ok(), "valid tile should pass: {:?}", result);
    }

    #[test]
    fn test_structural_verify_rejects_truncated_codes() {
        let weights = make_test_tile();
        let (codes, scales, biases) = pack_nf4_tile(&weights);
        let truncated = &codes[..codes.len() - 1];
        let result = structural_verify(truncated, &scales, &biases, 1, TILE_ELEMENTS as u32);
        assert!(result.is_err(), "truncated codes should fail");
    }

    #[test]
    fn test_structural_verify_rejects_nan_scale() {
        let weights = make_test_tile();
        let (codes, mut scales, biases) = pack_nf4_tile(&weights);
        scales[0] = f32::NAN;
        let result = structural_verify(&codes, &scales, &biases, 1, TILE_ELEMENTS as u32);
        assert!(result.is_err(), "NaN scale should fail");
    }

    #[test]
    fn test_structural_verify_rejects_inf_scale() {
        let weights = make_test_tile();
        let (codes, mut scales, biases) = pack_nf4_tile(&weights);
        scales[1] = f32::INFINITY;
        let result = structural_verify(&codes, &scales, &biases, 1, TILE_ELEMENTS as u32);
        assert!(result.is_err(), "inf scale should fail");
    }

    #[test]
    fn test_structural_verify_rejects_nan_bias() {
        let weights = make_test_tile();
        let (codes, scales, mut biases) = pack_nf4_tile(&weights);
        biases[2] = f32::NAN;
        let result = structural_verify(&codes, &scales, &biases, 1, TILE_ELEMENTS as u32);
        assert!(result.is_err(), "NaN bias should fail");
    }

    // ────────────────────────────────────────────────────────────────────
    // compute_matrix_metrics
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_compute_matrix_metrics_returns_sensible_values() {
        let weights = make_test_tile();
        let (codes, scales, biases) = pack_nf4_tile(&weights);
        let metrics = compute_matrix_metrics(
            "test_proj",
            &weights,
            &codes,
            &scales,
            &biases,
            1,
            TILE_ELEMENTS,
            PROFILE_ID_CANONICAL_NF4_V1.0,
        );
        assert_eq!(metrics.matrix_name, "test_proj");
        assert_eq!(metrics.profile_id, 0);
        // With a synthetic signal spanning [-1, 1], RMSE should be small
        // but non-zero due to NF4 quantization.
        assert!(
            metrics.weight_rmse >= 0.0,
            "RMSE should be non-negative"
        );
        assert!(
            metrics.weight_rmse < 0.5,
            "RMSE should be reasonable for NF4: got {}",
            metrics.weight_rmse
        );
        assert!(
            metrics.max_abs_error < 1.0,
            "max abs error should be bounded: got {}",
            metrics.max_abs_error
        );
        assert!(
            metrics.sqnr_db > 0.0,
            "SQNR should be positive: got {}",
            metrics.sqnr_db
        );
        assert!(
            (metrics.effective_bpw - 4.5).abs() < 1e-6,
            "effective BPW should be ~4.5: got {}",
            metrics.effective_bpw
        );
        // Should pass structural checks.
        assert_eq!(metrics.quality_status, QualityStatus::Passed);
    }

    // ────────────────────────────────────────────────────────────────────
    // classify_matrices_and_verify_role
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_classify_matrices_and_verify_role_single_role() {
        let weights = make_test_tile();
        let (codes, scales, biases) = pack_nf4_tile(&weights);

        let m1 = compute_matrix_metrics(
            "model.layers.0.q_proj",
            &weights,
            &codes,
            &scales,
            &biases,
            1,
            TILE_ELEMENTS,
            0,
        );
        let m2 = compute_matrix_metrics(
            "model.layers.1.q_proj",
            &weights,
            &codes,
            &scales,
            &biases,
            1,
            TILE_ELEMENTS,
            0,
        );

        let reports = classify_matrices_and_verify_role(&[m1, m2]);
        assert_eq!(reports.len(), 1, "both matrices are attention_q");
        let r = &reports[0];
        assert_eq!(r.role, "attention_q");
        assert_eq!(r.num_matrices, 2);
        assert_eq!(r.passes, 2);
        assert_eq!(r.failures, 0);
        assert_eq!(r.quality_status, QualityStatus::Passed);
    }

    #[test]
    fn test_classify_matrices_and_verify_role_multiple_roles() {
        let weights = make_test_tile();
        let (codes, scales, biases) = pack_nf4_tile(&weights);

        let m1 = compute_matrix_metrics(
            "model.layers.0.q_proj",
            &weights,
            &codes,
            &scales,
            &biases,
            1,
            TILE_ELEMENTS,
            0,
        );
        let m2 = compute_matrix_metrics(
            "model.layers.0.gate_proj",
            &weights,
            &codes,
            &scales,
            &biases,
            1,
            TILE_ELEMENTS,
            0,
        );

        let reports = classify_matrices_and_verify_role(&[m1, m2]);
        assert_eq!(reports.len(), 2, "two distinct roles");
        let role_names: Vec<&str> = reports.iter().map(|r| r.role.as_str()).collect();
        assert!(role_names.contains(&"attention_q"));
        assert!(role_names.contains(&"ffn_gate"));
    }

    #[test]
    fn test_classify_matrices_and_verify_role_failure_propagates() {
        let weights = make_test_tile();
        let (codes, scales, biases) = pack_nf4_tile(&weights);

        // Force a failure by truncating codes (hits packed_size check).
        let bad_codes = &codes[..codes.len() - 1];

        let m1 = compute_matrix_metrics(
            "model.layers.0.q_proj",
            &weights,
            bad_codes,
            &scales,
            &biases,
            1,
            640,
            0,
        );

        let reports = classify_matrices_and_verify_role(&[m1]);
        assert_eq!(reports.len(), 1);
        assert_eq!(
            reports[0].quality_status,
            QualityStatus::Failed,
            "role should fail if a matrix fails structural checks"
        );
        assert_eq!(reports[0].failures, 1);
    }

    // ────────────────────────────────────────────────────────────────────
    // full_structure_verify_run
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_full_structure_verify_run_produces_receipt() {
        // Create a multi-row weight matrix.
        let rows = 2;
        let cols = TILE_ELEMENTS * 2; // 2 tiles per row
        let mut weights = Vec::with_capacity(rows * cols);
        for i in 0..(rows * cols) {
            weights.push(((i as f32) / (rows * cols) as f32 - 0.5) * 2.0);
        }

        let (codes, scales, biases, _packed_rows, _packed_cols) =
            pack_nf4_weights(&weights, rows, cols);

        let matrices = [(
            "model.layers.0.q_proj",
            weights.as_slice(),
            codes.as_slice(),
            scales.as_slice(),
            biases.as_slice(),
            rows,
            cols,
            0u32,
        )];

        let receipt = full_structure_verify_run(
            &matrices,
            "abcd1234",
            "sha256:deadbeef",
        );

        assert_eq!(receipt.verification_level, VerificationLevel::Structural);
        assert_eq!(receipt.num_matrices_checked, 1);
        assert_eq!(receipt.compiler_revision, "abcd1234");
        assert_eq!(receipt.source_model_digest, "sha256:deadbeef");
        // Should pass since we packed from clean data.
        assert_eq!(receipt.overall_status, QualityStatus::Passed);
        assert_eq!(receipt.num_passed, 1);
        assert_eq!(receipt.num_failed, 0);
        assert_eq!(receipt.per_matrix.len(), 1);
        assert!(!receipt.per_role.is_empty());
    }

    // ────────────────────────────────────────────────────────────────────
    // apply_quality_policy
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_apply_quality_policy_default() {
        // Threshold: RMSE < 0.05, max_abs < 1.0, SQNR > 15 dB
        let m = MatrixQualityMetrics {
            matrix_name: "test".into(),
            role: "attention_q".into(),
            profile_id: 0,
            weight_rmse: 0.03,
            weight_nrmse: 0.01,
            max_abs_error: 0.5,
            sqnr_db: 25.0,
            effective_bpw: 4.5,
            quality_status: QualityStatus::Passed,
        };

        let result = apply_quality_policy(&[m], "default");
        assert_eq!(result[0].quality_status, QualityStatus::Passed);
    }

    #[test]
    fn test_apply_quality_policy_default_fails_high_rmse() {
        let m = MatrixQualityMetrics {
            matrix_name: "test".into(),
            role: "attention_q".into(),
            profile_id: 0,
            weight_rmse: 0.1,
            weight_nrmse: 0.02,
            max_abs_error: 0.5,
            sqnr_db: 25.0,
            effective_bpw: 4.5,
            quality_status: QualityStatus::Passed,
        };

        let result = apply_quality_policy(&[m], "default");
        assert_eq!(result[0].quality_status, QualityStatus::Failed);
    }

    #[test]
    fn test_apply_quality_policy_strict() {
        let m = MatrixQualityMetrics {
            matrix_name: "test".into(),
            role: "attention_q".into(),
            profile_id: 0,
            weight_rmse: 0.005,
            weight_nrmse: 0.001,
            max_abs_error: 0.2,
            sqnr_db: 30.0,
            effective_bpw: 4.5,
            quality_status: QualityStatus::Passed,
        };

        let result = apply_quality_policy(&[m], "strict");
        assert_eq!(result[0].quality_status, QualityStatus::Passed);
    }

    #[test]
    fn test_apply_quality_policy_strict_fails_loose_rmse() {
        let m = MatrixQualityMetrics {
            matrix_name: "test".into(),
            role: "attention_q".into(),
            profile_id: 0,
            weight_rmse: 0.02,
            weight_nrmse: 0.004,
            max_abs_error: 0.2,
            sqnr_db: 30.0,
            effective_bpw: 4.5,
            quality_status: QualityStatus::Passed,
        };

        let result = apply_quality_policy(&[m], "strict");
        assert_eq!(result[0].quality_status, QualityStatus::Failed);
    }

    #[test]
    fn test_apply_quality_policy_experimental_passes_structural() {
        let m = MatrixQualityMetrics {
            matrix_name: "test".into(),
            role: "attention_q".into(),
            profile_id: 0,
            weight_rmse: 10.0, // terrible, but experimental doesn't care
            weight_nrmse: 1.0,
            max_abs_error: 100.0,
            sqnr_db: -10.0,
            effective_bpw: 4.5,
            quality_status: QualityStatus::Passed,
        };

        let result = apply_quality_policy(&[m], "experimental");
        assert_eq!(result[0].quality_status, QualityStatus::Passed);
    }

    #[test]
    fn test_apply_quality_policy_experimental_fails_structural() {
        let m = MatrixQualityMetrics {
            matrix_name: "test".into(),
            role: "attention_q".into(),
            profile_id: 0,
            weight_rmse: 10.0,
            weight_nrmse: 1.0,
            max_abs_error: 100.0,
            sqnr_db: -10.0,
            effective_bpw: 4.5,
            quality_status: QualityStatus::Failed,
        };

        let result = apply_quality_policy(&[m], "experimental");
        assert_eq!(result[0].quality_status, QualityStatus::Failed);
    }

    #[test]
    fn test_apply_quality_policy_boundary_always_strict() {
        // Boundary roles (embedding, lm_head, multimodal_projection)
        // should use strict thresholds regardless of policy argument.
        let m = MatrixQualityMetrics {
            matrix_name: "test".into(),
            role: "lm_head".into(),
            profile_id: 0,
            weight_rmse: 0.02, // fails strict (< 0.01), passes default (< 0.05)
            weight_nrmse: 0.002,
            max_abs_error: 0.2,
            sqnr_db: 30.0,
            effective_bpw: 4.5,
            quality_status: QualityStatus::Passed,
        };

        // Even with "default" policy, boundary uses strict.
        let result = apply_quality_policy(&[m], "default");
        assert_eq!(result[0].quality_status, QualityStatus::Failed);
    }

    #[test]
    fn test_apply_quality_policy_keeps_existing_failure() {
        let m = MatrixQualityMetrics {
            matrix_name: "test".into(),
            role: "attention_q".into(),
            profile_id: 0,
            weight_rmse: 0.0,
            weight_nrmse: 0.0,
            max_abs_error: 0.0,
            sqnr_db: 100.0,
            effective_bpw: 4.5,
            quality_status: QualityStatus::Failed,
        };

        // Even with "experimental", already-failed stays failed.
        let result = apply_quality_policy(&[m], "experimental");
        assert_eq!(result[0].quality_status, QualityStatus::Failed);
    }
}

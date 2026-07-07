//! Weight-space and operator-space validation for quantized matrices.
//!
//! Validation is layered: weight-space metrics (RMSE, NRMSE, zero-collapse)
//! catch structural collapse, while operator-space metrics (matmul RMSE on
//! deterministic calibration inputs) catch behavioral degradation.

use super::contract::*;
#[cfg(all(not(feature = "metal-dispatch"), any(feature = "mlx-backend", feature = "prism-backend", feature = "prism-backend-ios")))]
use crate::backend::accelerate_ffi::{cblas_sgemm, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR};
use crate::nf4tile640::dequant_matmul_reference;
#[cfg(any(feature = "metal-dispatch", feature = "mlx-backend", feature = "prism-backend", feature = "prism-backend-ios"))]
use crate::nf4tile640::unpack_nf4_weights;

/// Number of vectors above which the Accelerate batched path is used.
const ACCELERATE_BATCH_THRESHOLD: usize = 8;

/// Validate operator-space quality using an explicit set of activation vectors.
/// Validate weight-space reconstruction quality.
pub fn validate_weight_space(
    source: &[f32],
    reconstructed: &[f32],
    _profile: &QuantizationValidationProfile,
) -> WeightValidationReport {
    let n = source.len();
    let mut sq_err = 0.0f64;
    let mut sq_source = 0.0f64;
    let mut max_abs_err = 0.0f64;
    let mut zero_collapsed = 0u64;
    let mut non_zero_source = 0u64;
    let src_threshold = 0.01f64;
    let recon_threshold = 0.001f64;

    for idx in 0..n {
        let diff = (source[idx] as f64 - reconstructed[idx] as f64).abs();
        let src_abs = (source[idx] as f64).abs();
        sq_err += diff * diff;
        sq_source += (source[idx] as f64).powi(2);
        if diff > max_abs_err {
            max_abs_err = diff;
        }
        if src_abs > src_threshold && (reconstructed[idx] as f64).abs() < recon_threshold {
            zero_collapsed += 1;
        }
        if src_abs > src_threshold {
            non_zero_source += 1;
        }
    }

    let rmse = (sq_err / n as f64).sqrt();
    let source_rms = (sq_source / n as f64).sqrt();
    let nrmse = rmse / (source_rms + 1e-30);
    let zero_collapse_ratio = if non_zero_source > 0 {
        zero_collapsed as f64 / non_zero_source as f64
    } else {
        0.0
    };

    WeightValidationReport {
        rmse,
        nrmse,
        max_abs_error: max_abs_err,
        zero_collapse_ratio,
    }
}

/// Validate operator-space quality using an explicit set of activation vectors.
///
/// For each vector, computes:
///   - Reference matmul: Y = X @ W^T
///   - Quantized matmul using NF4 dequant
///   - RMSE, operator NRMSE, cosine similarity, norm ratio drift
///
/// Returns the worst-case metrics across all vectors.
pub fn validate_operator_space_with_vectors(
    source: &[f32],
    in_features: usize,
    out_features: usize,
    packed_codes: &[u8],
    packed_scales: &[f32],
    packed_biases: &[f32],
    scale_vector: Option<&[f32]>,
    vectors: &[Vec<f32>],
    // Pre-unpacked quantized weights. If `Some`, used directly instead of
    // unpacking from packed_codes. Required for non-NF4 formats (INT8, etc.).
    pre_unpacked: Option<&[f32]>,
    deadline: Option<std::time::Instant>,
) -> ValidationOutcome {
    let num_vectors = vectors.len();
    if num_vectors < ACCELERATE_BATCH_THRESHOLD {
        return validate_operator_space_single(
            source,
            in_features,
            out_features,
            packed_codes,
            packed_scales,
            packed_biases,
            scale_vector,
            vectors,
            pre_unpacked,
            deadline,
        )
    }

    // ── Batched Accelerate path ──────────────────────────────────
    // Flatten activation vectors to contiguous [num_vectors, in_features].
    let mut inputs_flat = Vec::with_capacity(num_vectors * in_features);
    for v in vectors {
        inputs_flat.extend_from_slice(v);
    }

    // Check deadline before the batched matmul (can't be interrupted mid-kernel).
    if let Some(dl) = deadline {
        if std::time::Instant::now() >= dl {
            return ValidationOutcome::Interrupted(InterruptedValidationReport {
                phase: "batched".to_string(),
                processed_vectors: 0,
                partial_rmse: 0.0,
                partial_nrmse: 0.0,
                partial_cosine: 0.0,
                partial_ref_rms: 0.0,
            });
    }
    }

    // Unpack NF4 weights to f32 once for the quantized path.
    #[cfg(any(feature = "metal-dispatch", feature = "mlx-backend", feature = "prism-backend", feature = "prism-backend-ios"))]
    let quant_weights: Vec<f32> = match pre_unpacked {
        Some(w) => w.to_vec(),
        None => unpack_nf4_weights(packed_codes, packed_scales, packed_biases, in_features, out_features),
    };

    // ── Batched path: GPU (metal-dispatch) or Accelerate (CPU) ──
    #[cfg(all(
        feature = "metal-dispatch",
        any(
            feature = "mlx-backend",
            feature = "prism-backend",
            feature = "prism-backend-ios",
            feature = "ffi"
        )
    ))]
    {
        use crate::compute_image::compile::kernel_dispatch::GpuBatchMatmulDispatcher;
        use crate::compute_image::compile::kernel_registry::KernelRegistry;
        use metal::Device;
        use parking_lot::Mutex;
        use std::sync::Arc;
        use std::sync::LazyLock;

        static GPU_INIT: LazyLock<GpuBatchMatmulDispatcher> = LazyLock::new(|| {
            let device = Device::system_default().expect("Metal device not available");
            let registry = Arc::new(Mutex::new(KernelRegistry::new(&device)));
            GpuBatchMatmulDispatcher::new(&registry)
        });

        let ref_results = GPU_INIT.run(source, &inputs_flat, in_features, out_features, num_vectors);
        let mut quant_results = GPU_INIT.run(&quant_weights, &inputs_flat, in_features, out_features, num_vectors);

        // Post-scale quantized outputs by reduction-axis scale vector.
        if let Some(sv) = scale_vector {
            for k in 0..num_vectors {
                let base = k * out_features;
                for j in 0..out_features {
                    quant_results[base + j] *= sv[j];
                }
            }
        }

        return ValidationOutcome::Completed(compute_operator_metrics(&ref_results, &quant_results, num_vectors, out_features));
    }

    // CPU fallback (also reachable when metal-dispatch is not enabled).
    #[cfg(all(not(feature = "metal-dispatch"), any(feature = "mlx-backend", feature = "prism-backend", feature = "prism-backend-ios")))]
    {
    // Batch reference matmul: C = A @ B
    //   A = inputs [num_vectors × in_features], lda = in_features
    //   B = source [in_features × out_features], ldb = out_features
    //   C = refs   [num_vectors × out_features], ldc = out_features
    let mut refs_flat = vec![0.0f32; num_vectors * out_features];
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            num_vectors as i32,
            out_features as i32,
            in_features as i32,
            1.0,
            inputs_flat.as_ptr(),
            in_features as i32,
            source.as_ptr(),
            out_features as i32,
            0.0,
            refs_flat.as_mut_ptr(),
            out_features as i32,
        );
    }

    // Batch quantized matmul: same layout, using unpacked NF4 weights.
    let mut quants_flat = vec![0.0f32; num_vectors * out_features];
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            num_vectors as i32,
            out_features as i32,
            in_features as i32,
            1.0,
            inputs_flat.as_ptr(),
            in_features as i32,
            quant_weights.as_ptr(),
            out_features as i32,
            0.0,
            quants_flat.as_mut_ptr(),
            out_features as i32,
        );
    }

    // Post-scale quantized outputs by reduction-axis scale vector.
    if let Some(sv) = scale_vector {
        for k in 0..num_vectors {
            let base = k * out_features;
            for j in 0..out_features {
                quants_flat[base + j] *= sv[j];
            }
        }
    }

    // Compute per-vector metrics from batched outputs.
    let mut worst_rmse = 0.0f32;
    let mut worst_nrmse = 0.0f32;
    let mut worst_norm_drift = 0.0f32;
    let mut total_cosine = 0.0f32;
    let mut worst_cos = 1.0f32;
    let mut accumulated_ref_rms = 0.0f32;
    let mut total_sign_agreement = 0.0f32;
    let nf = num_vectors as f32;

    for k in 0..num_vectors {
        let base = k * out_features;
        let ref_row = &refs_flat[base..base + out_features];
        let quant_row = &quants_flat[base..base + out_features];
 
        let mut sq = 0.0f32;
        let mut ref_sq = 0.0f32;
        let mut quant_sq = 0.0f32;
        let mut ref_dot_quant = 0.0f32;
        let mut sign_match = 0u32;
        for j in 0..out_features {
            let diff = quant_row[j] - ref_row[j];
            sq += diff * diff;
            ref_sq += ref_row[j] * ref_row[j];
            quant_sq += quant_row[j] * quant_row[j];
            ref_dot_quant += ref_row[j] * quant_row[j];
            if quant_row[j].is_sign_positive() == ref_row[j].is_sign_positive() {
                sign_match += 1;
            }
        }
        let trial_rmse = (sq / out_features as f32).sqrt();
        let ref_rms = (ref_sq / out_features as f32).sqrt();
        let trial_nrmse = trial_rmse / (ref_rms + 1e-30);
        let quant_rms = (quant_sq / out_features as f32).sqrt();
        let trial_norm_drift = (quant_rms / (ref_rms + 1e-30) - 1.0).abs();
        let trial_cosine = ref_dot_quant / ((ref_sq).sqrt() * (quant_sq).sqrt() + 1e-30);

        if trial_rmse > worst_rmse {
            worst_rmse = trial_rmse;
        }
        if trial_nrmse > worst_nrmse {
            worst_nrmse = trial_nrmse;
        }
        if trial_norm_drift > worst_norm_drift {
            worst_norm_drift = trial_norm_drift;
        }
        if trial_cosine < worst_cos {
            worst_cos = trial_cosine;
        }
        total_cosine += trial_cosine;
        accumulated_ref_rms += ref_rms;
        total_sign_agreement += sign_match as f32 / out_features as f32;
    }

    ValidationOutcome::Completed(OperatorValidationReport {
        rmse: worst_rmse,
        operator_nrmse: worst_nrmse,
        cosine_similarity: total_cosine / nf,
        worst_cosine: worst_cos,
        ref_output_rms: accumulated_ref_rms / nf,
        norm_ratio_drift: worst_norm_drift,
        sign_agreement: total_sign_agreement / nf,
    })
    } // #[cfg(not(feature = "metal-dispatch"))]

    // Pure-Rust fallback when neither Metal nor Accelerate is available.
    #[cfg(not(any(
        feature = "metal-dispatch",
        feature = "mlx-backend",
        feature = "prism-backend",
        feature = "prism-backend-ios"
    )))]
    {
        return validate_operator_space_single(
            source,
            in_features,
            out_features,
            packed_codes,
            packed_scales,
            packed_biases,
            scale_vector,
            vectors,
            pre_unpacked,
            deadline,
        )
    }
}

/// Compute per-vector operator metrics from batched reference and quantized outputs.
///
/// Both `refs_flat` and `quants_flat` are [num_vectors × out_features] row-major flat arrays.
/// Returns the aggregate `OperatorValidationReport` (worst-case metrics across all
/// vectors plus averages for cosine similarity, ref_output_rms, and sign agreement).
#[cfg(feature = "metal-dispatch")]
fn compute_operator_metrics(
    refs_flat: &[f32],
    quants_flat: &[f32],
    num_vectors: usize,
    out_features: usize,
) -> OperatorValidationReport {
    let mut worst_rmse = 0.0f32;
    let mut worst_nrmse = 0.0f32;
    let mut worst_norm_drift = 0.0f32;
    let mut total_cosine = 0.0f32;
    let mut worst_cos = 1.0f32;
    let mut accumulated_ref_rms = 0.0f32;
    let mut total_sign_agreement = 0.0f32;
    let nf = num_vectors as f32;

    for k in 0..num_vectors {
        let base = k * out_features;
        let ref_row = &refs_flat[base..base + out_features];
        let quant_row = &quants_flat[base..base + out_features];

        let mut sq = 0.0f32;
        let mut ref_sq = 0.0f32;
        let mut quant_sq = 0.0f32;
        let mut ref_dot_quant = 0.0f32;
        let mut sign_match = 0u32;
        for j in 0..out_features {
            let diff = quant_row[j] - ref_row[j];
            sq += diff * diff;
            ref_sq += ref_row[j] * ref_row[j];
            quant_sq += quant_row[j] * quant_row[j];
            ref_dot_quant += ref_row[j] * quant_row[j];
            if quant_row[j].is_sign_positive() == ref_row[j].is_sign_positive() {
                sign_match += 1;
            }
        }
        let trial_rmse = (sq / out_features as f32).sqrt();
        let ref_rms = (ref_sq / out_features as f32).sqrt();
        let trial_nrmse = trial_rmse / (ref_rms + 1e-30);
        let quant_rms = (quant_sq / out_features as f32).sqrt();
        let trial_norm_drift = (quant_rms / (ref_rms + 1e-30) - 1.0).abs();
        let trial_cosine = ref_dot_quant / ((ref_sq).sqrt() * (quant_sq).sqrt() + 1e-30);

        if trial_rmse > worst_rmse {
            worst_rmse = trial_rmse;
        }
        if trial_nrmse > worst_nrmse {
            worst_nrmse = trial_nrmse;
        }
        if trial_norm_drift > worst_norm_drift {
            worst_norm_drift = trial_norm_drift;
        }
        if trial_cosine < worst_cos {
            worst_cos = trial_cosine;
        }
        total_cosine += trial_cosine;
        accumulated_ref_rms += ref_rms;
        total_sign_agreement += sign_match as f32 / out_features as f32;
    }

    OperatorValidationReport {
        rmse: worst_rmse,
        operator_nrmse: worst_nrmse,
        cosine_similarity: total_cosine / nf,
        worst_cosine: worst_cos,
        ref_output_rms: accumulated_ref_rms / nf,
        norm_ratio_drift: worst_norm_drift,
        sign_agreement: total_sign_agreement / nf,
    }
}

/// Single-vector validation (used for small batches where Accelerate overhead dominates).
fn validate_operator_space_single(
    source: &[f32],
    in_features: usize,
    out_features: usize,
    packed_codes: &[u8],
    packed_scales: &[f32],
    packed_biases: &[f32],
    scale_vector: Option<&[f32]>,
    vectors: &[Vec<f32>],
    pre_unpacked: Option<&[f32]>,
    deadline: Option<std::time::Instant>,
) -> ValidationOutcome {
    let mut worst_rmse = 0.0f32;
    let mut worst_nrmse = 0.0f32;
    let mut worst_norm_drift = 0.0f32;
    let mut total_cosine = 0.0f32;
    let mut worst_cos = 1.0f32;
    let mut accumulated_ref_rms = 0.0f32;
    let mut total_sign_agreement = 0.0f32;

    // Running accumulators for interrupted reporting.
    let mut sum_sq_error: f64 = 0.0;
    let mut sum_dot_product: f64 = 0.0;
    let mut sum_ref_sq: f64 = 0.0;
    let mut sum_quant_sq: f64 = 0.0;
    let mut max_abs_err: f32 = 0.0;
    let mut processed: u32 = 0;

    let num_vectors = vectors.len().max(1) as f32;

    for chunk in vectors.chunks(VALIDATION_BATCH_SIZE) {
    for input in chunk {
        // Reference: Y = X @ W^T (original source weights, no scaling).
        let mut ref_out = vec![0.0f32; out_features];
        for j in 0..out_features {
            let mut sum = 0.0f32;
            for i in 0..in_features {
                sum += source[i * out_features + j] * input[i];
            }
            ref_out[j] = sum;
        }

        // NF4 dequant matmul via the reference CPU implementation.
        let mut nf4_out = vec![0.0f32; out_features];
        match pre_unpacked {
            Some(qw) => {
                for j in 0..out_features {
                    let mut sum = 0.0f32;
                    for i in 0..in_features {
                        sum += qw[i * out_features + j] * input[i];
                    }
                    nf4_out[j] = sum;
                }
            }
            None => {
                let _ = dequant_matmul_reference(
                    input,
                    packed_codes,
                    packed_scales,
                    packed_biases,
                    1,
                    in_features,
                    out_features,
                    &mut nf4_out,
                );
            }
        }

        // Post-scale by the reduction-axis scale vector (applies per output column).
        if let Some(sv) = scale_vector {
            for j in 0..out_features {
                nf4_out[j] *= sv[j];
            }
        }

        let mut sq = 0.0f32;
        let mut ref_sq = 0.0f32;
        let mut quant_sq = 0.0f32;
        let mut ref_dot_quant = 0.0f32;
        let mut sign_match = 0u32;
        for j in 0..out_features {
            let diff = nf4_out[j] - ref_out[j];
            let diff_abs = diff.abs();
            if diff_abs > max_abs_err {
                max_abs_err = diff_abs;
            }
            sq += diff * diff;
            ref_sq += ref_out[j] * ref_out[j];
            quant_sq += nf4_out[j] * nf4_out[j];
            ref_dot_quant += ref_out[j] * nf4_out[j];
            if nf4_out[j].is_sign_positive() == ref_out[j].is_sign_positive() {
                sign_match += 1;
            }
        }
        let trial_rmse = (sq / out_features as f32).sqrt();
        let ref_rms = (ref_sq / out_features as f32).sqrt();
        let trial_nrmse = trial_rmse / (ref_rms + 1e-30);
        let quant_rms = (quant_sq / out_features as f32).sqrt();
        let trial_norm_drift = (quant_rms / (ref_rms + 1e-30) - 1.0).abs();
        let trial_cosine = ref_dot_quant / ((ref_sq).sqrt() * (quant_sq).sqrt() + 1e-30);

        if trial_rmse > worst_rmse {
            worst_rmse = trial_rmse;
        }
        if trial_nrmse > worst_nrmse {
            worst_nrmse = trial_nrmse;
        }
        if trial_norm_drift > worst_norm_drift {
            worst_norm_drift = trial_norm_drift;
        }
        if trial_cosine < worst_cos {
            worst_cos = trial_cosine;
        }
        total_cosine += trial_cosine;
        accumulated_ref_rms += ref_rms;
        total_sign_agreement += sign_match as f32 / out_features as f32;

        // Update running accumulators for interrupted reporting.
        processed += 1;
        sum_sq_error += sq as f64;
        sum_dot_product += ref_dot_quant as f64;
        sum_ref_sq += ref_sq as f64;
        sum_quant_sq += quant_sq as f64;
    }

    // Check deadline after each batch of VALIDATION_BATCH_SIZE vectors.
    if let Some(dl) = deadline {
        if std::time::Instant::now() >= dl {
            let total_elems = processed as f32 * out_features as f32;
            let partial_rmse = (sum_sq_error as f32 / total_elems).sqrt();
            let partial_ref_rms = (sum_ref_sq as f32 / total_elems).sqrt();
            let partial_nrmse = partial_rmse / (partial_ref_rms + 1e-30);
            let partial_cosine = sum_dot_product as f32
                / ((sum_ref_sq as f32).sqrt() * (sum_quant_sq as f32).sqrt() + 1e-30);

            return ValidationOutcome::Interrupted(InterruptedValidationReport {
                phase: "single_vector".to_string(),
                processed_vectors: processed,
                partial_rmse,
                partial_nrmse,
                partial_cosine,
                partial_ref_rms,
            });
        }
    }

    // End of chunk loop — next iteration or fall through to final report.
    }

    ValidationOutcome::Completed(OperatorValidationReport {
        rmse: worst_rmse,
        operator_nrmse: worst_nrmse,
        cosine_similarity: total_cosine / num_vectors,
        worst_cosine: worst_cos,
        ref_output_rms: accumulated_ref_rms / num_vectors,
        norm_ratio_drift: worst_norm_drift,
        sign_agreement: total_sign_agreement / num_vectors,
    })
}

/// Validate operator-space (matmul) quality on a deterministic calibration
/// suite of 5 synthetic test vectors.
pub fn validate_operator_space(
    source: &[f32],
    in_features: usize,
    out_features: usize,
    packed_codes: &[u8],
    packed_scales: &[f32],
    packed_biases: &[f32],
    scale_vector: Option<&[f32]>,
    _profile: &QuantizationValidationProfile,
) -> OperatorValidationReport {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let mut rng_state = seed;

    let synthetic_vectors: Vec<Vec<f32>> = (0..5)
        .map(|trial| {
            let state = &mut rng_state;
            (0..in_features)
                .map(|i| match trial {
                    0 => (i as f64 * 0.1).sin() as f32,
                    1 => (i as f64 * 0.07).cos() as f32,
                    2 => (i as f64 * 0.03).sin() as f32 + (i as f64 * 0.05).cos() as f32,
                    3 | _ => {
                        *state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((*state >> 33) as f32) / (1u64 << 31) as f32
                    }
                })
                .collect()
        })
        .collect();

    match validate_operator_space_with_vectors(
        source,
        in_features,
        out_features,
        packed_codes,
        packed_scales,
        packed_biases,
        scale_vector,
        &synthetic_vectors,
        None,
        None,
    ) {
        ValidationOutcome::Completed(report) => report,
        // Never interrupted because deadline is None.
        ValidationOutcome::Interrupted(_) => unreachable!(),
    }
}

/// Validate operator-space quality using a calibration activation bank's vectors.
pub fn validate_operator_space_with_bank(
    source: &[f32],
    in_features: usize,
    out_features: usize,
    packed_codes: &[u8],
    packed_scales: &[f32],
    packed_biases: &[f32],
    scale_vector: Option<&[f32]>,
    vectors: &[Vec<f32>],
    pre_unpacked: Option<&[f32]>,
    deadline: Option<std::time::Instant>,
) -> ValidationOutcome {
    validate_operator_space_with_vectors(
        source,
        in_features,
        out_features,
        packed_codes,
        packed_scales,
        packed_biases,
        scale_vector,
        vectors,
        pre_unpacked,
        deadline,
    )
}

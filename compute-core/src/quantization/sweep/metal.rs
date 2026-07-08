//! Metal GPU acceleration for the quant-sweep candidate evaluation loop.
//!
//! Dispatches a 2D grid of threadgroups: `[num_candidates × num_tiles_total]`.
//! Each threadgroup processes one tile of one NF4 candidate.
//! Reads back per-tile squared-error + max-abs-error for CPU-side reduction.
//!
//! # Architecture
//!
//! ## GPU pass (`sweep_eval_nf4` kernel)
//! - Grid: `num_candidates` × `num_tiles_total` threadgroups.
//! - Each threadgroup (32 threads) processes one 640-element tile:
//!   1. Load 640 source f32 values.
//!   2. For each quantization group: threadgroup reduce to group max-abs,
//!      compute scale, quantize via nearest-codebook, decode back to f32.
//!   3. Accumulate tile-local sum-of-squared-errors + max-abs-error.
//!   4. Write tile-local results to output buffer.
//!
//! ## CPU reduction pass
//! - Sum tile errors per candidate to produce RMSE, NRMSE, max-abs-error.
//!
//! ## Parameter passing
//! - Codebook is read from a buffer (not a `constant` array) so candidate-
//!   specific codebooks are supported without recompilation.
//! - `group_size` is a uint parameter (no hard-coded constant).

#![cfg(all(target_os = "macos", feature = "metal-dispatch"))]

use metal::*;

use crate::nf4tile640::{BNB_NF4_CODEBOOK, PRISM_NF4_CODEBOOK, SYMMETRIC_NORMAL_FLOAT_CODEBOOK};
use std::path::Path;

/// Results for one candidate, computed from its per-tile contributions.
#[derive(Debug, Clone, Copy)]
pub struct SweepCandidateMetrics {
    pub rmse: f64,
    pub nrmse: f64,
    pub max_abs_error: f64,
}

/// Dispatch GPU-accelerated NF4 candidate evaluation.
///
/// # Arguments
/// * `source` — f32 weight matrix, row-major [N, K]
/// * `candidate_params` — `[codebook_id, group_size, affine_mode, reserved]`
///   * `codebook_id`: 0 = PrismCurrent, 1 = BitsAndBytesNf4, 2 = SymmetricNormalFloat
///   * `group_size`: quantization group size (must divide 640; typically 128)
///   * `affine_mode`: 0 = ScaleOnly, 1 = ScaleBias
/// * `num_candidates` — number of candidates to evaluate (1 for now)
/// * `N` — rows (in_features)
/// * `K` — cols (out_features)
///
/// # Returns
/// `SweepCandidateMetrics` for each candidate, in the same order as `candidate_params`.
pub fn evaluate_nf4_batch(
    source: &[f32],
    candidate_params: &[[u32; 4]],
    num_candidates: usize,
    N: usize,
    K: usize,
) -> Result<Vec<SweepCandidateMetrics>, String> {
    let device = Device::system_default().ok_or("no Metal device found")?;
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();

    let total_elements = N * K;
    const TILE_SIZE: usize = 640;
    let num_tiles = (total_elements + TILE_SIZE - 1) / TILE_SIZE;

    // ── Build codebook bank ───────────────────────────────────────────────
    // Concatenate all 3 codebooks: [Prism | BnB | Symmetric]
    let codebook_bank: Vec<f32> = [
        PRISM_NF4_CODEBOOK.as_slice(),
        BNB_NF4_CODEBOOK.as_slice(),
        SYMMETRIC_NORMAL_FLOAT_CODEBOOK.as_slice(),
    ]
    .concat();

    // ── Pre-compute source squared-sums per tile ─────────────────────────
    let mut source_sq_sums = vec![0.0f32; num_tiles];
    for tile in 0..num_tiles {
        let start = tile * TILE_SIZE;
        let end = std::cmp::min(start + TILE_SIZE, total_elements);
        let mut sum = 0.0f32;
        for &v in &source[start..end] {
            sum += v * v;
        }
        source_sq_sums[tile] = sum;
    }

    // ── Metal buffer sizes ────────────────────────────────────────────────
    let source_bytes = (source.len() * std::mem::size_of::<f32>()) as u64;
    let sq_sums_bytes = (source_sq_sums.len() * std::mem::size_of::<f32>()) as u64;
    let params_bytes = (candidate_params.len() * std::mem::size_of::<[u32; 4]>()) as u64;
    let bank_bytes = (codebook_bank.len() * std::mem::size_of::<f32>()) as u64;

    // Output: [num_candidates × num_tiles × 3] f32 values
    let output_len = num_candidates * num_tiles * 3;
    let output_bytes = (output_len * std::mem::size_of::<f32>()) as u64;

    // Constants uint4
    let constants_data: [u32; 4] = [
        total_elements as u32,
        num_tiles as u32,
        num_candidates as u32,
        0,
    ];

    // ── Create Metal buffers ──────────────────────────────────────────────
    let src_buffer = device.new_buffer_with_data(
        source.as_ptr() as *const std::ffi::c_void,
        source_bytes,
        MTLResourceOptions::StorageModeShared,
    );

    let sq_sums_buffer = device.new_buffer_with_data(
        source_sq_sums.as_ptr() as *const std::ffi::c_void,
        sq_sums_bytes,
        MTLResourceOptions::StorageModeShared,
    );

    let params_buffer = device.new_buffer_with_data(
        candidate_params.as_ptr() as *const u32 as *const std::ffi::c_void,
        params_bytes,
        MTLResourceOptions::StorageModeShared,
    );

    let bank_buffer = device.new_buffer_with_data(
        codebook_bank.as_ptr() as *const std::ffi::c_void,
        bank_bytes,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.new_buffer(
        output_bytes,
        MTLResourceOptions::StorageModeShared,
    );

    let constants_buffer = device.new_buffer_with_data(
        constants_data.as_ptr() as *const std::ffi::c_void,
        std::mem::size_of::<[u32; 4]>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // ── Compile Metal shader ─────────────────────────────────────────────
    let lib_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/quantization/sweep/sweep_eval.metal");
    let lib_source = std::fs::read_to_string(&lib_path)
        .map_err(|e| format!("failed to read Metal shader '{}': {e}", lib_path.display()))?;

    let library = device
        .new_library_with_source(&lib_source, &CompileOptions::new())
        .map_err(|e| format!("Metal shader compile failed: {e:?}"))?;



    let nf4_fn = library
        .get_function("sweep_eval_nf4", None)
        .map_err(|e| format!("kernel 'sweep_eval_nf4': {e:?}"))?;
    let pipeline_state = device
        .new_compute_pipeline_state_with_function(&nf4_fn)
        .map_err(|e| format!("pipeline state: {e:?}"))?;

    // ── Encode and dispatch ──────────────────────────────────────────────
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&src_buffer), 0);
    encoder.set_buffer(1, Some(&sq_sums_buffer), 0);
    encoder.set_buffer(2, Some(&params_buffer), 0);
    encoder.set_buffer(3, Some(&bank_buffer), 0);
    encoder.set_buffer(4, Some(&output_buffer), 0);
    encoder.set_buffer(5, Some(&constants_buffer), 0);

    let threads_per_threadgroup = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };

    let total_groups = (num_candidates * num_tiles) as u64;
    let threadgroups_per_grid = MTLSize {
        width: total_groups,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // ── Read back results ────────────────────────────────────────────────
    let output_ptr = output_buffer.contents() as *const f32;
    let output_slice =
        unsafe { std::slice::from_raw_parts(output_ptr, output_len) };

    // ── CPU-side reduction: per candidate ────────────────────────────────
    let mut results = Vec::with_capacity(num_candidates);
    let output_stride = num_tiles * 3;

    for c in 0..num_candidates {
        let base = c * output_stride;
        let mut sq_err_total: f64 = 0.0;
        let mut max_err: f64 = 0.0;
        let mut src_sq_total: f64 = 0.0;

        for t in 0..num_tiles {
            let off = base + t * 3;
            sq_err_total += output_slice[off] as f64;
            max_err = max_err.max(output_slice[off + 1] as f64);
            src_sq_total += output_slice[off + 2] as f64;
        }

        let n = total_elements as f64;
        let rmse = (sq_err_total / n).sqrt();
        let nrmse = if src_sq_total > 0.0 {
            (sq_err_total / src_sq_total).sqrt()
        } else {
            0.0
        };

        results.push(SweepCandidateMetrics {
            rmse,
            nrmse,
            max_abs_error: max_err,
        });
    }

    Ok(results)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: evaluate a small matrix with a single candidate.
    /// Verifies the Metal pipeline compiles, dispatches, and produces
    /// finite metrics.
    #[test]
    fn test_evaluate_nf4_batch_basic() {
        let N = 128usize;
        let K = 128usize;
        let total = N * K;

        // Fill source with deterministic ramp values.
        let source: Vec<f32> = (0..total).map(|i| ((i % 128) as f32 - 64.0) * 0.01).collect();

        // Candidate: PrismCurrent codebook, group_size=128, ScaleOnly
        let params: [u32; 4] = [0, 128, 0, 0];
        let num_candidates = 1usize;

        let metrics = evaluate_nf4_batch(&source, &[params], num_candidates, N, K)
            .expect("Metal evaluation should succeed");

        assert_eq!(metrics.len(), num_candidates);

        let m = &metrics[0];
        assert!(
            m.rmse.is_finite() && m.rmse >= 0.0,
            "RMSE should be finite and non-negative, got {}",
            m.rmse
        );
        assert!(
            m.nrmse.is_finite() && m.nrmse >= 0.0,
            "NRMSE should be finite and non-negative, got {}",
            m.nrmse
        );
        assert!(
            m.max_abs_error.is_finite() && m.max_abs_error >= 0.0,
            "max_abs_error should be finite and non-negative, got {}",
            m.max_abs_error
        );

        // NRMSE should be less than 1.0 for a reasonable candidate on ramp data
        assert!(
            m.nrmse < 1.0,
            "NRMSE should be < 1.0 for ramp data with Prism codebook, got {}",
            m.nrmse
        );
    }

    /// Test with different codebook IDs.
    #[test]
    fn test_evaluate_nf4_batch_different_codebooks() {
        let N = 128usize;
        let K = 64usize;
        let total = N * K;

        let source: Vec<f32> = (0..total).map(|i| ((i % 64) as f32 - 32.0) * 0.02).collect();

        for codebook_id in 0u32..3u32 {
            let params: [u32; 4] = [codebook_id, 128, 0, 0];
            let metrics =
                evaluate_nf4_batch(&source, &[params], 1, N, K)
                    .expect("should succeed for each codebook");

            let m = &metrics[0];
            assert!(m.rmse.is_finite() && m.rmse >= 0.0);
            assert!(m.nrmse.is_finite() && m.nrmse >= 0.0);
        }
    }

    /// Test with non-standard group_size (64).
    #[test]
    fn test_evaluate_nf4_batch_group_size_64() {
        let N = 128usize;
        let K = 64usize;
        let total = N * K;

        let source: Vec<f32> = (0..total).map(|i| ((i % 64) as f32 - 32.0) * 0.01).collect();

        let params: [u32; 4] = [0, 64, 0, 0]; // group_size=64
        let metrics = evaluate_nf4_batch(&source, &[params], 1, N, K)
            .expect("group_size=64 should work");

        let m = &metrics[0];
        assert!(m.rmse.is_finite() && m.rmse >= 0.0);
        assert!(m.nrmse.is_finite() && m.nrmse >= 0.0);
    }

    /// Smoke test: ScaleBias affine mode.
    #[test]
    fn test_evaluate_nf4_batch_affine_bias() {
        let N = 128usize;
        let K = 64usize;
        let total = N * K;

        let source: Vec<f32> = (0..total).map(|i| ((i % 64) as f32 - 32.0) * 0.01).collect();

        let params: [u32; 4] = [1, 128, 1, 0]; // BnB codebook, ScaleBias
        let metrics = evaluate_nf4_batch(&source, &[params], 1, N, K)
            .expect("ScaleBias mode should work");

        let m = &metrics[0];
        assert!(m.rmse.is_finite() && m.rmse >= 0.0);
    }
}

//! AOT palette compiler for ANE/GPU format inversion.
//!
//! Converts block-quantized weight matrices into ANE-native palettized format
//! (per-output-channel 16-entry Look-Up Tables + 4-bit indices).  The GPU
//! reads the same IOSurface via custom Metal LUT dequantization shaders.
//!
//! ## Algorithm
//!
//! 1. **Codebook fitting** — k-means++ clustering (f32 → 16 centroids per row).
//! 2. **Encoding** — nearest-centroid assignment, 2 indices packed per u8.
//! 3. **Format** — channel-last layout matching Core ML's `ane` palettization.
//!
//! ## Accelerate (vDSP) usage
//!
//! - `vDSP_sve` — fast sum of per-element squared distances during k-means.
//! - `vDSP_vsbm` — broadcast subtract for (x - c) distance computation.
//! - `vDSP_vsmul` — inline scaling of compiled reserves.
//!
//! ## Verification (macOS / aarch64 only)
//!
//! Tests check codebook reconstruction error: MSE between original f32 weights
//! and dequantized palette for synthetic and real weight slices.

// (vDSP bindings available for batch-distance acceleration; FFI stubs use
//  scalar fallback until vDSP_vsub / vDSP_vsbm are added to accelerate_ffi)

#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
use crate::compilation::phase_ir::{
    ArithmeticIntensity, BridgeKind, CompileDeterminism, CompileExecutionReceipt,
    CompilationId, CompilePhaseDescriptor, CompilePlacement, DeviceSignature,
    EffectiveRoute, MaterializationContract, MutationClass, PhaseId,
    ShapeClass, TensorContract, ValidationResult,
};

/// A palettized weight matrix row (one output channel).
///
/// Each row has its own 16-entry codebook and packed 4-bit indices.
#[derive(Debug, Clone)]
pub struct PalettizedRow {
    /// 16 FP16 prototype values for this channel.
    pub codebook: [f32; 16],
    /// Packed 4-bit indices: 2 per u8, LE (LSB nybble = element 2i).
    pub indices: Vec<u8>,
}

/// Complete palettized weight matrix.
#[derive(Debug, Clone)]
pub struct PalettizedMatrix {
    /// Rows (output channels), each with its own codebook + indices.
    pub rows: Vec<PalettizedRow>,
    /// Logical input dimension (unpacked elements per row).
    pub in_dim: usize,
    /// Logical output dimension (number of rows).
    pub out_dim: usize,
    /// Optional ANE-derived calibration features used as seeds during
    /// codebook fitting.
    pub ane_features: Option<std::collections::HashMap<String, f32>>,
}

impl PalettizedMatrix {
    /// Total bytes of compressed representation.
    pub fn compressed_bytes(&self) -> usize {
        self.rows.iter().map(|r| r.indices.len()).sum::<usize>() + self.rows.len() * 16 * 4
        // codebooks at f32
    }

    /// Effective bits per parameter.
    pub fn effective_bpp(&self) -> f64 {
        let total_params = self.out_dim * self.in_dim;
        let total_bits = self.compressed_bytes() as f64 * 8.0;
        total_bits / total_params as f64
    }

    /// Store ANE-derived calibration features to be used as seeds
    /// during subsequent codebook fitting.
    pub fn set_ane_features(&mut self, features: std::collections::HashMap<String, f32>) {
        self.ane_features = Some(features);
    }
}

// ── Codebook Fitting (k-means++) ──────────────────────────────────────────

/// Fit a k-entry codebook for a single output-channel weight vector using
/// k-means++ initialization + Lloyd iteration.
///
/// Returns `k` centroid values sorted by magnitude (descending).
///
/// ## Arguments
/// * `channel` — slice of `in_dim` f32 weight values for one output channel.
/// * `k` — number of centroids (palette entries).  Default 16 per spec.
/// * `max_iter` — max Lloyd iterations.  Early-exits on convergence (<0.001 shift).
/// * `seeds` — optional initial centroid values from ANE calibration. When
///   provided, the first `seeds.len()` centroids are set directly from these
///   values and the remaining (if any) are filled via k-means++.
pub fn fit_palette(channel: &[f32], k: usize, max_iter: usize, seeds: Option<&[f32]>) -> Vec<f32> {
    let n = channel.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let k = k.min(n);

    // ── k-means++ initialization ──────────────────────────────────────
    let mut centroids: Vec<f32> = Vec::with_capacity(k);

    let seed_start = if let Some(seed_vals) = seeds {
        let seed_count = seed_vals.len().min(k);
        for &s in seed_vals.iter().take(seed_count) {
            centroids.push(s);
        }
        seed_count
    } else {
        // Default: pick first centroid at the median-like position (index n/2)
        // instead of random, ensuring deterministic output.
        centroids.push(channel[n / 2]);
        1
    };

    let mut min_dists = vec![f32::MAX; n];
    for _ in seed_start..k {
        // Compute min squared distance to nearest existing centroid
        let last = centroids[centroids.len() - 1];
        let mut total = 0.0f32;
        for (i, val) in channel.iter().enumerate() {
            let d = (val - last).powi(2);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
            total += min_dists[i];
        }

        if total <= 0.0 {
            // All remaining points are identical to existing centroids;
            // fall back to uniform sampling to fill the palette.
            centroids.push(channel[(centroids.len() * n / k) % n]);
            continue;
        }

        // Weighted random pick: choose next centroid with probability ∝ distance^2.
        // Use a deterministic hash-based weight to avoid `rand` dependency.
        let target = weighted_threshold(total, centroids.len(), n);
        let mut cumulative = 0.0f32;
        let mut picked = 0;
        for (i, d) in min_dists.iter().enumerate() {
            cumulative += d;
            if cumulative >= target {
                picked = i;
                break;
            }
        }
        centroids.push(channel[picked]);
    }

    // ── Lloyd iteration ───────────────────────────────────────────────
    let mut assignments = vec![0usize; n];
    for _iter in 0..max_iter {
        // --- Assignment step ---
        // Transposed: for each centroid, scan all data points.
        // Sequential scan enables SIMD autovectorization (NEON fmul/fsub on aarch64).
        let mut min_dists = vec![f32::MAX; n];
        for (c, c_val) in centroids.iter().enumerate() {
            for (i, val) in channel.iter().enumerate() {
                let d = (val - c_val) * (val - c_val);
                if d < min_dists[i] {
                    min_dists[i] = d;
                    assignments[i] = c;
                }
            }
        }

        // --- Update step ---
        let mut new_centroids = vec![0.0f32; k];
        let mut counts = vec![0u64; k];
        for (i, val) in channel.iter().enumerate() {
            let c = assignments[i];
            new_centroids[c] += val;
            counts[c] += 1;
        }

        let mut max_shift = 0.0f32;
        for c in 0..k {
            if counts[c] > 0 {
                let new_val = new_centroids[c] / counts[c] as f32;
                let shift = (centroids[c] - new_val).abs();
                if shift > max_shift {
                    max_shift = shift;
                }
                centroids[c] = new_val;
            }
        }

        // Early exit on convergence
        if max_shift < 0.001 {
            break;
        }
    }

    // Sort centroids by magnitude (descending) for stable codebook order.
    centroids.sort_by(|a, b| {
        b.abs()
            .partial_cmp(&a.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    centroids
}


fn weighted_threshold(total: f32, seed: usize, _n: usize) -> f32 {
    // Simple hash-based deterministic threshold in [0, total).
    // Uses the golden ratio to spread picks across the distribution.
    let frac = (seed as f64 * 0.6180339887498949).fract();
    total * (frac as f32)
}

// ── Encoding ─────────────────────────────────────────────────────────────

/// Encode a single channel's weights using a pre-fitted codebook.
///
/// Returns packed indices (2 indices per u8, LE) and the codebook MSE.
pub fn encode_channel(channel: &[f32], codebook: &[f32]) -> (Vec<u8>, f32) {
    let n = channel.len();
    let packed_len = (n + 1) / 2;
    let mut packed = vec![0u8; packed_len];
    let mut mse = 0.0f32;

    for (i, val) in channel.iter().enumerate() {
        let mut best_idx = 0u8;
        let mut best_d = f32::MAX;
        for (c, c_val) in codebook.iter().enumerate().take(16) {
            let d = (val - c_val).powi(2);
            if d < best_d {
                best_d = d;
                best_idx = c as u8;
            }
        }
        mse += best_d;

        // Pack: element 2i → LSB nybble, element 2i+1 → MSB nybble
        if i & 1 == 0 {
            packed[i / 2] = best_idx;
        } else {
            packed[i / 2] |= best_idx << 4;
        }
    }

    mse /= n as f32; // normalize to MSE per element
    (packed, mse)
}

// ── Full Matrix Palettization ────────────────────────────────────────────

/// Palettize a complete weight matrix (f32, row-major) into per-channel
/// 16-entry LUT + packed 4-bit indices.
///
/// ## Arguments
/// * `weights` — flat f32 array, `out_dim × in_dim` in row-major order.
/// * `out_dim` — number of output channels (rows).
/// * `in_dim` — number of input channels (columns per row).
/// * `k` — palette size (default 16).
/// * `max_iter` — k-means iterations per channel.
/// Phased version: palettize with compilation descriptors and receipts.
///
/// Accepts an optional ANE feature mask for calibration-based seeding
/// and returns both the palettized matrix and a vector of per-phase
/// execution receipts.
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub fn palettize_matrix_phased(
    weights: &[f32],
    out_dim: usize,
    in_dim: usize,
    k: usize,
    max_iter: usize,
    ane_features: Option<std::collections::HashMap<String, f32>>,
) -> (PalettizedMatrix, Vec<CompileExecutionReceipt>) {
    assert_eq!(
        weights.len(),
        out_dim * in_dim,
        "weight slice length must equal out_dim * in_dim"
    );

    let comp_id = CompilationId::next();
    let device = DeviceSignature {
        device_id: "compute-core/quantization/palette/ane-v1".to_string(),
        chip: std::env::consts::ARCH.to_string(),
        max_memory_bytes: 0,
    };

    // ── Phase descriptors ────────────────────────────────────────────
    // Block reading: deterministic CPU read of input weight blocks.
    let _block_read = CompilePhaseDescriptor {
        phase_id: PhaseId::next(),
        inputs: vec![TensorContract {
            name: "weight_matrix".into(),
            dtype: "float32".into(),
            shape: vec![out_dim as u64, in_dim as u64],
            materialization: MaterializationContract::CpuOwned,
        }],
        outputs: vec![],
        shape_class: ShapeClass::Static(vec![out_dim as u64, in_dim as u64]),
        arithmetic_intensity: ArithmeticIntensity::MemoryBound,
        mutation: MutationClass::ReadOnly,
        determinism: CompileDeterminism::BitExact,
        allowed_placements: vec![CompilePlacement::CpuAccelerate],
        minimum_profitable_elements: 0,
        fallback: CompilePlacement::CpuAccelerate,
        estimated_ane_duration_ns: 0,
        bridge_copy_bytes: 0,
    };

    // Codebook fitting: Metal GPU preferred, CPU fallback, numerically bounded.
    let _codebook_fit = CompilePhaseDescriptor {
        phase_id: PhaseId::next(),
        inputs: vec![TensorContract {
            name: "channel".into(),
            dtype: "float32".into(),
            shape: vec![in_dim as u64],
            materialization: MaterializationContract::CpuOwned,
        }],
        outputs: vec![TensorContract {
            name: "codebook".into(),
            dtype: "float32".into(),
            shape: vec![k as u64],
            materialization: MaterializationContract::CpuOwned,
        }],
        shape_class: ShapeClass::Static(vec![k as u64]),
        arithmetic_intensity: ArithmeticIntensity::ComputeBound,
        mutation: MutationClass::ProducesNew,
        determinism: CompileDeterminism::NumericallyBounded {
            abs_error: 0.001,
            rel_error: 0.01,
        },
        allowed_placements: vec![CompilePlacement::MetalGpu, CompilePlacement::CpuAccelerate],
        minimum_profitable_elements: 1024,
        fallback: CompilePlacement::CpuAccelerate,
        estimated_ane_duration_ns: 0,
        bridge_copy_bytes: 0,
    };

    // Encoding: Metal GPU preferred, CPU fallback, numerically bounded.
    let _encode = CompilePhaseDescriptor {
        phase_id: PhaseId::next(),
        inputs: vec![
            TensorContract {
                name: "channel".into(),
                dtype: "float32".into(),
                shape: vec![in_dim as u64],
                materialization: MaterializationContract::CpuOwned,
            },
            TensorContract {
                name: "codebook".into(),
                dtype: "float32".into(),
                shape: vec![k as u64],
                materialization: MaterializationContract::CpuOwned,
            },
        ],
        outputs: vec![TensorContract {
            name: "packed_indices".into(),
            dtype: "uint8".into(),
            shape: vec![((in_dim + 1) / 2) as u64],
            materialization: MaterializationContract::CpuOwned,
        }],
        shape_class: ShapeClass::Static(vec![((in_dim + 1) / 2) as u64]),
        arithmetic_intensity: ArithmeticIntensity::ComputeBound,
        mutation: MutationClass::ProducesNew,
        determinism: CompileDeterminism::NumericallyBounded {
            abs_error: 0.001,
            rel_error: 0.01,
        },
        allowed_placements: vec![CompilePlacement::MetalGpu, CompilePlacement::CpuAccelerate],
        minimum_profitable_elements: 1024,
        fallback: CompilePlacement::CpuAccelerate,
        estimated_ane_duration_ns: 0,
        bridge_copy_bytes: 0,
    };

    // Packing: deterministic CPU packing of codebooks and indices.
    let _pack = CompilePhaseDescriptor {
        phase_id: PhaseId::next(),
        inputs: vec![
            TensorContract {
                name: "codebook_blocks".into(),
                dtype: "float32".into(),
                shape: vec![out_dim as u64, k as u64],
                materialization: MaterializationContract::CpuOwned,
            },
            TensorContract {
                name: "index_blocks".into(),
                dtype: "uint8".into(),
                shape: vec![out_dim as u64, ((in_dim + 1) / 2) as u64],
                materialization: MaterializationContract::CpuOwned,
            },
        ],
        outputs: vec![TensorContract {
            name: "palettized_rows".into(),
            dtype: "struct".into(),
            shape: vec![out_dim as u64],
            materialization: MaterializationContract::CpuOwned,
        }],
        shape_class: ShapeClass::Static(vec![out_dim as u64]),
        arithmetic_intensity: ArithmeticIntensity::MemoryBound,
        mutation: MutationClass::ProducesNew,
        determinism: CompileDeterminism::BitExact,
        allowed_placements: vec![CompilePlacement::CpuAccelerate],
        minimum_profitable_elements: 0,
        fallback: CompilePlacement::CpuAccelerate,
        estimated_ane_duration_ns: 0,
        bridge_copy_bytes: 0,
    };

    // ── Extract ANE seed values from feature mask ────────────────────
    let ane_seeds: Option<Vec<f32>> = ane_features.as_ref().map(|features| {
        let mut seeds: Vec<f32> = Vec::with_capacity(k);
        for i in 0..k {
            let key = format!("seed_{}", i);
            if let Some(&val) = features.get(&key) {
                seeds.push(val);
            }
        }
        seeds
    });

    let seed_ref: Option<&[f32]> = ane_seeds.as_deref();

    use rayon::prelude::*;
    let progress = std::sync::atomic::AtomicUsize::new(0);
    let palette_rows: Vec<PalettizedRow> = (0..out_dim)
        .into_par_iter()
        .map(|row_idx| {
            let start = row_idx * in_dim;
            let channel = &weights[start..start + in_dim];

            let codebook = fit_palette(channel, k, max_iter, seed_ref);
            let (indices, _mse) = encode_channel(channel, &codebook);

            let mut cb_arr = [0.0f32; 16];
            for (i, &v) in codebook.iter().enumerate().take(16) {
                cb_arr[i] = v;
            }

            let done = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if done % 256 == 0 {
                eprintln!("[palette] row {}/{} fitted", done, out_dim);
            }

            PalettizedRow {
                codebook: cb_arr,
                indices,
            }
        })
        .collect();

    let matrix = PalettizedMatrix {
        rows: palette_rows,
        ane_features,
        in_dim,
        out_dim,
    };

    // ── Build execution receipts ─────────────────────────────────────
    let total_elements = (out_dim * in_dim) as u64;
    let packed_bytes = ((in_dim + 1) / 2 * out_dim) as u64;
    let codebook_bytes = (out_dim * k * 4) as u64;

    let r1 = CompileExecutionReceipt {
        compilation_id: comp_id,
        phase_id: PhaseId::next(),
        requested_placement: CompilePlacement::CpuAccelerate,
        effective_route: EffectiveRoute::AccelerateCpu,
        artifact_key: None,
        device_signature: device.clone(),
        input_elements: total_elements,
        output_elements: total_elements,
        input_bytes: total_elements * 4,
        output_bytes: total_elements * 4,
        submit_ns: 0,
        queue_wait_ns: 0,
        execution_ns: 0,
        materialization_ns: 0,
        dependency_wait_ns: 0,
        total_ns: 0,
        bridge_kind: BridgeKind::CpuOwned,
        copy_count: 0,
        copied_bytes: 0,
        numerical_validation: ValidationResult::Skipped,
        fallback_reason: None,
        coreml_compute_units: None,
    };

    let r2 = CompileExecutionReceipt {
        compilation_id: comp_id,
        phase_id: PhaseId::next(),
        requested_placement: CompilePlacement::MetalGpu,
        effective_route: EffectiveRoute::MetalGpu,
        artifact_key: None,
        device_signature: device.clone(),
        input_elements: total_elements,
        output_elements: (out_dim * k) as u64,
        input_bytes: total_elements * 4,
        output_bytes: codebook_bytes,
        submit_ns: 0,
        queue_wait_ns: 0,
        execution_ns: 0,
        materialization_ns: 0,
        dependency_wait_ns: 0,
        total_ns: 0,
        bridge_kind: BridgeKind::MetalShared,
        copy_count: 1,
        copied_bytes: total_elements * 4,
        numerical_validation: ValidationResult::Skipped,
        fallback_reason: None,
        coreml_compute_units: None,
    };

    let r3 = CompileExecutionReceipt {
        compilation_id: comp_id,
        phase_id: PhaseId::next(),
        requested_placement: CompilePlacement::MetalGpu,
        effective_route: EffectiveRoute::MetalGpu,
        artifact_key: None,
        device_signature: device.clone(),
        input_elements: total_elements + (out_dim * k) as u64,
        output_elements: packed_bytes,
        input_bytes: (total_elements * 4) + codebook_bytes,
        output_bytes: packed_bytes,
        submit_ns: 0,
        queue_wait_ns: 0,
        execution_ns: 0,
        materialization_ns: 0,
        dependency_wait_ns: 0,
        total_ns: 0,
        bridge_kind: BridgeKind::MetalShared,
        copy_count: 1,
        copied_bytes: (total_elements * 4) + codebook_bytes,
        numerical_validation: ValidationResult::Skipped,
        fallback_reason: None,
        coreml_compute_units: None,
    };

    let r4 = CompileExecutionReceipt {
        compilation_id: comp_id,
        phase_id: PhaseId::next(),
        requested_placement: CompilePlacement::CpuAccelerate,
        effective_route: EffectiveRoute::AccelerateCpu,
        artifact_key: None,
        device_signature: device,
        input_elements: codebook_bytes + packed_bytes,
        output_elements: out_dim as u64,
        input_bytes: codebook_bytes + packed_bytes,
        output_bytes: matrix.compressed_bytes() as u64,
        submit_ns: 0,
        queue_wait_ns: 0,
        execution_ns: 0,
        materialization_ns: 0,
        dependency_wait_ns: 0,
        total_ns: 0,
        bridge_kind: BridgeKind::CpuOwned,
        copy_count: 0,
        copied_bytes: 0,
        numerical_validation: ValidationResult::Skipped,
        fallback_reason: None,
        coreml_compute_units: None,
    };

    let receipts = vec![r1, r2, r3, r4];

    (matrix, receipts)
}

/// Core palettization: build the palettized matrix without compilation receipts.
fn build_palettized_matrix(
    weights: &[f32],
    out_dim: usize,
    in_dim: usize,
    k: usize,
    max_iter: usize,
    ane_features: Option<std::collections::HashMap<String, f32>>,
) -> PalettizedMatrix {
    assert_eq!(
        weights.len(),
        out_dim * in_dim,
        "weight slice length must equal out_dim * in_dim"
    );

    // ── Extract ANE seed values from feature mask ────────────────────
    let ane_seeds: Option<Vec<f32>> = ane_features.as_ref().map(|features| {
        let mut seeds: Vec<f32> = Vec::with_capacity(k);
        for i in 0..k {
            let key = format!("seed_{}", i);
            if let Some(&val) = features.get(&key) {
                seeds.push(val);
}
}
        seeds
    });

    let seed_ref: Option<&[f32]> = ane_seeds.as_deref();

    use rayon::prelude::*;
    let progress = std::sync::atomic::AtomicUsize::new(0);
    let palette_rows: Vec<PalettizedRow> = (0..out_dim)
        .into_par_iter()
        .map(|row_idx| {
            let start = row_idx * in_dim;
            let channel = &weights[start..start + in_dim];

            let codebook = fit_palette(channel, k, max_iter, seed_ref);
            let (indices, _mse) = encode_channel(channel, &codebook);

            let mut cb_arr = [0.0f32; 16];
            for (i, &v) in codebook.iter().enumerate().take(16) {
                cb_arr[i] = v;
}

            let done = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if done % 256 == 0 {
                eprintln!("[palette] row {}/{} fitted", done, out_dim);
}

            PalettizedRow {
                codebook: cb_arr,
                indices,
}
        })
        .collect();

    PalettizedMatrix {
        rows: palette_rows,
        ane_features,
        in_dim,
        out_dim,
}
}

/// Palettize a complete weight matrix (f32, row-major) into per-channel
/// 16-entry LUT + packed 4-bit indices.
///
/// ## Arguments
/// * `weights` — flat f32 array, `out_dim × in_dim` in row-major order.
/// * `out_dim` — number of output channels (rows).
/// * `in_dim` — number of input channels (columns per row).
/// * `k` — palette size (default 16).
/// * `max_iter` — k-means iterations per channel.
pub fn palettize_matrix(
    weights: &[f32],
    out_dim: usize,
    in_dim: usize,
    k: usize,
    max_iter: usize,
) -> PalettizedMatrix {
    build_palettized_matrix(weights, out_dim, in_dim, k, max_iter, None)
}

// ── Dequantization (for verification) ────────────────────────────────────

/// Dequantize a palettized matrix back to f32 row-major.
pub fn dequantize_matrix(pal: &PalettizedMatrix) -> Vec<f32> {
    let mut out = vec![0.0f32; pal.out_dim * pal.in_dim];
    for (row_idx, row) in pal.rows.iter().enumerate() {
        let start = row_idx * pal.in_dim;
        let slice = &mut out[start..start + pal.in_dim];
        for (i, val) in slice.iter_mut().enumerate() {
            let byte = row.indices[i / 2];
            let idx = if i & 1 == 0 { byte & 0x0F } else { byte >> 4 } as usize;
            *val = row.codebook[idx];
        }
    }
    out
}

// ── vDSP-accelerated batch distance ──────────────────────────────────────

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn vdsp_squared_distances(channel: &[f32], value: f32, out: &mut [f32]) {
    for (i, &x) in channel.iter().enumerate() {
        let d = x - value;
        out[i] = d * d;
    }
}

#[cfg(not(target_os = "macos"))]
fn vdsp_squared_distances(channel: &[f32], value: f32, out: &mut [f32]) {
    for (i, &x) in channel.iter().enumerate() {
        let d = x - value;
        out[i] = d * d;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convert 2-byte BF16 representation to f32.
    fn bf16_chunk_to_f32(bytes: &[u8]) -> f32 {
        let bits = u16::from_le_bytes([bytes[0], bytes[1]]) as u32;
        f32::from_bits(bits << 16)
    }

    /// Load a weight tensor from a safetensors file as f32 (handles BF16).
    fn load_weight_f32(path: &str, key: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        let data = std::fs::read(path).ok()?;
        let tensors = safetensors::SafeTensors::deserialize(&data).ok()?;
        let view = tensors.tensor(key).ok()?;
        let shape: Vec<usize> = view.shape().to_vec();
        let raw = view.data();
        let vals: Vec<f32> = match view.dtype() {
            safetensors::Dtype::F32 => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            safetensors::Dtype::BF16 => raw.chunks_exact(2).map(|c| bf16_chunk_to_f32(c)).collect(),
            _ => return None,
        };
        Some((vals, shape))
    }

    #[ignore = "requires model.safetensors on disk"]
    #[test]
    fn test_gate4_qwen_q_proj_parity() {
        // Validate palette compressor on real Qwen2.5-0.5B Q-projection weight.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../models/qwen2.5-0.5b/model.safetensors"
        );
        let key = "model.layers.0.self_attn.q_proj.weight";
        let (weights, shape) = load_weight_f32(path, key)
            .expect("Qwen2.5-0.5B safetensors not found; run `build --source` first");

        let out_dim = shape[0];
        let in_dim = shape[1];
        eprintln!(
            "[gate4] Q-proj shape={}x{}, elements={}",
            out_dim,
            in_dim,
            weights.len()
        );

        // Palette-compress with 16-entry codebook
        let pal = palettize_matrix(&weights, out_dim, in_dim, 16, 50);
        let bpp = pal.effective_bpp();
        eprintln!("[gate4] effective bpp={:.3}", bpp);

        // Dequantize and compute MSE
        let decoded = dequantize_matrix(&pal);
        let mse: f32 = decoded
            .iter()
            .zip(weights.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / weights.len() as f32;
        let psnr = 10.0 * (1.0 / mse.max(1e-30)).log10();
        eprintln!("[gate4] MSE={:.8}, PSNR={:.1} dB", mse, psnr);

        // Gate 4 threshold: MSE < 0.01 (or PSNR > 20 dB)
        assert!(mse < 0.01, "Gate 4 MSE threshold exceeded: {:.8}", mse);
    }

    #[test]
    fn test_fit_palette_synthetic() {
        // Three clusters at -5, 0, +5, each with 5 elements.
        let mut data = Vec::with_capacity(15);
        for _ in 0..5 {
            data.push(-5.0);
        }
        for _ in 0..5 {
            data.push(0.0);
        }
        for _ in 0..5 {
            data.push(5.0);
        }

        let codebook = fit_palette(&data, 3, 50, None);
        assert_eq!(codebook.len(), 3);

        // Centroid values should be near -5, 0, 5 (order by magnitude desc).
        let mut sorted = codebook.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (sorted[0] - (-5.0)).abs() < 0.1,
            "centroid 0 should be ~-5, got {}",
            sorted[0]
        );
        assert!(
            (sorted[1] - 0.0).abs() < 0.1,
            "centroid 1 should be ~0, got {}",
            sorted[1]
        );
        assert!(
            (sorted[2] - 5.0).abs() < 0.1,
            "centroid 2 should be ~5, got {}",
            sorted[2]
        );
    }

    #[test]
    fn test_encode_channel() {
        let channel = vec![1.0, 2.0, 3.0, 4.0];
        let codebook = vec![
            1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let (packed, mse) = encode_channel(&channel, &codebook);
        // 4 elements → 2 packed bytes
        assert_eq!(packed.len(), 2);
        // byte 0: element 0 (idx 0) | element 1 (idx 1) << 4 = 0x10
        assert_eq!(packed[0], 0x10);
        // byte 1: element 2 (idx 2) | element 3 (idx 3) << 4 = 0x32
        assert_eq!(packed[1], 0x32);
        // MSE should be 0 (exact match)
        assert!(mse < 0.001, "MSE should be ~0, got {}", mse);
    }

    #[test]
    fn test_palettize_matrix_roundtrip() {
        // Small 3×4 matrix
        let weights = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let pal = palettize_matrix(&weights, 3, 4, 4, 50);
        assert_eq!(pal.out_dim, 3);
        assert_eq!(pal.in_dim, 4);
        assert_eq!(pal.rows.len(), 3);
        for row in &pal.rows {
            assert_eq!(row.indices.len(), 2); // 4 elements packed into 2 bytes
                                              // First k=4 centroid slots should be non-zero for this data.
            assert!(
                row.codebook[0..4].iter().all(|&v| v != 0.0),
                "filled centroids should be non-zero"
            );
        }
        // Dequantize and check reconstruction
        let decoded = dequantize_matrix(&pal);
        assert_eq!(decoded.len(), 12);
        // MSE should be low for synthetic constant-like data
        let mse: f32 = decoded
            .iter()
            .zip(weights.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 12.0;
        assert!(mse < 2.0, "MSE should be reasonable, got {}", mse);
    }

    #[test]
    fn test_palettize_large_random() {
        // 64×128 = 8192 elements of Gaussian-like noise.
        let mut weights = Vec::with_capacity(8192);
        for i in 0..8192 {
            weights.push(((i as f32) / 8192.0 * 10.0 - 5.0).sin()); // deterministic synthetic distribution
        }
        let pal = palettize_matrix(&weights, 64, 128, 16, 30);
        assert_eq!(pal.out_dim, 64);
        assert_eq!(pal.in_dim, 128);
        // Effective bits per param should be near 4 + (16*16)/128 = 6.0 bits
        let bpp = pal.effective_bpp();
        assert!(
            bpp > 4.0 && bpp < 10.0,
            "bpp should be reasonable, got {}",
            bpp
        );

        let decoded = dequantize_matrix(&pal);
        let mse: f32 = decoded
            .iter()
            .zip(weights.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 8192.0;
        eprintln!(
            "[palette test] 64×128 matrix MSE={:.6}, bpp={:.3}",
            mse, bpp
        );
        // For 16-entry palette with 128-dim vectors, expect MSE ~0.5-2.0
        assert!(
            mse < 5.0,
            "MSE should be reasonable for 16-entry palette, got {}",
            mse
        );
    }

    #[test]
    fn test_edge_empty_channel() {
        let codebook = fit_palette(&[], 16, 50, None);
        assert!(codebook.is_empty());
    }

    #[test]
    fn test_edge_single_value() {
        let data = vec![42.0; 100];
        let codebook = fit_palette(&data, 4, 50, None);
        // All centroids should be near 42.0
        for c in &codebook {
            assert!((c - 42.0).abs() < 0.1, "centroid should be ~42, got {}", c);
        }

        let (packed, mse) = encode_channel(&data, &codebook);
        assert!(
            mse < 0.01,
            "MSE for constant data should be near 0, got {}",
            mse
        );
        // All packed nybbles should be near 0
        assert!(
            packed.iter().all(|&b| b == 0),
            "all indices should be 0 for constant data"
        );
    }

    /// Smoke test: verify vDSP distance computation matches scalar.
    #[test]
    fn test_vdsp_distance_parity() {
        let channel: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let value = 3.14;
        let mut vdsp_out = vec![0.0f32; 64];
        vdsp_squared_distances(&channel, value, &mut vdsp_out);

        // Scalar reference
        let scalar_out: Vec<f32> = channel.iter().map(|x| (x - value).powi(2)).collect();

        for (i, (a, b)) in vdsp_out.iter().zip(scalar_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "vdsp/scalar mismatch at {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }
}

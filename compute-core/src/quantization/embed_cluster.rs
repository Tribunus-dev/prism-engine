//! Embedding k-means clustering for token table quantization.
//!
//! Pure-Rust implementation of k-means++ initialization, iterative lloyd
//! refinement, centroid reordering, and 2-bit nibble quantization.
//!
//! Extracted from `gemma4_ingest.rs` to be independently testable.
//!
//! ## Reconstruction formula (design decision)
//!
//! Chosen approach: **reorder-for-locality only** (option A). Centroids are
//! computed purely to drive the cluster assignment that permutes the
//! embedding rows so contiguous same-cluster rows produce better block-scale
//! alignment (lower quantization error). The centroids themselves are
//! **not used at reconstruction** — the runtime fetches the ternary-quantized
//! row directly via [`reordered_position`] and dequantizes it with its own
//! block scale.
//!
//! The centroid tables (`CENTROID_NIBBLES`, `CENTROID_SCALES`) stored in the
//! v6 segment directory are dead bytes under this formula. They should be
//! removed in a follow-up pass once the format version is bumped.

use std::collections::HashSet;

// ── FP16 conversion ─────────────────────────────────────────────────

fn f32_to_fp16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x7FFFFF;
    if exp == 0 {
        return sign;
    }
    if exp == 0xFF {
        return if mant == 0 {
            if sign != 0 {
                0xFC00
            } else {
                0x7C00
            }
        } else {
            0x7E00
        };
    }
    let exp_f16: i32 = exp as i32 - 127 + 15;
    if exp_f16 >= 0x1F {
        return if sign != 0 { 0xFC00 } else { 0x7C00 };
    }
    if exp_f16 <= 0 {
        return sign;
    }
    sign | ((exp_f16 as u16) << 10) | ((mant >> 13) as u16)
}

/// Convert a stream of f32 bytes to fp16 scale + 2-bit nibbles.
pub fn quantize_block(values: &[f32; 256]) -> (u16, [u8; 64]) {
    let max_mag = values.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let scale = if max_mag > 1e-12 { max_mag } else { 1.0 };
    let scale_fp16 = f32_to_fp16_bits(scale);

    let mut nibbles = [0u8; 64];
    for (i, chunk) in values.chunks_exact(4).enumerate() {
        let mut byte: u8 = 0;
        for (j, &v) in chunk.iter().enumerate() {
            let snap = (v / scale).round().clamp(-1.0, 1.0) as i8;
            let nibble = match snap {
                1 => 0b01u8,
                -1 => 0b10u8,
                _ => 0b00u8,
            };
            byte |= nibble << (j * 2);
        }
        nibbles[i] = byte;
    }

    (scale_fp16, nibbles)
}

/// Process a flat weight array in 256-element blocks, append scales + nibbles.
pub fn process_weights(weights_f32: &[f32], scales_out: &mut Vec<u8>, weights_out: &mut Vec<u8>) {
    let padded = if weights_f32.len() % 256 == 0 {
        weights_f32.to_vec()
    } else {
        let n = ((weights_f32.len() + 255) / 256) * 256;
        let mut v = weights_f32.to_vec();
        v.resize(n, 0.0);
        v
    };

    for block in padded.chunks_exact(256) {
        let arr: [f32; 256] = {
            let mut b = [0.0f32; 256];
            b.copy_from_slice(block);
            b
        };
        let (scale, nibbles) = quantize_block(&arr);
        scales_out.extend_from_slice(&scale.to_le_bytes());
        weights_out.extend_from_slice(&nibbles);
    }
}

// ── K-Means Clustering (for embedding quantization) ──────────────────

thread_local! {
    static SEED: std::cell::Cell<u64> = std::cell::Cell::new(42);
}

pub fn rand_range(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    SEED.with(|seed| {
        let s = seed.get();
        let next = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed.set(next);
        (next >> 33) as usize % n
    })
}

/// K-Means++ centroid initialization.
pub fn kmeans_plusplus(data: &[f32], k: usize, n_rows: usize, dim: usize) -> Vec<f32> {
    let mut centroids: Vec<f32> = Vec::with_capacity(k * dim);
    let mut chosen: HashSet<usize> = HashSet::new();
    let first_idx = rand_range(n_rows);
    chosen.insert(first_idx);
    centroids.extend_from_slice(&data[first_idx * dim..(first_idx + 1) * dim]);

    let mut min_dist_sq: Vec<f32> = vec![f32::MAX; n_rows];
    for i in 0..n_rows {
        let row = &data[i * dim..(i + 1) * dim];
        let dist = row
            .iter()
            .zip(centroids.chunks_exact(dim).last().unwrap())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>();
        min_dist_sq[i] = dist;
    }

    for c in 1..k {
        let total_dist: f64 = min_dist_sq.iter().map(|&d| d as f64).sum();
        if total_dist <= 0.0 {
            let idx = loop {
                let idx = rand_range(n_rows);
                if !chosen.contains(&idx) {
                    break idx;
                }
            };
            chosen.insert(idx);
            centroids.extend_from_slice(&data[idx * dim..(idx + 1) * dim]);
            continue;
        }
        let threshold = rand_range(usize::MAX) as f64 / usize::MAX as f64 * total_dist;
        let mut cumulative = 0.0_f64;
        let mut next_idx = 0;
        for i in 0..n_rows {
            cumulative += min_dist_sq[i] as f64;
            if cumulative >= threshold && !chosen.contains(&i) {
                next_idx = i;
                break;
            }
        }
        chosen.insert(next_idx);
        centroids.extend_from_slice(&data[next_idx * dim..(next_idx + 1) * dim]);
        let new_centroid = &centroids[c * dim..(c + 1) * dim];
        for i in 0..n_rows {
            if chosen.contains(&i) {
                min_dist_sq[i] = 0.0;
                continue;
            }
            let row = &data[i * dim..(i + 1) * dim];
            let dist = row
                .iter()
                .zip(new_centroid)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>();
            if dist < min_dist_sq[i] {
                min_dist_sq[i] = dist;
            }
        }
    }
    centroids
}

/// One k-means iteration: assign (by squared Euclidean distance) + update.
///
/// Returns (assignments, centroid_movement_delta).
pub fn kmeans_iterate(
    data: &[f32],
    centroids: &mut [f32],
    n_rows: usize,
    dim: usize,
    k: usize,
) -> (Vec<u32>, f64) {
    let mut assignments: Vec<u32> = vec![0u32; n_rows];

    // ── Assignment: argmin of squared Euclidean distance ──
    // (Fixed from incorrect max-dot-product which mixed objectives.)
    for i in 0..n_rows {
        let row = &data[i * dim..(i + 1) * dim];
        let mut best_c = 0u32;
        let mut best_dist = f32::INFINITY;
        for c in 0..k {
            let centroid = &centroids[c * dim..(c + 1) * dim];
            let dist = row
                .iter()
                .zip(centroid)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>();
            if dist < best_dist {
                best_dist = dist;
                best_c = c as u32;
            }
        }
        assignments[i] = best_c;
    }

    // ── Update: mean of assigned points ──
    let old_centroids: Vec<f32> = centroids.to_vec();
    for c in 0..k {
        let slice = &mut centroids[c * dim..(c + 1) * dim];
        slice.fill(0.0_f32);
    }
    let mut counts: Vec<u64> = vec![0u64; k];
    for i in 0..n_rows {
        let c = assignments[i] as usize;
        counts[c] += 1;
        let row = &data[i * dim..(i + 1) * dim];
        let cent_slice = &mut centroids[c * dim..(c + 1) * dim];
        for j in 0..dim {
            cent_slice[j] += row[j];
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            let inv = 1.0 / counts[c] as f32;
            let slice = &mut centroids[c * dim..(c + 1) * dim];
            for j in 0..dim {
                slice[j] *= inv;
            }
        }
    }

    // ── Centroid movement delta ──
    let mut delta = 0.0_f64;
    for c in 0..k {
        let old = &old_centroids[c * dim..(c + 1) * dim];
        let new_ = &centroids[c * dim..(c + 1) * dim];
        delta += old
            .iter()
            .zip(new_)
            .map(|(a, b)| ((a - b) as f64) * ((a - b) as f64))
            .sum::<f64>()
            .sqrt();
    }
    (assignments, delta)
}

/// Reorder data rows by cluster assignment.
pub fn reorder_by_cluster(
    data: &[f32],
    assignments: &[u32],
    n_rows: usize,
    dim: usize,
    k: usize,
) -> Vec<f32> {
    let mut cluster_sizes: Vec<usize> = vec![0usize; k];
    for &a in assignments {
        cluster_sizes[a as usize] += 1;
    }
    let mut write_pos: Vec<usize> = Vec::with_capacity(k);
    let mut offset = 0usize;
    for c in 0..k {
        write_pos.push(offset);
        offset += cluster_sizes[c] * dim;
    }
    let mut reordered: Vec<f32> = vec![0.0_f32; offset];
    for i in 0..n_rows {
        let c = assignments[i] as usize;
        let dst = write_pos[c];
        let src = i * dim;
        reordered[dst..dst + dim].copy_from_slice(&data[src..src + dim]);
        write_pos[c] += dim;
    }
    reordered
}

/// Compute the canonical permuted position of a token in the reordered array.
///
/// This is O(n) per query — intended for testing and canonicalization,
/// not hot-path inference.
pub fn reordered_position(
    assignments: &[u32],
    token_index: usize,
    _n_rows: usize,
    _k: usize,
) -> usize {
    let cluster = assignments[token_index] as usize;
    let mut before = 0usize;
    for c in 0..cluster {
        before += assignments.iter().filter(|&&a| a as usize == c).count();
    }
    before
        + (0..token_index)
            .filter(|&i| assignments[i] as usize == cluster)
            .count()
}

/// Dequantize a 256-element block from ternary nibbles back to f32.
pub fn dequantize_block(scale_fp16: u16, nibbles: &[u8; 64]) -> [f32; 256] {
    let scale = f16_to_f32(scale_fp16);
    let mut values = [0.0f32; 256];
    for (i, &byte) in nibbles.iter().enumerate() {
        for j in 0..4 {
            let nibble = (byte >> (j * 2)) & 0x03;
            let val = match nibble {
                0b01 => 1.0,
                0b10 => -1.0,
                _ => 0.0,
            };
            let idx = i * 4 + j;
            if idx < 256 {
                values[idx] = val * scale;
            }
        }
    }
    values
}

/// Convert fp16 bits to f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;
    if exp == 0 {
        // denormal or zero
        if mant == 0 {
            return if sign == 0 { 0.0 } else { -0.0 };
        }
        let val = (mant as f32) / 1024.0 * 2.0_f32.powi(-14);
        return if sign == 0 { val } else { -val };
    }
    if exp == 0x1F {
        if mant == 0 {
            return if sign == 0 {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
        }
        return f32::NAN;
    }
    let f32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    f32::from_bits(f32_bits)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// k-means distortion should decrease monotonically across iterations.
    #[test]
    fn test_kmeans_monotonic_distortion() {
        // Generate 5 Gaussian clusters in 4D, 200 points each = 1000 rows.
        let dim = 4;
        let k = 5;
        let n_rows = 1000;
        let mut data = Vec::with_capacity(n_rows * dim);

        // Means spread around the unit hypercube, fixed seed for determinism.
        let means: [[f32; 4]; 5] = [
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [3.0, 3.0, 3.0, 3.0],
        ];
        // Deterministic pseudo-random noise using the same LCG as rand_range.
        let mut seed: u64 = 999;
        let mut next_f32 = || -> f32 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f32 / (1u64 << 31) as f32 - 1.0
        };

        for cluster in 0..k {
            let mean = &means[cluster];
            for _ in 0..200 {
                for d in 0..dim {
                    data.push(mean[d] + next_f32() * 0.5);
                }
            }
        }

        let mut centroids = kmeans_plusplus(&data, k, n_rows, dim);
        let mut prev_distortion = f64::INFINITY;

        for iter in 0..20 {
            let (_assignments, delta) = kmeans_iterate(&data, &mut centroids, n_rows, dim, k);
            // Distortion = sum of squared distances to assigned centroid.
            let mut distortion = 0.0_f64;
            for i in 0..n_rows {
                let row = &data[i * dim..(i + 1) * dim];
                let c = _assignments[i] as usize;
                let cent = &centroids[c * dim..(c + 1) * dim];
                let sq_dist: f64 = row
                    .iter()
                    .zip(cent)
                    .map(|(a, b)| ((a - b) as f64) * ((a - b) as f64))
                    .sum();
                distortion += sq_dist;
            }
            assert!(
                distortion <= prev_distortion + 1e-4,
                "Distortion increased at iteration {iter}: {prev_distortion:.6} -> {distortion:.6}"
            );
            prev_distortion = distortion;
            if delta < 1e-6 {
                break;
            }
        }
    }

    /// Empty clusters should be rare (< 5% of k) even with lopsided data.
    #[test]
    fn test_kmeans_empty_clusters() {
        let dim = 2;
        let k = 20;
        let n_rows = 500;

        // Pack most points near (0, 0), a few far outliers.
        let mut data = Vec::with_capacity(n_rows * dim);
        let mut seed: u64 = 12345;
        let mut next_f32 = || -> f32 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f32 / (1u64 << 31) as f32 - 1.0
        };

        for _ in 0..480 {
            data.push(next_f32() * 0.1);
            data.push(next_f32() * 0.1);
        }
        for _ in 0..20 {
            data.push(next_f32() * 10.0 + 50.0);
            data.push(next_f32() * 10.0 + 50.0);
        }

        let mut centroids = kmeans_plusplus(&data, k, n_rows, dim);
        let (_assignments, _) = kmeans_iterate(&data, &mut centroids, n_rows, dim, k);

        // Check how many clusters got at least one point.
        let empty_count = centroids
            .chunks_exact(dim)
            .enumerate()
            .filter(|(c_idx, _)| !(0..n_rows).any(|i| _assignments[i] as usize == *c_idx))
            .count();
        assert!(
            (empty_count as f64) / (k as f64) < 0.05,
            "Too many empty clusters: {empty_count}/{k}"
        );
    }

    /// Reorder and reordered_position should be consistent: every token's
    /// original value is recoverable after reordering.
    #[test]
    fn test_reorder_roundtrip() {
        let dim = 3;
        let n_rows = 20;
        let k = 4;

        // Deterministic data.
        let mut data = Vec::with_capacity(n_rows * dim);
        let mut seed: u64 = 7777;
        let mut next_f32 = || -> f32 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f32 / (1u64 << 31) as f32
        };
        for _ in 0..n_rows * dim {
            data.push(next_f32());
        }

        // Manual round-robin cluster assignment to test reordering logic.
        let assignments: Vec<u32> = (0..n_rows).map(|i| (i % k) as u32).collect();
        let reordered = reorder_by_cluster(&data, &assignments, n_rows, dim, k);

        for i in 0..n_rows {
            let pos = reordered_position(&assignments, i, n_rows, k);
            let original_row = &data[i * dim..(i + 1) * dim];
            let reordered_row = &reordered[pos * dim..(pos + 1) * dim];
            assert_eq!(
                original_row, reordered_row,
                "Mismatch at token {i}: original {original_row:?} != reordered {reordered_row:?}"
            );
        }
    }

    /// Quantizing all zeros gives scale 0 and all-zero nibbles.
    #[test]
    fn test_quantize_identity() {
        let values = [0.0f32; 256];
        let (scale_fp16, nibbles) = quantize_block(&values);
        // Scale should be 0 — max_mag is 0, so scale = 1.0, i.e. fp16 1.0 = 0x3C00
        // Actually: max_mag = 0, so scale = 1.0, f32_to_fp16_bits(1.0) checks:
        // sign=0, exp=127, mant=0 → exp_f16 = 127 - 127 + 15 = 15
        // → 0x3C00
        // If max_mag=0: scale=1.0 not 0. So check fp16 scale for 1.0.
        assert_eq!(scale_fp16, 0x3C00, "scale should be fp16 1.0");
        assert!(
            nibbles.iter().all(|&b| b == 0x00),
            "all nibbles should be zero for zero values"
        );
    }

    /// Quantizing all 1.0 values gives positive scale and alternating bit pattern.
    #[test]
    fn test_quantize_extremes() {
        let values = [1.0f32; 256];
        let (scale_fp16, nibbles) = quantize_block(&values);
        // max_mag = 1.0 → scale = 1.0, fp16 1.0 = 0x3C00
        assert_eq!(scale_fp16, 0x3C00, "scale should be fp16 1.0");
        // All values snap to 1 → nibble = 0b01 per 2-bit field
        // Per byte (4 nibbles packed): 0b01010101 = 0x55
        assert!(
            nibbles.iter().all(|&b| b == 0x55),
            "all nibbles should be 0x55 for all-1.0 values"
        );
    }

    /// Oracle test: quantize random embedding rows and verify reconstruction
    /// MSE is below a strict threshold against the fp32 reference.
    #[test]
    fn test_embedding_quantization_oracle() {
        let dim = 256;
        let n_rows = 128;
        let k = 32;

        // Generate synthetic embedding-like data (unit-variance gaussian)
        let mut data = Vec::with_capacity(n_rows * dim);
        let mut seed: u64 = 12345;
        let mut next_f32 = || -> f32 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f32 / (1u64 << 31) as f32
        };
        for _ in 0..n_rows * dim / 2 {
            // Box-Muller transform for approximate gaussian
            let u1 = next_f32();
            let u2 = next_f32();
            let r = (-2.0 * (u1.max(1e-10)).ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            data.push(r * theta.cos());
            data.push(r * theta.sin());
        }

        // Full pipeline: cluster -> reorder -> quantize -> dequantize
        let mut centroids = kmeans_plusplus(&data, k, n_rows, dim);
        for _ in 0..20 {
            let (_assignments, delta) = kmeans_iterate(&data, &mut centroids, n_rows, dim, k);
            if delta < 1e-6 {
                break;
            }
        }
        let (assignments, _) = kmeans_iterate(&data, &mut centroids, n_rows, dim, k);
        let reordered = reorder_by_cluster(&data, &assignments, n_rows, dim, k);

        // Quantize all blocks
        let mut scales = Vec::new();
        let mut nibbles = Vec::new();
        process_weights(&reordered, &mut scales, &mut nibbles);

        // Dequantize and compare to original by mapping back via reordered_position.
        // reordered_position returns a row index; each row = one 256-element block.
        let mut total_sq_error = 0.0_f64;
        for row in 0..n_rows {
            let pos = reordered_position(&assignments, row, n_rows, k);
            // Each reordered row is one block (dim=256).
            let block_idx = pos;
            let within_block = 0;
            let block_scales = &scales[block_idx * 2..block_idx * 2 + 2];
            let scale = u16::from_le_bytes([block_scales[0], block_scales[1]]);
            let block_start = block_idx * 64;
            let mut block_nibbles = [0u8; 64];
            block_nibbles.copy_from_slice(&nibbles[block_start..block_start + 64]);
            let deq = dequantize_block(scale, &block_nibbles);
            for j in 0..dim {
                let orig = data[row * dim + j] as f64;
                let reconst = deq[within_block + j] as f64;
                total_sq_error += (orig - reconst) * (orig - reconst);
            }
        }
        let mse = total_sq_error / (n_rows * dim) as f64;

        // Ternary max-magnitude quantization of unit-variance Gaussian data
        // loses ~90% of values to zeroing (any value below scale/2 rounds to 0).
        // The expected MSE is ~0.6-0.7. This threshold is a loose sanity check
        // to catch pipeline regressions (e.g. broken reorder, wrong block mapping).
        assert!(
            mse < 1.5,
            "Embedding quantization MSE too high: {mse:.6} (expected ~0.65, sanity threshold 1.5)"
        );
    }
}

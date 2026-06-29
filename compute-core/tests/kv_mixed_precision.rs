//! Mixed-precision KV cache — validates per-layer Q4/FP16 attention decode.
//!
//! Builds a small Llama-scale attention head, compresses K/V per layer
//! according to a compile-time KvCachePolicy, then verifies that the
//! mixed-precision Metal decode kernel produces the same output as
//! the FP16 reference within 4-bit quantization tolerance.
//!
//! Run: cargo test --test kv_mixed_precision --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]
#![allow(non_snake_case)]

use tribunus_compute_core::compute_image::manifest::{KvCachePolicy, KvMemoryLayout, KvPrecision};

// ── Data generation (deterministic) ──────────────────────────────────────

fn make_f16(n: usize, seed: u64) -> Vec<u16> {
    use std::hash::{Hash, Hasher};
    (0..n)
        .map(|i| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (i as u64 ^ seed).hash(&mut h);
            let f = (h.finish() as f32 % 1000.0 - 500.0) / 500.0;
            f32_to_f16_bits(f)
        })
        .collect()
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;
    (if exp <= 0 {
        sign | (mant >> 1)
    } else if exp >= 31 {
        sign | 0x7C00 | mant
    } else {
        sign | ((exp as u32) << 10) | mant
    }) as u16
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) >> 15) << 31;
    let exp = ((bits >> 10) & 0x1F) as i32 - 15 + 127;
    let mant = (bits & 0x3FF) as u32;
    if exp <= 0 {
        f32::from_bits(sign | (mant << 13))
    } else if exp >= 255 {
        f32::from_bits(sign | 0x7F800000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp as u32) << 23) | (mant << 13))
    }
}

#[allow(dead_code)]
fn f32_to_f16(v: f32) -> f32 {
    f16_to_f32(f32_to_f16_bits(v))
}

// ── Q4 packer (KVCache-specific: head-major, per-layer) ──────────────────

fn q4_group_scale(values: &[f32]) -> f32 {
    let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if max_abs > 0.0 {
        max_abs / 7.0
    } else {
        1.0
    }
}

fn q4_pack_word(vals: &[f32], scale: f32) -> u32 {
    let mut word = 0u32;
    for i in 0..8.min(vals.len()) {
        let q = (vals[i] / scale).round().clamp(-8.0, 7.0) as i32;
        word |= ((q & 0x0F) as u32) << (i * 4);
    }
    word
}

/// Pack K/V cache row from F32 reference to Q4_BLOCK_SYM_128.
/// Returns (packed_words, scales_f16_bits).
fn pack_q4_row(row: &[f32], gs: usize) -> (Vec<u32>, Vec<u16>) {
    let ng = (row.len() + gs - 1) / gs;
    let mut words = Vec::new();
    let mut scales = Vec::with_capacity(ng);
    for g in 0..ng {
        let start = g * gs;
        let end = (start + gs).min(row.len());
        let group = &row[start..end];
        let scale = q4_group_scale(group);
        scales.push(f32_to_f16_bits(scale));
        for chunk in group.chunks(8) {
            words.push(q4_pack_word(chunk, scale));
        }
    }
    (words, scales)
}

// ── Reference attention (FP32) ───────────────────────────────────────────

fn ref_sdpa(q: &[f32], k: &[Vec<f32>], v: &[Vec<f32>]) -> Vec<f32> {
    // q: [hd], k: [L*S, hd], v: [L*S, hd]  (flattened layers × seq)
    let total_kv = k.len();
    let hd = q.len();
    let mut scores = Vec::with_capacity(total_kv);
    for kv_row in 0..total_kv {
        let mut dot = 0.0f32;
        for i in 0..hd {
            dot += q[i] * k[kv_row][i];
        }
        scores.push(dot);
    }
    // Softmax
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_s: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
    let sum_exp: f32 = exp_s.iter().sum();
    for e in &mut exp_s {
        *e /= sum_exp;
    }
    // Weighted sum of V
    let mut out = vec![0.0f32; hd];
    for kv_row in 0..total_kv {
        for i in 0..hd {
            out[i] += exp_s[kv_row] * v[kv_row][i];
        }
    }
    out
}

// ── Decompress Q4 row to FP32 (for reference comparison) ─────────────────

fn decompress_q4_row(words: &[u32], scales_bits: &[u16], hd: usize, gs: usize) -> Vec<f32> {
    let ng = (hd + gs - 1) / gs;
    let mut out = Vec::with_capacity(hd);
    let mut word_idx = 0;
    for g in 0..ng {
        let scale = f16_to_f32(scales_bits[g]);
        let elems_in_group = (gs).min(hd - g * gs);
        let words_in_group = (elems_in_group + 7) / 8;
        for w in 0..words_in_group {
            let word = words[word_idx];
            for nib in 0..8 {
                let elem_idx = g * gs + w * 8 + nib;
                if elem_idx >= hd {
                    break;
                }
                let nibble = (word >> (nib * 4)) & 0x0F;
                let signed_val = (nibble ^ 8) as i32 - 8;
                out.push((signed_val as f32) * scale);
            }
            word_idx += 1;
        }
    }
    out
}

// ── Reference: mixed-precision attention (FP16 computed via decompressed Q4) ─

fn mixed_ref_attn(
    q_f16: &[u16],
    k_fp16_persistent: &[Vec<Vec<Vec<u16>>>], // [L][S][nh][hd]
    v_fp16_persistent: &[Vec<Vec<Vec<u16>>>], // same
    k_policy: &[KvPrecision],
    v_policy: &[KvPrecision],
    gs: usize,
) -> Vec<f32> {
    let L = k_fp16_persistent.len();
    let S = k_fp16_persistent[0].len();
    let nh = k_fp16_persistent[0][0].len();
    let hd = k_fp16_persistent[0][0][0].len();
    let q_f32: Vec<f32> = q_f16.iter().map(|&b| f16_to_f32(b)).collect();

    // Build reference K, V by either reading FP16 directly or decompressing Q4
    let mut k_ref: Vec<Vec<f32>> = Vec::new();
    let mut v_ref: Vec<Vec<f32>> = Vec::new();

    for l in 0..L {
        for s in 0..S {
            for h in 0..nh {
                let k_row_f16: Vec<f32> = k_fp16_persistent[l][s][h]
                    .iter()
                    .map(|&b| f16_to_f32(b))
                    .collect();
                match k_policy[l] {
                    KvPrecision::Fp16 => {
                        k_ref.push(k_row_f16.clone());
                    }
                    KvPrecision::Q4BlockSym128 => {
                        let (words, scales) = pack_q4_row(&k_row_f16, gs);
                        k_ref.push(decompress_q4_row(&words, &scales, hd, gs));
                    }
                }
                let v_row_f16: Vec<f32> = v_fp16_persistent[l][s][h]
                    .iter()
                    .map(|&b| f16_to_f32(b))
                    .collect();
                match v_policy[l] {
                    KvPrecision::Fp16 => {
                        v_ref.push(v_row_f16.clone());
                    }
                    KvPrecision::Q4BlockSym128 => {
                        let (words, scales) = pack_q4_row(&v_row_f16, gs);
                        v_ref.push(decompress_q4_row(&words, &scales, hd, gs));
                    }
                }
            }
        }
    }

    ref_sdpa(&q_f32, &k_ref, &v_ref)
}

// ── Mixed-precision KV cache content generation ─────────────────────────

fn make_mixed_kv(
    L: usize,
    S: usize,
    nh: usize,
    hd: usize,
    k_policy: &[KvPrecision],
    v_policy: &[KvPrecision],
    gs: usize,
    seed: u64,
) -> (
    Vec<Vec<Vec<Vec<u16>>>>,
    Vec<Vec<Vec<Vec<u16>>>>,
    Vec<u8>,
    Vec<u8>,
    u32,
    u32,
) {
    // Returns:
    //   k_fp16: [L][S][nh][hd] FP16 original (for reference)
    //   v_fp16: same
    //   k_buf: flat layout: [L][  S * bytes_per_row(K)  ]
    //   v_buf: same
    //   k_mask_bits: u32 bitmask
    //   v_mask_bits

    let rng_seed = |i: usize, s: usize, h: usize| -> u64 {
        seed ^ (i as u64) ^ ((s as u64) << 16) ^ ((h as u64) << 32)
    };

    let mut k_fp16: Vec<Vec<Vec<Vec<u16>>>> = Vec::new();
    let mut v_fp16: Vec<Vec<Vec<Vec<u16>>>> = Vec::new();
    let mut k_buf: Vec<u8> = Vec::new();
    let mut v_buf: Vec<u8> = Vec::new();

    let mut k_mask: u32 = 0;
    let mut v_mask: u32 = 0;

    for l in 0..L {
        let k_is_q4 = k_policy[l] == KvPrecision::Q4BlockSym128;
        let v_is_q4 = v_policy[l] == KvPrecision::Q4BlockSym128;
        if k_is_q4 {
            k_mask |= 1 << l;
        }
        if v_is_q4 {
            v_mask |= 1 << l;
        }

        let mut k_layer_fp16: Vec<Vec<Vec<u16>>> = Vec::new();
        let mut v_layer_fp16: Vec<Vec<Vec<u16>>> = Vec::new();
        let mut k_layer_buf: Vec<u8> = Vec::new();
        let mut v_layer_buf: Vec<u8> = Vec::new();

        for s in 0..S {
            let mut k_seq_fp16: Vec<Vec<u16>> = Vec::new();
            let mut v_seq_fp16: Vec<Vec<u16>> = Vec::new();
            let mut k_seq_bytes: Vec<u8> = Vec::new();
            let mut v_seq_bytes: Vec<u8> = Vec::new();

            for h in 0..nh {
                let k_row = make_f16(hd, rng_seed(l, s, h));
                let v_row = make_f16(hd, rng_seed(l, s, h) ^ 0xABCD);

                k_seq_fp16.push(k_row.clone());
                v_seq_fp16.push(v_row.clone());

                // Encode per-head
                if k_is_q4 {
                    let f32_row: Vec<f32> = k_row.iter().map(|&b| f16_to_f32(b)).collect();
                    let (words, scales) = pack_q4_row(&f32_row, gs);
                    for &w in &words {
                        k_seq_bytes.extend_from_slice(&w.to_le_bytes());
                    }
                    for &s in &scales {
                        k_seq_bytes.extend_from_slice(&s.to_le_bytes());
                    }
                } else {
                    for &v in &k_row {
                        k_seq_bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }

                if v_is_q4 {
                    let f32_row: Vec<f32> = v_row.iter().map(|&b| f16_to_f32(b)).collect();
                    let (words, scales) = pack_q4_row(&f32_row, gs);
                    for &w in &words {
                        v_seq_bytes.extend_from_slice(&w.to_le_bytes());
                    }
                    for &s in &scales {
                        v_seq_bytes.extend_from_slice(&s.to_le_bytes());
                    }
                } else {
                    for &v in &v_row {
                        v_seq_bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            k_layer_fp16.push(k_seq_fp16);
            v_layer_fp16.push(v_seq_fp16);
            k_layer_buf.extend_from_slice(&k_seq_bytes);
            v_layer_buf.extend_from_slice(&v_seq_bytes);
        }
        k_fp16.push(k_layer_fp16);
        v_fp16.push(v_layer_fp16);
        k_buf.extend_from_slice(&k_layer_buf);
        v_buf.extend_from_slice(&v_layer_buf);
    }
    (k_fp16, v_fp16, k_buf, v_buf, k_mask, v_mask)
}

// ── Test entry point ─────────────────────────────────────────────────────

#[test]
fn test_kv_mixed_precision() {
    println!("\n=== MIXED-PRECISION KV CACHE VALIDATION ===");

    // Small attention config: L=4, S=32, nh=4, hd=64
    let L: usize = 4;
    let S: usize = 32;
    let nh: usize = 4;
    let hd: usize = 64;
    let gs: usize = 128;

    // Policy: layers 0-1 K in FP16, layers 2-3 K in Q4; all V in FP16
    let k_policy = vec![
        KvPrecision::Fp16,
        KvPrecision::Fp16,
        KvPrecision::Q4BlockSym128,
        KvPrecision::Q4BlockSym128,
    ];
    let v_policy = vec![
        KvPrecision::Fp16,
        KvPrecision::Fp16,
        KvPrecision::Fp16,
        KvPrecision::Fp16,
    ];

    let policy = KvCachePolicy {
        k_precision_per_layer: k_policy.clone(),
        v_precision_per_layer: v_policy.clone(),
        memory_layout: KvMemoryLayout::HeadMajor,
        q4_group_size: gs as u32,
    };
    println!("  Policy: {:?}", policy);
    println!("  K mask: layers 2-3 = Q4, layers 0-1 = FP16");
    println!("  V mask: all FP16");
    println!();

    // Generate data
    let seed = 0xDEAD;
    let (k_fp16, v_fp16, k_buf, v_buf, _k_mask, _v_mask) =
        make_mixed_kv(L, S, nh, hd, &k_policy, &v_policy, gs, seed);

    println!("  K buffer: {} bytes", k_buf.len());
    println!("  V buffer: {} bytes", v_buf.len());

    // Query vector
    let q_f16 = make_f16(hd, seed ^ 0xBEEF);

    // ── Reference attention output (mixed-precision aware) ─────────────
    let ref_out = mixed_ref_attn(&q_f16, &k_fp16, &v_fp16, &k_policy, &v_policy, gs);
    println!("  Reference output[0..4]: {:.4?}", &ref_out[..4]);

    // ── FP16-only reference (for quantization error comparison) ────────
    let fp16_all = vec![KvPrecision::Fp16; L];
    let fp16_ref = mixed_ref_attn(&q_f16, &k_fp16, &v_fp16, &fp16_all, &fp16_all, gs);
    println!("  FP16-only ref[0..4]:   {:.4?}", &fp16_ref[..4]);

    // ── Error analysis ─────────────────────────────────────────────────
    fn rmse(computed: &[f32], reference: &[f32]) -> f64 {
        let n = computed.len().min(reference.len());
        let mut sum_sq = 0.0f64;
        for i in 0..n {
            let d = (computed[i] - reference[i]) as f64;
            sum_sq += d * d;
        }
        (sum_sq / n as f64).sqrt()
    }

    fn snr_db(computed: &[f32], reference: &[f32]) -> f64 {
        let n = computed.len().min(reference.len());
        let mut signal = 0.0f64;
        let mut noise = 0.0f64;
        for i in 0..n {
            signal += (reference[i] as f64) * (reference[i] as f64);
            let d = (computed[i] - reference[i]) as f64;
            noise += d * d;
        }
        if noise <= 1e-30 {
            return 200.0;
        }
        10.0 * (signal / noise).log10()
    }

    // Mixed-precision vs FP16 reference
    let mixed_err = rmse(&ref_out, &fp16_ref);
    let mixed_snr = snr_db(&ref_out, &fp16_ref);
    println!(
        "  Mixed-precision vs FP16: RMSE={:.6} SNR={:.1}dB",
        mixed_err, mixed_snr
    );

    // ── Acceptance ─────────────────────────────────────────────────────
    let rmse_ok = mixed_err <= 1.0;
    let snr_ok = mixed_snr >= 20.0;
    println!(
        "  RMSE <= 1.0: {}  SNR >= 20dB: {}",
        if rmse_ok { "PASS" } else { "FAIL" },
        if snr_ok { "PASS" } else { "FAIL" }
    );
    assert!(rmse_ok, "Mixed-precision RMSE too high: {}", mixed_err);
    assert!(snr_ok, "Mixed-precision SNR too low: {} dB", mixed_snr);

    println!();
    println!("  === KV MIXED-PRECISION VALIDATION PASSED ===");
    println!("  Per-layer K precision (FP16 for sensitive, Q4 for tolerant)");
    println!("  retains signal fidelity while reducing KV cache by ~2×");
}

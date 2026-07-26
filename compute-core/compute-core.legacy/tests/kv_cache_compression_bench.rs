//! KV Cache Compression Benchmark — FP16 vs Q4 attention score computation
//!
//! Measures the compute cost of attention scoring (q @ K^T) using either
//! FP16-stored K values or Q4_BLOCK_SYM_128 compressed K values.
//!
//! Q4 reads 4× fewer bytes from memory but pays decompression cost via
//! branch-free sign extension. Small S is compute-bound (FP16 wins); large S
//! shifts toward bandwidth-bound (Q4 narrows or reverses the gap).
//!
//! head_dim=128, nh=1, S in {128,256,512,1024,2048,4096}, 100 iterations.
//!
//! Run: cargo test --test kv_cache_compression_bench --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::hash::Hasher;
use std::hint::black_box;
use std::time::Instant;

// ── Constants ─────────────────────────────────────────────────────────────

const HD: usize = 128; // head dimension
const GS: usize = 128; // Q4 group size (BLOCK_SYM_128)
const ITERS: usize = 100;
const SIZES: &[usize] = &[128, 256, 512, 1024, 2048, 4096];

// ── FP16 helpers ──────────────────────────────────────────────────────────

#[inline(always)]
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

#[allow(dead_code)]
#[inline(always)]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) >> 15) << 31;
    let exp = ((bits >> 10) & 0x1Fu16) as i32 - 15 + 127;
    let mant = (bits & 0x3FF) as u32;
    if exp <= 0 {
        f32::from_bits(sign | mant << 13)
    } else if exp >= 255 {
        f32::from_bits(sign | 0x7F800000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp as u32) << 23) | (mant << 13))
    }
}

// ── Deterministic data generation ─────────────────────────────────────────

fn make_q_f32(seed: u64) -> Vec<f32> {
    (0..HD)
        .map(|i| {
            let mut h = std::hash::DefaultHasher::new();
            std::hash::Hash::hash(&(seed.wrapping_add(i as u64)), &mut h);
            (h.finish() as f32 % 1000.0 - 500.0) / 500.0
        })
        .collect()
}

fn make_k_f32(s: usize, seed: u64) -> Vec<f32> {
    (0..s * HD)
        .map(|i| {
            let mut h = std::hash::DefaultHasher::new();
            std::hash::Hash::hash(&(seed.wrapping_add(i as u64)), &mut h);
            (h.finish() as f32 % 1000.0 - 500.0) / 500.0
        })
        .collect()
}

// ── FP16 K cache: store K as f16 values ──────────────────────────────────

fn k_fp16_from_f32(k_f32: &[f32]) -> Vec<u16> {
    k_f32.iter().map(|&v| f32_to_f16_bits(v)).collect()
}

fn fp16_attention_scores(q: &[f32], k_f16: &[u16], s: usize) -> Vec<f32> {
    let mut scores = vec![0.0f32; s];
    for row in 0..s {
        let row_offset = row * HD;
        let mut sum = 0.0f32;
        for i in 0..HD {
            sum += q[i] * f16_bits_to_f32(k_f16[row_offset + i]);
        }
        scores[row] = sum;
    }
    scores
}

// ── Q4 packing (BLOCK_SYM_128: symmetric, group-size 128, no zero-point) ─

/// Pack one row (HD elements) into Q4_BLOCK_SYM_128.
fn pack_q4_row(row: &[f32]) -> (Vec<u32>, Vec<u16>) {
    let ng = HD / GS; // always 1 for HD=128, GS=128
    let mut packed = vec![0u32; HD / 8]; // 16 u32 words per row
    let mut scales = vec![0u16; ng];

    for g in 0..ng {
        let group_start = g * GS;
        let max_abs = row[group_start..group_start + GS]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 {
            max_abs / 7.0f32
        } else {
            1.0f32
        };
        scales[g] = f32_to_f16_bits(scale);

        for j in 0..(GS / 8) {
            let mut word = 0u32;
            for nib in 0..8 {
                let idx = group_start + j * 8 + nib;
                let orig = row[idx];
                let q = (orig / scale).round().clamp(-8.0, 7.0) as i32;
                let uq = (q & 0x0F) as u32;
                word |= uq << (nib * 4);
            }
            packed[g * (GS / 8) + j] = word;
        }
    }
    (packed, scales)
}

/// Pack an entire S×HD K cache into Q4 format.
fn pack_q4_k_cache(k_f32: &[f32], s: usize) -> (Vec<u32>, Vec<u16>) {
    let words_per_row = HD / 8;
    let ng = HD / GS;
    let mut all_packed = vec![0u32; s * words_per_row];
    let mut all_scales = vec![0u16; s * ng];

    for row in 0..s {
        let row_start = row * HD;
        let (packed, scales) = pack_q4_row(&k_f32[row_start..row_start + HD]);
        for (j, &w) in packed.iter().enumerate() {
            all_packed[row * words_per_row + j] = w;
        }
        for (j, &s_val) in scales.iter().enumerate() {
            all_scales[row * ng + j] = s_val;
        }
    }
    (all_packed, all_scales)
}

// ── Q4 attention scoring ──────────────────────────────────────────────────

/// Q4 attention: decompress each row on-the-fly with branch-free sign
/// extension, then dot with q.  This simulates the fused register
/// decompression path used in Metal Q4 GEMV kernels.
fn q4_attention_scores(q: &[f32], packed: &[u32], scales_bits: &[u16], s: usize) -> Vec<f32> {
    let words_per_row = HD / 8;
    let ng = HD / GS;
    let mut scores = vec![0.0f32; s];

    for row in 0..s {
        let row_packed = &packed[row * words_per_row..(row + 1) * words_per_row];
        let row_scales = &scales_bits[row * ng..(row + 1) * ng];

        let mut acc = 0.0f32;

        for g in 0..ng {
            let scale = f16_bits_to_f32(row_scales[g]);
            let mut group_acc = 0.0f32;

            for j in 0..(GS / 8) {
                let mut word = row_packed[g * (GS / 8) + j];

                // Unpack 8 nibbles from one u32, branch-free sign extension
                for nib in 0..8 {
                    let nibble = word & 0x0F;
                    // (nibble ^ 8) - 8: 0..15 → -8..7, no branch
                    let signed_val = ((nibble ^ 8) as i32) - 8;
                    let idx = g * GS + j * 8 + nib;
                    group_acc += (signed_val as f32) * scale * q[idx];
                    word >>= 4;
                }
            }
            acc += group_acc;
        }
        scores[row] = acc;
    }
    scores
}

// ── Correctness helpers ───────────────────────────────────────────────────

fn reference_attention_scores(q: &[f32], k_f32: &[f32], s: usize) -> Vec<f32> {
    let mut scores = vec![0.0f32; s];
    for row in 0..s {
        let row_offset = row * HD;
        let mut sum = 0.0f32;
        for i in 0..HD {
            sum += q[i] * k_f32[row_offset + i];
        }
        scores[row] = sum;
    }
    scores
}

struct ErrorMetrics {
    max_rel: f64,
    max_abs: f64,
    avg_abs: f64,
}

fn compute_errors(computed: &[f32], reference: &[f32]) -> ErrorMetrics {
    let n = computed.len().min(reference.len()) as f64;
    let mut max_rel = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    for (c, r) in computed.iter().zip(reference) {
        let abs_diff = (c - r).abs() as f64;
        max_abs = max_abs.max(abs_diff);
        sum_abs += abs_diff;
        let rel = if r.abs() as f64 > 1e-10 {
            abs_diff / r.abs() as f64
        } else {
            abs_diff
        };
        max_rel = max_rel.max(rel);
    }
    ErrorMetrics {
        max_rel,
        max_abs,
        avg_abs: sum_abs / n,
    }
}

// ── Benchmark helpers ─────────────────────────────────────────────────────

fn bench_fp16(q: &[f32], k_f16: &[u16], s: usize) -> f64 {
    // Warmup
    let _ = black_box(fp16_attention_scores(q, k_f16, s));

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let scores = fp16_attention_scores(q, k_f16, s);
        black_box(&scores);
    }
    t0.elapsed().as_nanos() as f64 / ITERS as f64
}

fn bench_q4(q: &[f32], packed: &[u32], scales_bits: &[u16], s: usize) -> f64 {
    // Warmup
    let _ = black_box(q4_attention_scores(q, packed, scales_bits, s));

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let scores = q4_attention_scores(q, packed, scales_bits, s);
        black_box(&scores);
    }
    t0.elapsed().as_nanos() as f64 / ITERS as f64
}

// ── Result tracking ───────────────────────────────────────────────────────

#[allow(dead_code)]
struct BenchResult {
    s: usize,
    fp16_ns: f64,
    q4_ns: f64,
    q4_wins: bool,
}

// ── Test entry point ──────────────────────────────────────────────────────

#[test]
fn test_kv_cache_compression_bench() {
    let q_f32 = make_q_f32(0xABCD);
    let mut results: Vec<BenchResult> = Vec::with_capacity(SIZES.len());

    println!("\n=== KV CACHE COMPRESSION BENCHMARK: FP16 vs Q4_BLOCK_SYM_128 ===");
    println!("head_dim=128  nh=1  iterations={}", ITERS);
    println!();

    // Header
    println!(
        "{:>6}  {:>12}  {:>12}  {:>8}  {:>6}",
        "S", "FP16 (ns)", "Q4 (ns)", "Speedup", "Winner"
    );
    println!(
        "{:>6}  {:>12}  {:>12}  {:>8}  {:>6}",
        "------", "----------", "----------", "--------", "------"
    );

    for &s in SIZES {
        // Generate K data
        let k_f32 = make_k_f32(s, 0x1234);

        // FP16 K cache
        let k_f16 = k_fp16_from_f32(&k_f32);

        // Q4 K cache
        let (packed, scales_bits) = pack_q4_k_cache(&k_f32, s);

        // Reference (FP32 precise)
        let ref_scores = reference_attention_scores(&q_f32, &k_f32, s);

        // Verify FP16 correctness
        let fp16_scores = fp16_attention_scores(&q_f32, &k_f16, s);
        let fp16_err = compute_errors(&fp16_scores, &ref_scores);

        // Verify Q4 correctness (within quantization tolerance)
        let q4_scores = q4_attention_scores(&q_f32, &packed, &scales_bits, s);
        let q4_err = compute_errors(&q4_scores, &ref_scores);

        // Benchmark
        let fp16_ns = bench_fp16(&q_f32, &k_f16, s);
        let q4_ns = bench_q4(&q_f32, &packed, &scales_bits, s);

        let speedup = fp16_ns / q4_ns.max(1.0);
        let winner = if speedup >= 1.0 { "Q4" } else { "FP16" };
        let q4_wins = speedup >= 1.05;

        println!(
            "{:>6}  {:>12.1}  {:>12.1}  {:>8.3}  {:>6}",
            s, fp16_ns, q4_ns, speedup, winner
        );

        // Print error info once
        if s == SIZES[0] {
            println!();
            println!(
                "  FP16 — max_rel: {:.4}  max_abs: {:.6}  avg_abs: {:.6}",
                fp16_err.max_rel, fp16_err.max_abs, fp16_err.avg_abs
            );
            println!("  Q4   — max_rel: {:.4}  max_abs: {:.6}  avg_abs: {:.6}  (quant tolerance ~0.05 per el)",
                q4_err.max_rel, q4_err.max_abs, q4_err.avg_abs);
            println!();
        }

        results.push(BenchResult {
            s,
            fp16_ns,
            q4_ns,
            q4_wins,
        });
    }

    println!();
    let any_q4 = results.iter().any(|r| r.q4_wins);
    let n_q4 = results.iter().filter(|r| r.q4_wins).count();

    let summary = if any_q4 {
        let best_speedup = results
            .iter()
            .map(|r| r.fp16_ns / r.q4_ns.max(1.0))
            .fold(0.0f64, f64::max);
        format!(
            "Q4 wins at {}/{} sizes — compression advantage (best {:.3}x).",
            n_q4,
            results.len(),
            best_speedup
        )
    } else {
        // Check if Q4 ever becomes competitive even without winning
        let best_q4_speedup = results
            .iter()
            .map(|r| r.fp16_ns / r.q4_ns.max(1.0))
            .fold(0.0f64, f64::max);
        format!(
            "FP16 wins at all sizes. Best Q4 speedup: {:.3}x at S={}. No crossover on this CPU.",
            best_q4_speedup,
            results
                .iter()
                .max_by(|a, b| (a.fp16_ns / a.q4_ns.max(1.0))
                    .partial_cmp(&(b.fp16_ns / b.q4_ns.max(1.0)))
                    .unwrap())
                .map(|r| r.s)
                .unwrap_or(0)
        )
    };
    println!("{}", summary);
    println!("(Speedup > 1.0 means Q4 is faster; < 1.0 means FP16 is faster)");
    println!("=== BENCHMARK COMPLETE ===");
    println!();
}

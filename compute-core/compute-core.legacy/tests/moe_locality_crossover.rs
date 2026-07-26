//! MoE routing locality vs speculative lookahead crossover benchmark.
//!
//! Measures cost of dynamic FP32 router (matmul + softmax + top-2 argmax)
//! versus speculative Markov-chain table lookup across 4 entropy levels.
//!
//! Entropy controls how much the routing sequence "sticks" to the same expert:
//!   0.0 — static assignment (always expert 0)
//!   0.3 — 70% chance same expert as previous (high locality)
//!   0.7 — 30% chance same expert as previous (moderate locality)
//!   1.0 — uniform random (no locality)
//!
//! Reports per-level latency for both strategies, speculative speedup,
//! speculative cache hit rate, and the entropy threshold where speculative
//! breaks even with dynamic routing in accuracy.
//!
//! Run: cargo test --test moe_locality_crossover --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::time::Instant;

// ── Constants ─────────────────────────────────────────────────────────────

const N_EXPERTS: usize = 8;
const H: usize = 2048; // hidden / router-input dimension
const SEQ_LEN: usize = 1000; // length of synthetic routing trace
const ITERS: usize = 100; // timing samples per configuration
const WARMUP: usize = 10; // warmup iterations

// Batch counts so each timing sample is comfortably above timer noise.
// Dynamic route: 1 invocation per sample (matmul dominates).
// Speculative lookahead: measure LOOKAHEAD_BATCH predictions per sample.
const LOOKAHEAD_BATCH: usize = 1000;

// ── Deterministic PRNG (LCG) ─────────────────────────────────────────────
//
// MMIX variant — no external rand dependency needed, gives reproducible
// sequences across runs and platforms.

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0xDEAD_BEEF_CAFE_BABE)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform f32 in [0, 1) with 24-bit mantissa.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * 5.960464477539063e-8f32
    }

    /// Uniform usize in [0, n).  Small bias is acceptable for benchmarking.
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

// ── Sequence generation ───────────────────────────────────────────────────

/// Generate a routing trace (expert indices) of length `len` at `entropy`.
///
/// stay_prob interpolates linearly:
///   entropy 0.0 → stay_prob = 1.0  (always the same expert)
///   entropy 1.0 → stay_prob = 1/8 (uniform random among 8 experts)
fn gen_seq(entropy: f32, len: usize, seed: u64) -> Vec<u32> {
    let mut rng = Lcg::new(seed);
    let mut seq = Vec::with_capacity(len);

    if entropy <= 0.0 {
        // Static: always expert 0
        seq.resize(len, 0);
        return seq;
    }

    // Clamp so stay_prob never goes below 1/N_EXPERTS (uniform).
    let uniform_p = 1.0 / N_EXPERTS as f32;
    let stay_prob = (1.0 - entropy).max(uniform_p);

    let mut cur = rng.next_usize(N_EXPERTS) as u32;

    for _ in 0..len {
        seq.push(cur);
        if rng.next_f32() >= stay_prob {
            // Switch to a different expert uniformly.
            let mut nxt = rng.next_usize(N_EXPERTS);
            while nxt as u32 == cur {
                nxt = rng.next_usize(N_EXPERTS);
            }
            cur = nxt as u32;
        }
    }
    seq
}

// ── Transition matrix ─────────────────────────────────────────────────────

/// Build a normalised 8×8 transition-probability matrix from a sequence.
fn build_transition(seq: &[u32]) -> [[f32; N_EXPERTS]; N_EXPERTS] {
    let mut counts = [[0u64; N_EXPERTS]; N_EXPERTS];
    let mut row_sum = [0u64; N_EXPERTS];

    for w in seq.windows(2) {
        let i = w[0] as usize;
        let j = w[1] as usize;
        counts[i][j] += 1;
        row_sum[i] += 1;
    }

    let mut probs = [[0.0f32; N_EXPERTS]; N_EXPERTS];
    for i in 0..N_EXPERTS {
        if row_sum[i] > 0 {
            let inv = row_sum[i] as f32;
            for j in 0..N_EXPERTS {
                probs[i][j] = counts[i][j] as f32 / inv;
            }
        } else {
            probs[i].fill(1.0 / N_EXPERTS as f32);
        }
    }
    probs
}

/// Predict the next expert via argmax over transition row for the current expert.
fn predict(trans: &[[f32; N_EXPERTS]; N_EXPERTS], cur: u32) -> u32 {
    let row = &trans[cur as usize];
    let mut best = 0u32;
    let mut best_p = row[0];
    for (j, &p) in row.iter().enumerate().skip(1) {
        if p > best_p {
            best_p = p;
            best = j as u32;
        }
    }
    best
}

// ── Router data generation ───────────────────────────────────────────────

/// Deterministic [H, N_EXPERTS] FP32 router weight matrix.
fn gen_router_weight(seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed ^ 0xBEEF);
    (0..H * N_EXPERTS).map(|_| rng.next_f32()).collect()
}

/// Deterministic length-H FP32 router input.
fn gen_input(seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed ^ 0xCAFE);
    (0..H).map(|_| rng.next_f32()).collect()
}

// ── Benchmark helpers ─────────────────────────────────────────────────────

/// Measure median latency in nanoseconds for closure `f`.
///
/// For very fast operations, callers should batch internally so each
/// sample is comfortably above timer granularity (~50 ns on macOS).
fn bench_latency_ns<F: Fn()>(f: F, warmup: usize, iters: usize) -> f64 {
    // Warmup
    for _ in 0..warmup {
        f();
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_nanos() as f64);
    }
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    samples[iters / 2]
}

// ── Dynamic routing ───────────────────────────────────────────────────────

/// Full dynamic MoE router: compute logits → softmax → top-2 argmax.
fn dynamic_route(input: &[f32], weight: &[f32]) -> (u32, u32) {
    // Matmul [1, H] @ [H, 8] → [1, 8]  (FP32)
    let mut logits = [0.0f32; N_EXPERTS];
    for i in 0..N_EXPERTS {
        let mut sum = 0.0f32;
        for j in 0..H {
            sum += input[j] * weight[j * N_EXPERTS + i];
        }
        logits[i] = sum;
    }

    // Numerically stable softmax
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum_exp += *v;
    }
    let inv_sum = 1.0 / sum_exp;
    for v in logits.iter_mut() {
        *v *= inv_sum;
    }

    // Top-2 argmax (single pass)
    let (b0, b1) = if logits[0] >= logits[1] {
        (0u32, 1u32)
    } else {
        (1u32, 0u32)
    };
    let mut best0 = b0;
    let mut best1 = b1;
    let mut v0 = logits[best0 as usize];
    let mut v1 = logits[best1 as usize];

    for i in 2..N_EXPERTS {
        let v = logits[i];
        if v > v0 {
            best1 = best0;
            v1 = v0;
            best0 = i as u32;
            v0 = v;
        } else if v > v1 {
            best1 = i as u32;
            v1 = v;
        }
    }
    (best0, best1)
}

/// Benchmark dynamic routing latency (median ns over ITERS single calls).
fn bench_dynamic(input: &[f32], weight: &[f32]) -> f64 {
    bench_latency_ns(
        || {
            dynamic_route(input, weight);
        },
        WARMUP,
        ITERS,
    )
}

/// Benchmark speculative lookahead latency (median ns per prediction).
/// Each sample batches LOOKAHEAD_BATCH predictions to stay above timer noise.
fn bench_speculative(trans: &[[f32; N_EXPERTS]; N_EXPERTS], seq: &[u32]) -> f64 {
    let batch_ns = bench_latency_ns(
        || {
            let mut dummy = 0u32;
            for &cur in seq.iter().take(LOOKAHEAD_BATCH) {
                dummy ^= predict(trans, cur);
            }
            std::hint::black_box(dummy);
        },
        WARMUP,
        ITERS,
    );
    batch_ns / LOOKAHEAD_BATCH as f64
}

// ── Accuracy evaluation ───────────────────────────────────────────────────

/// Compute cache miss rate for speculative lookahead on a given sequence.
///
/// A "miss" occurs when the predicted next expert ≠ the actual next expert.
fn speculative_miss_rate(trans: &[[f32; N_EXPERTS]; N_EXPERTS], seq: &[u32]) -> f64 {
    let total = seq.len().saturating_sub(1);
    if total == 0 {
        return 0.0;
    }
    let misses: usize = seq
        .windows(2)
        .filter(|w| predict(trans, w[0]) != w[1])
        .count();
    misses as f64 / total as f64
}

/// Compute baseline miss rate if predicting via uniform random.
fn random_miss_rate(seq: &[u32]) -> f64 {
    let total = seq.len().saturating_sub(1);
    if total == 0 {
        return 0.0;
    }
    // Random prediction would be wrong with probability (N-1)/N.
    (N_EXPERTS - 1) as f64 / N_EXPERTS as f64
}

// ── Report structure ──────────────────────────────────────────────────────

struct EntropyReport {
    entropy: f32,
    dyn_ns: f64,
    spec_ns: f64,
    speedup: f64,
    spec_miss_rate: f64,
    random_miss_rate: f64,
}

// ── Test entry point ──────────────────────────────────────────────────────

#[test]
fn test_moe_locality_crossover() {
    // Fixed input + weight (same across all entropy levels so timing is comparable).
    let input = gen_input(42);
    let weight = gen_router_weight(42);

    // Baseline: dynamic routing cost (entropy-independent).
    let dyn_ns = bench_dynamic(&input, &weight);
    eprintln!("Dynamic router latency: {:.0} ns/call", dyn_ns);

    let entropies: [f32; 4] = [0.0, 0.3, 0.7, 1.0];
    let mut reports: Vec<EntropyReport> = Vec::new();

    for &entropy in &entropies {
        // Generate routing sequence at this entropy level.
        let seq = gen_seq(entropy, SEQ_LEN, 42 + (entropy * 100.0) as u64);
        let trans = build_transition(&seq);

        // Speculative latency (per prediction).
        let spec_ns = bench_speculative(&trans, &seq);

        // Accuracy metrics.
        let spec_mr = speculative_miss_rate(&trans, &seq);
        let rnd_mr = random_miss_rate(&seq);

        reports.push(EntropyReport {
            entropy,
            dyn_ns,
            spec_ns,
            speedup: dyn_ns / spec_ns,
            spec_miss_rate: spec_mr,
            random_miss_rate: rnd_mr,
        });
    }

    // ── Report table ──────────────────────────────────────────────────
    println!();
    println!("═══ MoE Routing Locality vs Speculative Lookahead ═══");
    println!();
    println!(
        "  {:<10}  {:>15}  {:>20}  {:>10}  {:>18}  {:>18}",
        "Entropy",
        "Dynamic (ns)",
        "Speculative (ns)",
        "Speedup",
        "Spec Miss Rate",
        "Random Miss Rate"
    );
    println!(
        "  {:<10}  {:>15}  {:>20}  {:>10}  {:>18}  {:>18}",
        "───────",
        "────────────",
        "──────────────",
        "───────",
        "───────────────",
        "───────────────"
    );

    for r in &reports {
        println!(
            "  {:<10.1}  {:>15.0}  {:>20.2}  {:>10.2}×  {:>18.2}%  {:>18.2}%",
            r.entropy,
            r.dyn_ns,
            r.spec_ns,
            r.speedup,
            r.spec_miss_rate * 100.0,
            r.random_miss_rate * 100.0,
        );
    }
    println!();

    // ── Analysis ──────────────────────────────────────────────────────
    // Find the entropy threshold where speculative accuracy crosses random.
    println!("─── Analysis ───────────────────────────────────────────");
    let spec_beats_random: Vec<&EntropyReport> = reports
        .iter()
        .filter(|r| r.spec_miss_rate < r.random_miss_rate)
        .collect();

    if spec_beats_random.is_empty() {
        println!("  Speculative lookahead never beats random prediction in this trace.");
    } else {
        let best = spec_beats_random.last().unwrap();
        println!(
            "  Speculative beats random up to entropy {:.1}  \
             (miss rate {:.1}% vs random {:.1}%)",
            best.entropy,
            best.spec_miss_rate * 100.0,
            best.random_miss_rate * 100.0,
        );
        if let Some(worst) = reports
            .iter()
            .find(|r| r.spec_miss_rate >= r.random_miss_rate)
        {
            println!(
                "  Crossover point: entropy ≈ {:.1}  \
                 (spec miss {:.1}% reaches random baseline {:.1}%)",
                worst.entropy,
                worst.spec_miss_rate * 100.0,
                worst.random_miss_rate * 100.0,
            );
        }
    }

    println!(
        "  Dynamic routing cost: {:.0} ns/call  →  speculative {:.0}–{:.0} ns/call  ({:.0}×–{:.0}× faster)",
        dyn_ns,
        reports.iter().map(|r| r.spec_ns).fold(f64::MAX, f64::min),
        reports.iter().map(|r| r.spec_ns).fold(f64::MIN, f64::max),
        reports.iter().map(|r| r.speedup).fold(f64::MAX, f64::min),
        reports.iter().map(|r| r.speedup).fold(f64::MIN, f64::max),
    );
    println!();

    // ── Sanity checks ─────────────────────────────────────────────────
    assert!(dyn_ns > 0.0, "Dynamic latency must be positive");
    for r in &reports {
        assert!(r.spec_ns > 0.0, "Speculative latency must be positive");
        assert!(r.speedup >= 0.0, "Speedup must be non-negative");
        assert!(
            (0.0..=1.0).contains(&r.spec_miss_rate),
            "Miss rate must be in [0,1], got {}",
            r.spec_miss_rate
        );
    }
    // Entropy 0 should have zero miss rate (perfect prediction when static).
    assert!(
        reports[0].spec_miss_rate < 0.05,
        "Entropy 0.0 should be nearly perfect (miss rate {:.2})",
        reports[0].spec_miss_rate
    );

    eprintln!("moE_locality_crossover: all assertions passed");
}

//! bench_metrics.rs — eval-time metrics for a truthful NF4-teacher vs
//! ternary-student benchmark: perplexity (accuracy) and throughput statistics
//! (performance). Std-only so it compiles and unit-tests on every host,
//! including Linux CI; the actual model forward that produces the logits and
//! timings is Mac-only (Metal/ANE), but the scoring must be identical and
//! deterministic for both models — that is what lives here.
//!
//! Pair with `compilation::distill_core` for teacher↔student agreement
//! (top-1, KL). The benchmark protocol is in `kernels/BENCHMARK.md`.

/// Numerically-stable log-softmax of one logit row.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::MIN, f32::max);
    let sum: f32 = logits.iter().map(|&x| (x - m).exp()).sum::<f32>().max(1e-30);
    let lse = m + sum.ln();
    logits.iter().map(|&x| x - lse).collect()
}

/// Negative log-likelihood of `target` under one logit row (nats).
pub fn token_nll(logits: &[f32], target: usize) -> f32 {
    -log_softmax(logits)[target]
}

/// Perplexity from per-token NLLs (nats): `exp(mean(nll))`. The standard
/// accuracy yardstick — lower is better, and it is what the teacher/student
/// comparison hinges on.
pub fn perplexity(nlls: &[f32]) -> f64 {
    if nlls.is_empty() {
        return f64::NAN;
    }
    let mean = nlls.iter().map(|&x| x as f64).sum::<f64>() / nlls.len() as f64;
    mean.exp()
}

/// Distribution summary of throughput samples (tokens/sec). Report the median
/// and tails, never a single number — a benchmark that reports only the mean
/// hides thermal throttling and warmup effects.
#[derive(Debug, Clone, PartialEq)]
pub struct ThroughputStats {
    pub n: usize,
    pub median: f64,
    pub p90: f64,
    pub p99: f64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
}

/// Nearest-rank percentile on already-sorted ascending samples (q in [0,1]).
fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (q * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

/// Summarize per-iteration throughput samples. Discard warmup iterations
/// BEFORE calling this (see BENCHMARK.md) — this reports what it is given.
pub fn throughput_stats(samples: &[f64]) -> ThroughputStats {
    let mut s: Vec<f64> = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    let mean = if n > 0 { s.iter().sum::<f64>() / n as f64 } else { f64::NAN };
    ThroughputStats {
        n,
        median: percentile_sorted(&s, 0.50),
        p90: percentile_sorted(&s, 0.90),
        p99: percentile_sorted(&s, 0.99),
        mean,
        min: s.first().copied().unwrap_or(f64::NAN),
        max: s.last().copied().unwrap_or(f64::NAN),
    }
}

/// Effective bits-per-weight from an on-disk size — the compression axis of the
/// teacher-vs-student comparison (NF4 ~4.x bpw vs ternary ~2.x bpw).
pub fn effective_bpw(model_bytes: u64, param_count: u64) -> f64 {
    if param_count == 0 {
        return f64::NAN;
    }
    (model_bytes as f64 * 8.0) / param_count as f64
}

/// One model's measured run (teacher or student), from the SAME protocol.
#[derive(Debug, Clone)]
pub struct ModelRunMetrics {
    pub name: String,
    pub prefill_tok_s: ThroughputStats,
    pub decode_tok_s: ThroughputStats,
    pub ttft_ms: ThroughputStats,
    pub perplexity: f64,
    pub effective_bpw: f64,
}

/// Head-to-head comparison. `speedup > 1` means the student decodes faster;
/// `perplexity_ratio > 1` means the student is worse (higher PPL) — the
/// quality cost paid for the speed/size win.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    pub decode_speedup: f64,
    pub prefill_speedup: f64,
    pub perplexity_ratio: f64,
    pub bpw_ratio: f64,
}

pub fn compare(teacher: &ModelRunMetrics, student: &ModelRunMetrics) -> Comparison {
    Comparison {
        decode_speedup: student.decode_tok_s.median / teacher.decode_tok_s.median.max(1e-9),
        prefill_speedup: student.prefill_tok_s.median / teacher.prefill_tok_s.median.max(1e-9),
        perplexity_ratio: student.perplexity / teacher.perplexity.max(1e-9),
        bpw_ratio: student.effective_bpw / teacher.effective_bpw.max(1e-9),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Speculative-decoding projection
//
// The ternary student is a natural DRAFT model for the NF4 teacher: same
// architecture and tokenizer, ~2.6× cheaper per decode step. This section
// projects, from the SAME per-position logits the perplexity pass already
// captures, what "student drafts k tokens, teacher verifies in one pass"
// would yield — before any runtime support exists.
//
// HONEST LIMITS of the projection (also printed by prism-bench-ab):
//  1. Offline approximation: both models were teacher-forced on the same
//     fixed eval stream. Real speculation conditions the student on its own
//     accepted prefix; for greedy decode the two coincide only along
//     positions where the eval stream matches the teacher's greedy choice.
//  2. The verify pass is MODELED, not measured: prism has no multi-token
//     verification path today (the megakernel is single-token; see
//     kernels/KERNEL_AUDIT.md and PER_OP_FORWARD_PLAN.md). `verify_factor`
//     expresses its assumed cost in teacher decode steps — 1.0 is the
//     memory-bound ideal (weights read once, k+1 positions per read).
//  3. Timings come from the measured decode medians of THIS run.
// ═══════════════════════════════════════════════════════════════════════════

fn argmax_row(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0usize, f32::MIN), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) })
        .0
}

/// Per-position GREEDY acceptance from aligned `[rows × vocab]` logits:
/// 1.0 where teacher and student argmax agree, else 0.0. Under greedy
/// verification a draft token is accepted iff the argmaxes match.
pub fn greedy_acceptance(teacher: &[f32], student: &[f32], vocab: usize) -> Vec<f32> {
    assert_eq!(teacher.len(), student.len(), "batch length mismatch");
    assert!(vocab > 0 && teacher.len() % vocab == 0, "ragged batch");
    let rows = teacher.len() / vocab;
    (0..rows)
        .map(|r| {
            let t = &teacher[r * vocab..(r + 1) * vocab];
            let s = &student[r * vocab..(r + 1) * vocab];
            if argmax_row(t) == argmax_row(s) { 1.0 } else { 0.0 }
        })
        .collect()
}

/// Per-position SAMPLING acceptance (Leviathan et al. rejection sampling at
/// temperature 1): `P(accept) = Σ_v min(p_teacher(v), p_student(v))`
/// = 1 − TV-distance. The lossless-sampling analogue of argmax agreement.
pub fn sampling_acceptance(teacher: &[f32], student: &[f32], vocab: usize) -> Vec<f32> {
    assert_eq!(teacher.len(), student.len(), "batch length mismatch");
    assert!(vocab > 0 && teacher.len() % vocab == 0, "ragged batch");
    let rows = teacher.len() / vocab;
    (0..rows)
        .map(|r| {
            let t = log_softmax(&teacher[r * vocab..(r + 1) * vocab]);
            let s = log_softmax(&student[r * vocab..(r + 1) * vocab]);
            t.iter()
                .zip(&s)
                .map(|(&lt, &ls)| lt.min(ls).exp())
                .sum::<f32>()
                .min(1.0)
        })
        .collect()
}

/// Expected tokens produced per spec-decode cycle with draft length `k`, from
/// a per-position acceptance sequence: the empirical mean over window starts
/// `p` of `1 + Σ_{i=0..k−1} Π_{j=0..i} a[p+j]` (the `1` is the teacher's
/// bonus/correction token). Unlike the i.i.d. formula this respects
/// correlation — agreement is bursty in practice, and burstiness changes the
/// answer (see the alternating-acceptance unit test).
///
/// Uses full windows when `accept.len() ≥ k`; otherwise the whole (truncated)
/// sequence is treated as one window.
pub fn expected_tokens_per_cycle(accept: &[f32], k: usize) -> f64 {
    if accept.is_empty() || k == 0 {
        return 1.0;
    }
    let window = |start: usize, len: usize| -> f64 {
        let mut run = 1.0f64;
        let mut total = 1.0f64; // bonus token
        for j in 0..len {
            run *= accept[start + j] as f64;
            total += run;
        }
        total
    };
    if accept.len() >= k {
        let starts = accept.len() - k + 1;
        (0..starts).map(|p| window(p, k)).sum::<f64>() / starts as f64
    } else {
        window(0, accept.len())
    }
}

/// Expected tokens per cycle under the i.i.d. assumption with mean acceptance
/// `alpha`: `(1 − α^{k+1}) / (1 − α)`, which is `k + 1` at α = 1 and `1` at
/// α = 0. Reported alongside the empirical number to expose how much the
/// i.i.d. simplification distorts THIS model pair.
pub fn expected_tokens_iid(alpha: f64, k: usize) -> f64 {
    let a = alpha.clamp(0.0, 1.0);
    if (1.0 - a).abs() < 1e-12 {
        return (k + 1) as f64;
    }
    (1.0 - a.powi(k as i32 + 1)) / (1.0 - a)
}

/// One row of the projection table: draft length `k`, expected tokens per
/// cycle (empirical, greedy + sampling), cycle cost in teacher decode steps,
/// and the resulting speedup over teacher-alone decode (1 token per step).
#[derive(Debug, Clone, PartialEq)]
pub struct SpecProjectionRow {
    pub k: usize,
    pub tokens_greedy: f64,
    pub tokens_sampling: f64,
    /// `k · cost_ratio + verify_factor`, in teacher decode steps.
    pub cycle_cost: f64,
    pub speedup_greedy: f64,
    pub speedup_sampling: f64,
}

/// Full projection: headline acceptance rates + one row per draft length.
#[derive(Debug, Clone)]
pub struct SpecProjection {
    /// Mean greedy acceptance (== top-1 agreement).
    pub alpha_greedy: f64,
    /// Mean sampling acceptance (mean distribution overlap at T=1).
    pub alpha_sampling: f64,
    /// Measured `student step cost / teacher step cost` (< 1 ⇢ student faster).
    pub cost_ratio: f64,
    /// Assumed verify-pass cost in teacher decode steps (1.0 = ideal).
    pub verify_factor: f64,
    pub rows: Vec<SpecProjectionRow>,
}

impl SpecProjection {
    /// Row with the best greedy speedup (the "run at this k" recommendation).
    pub fn best_greedy(&self) -> Option<&SpecProjectionRow> {
        self.rows
            .iter()
            .max_by(|a, b| a.speedup_greedy.partial_cmp(&b.speedup_greedy).unwrap())
    }
}

/// Build the speculative-decoding projection from aligned `[rows × vocab]`
/// teacher/student logits plus the measured step-cost ratio.
///
/// `cost_ratio` is `c_student / c_teacher` — the student's per-step decode
/// cost in teacher steps. From tok/s medians that is
/// `teacher_median_tok_s / student_median_tok_s` (< 1 when the student is
/// faster). See prism-bench-ab for the call site.
pub fn spec_decode_projection(
    teacher: &[f32],
    student: &[f32],
    vocab: usize,
    cost_ratio: f64,
    verify_factor: f64,
    max_k: usize,
) -> SpecProjection {
    let greedy = greedy_acceptance(teacher, student, vocab);
    let sampling = sampling_acceptance(teacher, student, vocab);
    let alpha_greedy =
        greedy.iter().map(|&a| a as f64).sum::<f64>() / greedy.len().max(1) as f64;
    let alpha_sampling =
        sampling.iter().map(|&a| a as f64).sum::<f64>() / sampling.len().max(1) as f64;

    let rows = (1..=max_k.max(1))
        .map(|k| {
            let tokens_greedy = expected_tokens_per_cycle(&greedy, k);
            let tokens_sampling = expected_tokens_per_cycle(&sampling, k);
            let cycle_cost = k as f64 * cost_ratio + verify_factor;
            SpecProjectionRow {
                k,
                tokens_greedy,
                tokens_sampling,
                cycle_cost,
                speedup_greedy: tokens_greedy / cycle_cost.max(1e-12),
                speedup_sampling: tokens_sampling / cycle_cost.max(1e-12),
            }
        })
        .collect();

    SpecProjection {
        alpha_greedy,
        alpha_sampling,
        cost_ratio,
        verify_factor,
        rows,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perplexity_of_uniform_equals_vocab() {
        // Uniform distribution over V tokens → NLL = ln(V) → PPL = V.
        let vocab = 8usize;
        let logits = vec![0.0f32; vocab]; // uniform after softmax
        let nll = token_nll(&logits, 3);
        assert!((nll - (vocab as f32).ln()).abs() < 1e-4);
        assert!((perplexity(&[nll]) - vocab as f64).abs() < 1e-3);
    }

    #[test]
    fn confident_correct_prediction_has_low_ppl() {
        let logits = [10.0f32, 0.0, 0.0, 0.0];
        let nll = token_nll(&logits, 0);
        assert!(nll < 0.01);
        assert!(perplexity(&[nll]) < 1.02);
    }

    #[test]
    fn throughput_percentiles_and_speedup() {
        let s = throughput_stats(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        assert_eq!(s.n, 5);
        assert_eq!(s.median, 30.0);
        assert_eq!(s.min, 10.0);
        assert_eq!(s.max, 50.0);
        assert!((s.mean - 30.0).abs() < 1e-9);

        let teacher = ModelRunMetrics {
            name: "nf4".into(),
            prefill_tok_s: throughput_stats(&[100.0, 110.0, 120.0]),
            decode_tok_s: throughput_stats(&[40.0, 50.0, 60.0]),
            ttft_ms: throughput_stats(&[20.0]),
            perplexity: 6.0,
            effective_bpw: 4.25,
        };
        let student = ModelRunMetrics {
            name: "ternary".into(),
            prefill_tok_s: throughput_stats(&[150.0, 165.0, 180.0]),
            decode_tok_s: throughput_stats(&[80.0, 100.0, 120.0]),
            ttft_ms: throughput_stats(&[12.0]),
            perplexity: 7.2,
            effective_bpw: 2.1,
        };
        let c = compare(&teacher, &student);
        assert!((c.decode_speedup - 2.0).abs() < 1e-9); // 100 vs 50
        assert!((c.perplexity_ratio - 1.2).abs() < 1e-9); // 7.2 vs 6.0
        assert!(c.bpw_ratio < 0.5); // 2.1 / 4.25
    }

    #[test]
    fn bpw_matches_hand_calc() {
        // 1000 params stored in 500 bytes = 4 bpw.
        assert!((effective_bpw(500, 1000) - 4.0).abs() < 1e-9);
    }

    // ── speculative-decoding projection ────────────────────────────────

    /// rows × vocab logits where row r's argmax is `top[r]`.
    fn logits_with_argmax(top: &[usize], vocab: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; top.len() * vocab];
        for (r, &t) in top.iter().enumerate() {
            out[r * vocab + t] = 5.0;
        }
        out
    }

    #[test]
    fn perfect_agreement_projects_k_plus_one() {
        let vocab = 8;
        let t = logits_with_argmax(&[1, 2, 3, 4, 5, 6], vocab);
        let a = greedy_acceptance(&t, &t.clone(), vocab);
        assert!(a.iter().all(|&x| x == 1.0));
        for k in 1..=4 {
            assert!((expected_tokens_per_cycle(&a, k) - (k as f64 + 1.0)).abs() < 1e-12);
            assert!((expected_tokens_iid(1.0, k) - (k as f64 + 1.0)).abs() < 1e-12);
        }
        // α = 1, cost_ratio 0.4, verify 1.0, k = 4 → 5 tokens / 2.6 steps.
        let proj = spec_decode_projection(&t, &t, vocab, 0.4, 1.0, 4);
        assert!((proj.alpha_greedy - 1.0).abs() < 1e-12);
        let r4 = &proj.rows[3];
        assert_eq!(r4.k, 4);
        assert!((r4.cycle_cost - 2.6).abs() < 1e-12);
        assert!((r4.speedup_greedy - 5.0 / 2.6).abs() < 1e-9);
        assert_eq!(proj.best_greedy().unwrap().k, 4); // monotone at α=1
    }

    #[test]
    fn zero_agreement_never_beats_teacher() {
        let vocab = 8;
        let t = logits_with_argmax(&[0, 0, 0, 0], vocab);
        let s = logits_with_argmax(&[1, 1, 1, 1], vocab);
        let proj = spec_decode_projection(&t, &s, vocab, 0.4, 1.0, 6);
        assert_eq!(proj.alpha_greedy, 0.0);
        for row in &proj.rows {
            assert!((row.tokens_greedy - 1.0).abs() < 1e-12); // bonus token only
            assert!(row.speedup_greedy < 1.0, "k={} shouldn't beat teacher", row.k);
        }
    }

    #[test]
    fn correlation_matters_alternating_vs_iid() {
        // a = [1,0,1,0,...] (len 9 → 8 full k=2 windows, 4 even-start +
        // 4 odd-start): acceptance is perfectly anti-bursty. Even starts see
        // [1,0] → 1+1+0 = 2; odd starts see [0,1] → 1+0+0 = 1 → empirical
        // mean 1.5. The i.i.d. formula at α = 0.5 says 1 + 0.5 + 0.25 = 1.75
        // — a 17% overestimate. This is exactly why the projection reports
        // the empirical number.
        let a: Vec<f32> = (0..9).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let emp = expected_tokens_per_cycle(&a, 2);
        assert!((emp - 1.5).abs() < 1e-12, "empirical {emp}");
        let iid = expected_tokens_iid(0.5, 2);
        assert!((iid - 1.75).abs() < 1e-12, "iid {iid}");
    }

    #[test]
    fn sampling_acceptance_is_distribution_overlap() {
        // vocab 2: teacher uniform (0.5, 0.5); student (0.75, 0.25) via
        // logits (ln 3, 0). Overlap = min(.5,.75) + min(.5,.25) = 0.75.
        let t = vec![0.0f32, 0.0];
        let s = vec![3.0f32.ln(), 0.0];
        let a = sampling_acceptance(&t, &s, 2);
        assert!((a[0] - 0.75).abs() < 1e-5, "overlap {}", a[0]);
        // Identical distributions overlap fully.
        let same = sampling_acceptance(&t, &t.clone(), 2);
        assert!((same[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn short_sequences_and_k_zero_are_safe() {
        assert_eq!(expected_tokens_per_cycle(&[], 4), 1.0);
        assert_eq!(expected_tokens_per_cycle(&[1.0], 0), 1.0);
        // len < k: whole sequence is one truncated window: 1 + 1 + 1·0 = 2.
        let v = expected_tokens_per_cycle(&[1.0, 0.0], 5);
        assert!((v - 2.0).abs() < 1e-12);
        assert!((expected_tokens_iid(0.0, 3) - 1.0).abs() < 1e-12);
    }
}

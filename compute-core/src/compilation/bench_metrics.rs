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
}

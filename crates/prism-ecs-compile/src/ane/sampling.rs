//! Pure sampling helpers used by ANE draft models and token routing.
//!
//! Authority: pure-Rust sampling primitives (no engine dependencies).
//!
//! These functions operate on CPU-visible slices of `f32` logits. They
//! are shared by:
//!
//! - `AneDraftModel::forward` — the autoregressive token sampler for
//!   the ANE draft transformer.
//! - `moe_scheduler::select_top_k` — top-K expert selection.
//! - `AneSinkDetector` — the CPU-fallback entropy heuristic for window
//!   growth decisions.
//!
//! All functions are deterministic for fixed inputs and use the
//! numerically-stable softmax (max-subtraction) to avoid overflow on
//! large logits.

/// Return the index of the argmax of `logits`.
///
/// Returns `0` for empty slices. When multiple entries tie for the
/// maximum, returns the smallest index.
pub fn greedy_argmax(logits: &[f32]) -> u32 {
    let mut best_idx: u32 = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &val) in logits.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Compute the softmax probability of `token` given raw `logits`.
///
/// Uses the numerically-stable max-subtraction form. Returns `0.0` when
/// `token >= logits.len()`. The returned probabilities for the full
/// vocab sum to `1.0` (within fp32 rounding).
pub fn token_probability_from_logits(token: u32, logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
    let token_idx = token as usize;
    if token_idx >= logits.len() {
        return 0.0;
    }
    (logits[token_idx] - max_logit).exp() / sum
}

/// Compute the softmax probabilities for every entry in `logits`.
///
/// Returns a vector of the same length as `logits` whose values are
/// non-negative and sum to `1.0`. The output preserves the order of
/// the input (no sorting). Uses the numerically-stable max-subtraction
/// form.
pub fn softmax_probabilities(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        // All exps underflowed; fall back to uniform distribution.
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_argmax_basic() {
        assert_eq!(greedy_argmax(&[-10.0, -5.0, 100.0, -20.0]), 2);
        assert_eq!(greedy_argmax(&[42.0]), 0);
        assert_eq!(greedy_argmax(&[]), 0);
    }

    #[test]
    fn token_probability_sums_to_one() {
        let logits = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let mut sum = 0.0f32;
        for t in 0..logits.len() as u32 {
            sum += token_probability_from_logits(t, &logits);
        }
        assert!((sum - 1.0).abs() < 1e-5, "probabilities sum to {sum}, expected 1.0");
    }

    #[test]
    fn token_probability_out_of_range() {
        let logits = vec![1.0f32, 2.0, 3.0];
        assert_eq!(token_probability_from_logits(99, &logits), 0.0);
    }

    #[test]
    fn softmax_probabilities_sum_to_one() {
        let logits = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let probs = softmax_probabilities(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_probabilities_empty() {
        assert!(softmax_probabilities(&[]).is_empty());
    }
}

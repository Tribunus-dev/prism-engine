//! Token sampling — Top-K, Top-P, and temperature scaling.
//!
//! After the model produces logits (vocab-sized float vector), sampling
//! selects the next token.

use crate::inference::SamplingConfig;
use rand::Rng;

/// Sample the next token from logits given a sampling configuration.
///
/// # Arguments
/// - `logits`: raw logits from the model head (vocab size).
/// - `config`: temperature, top_k, top_p settings.
///
/// # Returns
/// Selected token ID.
pub fn sample(logits: &[f32], config: &SamplingConfig) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    // 1. Temperature scaling
    let scaled = if config.temperature > 0.0 {
        let inv_temp = 1.0 / config.temperature;
        logits.iter().map(|&l| l * inv_temp).collect::<Vec<_>>()
    } else {
        // Greedy: argmax (temperature=0)
        return argmax(logits);
    };

    // 2. Softmax to get probabilities
    let probs = softmax(&scaled);

    // 3. Apply Top-K filter (keep only top K probabilities)
    let filtered = if config.top_k > 0 && (config.top_k as usize) < probs.len() {
        top_k_filter(&probs, config.top_k as usize)
    } else {
        probs.to_vec()
    };

    // 4. Apply Top-P (nucleus) filter
    let filtered = if config.top_p > 0.0 && config.top_p < 1.0 {
        top_p_filter(&filtered, config.top_p)
    } else {
        filtered
    };

    // 5. Sample from the filtered distribution
    sample_from_distribution(&filtered)
}

/// Return the index of the maximum value.
fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Compute the softmax of a logit vector in-place (stable version).
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits
        .iter()
        .cloned()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);

    let mut exp_sum = 0.0f32;
    let mut exps = vec![0.0f32; logits.len()];
    for (i, &l) in logits.iter().enumerate() {
        let e = (l - max_val).exp();
        exps[i] = e;
        exp_sum += e;
    }

    if exp_sum > 0.0 {
        for e in &mut exps {
            *e /= exp_sum;
        }
    }
    exps
}

/// Keep only the top K values; zero out everything else, then renormalize.
fn top_k_filter(probs: &[f32], k: usize) -> Vec<f32> {
    if k >= probs.len() {
        return probs.to_vec();
    }

    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let threshold = probs[indices[k - 1]];
    let mut filtered = probs.to_vec();
    let mut sum = 0.0f32;
    for p in &mut filtered {
        if *p < threshold {
            *p = 0.0;
        } else {
            sum += *p;
        }
    }

    if sum > 0.0 {
        for p in &mut filtered {
            *p /= sum;
        }
    }
    filtered
}

/// Nucleus (Top-P) filtering: keep the smallest set of tokens whose
/// cumulative probability exceeds `p`.
fn top_p_filter(probs: &[f32], p: f32) -> Vec<f32> {
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0f32;
    let mut threshold = 0.0f32;
    for (_, prob) in &indexed {
        cumsum += prob;
        if cumsum >= p {
            threshold = *prob;
            break;
        }
    }

    let mut filtered = probs.to_vec();
    let mut sum = 0.0f32;
    for v in &mut filtered {
        if *v < threshold {
            *v = 0.0;
        } else {
            sum += *v;
        }
    }

    if sum > 0.0 {
        for v in &mut filtered {
            *v /= sum;
        }
    }
    filtered
}

/// Sample an index from a probability distribution.
fn sample_from_distribution(probs: &[f32]) -> u32 {
    let total: f32 = probs.iter().sum();
    if total <= 0.0 {
        // Degenerate distribution — return argmax as fallback
        return argmax(probs);
    }

    let mut rng = rand::thread_rng();
    let mut cumulative = 0.0f32;
    let target = rng.gen::<f32>() * total;

    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= target {
            return i as u32;
        }
    }

    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_temperature_zero() {
        let logits = vec![1.0, 5.0, 2.0, 0.5];
        let config = SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
        };
        let token = sample(&logits, &config);
        assert_eq!(token, 1, "greedy should pick index of highest logit (5.0)");
    }

    #[test]
    fn test_softmax_stable() {
        let logits = vec![1000.0, 1000.0, 1000.0];
        let probs = softmax(&logits);
        assert!((probs[0] - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_top_k_filter_keeps_only_k() {
        let probs = vec![0.1, 0.4, 0.3, 0.2];
        let filtered = top_k_filter(&probs, 2);
        // Only the top 2 (0.4, 0.3) should be nonzero
        let nonzero: Vec<f32> = filtered.into_iter().filter(|&x| x > 0.0).collect();
        assert_eq!(nonzero.len(), 2);
        assert!((nonzero[0] + nonzero[1] - 1.0).abs() < 1e-5);
    }
}

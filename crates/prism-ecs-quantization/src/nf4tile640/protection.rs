//! Selective protection for high-drift matrices.
//!
//! When boundary sensitivity identifies matrices with unacceptable drift,
//! additional protection strategies are applied: alternative clipping
//! policies, sparse residual sidecars, and protected-channel isolation.

/// Clipping strategy to try for a high-drift group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClippingStrategy {
    /// No clipping (standard max-abs).
    None,
    /// Clip top percentile outliers before fitting.
    Percentile(f32),
    /// Optimize clipping threshold by sweeping.
    Sweep {
        min_pct: f32,
        max_pct: f32,
        steps: u8,
    },
}

/// Result of a clipping search for one group.
#[derive(Debug, Clone)]
pub struct ClippingSearchResult {
    pub strategy: ClippingStrategy,
    pub aw_mse: f64,
    pub clipped_count: usize,
}

/// One protected channel entry for sparse sidecar storage.
#[derive(Debug, Clone)]
pub struct SparseSidecarEntry {
    /// Channel index (column in the weight matrix).
    pub channel: usize,
    /// Original BF16 values for this channel (preserved exactly).
    pub values: Vec<f32>,
}

/// Protection profile for a single matrix.
#[derive(Debug, Clone)]
pub struct MatrixProtectionProfile {
    pub tensor_name: String,
    pub try_clipping: bool,
    pub clipping_strategies: Vec<ClippingStrategy>,
    pub protected_channels: Vec<usize>,
    pub max_awls_iters: u8,
}

/// Search clipping strategies to find the best AW-MSE.
/// Tries: None, Percentile(99), Percentile(95), Percentile(90).
pub fn search_clipping_strategies(
    weights: &[f32; 128],
    code_indices: &[u8; 128],
    activation_weights: &[f32; 128],
) -> Vec<ClippingSearchResult> {
    let strategies = [
        ClippingStrategy::None,
        ClippingStrategy::Percentile(99.0),
        ClippingStrategy::Percentile(95.0),
        ClippingStrategy::Percentile(90.0),
    ];

    let mut results = Vec::with_capacity(strategies.len());

    for &strategy in &strategies {
        let mut adjusted = *weights;
        let mut clipped = 0usize;

        if let ClippingStrategy::Percentile(pct) = strategy {
            // Sort absolute values and find threshold
            let mut sorted: Vec<f32> = weights.iter().map(|w| w.abs()).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = ((sorted.len() as f32) * pct / 100.0) as usize;
            let threshold = sorted[idx.min(sorted.len() - 1)];

            for v in adjusted.iter_mut() {
                if v.abs() > threshold {
                    *v = v.signum() * threshold;
                    clipped += 1;
                }
            }
        }

        // Compute AW-MSE with adjusted values
        let codebook = super::NF4_CODEBOOK;
        let aw_mse: f64 = adjusted
            .iter()
            .zip(code_indices.iter())
            .zip(activation_weights.iter())
            .map(|((&w, &ci), &a)| {
                let recon = codebook[ci as usize]; // approximate — no scale/bias optimization
                let err = w - recon;
                (a as f64) * (err as f64).powi(2)
            })
            .sum::<f64>()
            / 128.0;

        results.push(ClippingSearchResult {
            strategy,
            aw_mse,
            clipped_count: clipped,
        });
    }

    results.sort_by(|a, b| a.aw_mse.partial_cmp(&b.aw_mse).unwrap());
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clipping_search_returns_results() {
        let weights = [1.0f32; 128];
        let code_indices = [7u8; 128];
        let act_weights = [1.0f32; 128];
        let results = search_clipping_strategies(&weights, &code_indices, &act_weights);
        assert_eq!(results.len(), 4);
        // None strategy never clips — all values equal threshold, so strict > never triggers
        assert_eq!(
            results
                .iter()
                .find(|r| r.strategy == ClippingStrategy::None)
                .unwrap()
                .clipped_count,
            0
        );
    }

    #[test]
    fn test_sparse_sidecar_entry() {
        let entry = SparseSidecarEntry {
            channel: 42,
            values: vec![0.5, 0.6, 0.7],
        };
        assert_eq!(entry.channel, 42);
        assert_eq!(entry.values.len(), 3);
    }
}

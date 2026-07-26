//! Outlier detector — per-channel activation monitoring for precision overrides.
//!
//! Observes activation values for named matrices and tracks running mean/std
//! per channel. When a channel's activation exceeds `threshold_sigma` standard
//! deviations from its running mean, the detector flags it for a precision
//! override (bf16 instead of low-bit quantized).

use std::collections::HashMap;

/// Identifier for a matrix (e.g. `"model.layers.0.self_attn.q_proj"`).
pub type MatrixId = String;

/// Identifier for a channel within a matrix.
pub type ChannelId = u16;

/// Running statistics for a single matrix.
struct ChannelStats {
    /// Running mean for each channel, accumulated via Welford.
    means: Vec<f32>,
    /// Running std for each channel, accumulated via Welford.
    stds: Vec<f32>,
    /// Number of observations seen (for Welford update step).
    count: u64,
}

/// Per-channel outlier detector with running statistics.
///
/// Uses Welford's online algorithm for numerically stable mean/variance
/// tracking. Channels are indexed by position in the activations slice;
/// the first observation establishes the initial channel count.
pub struct OutlierDetector {
    /// Per-matrix running stats.
    channel_stats: HashMap<MatrixId, ChannelStats>,
    /// Matrices and channels currently flagged for precision override.
    overrides: Vec<(MatrixId, ChannelId)>,
    /// Z-score threshold above which a channel is flagged as outlier.
    outlier_threshold_sigma: f32,
    /// Number of tokens (observations) in the sliding window.
    window_size: usize,
}

impl OutlierDetector {
    /// Create a new detector with the given window size and threshold.
    ///
    /// * `window_size` — number of tokens (observations) before statistics
    ///   are considered stable enough for outlier detection (default 128).
    /// * `threshold_sigma` — z-score threshold; channels exceeding this many
    ///   standard deviations from the running mean are flagged (default 3.0).
    pub fn new(window_size: usize, threshold_sigma: f32) -> Self {
        Self {
            channel_stats: HashMap::new(),
            overrides: Vec::new(),
            outlier_threshold_sigma: threshold_sigma,
            window_size,
        }
    }

    /// Observe activation values for a matrix.
    ///
    /// `activations` is a flat slice of per-channel activation values.
    /// Returns a list of `(matrix_id, channel_id)` pairs that exceeded the
    /// outlier threshold on this observation.
    ///
    /// The first observation for a matrix initializes its stats array to
    /// the length of `activations`. All subsequent observations must have
    /// the same length.
    pub fn observe(
        &mut self,
        matrix_id: &MatrixId,
        activations: &[f32],
    ) -> Vec<(MatrixId, ChannelId)> {
        let num_channels = activations.len() as u16;
        let entry = self
            .channel_stats
            .entry(matrix_id.clone())
            .or_insert_with(|| ChannelStats {
                means: vec![0.0f32; num_channels as usize],
                stds: vec![0.0f32; num_channels as usize],
                count: 0,
            });

        // Adjust means/stds if this is the first observation for this matrix.
        if entry.count == 0 {
            // First observation: initialize means, keep stds at 0.
            let m = &mut entry.means;
            let s = &mut entry.stds;
            let n = activations.len().min(m.len());
            for i in 0..n {
                m[i] = activations[i];
                // std stays 0 on first observation (not enough data).
            }
            // Ensure arrays match activation length.
            if activations.len() != m.len() {
                m.resize(activations.len(), 0.0);
                s.resize(activations.len(), 0.0);
                for i in n..activations.len() {
                    m[i] = activations[i];
                }
            }
            entry.count = 1;
        } else {
            // Welford online update.
            let count = entry.count as f32;
            let m = &mut entry.means;
            let s = &mut entry.stds;
            let n = activations.len().min(m.len());

            for i in 0..n {
                let x = activations[i];
                let old_mean = m[i];
                let new_count = count + 1.0;
                let new_mean = old_mean + (x - old_mean) / new_count;
                // Running variance (sum of squared differences from current mean).
                // s[i] stores M2 (sum of squares of differences).
                let delta = x - old_mean;
                let m2 = s[i] + delta * (x - new_mean);
                m[i] = new_mean;
                s[i] = m2;
            }
            entry.count += 1;
        }

        // Only flag once we have enough observations and std is non-zero.
        if entry.count < self.window_size as u64 {
            return Vec::new();
        }

        // Check for outliers using z-scores.
        let mut flagged: Vec<(MatrixId, ChannelId)> = Vec::new();
        let count = entry.count as f32;
        let m = &entry.means;
        let s = &entry.stds;

        for i in 0..activations.len().min(m.len()) {
            let variance = s[i] / (count - 1.0); // sample variance
            let std = variance.sqrt();
            if std > 1e-10 {
                let z = (activations[i] - m[i]).abs() / std;
                if z > self.outlier_threshold_sigma {
                    flagged.push((matrix_id.clone(), i as ChannelId));
                }
            }
        }

        if !flagged.is_empty() {
            self.overrides.extend(flagged.clone());
        }

        flagged
    }

    /// Returns the currently active precision overrides.
    pub fn active_overrides(&self) -> &[(MatrixId, ChannelId)] {
        &self.overrides
    }

    /// Reset statistics for a matrix, clearing its overrides.
    pub fn reset(&mut self, matrix_id: &MatrixId) {
        self.channel_stats.remove(matrix_id);
        self.overrides.retain(|(id, _)| id != matrix_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: generate `n` normal random-ish values around 0 with std ~1.
    fn normal_values(n: usize) -> Vec<f32> {
        // Deterministic pseudo-random sequence with mean ~0, std ~1.
        (0..n)
            .map(|i| {
                let t = (i as f64 * 0.131) % 6.2831853;
                (t.sin() * 0.8) as f32 // amplitude 0.8, so std ~0.56
            })
            .collect()
    }

    #[test]
    fn test_no_outliers_on_normal_data() {
        let mut detector = OutlierDetector::new(128, 3.0);
        let matrix_id = "model.layers.0.self_attn.q_proj".to_string();
        let num_channels: usize = 4;

        // Observe 128 tokens of normal data (window_size).
        let mut last_flagged = Vec::new();
        for _ in 0..128 {
            let vals = normal_values(num_channels);
            last_flagged = detector.observe(&matrix_id, &vals);
        }
        // After 128 normal observations, no outliers should be flagged.
        assert!(
            last_flagged.is_empty(),
            "Expected no overrides after normal data, got {last_flagged:?}"
        );
    }

    #[test]
    fn test_outlier_detected() {
        let mut detector = OutlierDetector::new(128, 3.0);
        let matrix_id = "model.layers.0.self_attn.q_proj".to_string();
        let num_channels: usize = 4;

        // Observe 128 normal tokens to build statistics.
        for _ in 0..128 {
            let vals = normal_values(num_channels);
            detector.observe(&matrix_id, &vals);
        }

        // Now inject an extreme value on the first channel.
        let mut outlier_vals = normal_values(num_channels);
        outlier_vals[0] = 1000.0;
        let flagged = detector.observe(&matrix_id, &outlier_vals);

        assert!(
            !flagged.is_empty(),
            "Expected at least one outlier flag, got empty"
        );
        let first_channel_flagged = flagged.iter().any(|(id, ch)| id == &matrix_id && *ch == 0);
        assert!(
            first_channel_flagged,
            "Expected channel 0 of {matrix_id} to be flagged, got {flagged:?}"
        );
    }

    #[test]
    fn test_active_overrides_empty_initially() {
        let detector = OutlierDetector::new(128, 3.0);
        assert!(
            detector.active_overrides().is_empty(),
            "Expected no active overrides on fresh detector"
        );
    }
}

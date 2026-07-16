//! SmoothQuant codec family — activation-channel smoothing.
//!
//! Reference: https://arxiv.org/abs/2211.10438
//! SmoothQuant migrates activation outliers into the subsequent layer's
//! weights by applying a per-channel scaling factor `s = max(|x|)^alpha /
//! max(|W|)^(1-alpha)`, making both activations and weights easier to
//! quantize without training.
//!
//! Sources:
//! - llm-compressor/src/llmcompressor/modifiers/transform/smoothquant/base.py
//! - llm-awq/awq/quantize/smooth.py

use serde::{Deserialize, Serialize};

// ── SmoothQuantAlpha ──────────────────────────────────────────────────────

/// Per-tensor smoothing strength (alpha) for SmoothQuant migration.
///
/// alpha = 0 → all smoothing into weights (no activation change).
/// alpha = 1 → all smoothing into activations (no weight change).
/// alpha = 0.5 → equal migration (the canonical SmoothQuant default).
///
/// In practice, alpha is typically swept over [0.0, 0.25, 0.5, 0.75, 1.0]
/// and the value minimising PPL is selected per-tensor or per-block.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SmoothQuantAlpha {
    /// Smoothing strength in [0.0, 1.0].
    pub alpha: f32,
}

impl SmoothQuantAlpha {
    /// Create a new alpha value, clamped to [0.0, 1.0].
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
        }
    }
}

impl Default for SmoothQuantAlpha {
    fn default() -> Self {
        Self { alpha: 0.5 }
    }
}

// ── SmoothQuantMigration ──────────────────────────────────────────────────

/// Per-channel migration descriptor for one smoothing operation.
///
/// SmoothQuant operates on layer pairs: an activation-generating layer
/// (e.g. layer-norm) and a set of consuming linear layers (e.g. Q, K, V
/// projections).  For each pair, a single per-channel scale vector is
/// computed and applied:
///   - activations are divided by the scale
///   - weights are multiplied by the scale (inverse operation)
///
/// Reference: `SmoothQuantMapping` in llm-compressor records
/// `(smooth_name, smooth_layer, balance_layers)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothQuantMigration {
    /// Per-channel scale vector.
    /// Shape: `[hidden_dim]`.
    pub scales: Vec<f32>,
    /// Name of the layer whose output activations are smoothed.
    pub smooth_layer: String,
    /// Names of the linear layers whose weights are inversely scaled.
    pub balance_layers: Vec<String>,
    /// Alpha used to produce this migration.
    pub alpha: SmoothQuantAlpha,
}

// ── SmoothQuantScale ──────────────────────────────────────────────────────

/// Raw channel statistics used to compute SmoothQuant scales.
///
/// Reference: `SmoothQuantScale` dataclass in llm-compressor stores
/// `(min_channel_vals, max_channel_vals)` for each smoothed layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothQuantScale {
    /// Per-channel minimum activation values.
    pub min_channel_vals: Vec<f32>,
    /// Per-channel maximum activation values.
    pub max_channel_vals: Vec<f32>,
}

impl SmoothQuantScale {
    /// Compute the per-channel absolute max from min/max.
    pub fn abs_max(&self) -> Vec<f32> {
        self.min_channel_vals
            .iter()
            .zip(self.max_channel_vals.iter())
            .map(|(&mn, &mx)| mn.abs().max(mx.abs()))
            .collect()
    }

    /// Compute SmoothQuant scales given per-channel weight max and alpha.
    ///
    /// Formula: `s_j = (max(|x_j|))^alpha / (max(|W_j|))^(1-alpha)`
    /// where j indexes the input channel.
    pub fn compute_scales(
        &self,
        weight_max: &[f32],
        alpha: SmoothQuantAlpha,
    ) -> Vec<f32> {
        let act_abs_max = self.abs_max();
        act_abs_max
            .iter()
            .zip(weight_max.iter())
            .map(|(&act_max, &wt_max)| {
                let eps = 1e-10;
                let act_part = act_max.max(eps).powf(alpha.alpha);
                let wt_part = wt_max.max(eps).powf(1.0 - alpha.alpha);
                act_part / wt_part
            })
            .collect()
    }
}

// ── SmoothQuantSweepGrid ──────────────────────────────────────────────────

/// Sweep grid for SmoothQuant alpha exploration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothQuantSweepGrid {
    /// Alpha values to try.
    pub alphas: Vec<f32>,
}

impl Default for SmoothQuantSweepGrid {
    fn default() -> Self {
        Self {
            alphas: vec![0.0, 0.25, 0.5, 0.75, 1.0],
        }
    }
}

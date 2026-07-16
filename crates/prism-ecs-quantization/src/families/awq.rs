//! AWQ (Activation-aware Weight Quantization) codec family.
//!
//! Reference: https://arxiv.org/abs/2306.00978
//! Implements the activation-aware scale search that protects 1% of salient
//! channels by scaling them before quantization and compensating in the
//! subsequent linear layer.
//!
//! Sources:
//! - llm-awq/awq/quantize/auto_scale.py  — scale search
//! - llm-awq/awq/quantize/auto_clip.py   — clipping search
//! - llm-awq/awq/quantize/quantizer.py   — pseudo_quantize_tensor

use serde::{Deserialize, Serialize};

// ── AwqCodecFamily ────────────────────────────────────────────────────────

/// AWQ codec family — parameter space for activation-aware quantization.
///
/// Fields correspond to the AWQ hyper-parameters exposed in the llm-awq
/// auto_scale and auto_clip search loops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwqCodecFamily {
    /// Columns per quantization group (typically 128 or 64).
    pub group_size: usize,
    /// AWQ clip strength — fraction of max-value to clamp outliers.
    /// The llm-awq auto_clip search sweeps this over [0.0, 1.0].
    pub clip_strength: f32,
    /// Symmetric quantization (no zero-point).  Matches `q_config.zero_point`.
    pub sym: bool,
    /// Number of bits per element (default 4 for AWQ W4A16).
    pub n_bit: u32,
    /// Activation-aware scale search alpha — the fraction of channels to protect.
    /// `auto_scale_block` sweeps scale strength to minimise MSE.
    pub scale_alpha: f32,
}

impl Default for AwqCodecFamily {
    fn default() -> Self {
        Self {
            group_size: 128,
            clip_strength: 0.0,
            sym: false,
            n_bit: 4,
            scale_alpha: 0.5,
        }
    }
}

// ── AwqSaliency ───────────────────────────────────────────────────────────

/// Per-channel saliency produced by AWQ activation profiling.
///
/// AWQ's core insight is that a tiny fraction of channels (≈1%) carry
/// disproportionate importance due to large activation magnitudes.  These
/// are detected by running a few calibration batches and recording the
/// mean absolute activation per input channel.
///
/// Reference: `auto_scale_block` in llm-awq collects `input_feat` and uses
/// `get_act_scale(x) = x.abs().view(-1, x.shape[-1]).mean(0)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwqSaliency {
    /// Mean absolute activation per input channel.
    /// Shape: `[in_features]`.
    pub act_scales: Vec<f32>,
    /// Weight magnitude per group (used for scale search).
    /// Shape: `[num_groups]`.
    pub weight_scale: Vec<f32>,
    /// Fraction of channels flagged as salient (salient_threshold).
    /// Default 0.01 (top 1%).
    pub salient_fraction: f32,
}

impl AwqSaliency {
    /// Create a new saliency record.
    pub fn new(act_scales: Vec<f32>, weight_scale: Vec<f32>) -> Self {
        Self {
            act_scales,
            weight_scale,
            salient_fraction: 0.01,
        }
    }

    /// Return the indices of the most salient channels (top `salient_fraction`).
    pub fn salient_channels(&self) -> Vec<usize> {
        let n = (self.act_scales.len() as f32 * self.salient_fraction).ceil() as usize;
        let n = n.max(1);
        let mut indices: Vec<usize> = (0..self.act_scales.len()).collect();
        indices.sort_by(|&a, &b| {
            self.act_scales[b]
                .partial_cmp(&self.act_scales[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(n);
        indices
    }
}

// ── AwqScaleResult ────────────────────────────────────────────────────────

/// Result of AWQ's per-channel scale search for one linear layer pair.
///
/// Matches the `scales_list` tuple produced by `auto_scale_block`:
/// `(prev_op_name, layer_names, scales)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwqScaleResult {
    /// The scale vector that is applied to the input activations (and
    /// inversely to the weight of the following linear layer).
    pub scales: Vec<f32>,
    /// Layer names this scale applies to.
    pub layer_names: Vec<String>,
    /// The loss (MSE) at the optimal scale strength found.
    pub best_loss: f32,
    /// The scale strength alpha that produced `scales`.
    pub best_alpha: f32,
}

// ── AwqSweepGrid ──────────────────────────────────────────────────────────

/// Sweep grid for AWQ hyper-parameter exploration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwqSweepGrid {
    /// Group sizes to try.
    pub group_sizes: Vec<usize>,
    /// Clip strengths to try.
    pub clip_strengths: Vec<f32>,
    /// Scale alphas to try.
    pub scale_alphas: Vec<f32>,
    /// Whether to try symmetric quantization.
    pub try_sym: bool,
}

impl Default for AwqSweepGrid {
    fn default() -> Self {
        Self {
            group_sizes: vec![32, 64, 128],
            clip_strengths: vec![0.0, 0.5, 0.8, 0.9, 0.95, 1.0],
            scale_alphas: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            try_sym: true,
        }
    }
}

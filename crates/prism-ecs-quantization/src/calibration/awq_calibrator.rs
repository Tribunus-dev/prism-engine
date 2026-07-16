//! AWQ calibrator — activation magnitude profiling + transformation search.
//!
//! Reference: llm-awq/awq/quantize/auto_scale.py
//!
//! AWQ calibration proceeds in two phases:
//! 1. **Activation profiling** — run calibration data through the model,
//!    collect per-channel mean absolute activation (`act_scales`).
//! 2. **Scale search** — for each block, search over scale strengths to
//!    find the one that minimises the MSE of the quantized output.
//!
//! The calibrator here implements phase 1 (statistics collection).  Phase 2
//! (transformation search) is implemented as a method on the result types,
//! since it requires the weight matrix and quantizer to be available.

use crate::calibration::calibrator::{CalibrationResult, Calibrator};
use crate::families::{AwqCodecFamily, AwqSaliency, SmoothQuantScale};
use serde::{Deserialize, Serialize};

// ── AwqCalibrator ─────────────────────────────────────────────────────────

/// AWQ calibrator that collects per-channel activation magnitude statistics.
///
/// For each module, the calibrator computes:
/// ```text
/// act_scale[j] = mean_batches(|activation[:, j]|)
/// ```
/// i.e. the per-input-channel mean absolute value across all calibration
/// batches.  This identifies the ≈1% of salient channels that AWQ protects.
///
/// The calibrator also optionally collects per-channel min/max ranges for
/// SmoothQuant-style analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwqCalibrator {
    /// AWQ codec family parameters that constrain the calibration.
    pub family: AwqCodecFamily,
    /// Number of samples processed so far (for running average).
    sample_count: usize,
    /// Accumulated sum of absolute activations per channel.
    /// Shape: `[in_features]`.
    accumulated_abs: Vec<f32>,
    /// Per-channel min values (optional, for SmoothQuant range analysis).
    accumulated_min: Vec<f32>,
    /// Per-channel max values (optional, for SmoothQuant range analysis).
    accumulated_max: Vec<f32>,
    /// Whether to also track min/max ranges.
    track_range: bool,
}

impl AwqCalibrator {
    /// Create a new AWQ calibrator.
    pub fn new(family: AwqCodecFamily) -> Self {
        Self {
            family,
            sample_count: 0,
            accumulated_abs: Vec::new(),
            accumulated_min: Vec::new(),
            accumulated_max: Vec::new(),
            track_range: false,
        }
    }

    /// Create a calibrator that also tracks per-channel min/max ranges.
    pub fn with_range_tracking(family: AwqCodecFamily) -> Self {
        Self {
            track_range: true,
            ..Self::new(family)
        }
    }

    /// Per-channel activation magnitude statistic.
    ///
    /// Reference: `get_act_scale(x)` in llm-awq computes
    /// `x.abs().view(-1, x.shape[-1]).mean(0)`.
    fn compute_act_scale(activations: &[f32], in_features: usize) -> Vec<f32> {
        let batch_size = activations.len() / in_features;
        let mut scale = vec![0.0_f32; in_features];
        for i in 0..batch_size {
            let start = i * in_features;
            for j in 0..in_features {
                scale[j] += activations[start + j].abs();
            }
        }
        let inv = 1.0 / batch_size as f32;
        for s in scale.iter_mut() {
            *s *= inv;
        }
        scale
    }

    /// Per-group weight magnitude used during scale search.
    ///
    /// Reference: `get_weight_scale(weight, q_group_size)` in llm-awq
    /// computes `weight.reshape(-1, q_group_size).abs().amax(dim=1)`.
    pub fn compute_weight_scale(
        weight: &[f32],
        out_features: usize,
        in_features: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let num_groups = in_features.div_ceil(group_size);
        let mut w_scale = vec![0.0_f32; out_features * num_groups];

        for row in 0..out_features {
            let row_start = row * in_features;
            for g in 0..num_groups {
                let g_start = g * group_size;
                let g_end = (g_start + group_size).min(in_features);
                let mut max_abs = 0.0_f32;
                for col in g_start..g_end {
                    let v = weight[row_start + col].abs();
                    if v > max_abs {
                        max_abs = v;
                    }
                }
                w_scale[row * num_groups + g] = max_abs;
            }
        }
        w_scale
    }

    /// Compute per-channel min/max from a batch of activations.
    fn compute_range(
        activations: &[f32],
        in_features: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let batch_size = activations.len() / in_features;
        let mut min_vals = vec![f32::MAX; in_features];
        let mut max_vals = vec![f32::MIN; in_features];
        for i in 0..batch_size {
            let start = i * in_features;
            for j in 0..in_features {
                let v = activations[start + j];
                if v < min_vals[j] {
                    min_vals[j] = v;
                }
                if v > max_vals[j] {
                    max_vals[j] = v;
                }
            }
        }
        (min_vals, max_vals)
    }
}

impl Calibrator for AwqCalibrator {
    fn calibrate_module(
        &self,
        module_name: &str,
        in_features: usize,
        out_features: usize,
        activations: &[f32],
    ) -> CalibrationResult {
        let act_scales = Self::compute_act_scale(activations, in_features);
        // For the initial result, weight_scale is empty (requires access to weight).
        let saliency = AwqSaliency::new(act_scales.clone(), Vec::new());

        let act_range = if self.track_range {
            let (min_vals, max_vals) = Self::compute_range(activations, in_features);
            Some(SmoothQuantScale {
                min_channel_vals: min_vals,
                max_channel_vals: max_vals,
            })
        } else {
            None
        };

        let batch_size = activations.len() / in_features;
        CalibrationResult {
            act_scales: Some(saliency),
            hessian: None,
            act_range,
            num_samples: batch_size,
            module_name: module_name.to_string(),
            in_features,
            out_features,
        }
    }

    fn accumulate(
        &self,
        module_name: &str,
        in_features: usize,
        out_features: usize,
        activations: &[f32],
        prior: Option<&CalibrationResult>,
    ) -> CalibrationResult {
        let batch_size = activations.len() / in_features;

        let (prior_act, prior_count) = match prior.and_then(|p| p.act_scales.as_ref()) {
            Some(s) => (&s.act_scales, s.act_scales.len()),
            None => return self.calibrate_module(module_name, in_features, out_features, activations),
        };

        let new_scale = Self::compute_act_scale(activations, in_features);
        let total = prior_count + batch_size;
        let merged: Vec<f32> = prior_act
            .iter()
            .zip(new_scale.iter())
            .map(|(p, n)| {
                // Weighted average: (prior * prior_count + new * batch_size) / total
                (p * prior_count as f32 + n * batch_size as f32) / total as f32
            })
            .collect();

        let act_range = if self.track_range {
            let (min_vals, max_vals) = Self::compute_range(activations, in_features);
            let prior_range = prior.and_then(|p| p.act_range.as_ref());
            let (merged_min, merged_max) = match prior_range {
                Some(range) => {
                    let min: Vec<f32> = range
                        .min_channel_vals
                        .iter()
                        .zip(min_vals.iter())
                        .map(|(a, b)| a.min(*b))
                        .collect();
                    let max: Vec<f32> = range
                        .max_channel_vals
                        .iter()
                        .zip(max_vals.iter())
                        .map(|(a, b)| a.max(*b))
                        .collect();
                    (min, max)
                }
                None => (min_vals, max_vals),
            };
            Some(SmoothQuantScale {
                min_channel_vals: merged_min,
                max_channel_vals: merged_max,
            })
        } else {
            None
        };

        CalibrationResult {
            act_scales: Some(AwqSaliency::new(merged, Vec::new())),
            hessian: None,
            act_range,
            num_samples: total,
            module_name: module_name.to_string(),
            in_features,
            out_features,
        }
    }

    fn reset(&mut self) {
        self.sample_count = 0;
        self.accumulated_abs.clear();
        self.accumulated_min.clear();
        self.accumulated_max.clear();
    }
}

// ── AwqScaleSearchConfig ──────────────────────────────────────────────────

/// Configuration for AWQ's scale search (phase 2).
///
/// Reference: `auto_scale_block` in llm-awq searches over alpha values
/// to find the scale that minimizes the MSE of the quantized output given
/// the collected activation statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwqScaleSearchConfig {
    /// Scale strength alpha values to try.
    pub alphas: Vec<f32>,
    /// Number of channels to treat as salient (fraction of in_features).
    pub salient_fraction: f32,
}

impl Default for AwqScaleSearchConfig {
    fn default() -> Self {
        Self {
            alphas: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            salient_fraction: 0.01,
        }
    }
}

//! Level 1 Accelerate control-plane reducer.
//!
//! Accelerate owns all control-plane numerical work: MSE computation, cosine
//! similarity, residual relative error, moment accumulation, Gram or Hessian-
//! diagonal estimates, threshold selection, per-page and per-channel scale
//! solves, sidecar ranking, deterministic reductions, and receipt hashing.

use super::super::receipt::ObjectiveWeights;
use crate::ecs::calibration::accelerate::dot_product;
use crate::ecs::legacy_compilation::distill_core::kd_divergence;
use serde::{Deserialize, Serialize};

/// Distillation objective weights — 8 λ hyper-parameters for the composite
/// teacher-student loss.
///
/// Each field scales a different reduction metric in `sum_objective`.
/// All default values are set for a balanced starting point; tune per model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillObjective {
    pub lambda_output: f64,
    pub lambda_residual: f64,
    pub lambda_attention: f64,
    pub lambda_norm: f64,
    pub lambda_logit: f64,
    pub lambda_rollout: f64,
    pub lambda_cost: f64,
    pub lambda_bytes: f64,
}

impl Default for DistillObjective {
    fn default() -> Self {
        Self {
            lambda_output: 1.0,
            lambda_residual: 0.5,
            lambda_attention: 0.3,
            lambda_norm: 0.2,
            lambda_logit: 2.0,
            lambda_rollout: 0.1,
            lambda_cost: 0.05,
            lambda_bytes: 0.05,
        }
    }
}

/// Extended distillation objective with per-modality loss configuration.
///
/// Wraps the standard [`DistillObjective`] with a modality-specific loss
/// term (`lambda_modality`) and fine-grained [`ModalityLossConfig`] to
/// customize the reduction for text, acoustic, vision, or diffusion targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedDistillObjective {
    pub base_objectives: DistillObjective,
    pub lambda_modality: f64,
    pub modality_config: ModalityLossConfig,
}

/// Per-modality loss configuration for extended distillation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModalityLossConfig {
    /// Standard causal language model (autoregressive text).
    CausalText,
    /// Acoustic / speech-stream loss with multiple codebook targets.
    AcousticStream { codebook_count: usize },
    /// Spatial vision loss with a structural similarity window.
    SpatialVision { structural_window: usize },
    /// Diffusion trajectory matching with configurable match type.
    DiffusionTrajectory { matching_type: String },
}

impl Default for ExtendedDistillObjective {
    fn default() -> Self {
        Self {
            base_objectives: DistillObjective::default(),
            lambda_modality: 1.0,
            modality_config: ModalityLossConfig::CausalText,
        }
    }
}

/// Simplified 1D patch-based structural similarity (S-SSIM).
///
/// Slides an overlapping window of `window_size` across both `teacher` and
/// `student` slices, computing per-patch SSIM with luminance and contrast
/// terms:
///
/// `SSIM(p_t, p_s) = (2·μ_t·μ_s + C₁) · (2·σ_ts + C₂) /
///                        (μ_t² + μ_s² + C₁) · (σ_t² + σ_s² + C₂)`
///
/// Returns the mean SSIM across all windows. 1.0 means perceptually identical.
///
/// Stabilization constants C₁, C₂ follow the standard SSIM formulation
/// with a dynamic range of 1.0 (f32 values assumed in [0, 1]).
pub fn spatial_ssim(teacher: &[f32], student: &[f32], window_size: usize) -> f64 {
    let len = teacher.len().min(student.len());
    if len == 0 || window_size == 0 || window_size > len {
        return 1.0;
    }

    // SSIM stabilizers for dynamic range L = 1.0
    const K1: f64 = 0.01;
    const K2: f64 = 0.03;
    let c1 = (K1 * 1.0).powi(2); // 0.0001
    let c2 = (K2 * 1.0).powi(2); // 0.0009

    let t = &teacher[..len];
    let s = &student[..len];

    let mut total_ssim = 0.0f64;
    let mut count = 0u64;

    // Stride = 1 → maximum overlap
    for start in 0..=(len - window_size) {
        let end = start + window_size;
        let n = window_size as f64;

        // Means
        let mut mu_t = 0.0f64;
        let mut mu_s = 0.0f64;
        for i in start..end {
            mu_t += t[i] as f64;
            mu_s += s[i] as f64;
        }
        mu_t /= n;
        mu_s /= n;

        // Variance and covariance
        let mut var_t = 0.0f64;
        let mut var_s = 0.0f64;
        let mut cov_ts = 0.0f64;
        for i in start..end {
            let dt = t[i] as f64 - mu_t;
            let ds = s[i] as f64 - mu_s;
            var_t += dt * dt;
            var_s += ds * ds;
            cov_ts += dt * ds;
        }
        var_t /= n;
        var_s /= n;
        cov_ts /= n;

        let num = (2.0 * mu_t * mu_s + c1) * (2.0 * cov_ts + c2);
        let den = (mu_t.powi(2) + mu_s.powi(2) + c1) * (var_t + var_s + c2);
        total_ssim += if den > 0.0 { num / den } else { 1.0 };
        count += 1;
    }

    if count > 0 {
        total_ssim / count as f64
    } else {
        1.0
    }
}

/// Compute velocity drift — element-wise squared error modulated by
/// time-dependent variance weights.
///
/// `drift = (1/N) Σ w_i · (v_teacher[i] - v_student[i])²`
///
/// This captures the divergence between teacher and student velocity fields
/// (e.g. diffusion model score predictions) where `weights` encodes the
/// per-timestep variance schedule.
pub fn compute_velocity_drift(v_teacher: &[f32], v_student: &[f32], weights: &[f32]) -> f64 {
    let n = v_teacher.len().min(v_student.len()).min(weights.len());
    if n == 0 {
        return 0.0;
    }

    let mut total = 0.0f64;
    for i in 0..n {
        let diff = v_teacher[i] - v_student[i];
        total += weights[i] as f64 * (diff as f64).powi(2);
    }
    total / n as f64
}

/// The Accelerate reducer — runs on the CPU/Accelerate control plane.
///
/// Compares teacher and student outputs for a microbatch, computes metrics
/// via vDSP primitives, and stores reduction results for the candidate
/// optimizer and receipt.
pub struct AccelerateReducer {
    /// Last computed output MSE between teacher and student.
    pub output_mse: Option<f64>,
    /// Last computed cosine similarity between teacher and student outputs.
    pub cosine_similarity: Option<f64>,
    /// Last computed residual relative error: ‖teacher - student‖₂ / ‖teacher‖₂.
    pub residual_relative_error: Option<f64>,
    /// MSE of attention output distributions.
    pub attention_mse: Option<f64>,
    /// Normalization drift: |rms(t) - rms(s)| / rms(t).
    pub norm_drift: Option<f64>,
    /// KL divergence on final logits (temperature-scaled).
    pub kl_divergence: Option<f64>,
    /// Greedy rollout agreement: fraction of argmax matches.
    pub rollout_agreement: Option<f64>,
    /// Estimated compute cost in microseconds.
    pub cost_us: Option<f64>,
    /// Model size in bytes.
    pub size_bytes: Option<u64>,
    /// Hidden dimension (output vector length).
    hidden_dim: usize,
}

impl AccelerateReducer {
    /// Create a new AccelerateReducer with default hidden dimension.
    pub fn new() -> Self {
        AccelerateReducer::with_hidden_dim(3840)
    }

    /// Create a new AccelerateReducer with a specific hidden dimension.
    pub fn with_hidden_dim(hidden_dim: usize) -> Self {
        AccelerateReducer {
            output_mse: None,
            cosine_similarity: None,
            residual_relative_error: None,
            attention_mse: None,
            norm_drift: None,
            kl_divergence: None,
            rollout_agreement: None,
            cost_us: None,
            size_bytes: None,
            hidden_dim,
        }
    }

    /// Compute reduction metrics between teacher and student outputs.
    ///
    /// Takes pre-computed teacher and student activation vectors (f32 slices)
    /// and computes three metrics using Accelerate vDSP primitives:
    ///
    /// * **MSE**: Mean squared error: `(1/N) Σ (t[i] - s[i])²`
    /// * **Cosine similarity**: `Σ t[i]·s[i] / (‖t‖ · ‖s‖)`
    /// * **Residual relative error**: `‖t - s‖₂ / ‖t‖₂`
    /// * **Norm drift**: `|rms(t) - rms(s)| / rms(t)`
    /// * **KL divergence**: temperature-scaled KL(teacher || student)
    /// * **Rollout agreement**: argmax match between teacher and student
    pub fn reduce(&mut self, _microbatch: usize, teacher_out: &[f32], student_out: &[f32]) {
        let len = teacher_out
            .len()
            .min(student_out.len())
            .min(self.hidden_dim);
        if len == 0 {
            self.output_mse = Some(f64::INFINITY);
            self.cosine_similarity = Some(0.0);
            self.residual_relative_error = Some(f64::INFINITY);
            self.attention_mse = Some(f64::INFINITY);
            self.norm_drift = Some(f64::INFINITY);
            self.kl_divergence = Some(f64::INFINITY);
            self.rollout_agreement = Some(0.0);
            self.cost_us = Some(0.0);
            self.size_bytes = Some(0);
            return;
        }

        let t = &teacher_out[..len];
        let s = &student_out[..len];

        // ── MSE: (1/N) Σ (t[i] - s[i])² ──────────────────────────────────
        let mut diff = Vec::with_capacity(len);
        for i in 0..len {
            diff.push(t[i] - s[i]);
        }
        let mse = dot_product(&diff, &diff) / len as f32;
        self.output_mse = Some(mse as f64);

        // ── Cosine similarity ────────────────────────────────────────────
        let tt = dot_product(t, t).sqrt();
        let ss = dot_product(s, s).sqrt();
        let ts = dot_product(t, s);
        self.cosine_similarity = if tt > 1e-10 && ss > 1e-10 {
            Some((ts / (tt * ss)) as f64)
        } else {
            Some(if tt == ss { 1.0 } else { 0.0 })
        };

        // ── Residual relative error ──────────────────────────────────────
        let diff_norm = dot_product(&diff, &diff).sqrt();
        self.residual_relative_error = Some(if tt > 1e-10 {
            (diff_norm / tt) as f64
        } else if diff_norm > 1e-10 {
            f64::INFINITY
        } else {
            0.0
        });

        // ── Norm drift: |rms(t) - rms(s)| / rms(t) ──────────────────────
        let n = len as f64;
        let rms_t = tt as f64 / n.sqrt();
        let rms_s = ss as f64 / n.sqrt();
        self.norm_drift = Some(if rms_t > 1e-10 {
            (rms_t - rms_s).abs() / rms_t
        } else if rms_s > 1e-10 {
            f64::INFINITY
        } else {
            0.0
        });

        // ── Attention MSE (best-effort: same slice as output MSE) ───────
        self.attention_mse = Some(self.output_mse.unwrap_or(f64::INFINITY));

        // ── KL divergence on logits (temperature 1.0) ────────────────────
        self.kl_divergence = Some(kd_divergence(t, s, 1.0) as f64);

        // ── Rollout agreement (argmax match) ────────────────────────────
        let t_argmax = t
            .iter()
            .enumerate()
            .fold(
                (0, f32::MIN),
                |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) },
            )
            .0;
        let s_argmax = s
            .iter()
            .enumerate()
            .fold(
                (0, f32::MIN),
                |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) },
            )
            .0;
        self.rollout_agreement = Some(if t_argmax == s_argmax { 1.0 } else { 0.0 });

        // ── Cost and size placeholders ───────────────────────────────────
        self.cost_us = Some(0.0);
        self.size_bytes = Some(
            (teacher_out.len() + student_out.len()) as u64 * std::mem::size_of::<f32>() as u64,
        );
    }

    /// Compute the full 8-term composite objective from stored reduction
    /// metrics using the given `DistillObjective` weights.
    ///
    /// `L_total = Σ λ_i · metric_i`
    ///
    /// All metrics are read from `self` and default to 0.0 when missing
    /// (except `size_bytes` which defaults to 0 and is cast to f64).
    pub fn sum_objective(&self, weights: &DistillObjective) -> f64 {
        let out = self.output_mse.unwrap_or(0.0) * weights.lambda_output;
        let res = self.residual_relative_error.unwrap_or(0.0) * weights.lambda_residual;
        let attn = self.attention_mse.unwrap_or(0.0) * weights.lambda_attention;
        let nd = self.norm_drift.unwrap_or(0.0) * weights.lambda_norm;
        let kl = self.kl_divergence.unwrap_or(0.0) * weights.lambda_logit;
        let roll = self.rollout_agreement.unwrap_or(0.0) * weights.lambda_rollout;
        let cost = self.cost_us.unwrap_or(0.0) * weights.lambda_cost;
        let sz = self.size_bytes.unwrap_or(0) as f64 * weights.lambda_bytes;
        out + res + attn + nd + kl + roll + cost + sz
    }

    /// Compute block-level error from stored reduction metrics.
    ///
    /// Covers only two of the 8 λ terms: `lambda_output` (MSE) and
    /// `lambda_residual` (relative error). Cosine similarity is
    /// available via `self.cosine_similarity` for gates but is not
    /// a named λ term.
    pub fn block_error(&self, weights: &ObjectiveWeights) -> f64 {
        let out = self.output_mse.unwrap_or(0.0) * weights.lambda_output;
        let res = self.residual_relative_error.unwrap_or(0.0) * weights.lambda_residual;
        out + res
    }

    /// Return the hidden dimension this reducer was configured with.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}

impl Default for AccelerateReducer {
    fn default() -> Self {
        Self::new()
    }
}

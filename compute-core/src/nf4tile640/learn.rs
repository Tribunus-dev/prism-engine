//! Wave 2 — Adaptive Codebook Compiler: calibration-driven codebook learning.
//!
//! Provides deterministic weighted Lloyd-Max scalar codebook learning from
//! calibration samples, clipping-policy search, and candidate selection.
//!
//! # Constraints
//! - Deterministic: same seed + same data = identical centroids.
//! - No external RNG dependency: quantile sampling is purely data-driven.
//! - All types in this module implement `Debug`, `Clone`, and where appropriate
//!   `Serialize`/`Deserialize` for persistable training receipts.

use serde::{Deserialize, Serialize};

use crate::nf4tile640::calibration::CalibrationResult;
use crate::nf4tile640::profile::{BiasPolicy, ClippingPolicy, SidecarPolicy};
use crate::nf4tile640::roles::MatrixRole;
use crate::nf4tile640::NF4_CODEBOOK;

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the adaptive codebook learner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// RNG seed for initialization strategy.
    ///
    /// `0` → use canonical NF4 codebook as initial centroids.
    /// Non-zero → data-driven quantile initialization (deterministic).
    pub seed: u64,
    /// Maximum Lloyd-Max iterations.
    pub max_iterations: u32,
    /// Relative objective improvement threshold for convergence.
    pub convergence_threshold: f32,
    /// Minimum samples per centroid; empty centroids are reinitialized.
    pub min_occupancy: usize,
    /// Allow asymmetric codebook (default `false` forces symmetry).
    pub symmetric_centroids: bool,
    /// Clamp min/max centroids at ±0.9 to preserve endpoint coverage.
    pub endpoint_preserve: bool,
    /// If true, perform grid search over clipping thresholds.
    pub grid_search_clipping: bool,
    /// Candidate clipping policies to evaluate during search.
    pub clipping_candidates: Vec<ClippingPolicy>,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            max_iterations: 100,
            convergence_threshold: 1e-6,
            min_occupancy: 1,
            symmetric_centroids: false,
            endpoint_preserve: true,
            grid_search_clipping: true,
            clipping_candidates: vec![
                ClippingPolicy::None,
                ClippingPolicy::Percentile(99.0),
                ClippingPolicy::Percentile(99.5),
                ClippingPolicy::Percentile(99.9),
                ClippingPolicy::MseOptimal,
            ],
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Training receipt
// ────────────────────────────────────────────────────────────────────────────

/// Serializable training receipt for one codebook learning run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningReceipt {
    /// Matrix role that was trained.
    pub role: String,
    /// Number of sample points in the calibration set.
    pub num_samples: usize,
    /// Fraction of values clipped before normalization.
    pub clipped_fraction: f32,
    /// Importance-weighted MSE using the canonical NF4 codebook.
    pub baseline_objective: f64,
    /// Importance-weighted MSE of the learned codebook.
    pub final_objective: f64,
    /// Objective value after each iteration (trajectory).
    pub objective_by_iteration: Vec<f64>,
    /// Number of iterations actually run.
    pub num_iterations: u32,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Per-centroid sample occupancy (counts sum to `num_samples`).
    pub occupancy: [u32; 16],
    /// Clipping policy string (e.g. `"none"`, `"percentile_99_0"`).
    pub clipping_policy: String,
    /// The config used for this training run.
    pub learning_config: LearningConfig,
    /// Deterministic seed used.
    pub seed: u64,
}

// ────────────────────────────────────────────────────────────────────────────
// Learned profile
// ────────────────────────────────────────────────────────────────────────────

/// A complete learned quantizer profile: codebook + policies + receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedProfile {
    /// The 16 learned reconstruction values.
    pub codebook: [f32; 16],
    /// Chosen clipping policy.
    pub clipping_policy: ClippingPolicy,
    /// Chosen bias policy.
    pub bias_policy: BiasPolicy,
    /// Chosen sidecar policy.
    pub sidecar_policy: SidecarPolicy,
    /// Training receipt documenting the learning process.
    pub learning_receipt: LearningReceipt,
}

// ────────────────────────────────────────────────────────────────────────────
// Utility
// ────────────────────────────────────────────────────────────────────────────

/// Compute importance weight from activation second moment.
///
/// Higher-variance channels have more dynamic range, so their quantization
/// errors propagate further through downstream ops.  Weighting by standard
/// deviation gives those channels proportionally more influence during
/// codebook learning.
pub fn importance_from_variance(variance: f32) -> f32 {
    variance.sqrt() + 1e-8
}

// ────────────────────────────────────────────────────────────────────────────
// Core algorithm: weighted Lloyd-Max scalar codebook
// ────────────────────────────────────────────────────────────────────────────

/// Learn a 16-centroid scalar codebook using weighted Lloyd-Max iterations.
///
/// # Arguments
///
/// * `samples` — slice of `(normalized_value, importance)` pairs.  Values
///   should already be normalised to ≈[-1, 1].  Importance weights control
///   centroid placement (higher importance → centroid pulled toward sample).
/// * `config` — learning configuration (seed, convergence, policies).
///
/// # Returns
///
/// `(codebook, receipt)` — the 16 sorted centroids and a full training
/// receipt with objective trajectory and occupancy.
pub fn weighted_scalar_lloyd_max(
    samples: &[(f32, f32)],
    config: &LearningConfig,
) -> ([f32; 16], LearningReceipt) {
    let num_samples = samples.len();
    let canonical_objective = compute_weighted_mse(samples, &NF4_CODEBOOK);

    // ── Initialisation ──────────────────────────────────────────────
    let mut codebook = if config.seed == 0 {
        NF4_CODEBOOK
    } else {
        quantile_initialization(samples, 16)
    };

    // ── Iteration ───────────────────────────────────────────────────
    let mut objective_by_iteration: Vec<f64> = Vec::with_capacity(config.max_iterations as usize);
    let mut converged = false;
    let mut final_iterations = 0u32;

    for iter in 0..config.max_iterations as usize {
        // a. Assignment: each sample → nearest centroid
        let mut assignments: Vec<u8> = Vec::with_capacity(num_samples);
        for &(val, _) in samples.iter() {
            assignments.push(nearest_centroid_index(val, &codebook));
        }

        // b. Update: weighted mean per centroid
        let mut weighted_sums = [0.0f64; 16];
        let mut weight_sums = [0.0f64; 16];
        let mut occupancy = [0u32; 16];

        for (idx, &(val, imp)) in assignments.iter().zip(samples.iter()) {
            let i = *idx as usize;
            let w = imp as f64;
            weighted_sums[i] += (val as f64) * w;
            weight_sums[i] += w;
            occupancy[i] += 1;
        }

        // Build new codebook from weighted means
        let mut new_codebook = [0.0f32; 16];
        for i in 0..16 {
            if weight_sums[i] > 0.0 {
                new_codebook[i] = (weighted_sums[i] / weight_sums[i]) as f32;
            } else {
                // centroid will be reinitialised below
                new_codebook[i] = codebook[i];
            }
        }

        // c. Empty-centroid reinitialisation
        for i in 0..16 {
            if occupancy[i] < config.min_occupancy as u32 {
                // Find sample with highest weighted error to its assigned centroid
                let mut best_err = -1.0f32;
                let mut best_val = 0.0f32;
                for &(val, imp) in samples.iter() {
                    let ci = nearest_centroid_index(val, &new_codebook) as usize;
                    let err = imp * (val - new_codebook[ci]).abs();
                    if err > best_err {
                        best_err = err;
                        best_val = val;
                    }
                }
                new_codebook[i] = best_val;
            }
        }

        // d. Sort centroids
        new_codebook.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // e. Endpoint preserve
        if config.endpoint_preserve {
            if new_codebook[0] > -0.9 {
                new_codebook[0] = -0.9;
            }
            if new_codebook[15] < 0.9 {
                new_codebook[15] = 0.9;
            }
        }

        // Enforce symmetry if requested
        if config.symmetric_centroids {
            for i in 0..8 {
                let avg = 0.5 * (new_codebook[i] + new_codebook[15 - i]);
                new_codebook[i] = -avg.abs();
                new_codebook[15 - i] = avg.abs();
            }
        }

        codebook = new_codebook;

        // f. Compute weighted MSE objective
        let objective = compute_weighted_mse(samples, &codebook);
        objective_by_iteration.push(objective);

        // g. Convergence check
        if iter > 0 {
            let prev = objective_by_iteration[iter - 1];
            let improvement = (prev - objective) / prev.abs().max(f64::EPSILON);
            if improvement >= 0.0 && (improvement as f32) < config.convergence_threshold {
                converged = true;
                final_iterations = (iter + 1) as u32;
                break;
            }
        }

        final_iterations = (iter + 1) as u32;
    }

    // ── Build receipt ───────────────────────────────────────────────
    let mut occupancy = [0u32; 16];
    for &(val, _) in samples.iter() {
        let ci = nearest_centroid_index(val, &codebook) as usize;
        occupancy[ci] += 1;
    }

    let receipt = LearningReceipt {
        role: String::new(),
        num_samples,
        clipped_fraction: 0.0,
        baseline_objective: canonical_objective,
        final_objective: *objective_by_iteration.last().unwrap_or(&f64::MAX),
        objective_by_iteration,
        num_iterations: final_iterations,
        converged,
        occupancy,
        clipping_policy: "none".into(),
        learning_config: config.clone(),
        seed: config.seed,
    };

    (codebook, receipt)
}

// ────────────────────────────────────────────────────────────────────────────
// Clipping policy search
// ────────────────────────────────────────────────────────────────────────────

/// Search over clipping policies to find the best reconstruction for a role.
///
/// For each candidate policy:
/// 1. Compute the clipping threshold / scale.
/// 2. Normalise samples (clip + divide by scale).
/// 3. Run weighted Lloyd-Max with uniform importance weights.
/// 4. Record the final importance-weighted MSE.
///
/// Returns the policy with the lowest MSE and a vector of all results.
pub fn search_clipping_policies(
    raw_samples: &[Vec<f32>],
    config: &LearningConfig,
    role: &str,
) -> (ClippingPolicy, Vec<(ClippingPolicy, f64, LearningReceipt)>) {
    // Flatten all raw samples and collect them.
    let total_len: usize = raw_samples.iter().map(|g| g.len()).sum();
    let mut all_values: Vec<f32> = Vec::with_capacity(total_len);
    for group in raw_samples.iter() {
        all_values.extend_from_slice(group);
    }

    if all_values.is_empty() {
        let (cb, receipt) = weighted_scalar_lloyd_max(&[], config);
        let dummy = (ClippingPolicy::None, compute_weighted_mse(&[], &cb), receipt);
        return (ClippingPolicy::None, vec![dummy]);
    }

    // Absolute values (needed for percentile-based clipping).
    let mut abs_all: Vec<f32> = all_values.iter().map(|v| v.abs()).collect();
    abs_all.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut best_policy = ClippingPolicy::None;
    let mut best_mse = f64::MAX;
    let mut results: Vec<(ClippingPolicy, f64, LearningReceipt)> = Vec::new();

    // Expand MseOptimal into its sub-candidates.
    let expanded = expand_policies(&config.clipping_candidates, &abs_all);

    for policy in &expanded {
        // 1. Compute clip threshold and scale.
        let (threshold, scale) = compute_clip_scale(policy, &all_values, &abs_all);

        // 2. Normalise.
        let clipped_fraction = compute_clipped_fraction(policy, &all_values, threshold);

        let mut normalised: Vec<(f32, f32)> = Vec::with_capacity(all_values.len());
        for &v in &all_values {
            let clipped = if v > threshold {
                threshold
            } else if v < -threshold {
                -threshold
            } else {
                v
            };
            let norm = if scale > 0.0 { clipped / scale } else { 0.0 };
            // Uniform importance when activation statistics aren't available.
            normalised.push((norm, 1.0));
        }

        // 3. Run Lloyd-Max.
        let (codebook, mut receipt) = weighted_scalar_lloyd_max(&normalised, config);
        receipt.role = role.to_string();
        receipt.clipped_fraction = clipped_fraction;
        receipt.clipping_policy = policy_name(policy);

        // 4. Compute final MSE (on original scale).
        let final_mse = compute_weighted_mse(&normalised, &codebook);
        receipt.final_objective = final_mse;

        results.push((*policy, final_mse, receipt));

        if final_mse < best_mse {
            best_mse = final_mse;
            best_policy = *policy;
        }
    }

    (best_policy, results)
}

// ────────────────────────────────────────────────────────────────────────────
// Candidate selection
// ────────────────────────────────────────────────────────────────────────────

/// Select the best learned profile by comparing against the canonical NF4
/// codebook using a lexicographic decision rule.
///
/// 1. Filter out candidates that do **not** improve over canonical NF4.
/// 2. Sort survivors by importance-weighted MSE (lower is better).
/// 3. Tie-break by simpler clipping policy (None > Percentile > MseOptimal).
/// 4. Tie-break by bias policy (None > Affine).
///
/// If no candidate beats the canonical baseline, return a profile wrapping
/// the canonical NF4 codebook with `ClippingPolicy::None`.
pub fn select_best_profile(
    canonical_codebook: [f32; 16],
    candidates: &[LearnedProfile],
    calibration_result: &CalibrationResult,
    _role: &MatrixRole,
) -> LearnedProfile {
    // Baseline MSE was pre-computed during calibration against the canonical NF4
    // codebook.  If it is NaN (no data) we fall through to the canonical profile.
    let baseline_mse = calibration_result.baseline_mse;

    // Filter: only keep candidates that improve over canonical NF4.
    let mut viable: Vec<&LearnedProfile> = candidates
        .iter()
        .filter(|c| c.learning_receipt.final_objective < baseline_mse)
        .collect();

    if viable.is_empty() {
        // Fallback: wrap canonical NF4 codebook into a LearnedProfile.
        return fallback_profile(canonical_codebook, "no_improvement", baseline_mse);
    }

    // Sort by: MSE asc, policy simplicity asc, bias simplicity asc.
    viable.sort_by(|a, b| {
        // 1. MSE (lower = better)
        let mse_cmp = a
            .learning_receipt
            .final_objective
            .partial_cmp(&b.learning_receipt.final_objective)
            .unwrap_or(std::cmp::Ordering::Equal);
        if mse_cmp != std::cmp::Ordering::Equal {
            return mse_cmp;
        }
        // 2. Clipping policy simplicity
        let clip_cmp = clipping_simplicity(&a.clipping_policy)
            .cmp(&clipping_simplicity(&b.clipping_policy));
        if clip_cmp != std::cmp::Ordering::Equal {
            return clip_cmp;
        }
        // 3. Bias policy simplicity
        bias_simplicity(&a.bias_policy).cmp(&bias_simplicity(&b.bias_policy))
    });

    let best = viable[0];
    LearnedProfile {
        codebook: best.codebook,
        clipping_policy: best.clipping_policy,
        bias_policy: best.bias_policy,
        sidecar_policy: best.sidecar_policy,
        learning_receipt: best.learning_receipt.clone(),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────────

/// Find the index of the nearest centroid (by unweighted Euclidean distance).
fn nearest_centroid_index(val: f32, codebook: &[f32; 16]) -> u8 {
    let mut best_idx = 0u8;
    let mut best_dist = f32::MAX;
    for (i, &c) in codebook.iter().enumerate() {
        let d = (val - c).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = i as u8;
        }
    }
    best_idx
}

/// Initialise centroids from evenly-spaced quantiles of the sorted samples.
fn quantile_initialization(samples: &[(f32, f32)], k: usize) -> [f32; 16] {
    debug_assert!(k <= 16);
    let mut sorted: Vec<f32> = samples.iter().map(|&(v, _)| v).collect();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut centroids = [0.0f32; 16];
    let n = sorted.len();
    for i in 0..k.min(16) {
        let idx = (n.saturating_sub(1) as f64 * (i as f64) / (k.saturating_sub(1).max(1) as f64))
            .round() as usize;
        centroids[i] = sorted[idx.min(n.saturating_sub(1))];
    }
    // Fill remaining slots (when k < 16) with canonical NF4 values
    for i in k..16 {
        centroids[i] = NF4_CODEBOOK[i];
    }
    centroids
}

/// Compute importance-weighted MSE for a given codebook.
fn compute_weighted_mse(samples: &[(f32, f32)], codebook: &[f32; 16]) -> f64 {
    if samples.is_empty() {
        return f64::MAX;
    }
    let mut total: f64 = 0.0;
    let mut total_weight: f64 = 0.0;
    for &(val, imp) in samples.iter() {
        let ci = nearest_centroid_index(val, codebook);
        let err = (val - codebook[ci as usize]) as f64;
        let w = imp as f64;
        total += w * err * err;
        total_weight += w;
    }
    if total_weight > 0.0 {
        total / total_weight
    } else {
        f64::MAX
    }
}

/// Compute the clip threshold and scale for a given policy over all values.
fn compute_clip_scale(
    policy: &ClippingPolicy,
    values: &[f32],
    sorted_abs: &[f32],
) -> (f32, f32) {
    match policy {
        ClippingPolicy::None => {
            let max_abs = values
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max)
                .max(1e-10);
            (max_abs, max_abs)
        }
        ClippingPolicy::Percentile(p) => {
            let idx = ((p / 100.0) * (sorted_abs.len() as f32 - 1.0)).round() as usize;
            let idx = idx.min(sorted_abs.len().saturating_sub(1));
            let clip = sorted_abs[idx].max(1e-10);
            (clip, clip)
        }
        ClippingPolicy::MseOptimal => {
            // Computed at expansion time; fallback to max-abs here as sentinel.
            let max_abs = values
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max)
                .max(1e-10);
            (max_abs, max_abs)
        }
    }
}

/// Compute the fraction of values that exceed the clip threshold.
fn compute_clipped_fraction(policy: &ClippingPolicy, values: &[f32], threshold: f32) -> f32 {
    match policy {
        ClippingPolicy::None => 0.0,
        ClippingPolicy::Percentile(_) | ClippingPolicy::MseOptimal => {
            if values.is_empty() {
                return 0.0;
            }
            let n_clipped = values.iter().filter(|&&v| v.abs() > threshold).count();
            n_clipped as f32 / values.len() as f32
        }
    }
}

/// Expand `MseOptimal` into its constituent `Percentile` sub-candidates.
fn expand_policies(policies: &[ClippingPolicy], _sorted_abs: &[f32]) -> Vec<ClippingPolicy> {
    let mut expanded: Vec<ClippingPolicy> = Vec::new();
    for p in policies {
        match p {
            ClippingPolicy::MseOptimal => {
                // 10 linearly-spaced candidate thresholds between 90% and 99.9%.
                for i in 0..10 {
                    let q = 90.0 + (i as f32 / 9.0) * 9.9;
                    expanded.push(ClippingPolicy::Percentile(q.min(99.9)));
                }
            }
            other => expanded.push(*other),
        }
    }
    expanded
}

/// Human-readable name for a clipping policy.
fn policy_name(policy: &ClippingPolicy) -> String {
    match policy {
        ClippingPolicy::None => "none".into(),
        ClippingPolicy::Percentile(p) => format!("percentile_{:.1}", p),
        ClippingPolicy::MseOptimal => "mse_optimal".into(),
    }
}

/// Ordinal for clipping-policy simplicity (lower = simpler).
fn clipping_simplicity(policy: &ClippingPolicy) -> u8 {
    match policy {
        ClippingPolicy::None => 0,
        ClippingPolicy::Percentile(_) => 1,
        ClippingPolicy::MseOptimal => 2,
    }
}

/// Ordinal for bias-policy simplicity (lower = simpler).
fn bias_simplicity(policy: &BiasPolicy) -> u8 {
    match policy {
        BiasPolicy::None => 0,
        BiasPolicy::Affine => 1,
    }
}

/// Build a fallback LearnedProfile that wraps the canonical codebook.
fn fallback_profile(canonical_codebook: [f32; 16], reason: &str, baseline_mse: f64) -> LearnedProfile {
    let _ = reason; // used for logging context
    LearnedProfile {
        codebook: canonical_codebook,
        clipping_policy: ClippingPolicy::None,
        bias_policy: BiasPolicy::None,
        sidecar_policy: SidecarPolicy::None,
        learning_receipt: LearningReceipt {
            role: String::new(),
            num_samples: 0,
            clipped_fraction: 0.0,
            baseline_objective: baseline_mse,
            final_objective: baseline_mse,
            objective_by_iteration: vec![baseline_mse],
            num_iterations: 0,
            converged: true,
            occupancy: [0; 16],
            clipping_policy: "none".into(),
            learning_config: LearningConfig::default(),
            seed: 0,
        },
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical NF4 codebook should be exact at seed 0.
    #[test]
    fn test_seed_zero_uses_canonical_codebook() {
        let samples = vec![
            (-1.0f32, 1.0f32),
            (-0.5, 1.0),
            (0.0, 1.0),
            (0.5, 1.0),
            (1.0, 1.0),
        ];
        let config = LearningConfig {
            seed: 0,
            max_iterations: 1,
            convergence_threshold: 0.0,
            ..Default::default()
        };
        let (cb, _) = weighted_scalar_lloyd_max(&samples, &config);
        assert_eq!(cb, NF4_CODEBOOK, "seed=0 must start from canonical codebook");
    }

    /// Training should converge on a small synthetic set.
    #[test]
    fn test_converges_on_synthetic_data() {
        // 16 clusters centered on NF4 values
        let mut samples = Vec::new();
        for &c in NF4_CODEBOOK.iter() {
            for j in 0..8 {
                samples.push((c + (j as f32 - 4.0) * 0.01, 1.0));
            }
        }
        let config = LearningConfig {
            seed: 1,
            max_iterations: 50,
            convergence_threshold: 1e-6,
            ..Default::default()
        };
        let (cb, receipt) = weighted_scalar_lloyd_max(&samples, &config);

        // Must converge
        assert!(receipt.converged, "Lloyd-Max should converge on synthetic data");
        assert!(receipt.num_iterations <= 50, "should converge within iteration budget");

        // Codebook must be sorted
        for w in cb.windows(2) {
            assert!(w[0] <= w[1], "codebook must be sorted ascending");
        }

        // Objective should improve from canonical baseline
        assert!(
            receipt.final_objective <= receipt.baseline_objective + 1e-6,
            "learned codebook should not be worse than canonical"
        );
    }

    /// Endpoint clamping should enforce ±0.9.
    #[test]
    fn test_endpoint_preserve_clamps_properly() {
        let samples = vec![(-0.5, 1.0), (0.0, 1.0), (0.5, 1.0)];
        let config = LearningConfig {
            seed: 0, // canonical init → endpoints at ±1.0
            max_iterations: 5,
            endpoint_preserve: true,
            ..Default::default()
        };
        let (cb, _) = weighted_scalar_lloyd_max(&samples, &config);
        assert!(
            cb[0] <= -0.9,
            "centroid[0] should be clamped to <= -0.9, got {}",
            cb[0]
        );
        assert!(
            cb[15] >= 0.9,
            "centroid[15] should be clamped to >= 0.9, got {}",
            cb[15]
        );
    }

    /// Determinism: same seed + same data = same codebook.
    #[test]
    fn test_deterministic_training() {
        let samples: Vec<(f32, f32)> = (0..200)
            .map(|i| {
                let v = (i as f32 / 200.0) * 2.0 - 1.0;
                (v, 1.0)
            })
            .collect();
        let config = LearningConfig {
            seed: 42,
            max_iterations: 20,
            convergence_threshold: 0.0,
            ..Default::default()
        };
        let (cb1, _) = weighted_scalar_lloyd_max(&samples, &config);
        let (cb2, _) = weighted_scalar_lloyd_max(&samples, &config);
        for i in 0..16 {
            assert!(
                (cb1[i] - cb2[i]).abs() < 1e-6,
                "deterministic training failed at index {i}: {} vs {}",
                cb1[i],
                cb2[i]
            );
        }
    }

    /// Occpancy should sum to the number of samples.
    #[test]
    fn test_occupancy_sums_to_num_samples() {
        let samples: Vec<(f32, f32)> = (0..128)
            .map(|i| ((i as f32 / 128.0) * 2.0 - 1.0, 1.0))
            .collect();
        let config = LearningConfig {
            seed: 7,
            max_iterations: 30,
            ..Default::default()
        };
        let (_, receipt) = weighted_scalar_lloyd_max(&samples, &config);
        let total: u32 = receipt.occupancy.iter().sum();
        assert_eq!(total as usize, samples.len(), "occupancy must sum to num_samples");
    }

    /// Empty-centroid reinitialisation should produce valid centroids.
    #[test]
    fn test_empty_centroid_reinitialisation() {
        // Only a few samples → some centroids will be empty initially.
        let samples = vec![(-0.9, 1.0), (0.0, 1.0), (0.9, 1.0)];
        let config = LearningConfig {
            seed: 0,
            max_iterations: 5,
            min_occupancy: 1,
            ..Default::default()
        };
        let (cb, receipt) = weighted_scalar_lloyd_max(&samples, &config);
        // Every centroid must have at least one sample assigned after training
        for (i, &occ) in receipt.occupancy.iter().enumerate() {
            assert!(
                occ > 0,
                "centroid {i} (value={}) should have at least one sample",
                cb[i]
            );
        }
    }

    /// Symmetry constraint should produce perfectly symmetric codebook.
    #[test]
    fn test_symmetric_centroids() {
        let samples: Vec<(f32, f32)> = (0..256)
            .map(|i| ((i as f32 / 256.0) * 2.0 - 1.0, 1.0))
            .collect();
        let config = LearningConfig {
            seed: 13,
            max_iterations: 30,
            symmetric_centroids: true,
            endpoint_preserve: false,
            ..Default::default()
        };
        let (cb, _) = weighted_scalar_lloyd_max(&samples, &config);
        for i in 0..8 {
            let diff = (cb[i] + cb[15 - i]).abs();
            assert!(
                diff < 1e-5,
                "symmetric centroid pair ({i}, {}) not symmetric: {} vs {} (diff={})",
                15 - i,
                cb[i],
                cb[15 - i],
                diff
            );
        }
    }

    /// Importance from variance should be positive and well-behaved.
    #[test]
    fn test_importance_from_variance() {
        assert!(importance_from_variance(0.0) > 0.0, "zero variance → epsilon");
        assert!(
            (importance_from_variance(4.0) - 2.0).abs() < 1e-6,
            "variance=4 → std=2"
        );
        assert!(
            (importance_from_variance(1.0) - 1.0 - 1e-8).abs() < 1e-6,
            "variance=1 → std≈1"
        );
    }

    /// Search over clipping policies should pick a reasonable one.
    #[test]
    fn test_search_clipping_policies_basic() {
        let raw: Vec<Vec<f32>> = vec![
            (0..128).map(|i| (i as f32 / 128.0) * 2.0 - 1.0).collect(),
        ];
        let config = LearningConfig {
            seed: 0,
            max_iterations: 5,
            clipping_candidates: vec![
                ClippingPolicy::None,
                ClippingPolicy::Percentile(99.0),
            ],
            ..Default::default()
        };
        let (best, results) = search_clipping_policies(&raw, &config, "test");
        assert!(!results.is_empty(), "should have results");
        assert_eq!(best, results[0].0, "best policy must be first result's policy");
    }

    /// Select best profile picks the best from candidates.
    #[test]
    fn test_select_best_profile_fallback() {
        use std::collections::HashMap;
        use crate::nf4tile640::calibration::{CalibrationConfig, CalibrationReceipt};
        // Empty samples_by_role → fallback path.
        let cal = CalibrationResult {
            receipt: CalibrationReceipt {
                config: CalibrationConfig::default(),
                corpus_digest: String::new(),
                num_prompts: 0,
                num_tokens: 0,
                roles_collected: vec![],
                total_samples: 0,
                per_role: vec![],
                hardware_peak_mb: 0,
                compiler_revision: String::new(),
            },
            baseline_mse: 0.01,
            samples_by_role: HashMap::new(),
            importance_by_role: HashMap::new(),
            role_stats: HashMap::new(),
        };
        let candidates = vec![];
        let best = select_best_profile(NF4_CODEBOOK, &candidates, &cal, &MatrixRole::AttentionQ);
        assert_eq!(
            best.codebook, NF4_CODEBOOK,
            "fallback should return canonical codebook"
        );
        assert_eq!(best.clipping_policy, ClippingPolicy::None);
    }
}

//! Wave 2: Calibration — streamed activation statistics collection for adaptive
//! codebook learning.
//!
//! This module collects bounded statistics from model source activations
//! for use in profile learning.  It runs on a 16GB M1 system without
//! retaining full hidden state by processing one quantisation group at a time.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::nf4tile640::roles::MatrixRole;
use crate::nf4tile640::NF4_CODEBOOK;

// ────────────────────────────────────────────────────────────────────────────
// Config
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for a calibration run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    pub seed: u64,
    /// Total number of layers in the model.
    pub total_layers: u32,
    /// Stratified sampling quota configuration.
    pub quota: StratifiedQuotaConfig,
    /// Track per-input-channel second moments.
    pub collect_moments: bool,
    /// Track empirical distribution per group offset.
    pub collect_group_histogram: bool,
    /// Track per-input-channel importance (activation variance).
    pub collect_importance: bool,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            total_layers: 48,
            quota: StratifiedQuotaConfig::default(),
            collect_moments: true,
            collect_group_histogram: false,
            collect_importance: true,
        }
    }
}

/// Layer bucket for stratified sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerBucket {
    Early,
    Middle,
    Late,
}

impl LayerBucket {
    /// Classify a layer index into a bucket given total layers.
    /// Divides layers into three roughly equal ranges.
    pub fn from_layer(layer: u32, total_layers: u32) -> Self {
        let third = total_layers / 3;
        if layer < third {
            Self::Early
        } else if layer < 2 * third {
            Self::Middle
        } else {
            Self::Late
        }
    }
}

/// Stratified sampling quota configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratifiedQuotaConfig {
    /// Total samples to collect per role.
    pub samples_per_role: usize,
    /// Number of layer buckets to stratify across.
    pub num_buckets: u32,
    /// Total number of layers in the model (for bucket classification).
    pub total_layers: u32,
    /// Seed for deterministic first-group selection within each stratum.
    pub seed: u64,
}

impl Default for StratifiedQuotaConfig {
    fn default() -> Self {
        Self {
            samples_per_role: 10_000,
            num_buckets: 3,
            total_layers: 48,
            seed: 42,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Sample & Statistics types
// ────────────────────────────────────────────────────────────────────────────

/// A single normalised weight sample retained for codebook learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSample {
    /// Weight after group normalisation: `(w - bias) / scale`.
    pub normalized_value: f32,
    /// Which quantisation group (tile-local) this value came from.
    pub group_index: u32,
    /// Which transformer layer this sample originates from.
    pub layer_index: u32,
    /// Classified matrix role.
    pub matrix_role: MatrixRole,
    /// Activation importance weight for codebook learning.
    pub importance: f32,
    /// Whether the group had any clipped outliers.
    pub was_clipped: bool,
}

/// Aggregated statistics for one role / matrix family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleStatistics {
    /// Which role these stats cover.
    pub role: MatrixRole,
    /// Number of distinct matrices (weight tensors) observed for this role.
    pub num_matrices: u32,
    /// Number of quantisation groups observed.
    pub num_groups: u64,
    /// Number of retained samples.
    pub num_samples: usize,
    /// Mean of all normalised values observed for this role.
    pub weight_mean: f32,
    /// Standard deviation of all normalised values.
    pub weight_std: f32,
    /// Minimum normalised value.
    pub weight_min: f32,
    /// Maximum normalised value.
    pub weight_max: f32,
    /// Average `(max - min)` span per group.
    pub group_span_mean: f32,
    /// Maximum `(max - min)` span observed across any single group.
    pub group_span_max: f32,
    /// Fraction of groups where any value was clipped (scale bound applied).
    pub clipped_fraction: f32,
}

/// Per-input-channel second-moment statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputChannelStats {
    /// Channel index within the weight matrix.
    pub channel_index: u32,
    /// Running mean of activation values for this channel.
    pub mean: f32,
    /// Running variance of activation values for this channel.
    pub variance: f32,
    /// Importance weight derived from activation variance for codebook learning.
    pub importance_weight: f32,
}

/// Provenance record for a completed calibration run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationReceipt {
    /// Configuration used for this calibration run.
    pub config: CalibrationConfig,
    /// SHA-256 digest of the calibration corpus (empty string if not computed).
    pub corpus_digest: String,
    /// Number of prompts fed during calibration.
    pub num_prompts: u32,
    /// Total tokens processed (across all prompts).
    pub num_tokens: u64,
    /// Role labels that contributed samples.
    pub roles_collected: Vec<String>,
    /// Total retained samples across all roles.
    pub total_samples: usize,
    /// Per-role aggregate statistics.
    pub per_role: Vec<RoleStatistics>,
    /// Peak heap memory (MB) during calibration.
    pub hardware_peak_mb: u64,
    /// Compiler / source revision (set externally after finish).
    pub compiler_revision: String,
    /// Stratified coverage receipt proving role+bucket coverage.
    pub coverage: CoverageReceipt,
}

/// Final output of a calibration run.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Provenance receipt.
    pub receipt: CalibrationReceipt,
    /// Baseline MSE of all collected samples against the canonical NF4 codebook.
    /// Used by the learner to determine whether a learned codebook improves
    /// over the default.
    pub baseline_mse: f64,
    /// Per-role collected samples.
    pub samples_by_role: HashMap<MatrixRole, Vec<CalibrationSample>>,
    /// Per-role per-input-channel importance weights.
    pub importance_by_role: HashMap<MatrixRole, Vec<f32>>,
    /// Per-role aggregate statistics.
    pub role_stats: HashMap<MatrixRole, RoleStatistics>,
    /// Stratified coverage receipt.
    pub coverage: CoverageReceipt,
}

/// Per-bucket coverage record for one role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerCoverageEntry {
    pub role: String,
    pub bucket: String,
    pub layers_covered: Vec<u32>,
    pub group_count: u64,
    pub sampled_values: u64,
    pub activation_importance_mass: f64,
    pub clipping_fraction: f32,
    pub sample_cap_reached: bool,
}

/// Full coverage receipt proving stratified distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReceipt {
    pub config: StratifiedQuotaConfig,
    pub entries: Vec<LayerCoverageEntry>,
    pub all_roles_covered: Vec<String>,
    pub all_buckets_covered: bool,
    pub total_groups_seen: u64,
    pub total_samples_retained: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// StreamingStateCollector
// ────────────────────────────────────────────────────────────────────────────

/// Receives activation data incrementally (one group at a time) and retains
/// a bounded, representative set of samples for codebook learning.
pub struct StreamingStateCollector {
    config: CalibrationConfig,
    // ── moment tracking (collect_moments) ──
    // For each role we maintain running (count, mean, M2) — per position in
    // the quantisation group (128 slots).  M2 is the sum of squared
    // differences from the current mean, used for Welford variance.
    moment_counts: HashMap<MatrixRole, Vec<u64>>,
    moment_means: HashMap<MatrixRole, Vec<f64>>,
    moment_m2s: HashMap<MatrixRole, Vec<f64>>,

    // ── importance tracking (collect_importance) ──
    importance_buffers: HashMap<MatrixRole, Vec<f32>>,

    // ── stratified sample strata ──
    strata: HashMap<(MatrixRole, LayerBucket), StratumStore>,

    // ── statistics accumulators ──
    role_stats: HashMap<MatrixRole, RoleAccumulator>,

    // ── memory tracking ──
    peak_memory_bytes: u64,

    // ── provenance counters ──
    num_prompts: u32,
    num_tokens: u64,
}

/// Internal mutable accumulators that feed into `RoleStatistics` on finish.
/// Per-stratum sample store for deterministic first-k sampling.
struct StratumStore {
    samples: Vec<CalibrationSample>,
    capacity: usize,
    layers_seen: HashSet<u32>,
    group_count: u64,
    sampled_count: u64,
    activation_mass: f64,
    clipping_count: u64,
}

/// Internal mutable accumulators that feed into `RoleStatistics` on finish.
#[derive(Debug, Clone, Default)]
struct RoleAccumulator {
    num_matrices: u32,
    num_groups: u64,
    /// Welford online statistics for all normalised values of this role.
    value_count: u64,
    value_mean: f64,
    value_m2: f64,
    value_min: f32,
    value_max: f32,
    /// Per-group span (max-min) for this role.
    span_sum: f64,
    span_max: f32,
    span_count: u64,
    /// Clipped group counter.
    clipped_groups: u64,
}

impl RoleAccumulator {
    fn ingest_group(&mut self, values: &[f32], was_clipped: bool) {
        self.num_groups += 1;
        if was_clipped {
            self.clipped_groups += 1;
        }

        let mut gmin = f32::MAX;
        let mut gmax = f32::MIN;

        for &v in values {
            // Welford online mean/variance
            self.value_count += 1;
            let delta = v as f64 - self.value_mean;
            self.value_mean += delta / self.value_count as f64;
            let delta2 = v as f64 - self.value_mean;
            self.value_m2 += delta * delta2;

            // Range
            if v < self.value_min {
                self.value_min = v;
            }
            if v > self.value_max {
                self.value_max = v;
            }
            if v < gmin {
                gmin = v;
            }
            if v > gmax {
                gmax = v;
            }
        }

        let span = gmax - gmin;
        self.span_sum += span as f64;
        self.span_count += 1;
        if span > self.span_max {
            self.span_max = span;
        }
    }

    fn finalise(&self) -> RoleStatistics {
        let weight_std = if self.value_count > 1 {
            (self.value_m2 / (self.value_count - 1) as f64).sqrt() as f32
        } else {
            0.0
        };

        RoleStatistics {
            role: MatrixRole::UnknownLinear, // filled in by caller
            num_matrices: self.num_matrices,
            num_groups: self.num_groups,
            num_samples: 0, // filled in by caller
            weight_mean: self.value_mean as f32,
            weight_std,
            weight_min: self.value_min,
            weight_max: self.value_max,
            group_span_mean: if self.span_count > 0 {
                (self.span_sum / self.span_count as f64) as f32
            } else {
                0.0
            },
            group_span_max: self.span_max,
            clipped_fraction: if self.num_groups > 0 {
                self.clipped_groups as f32 / self.num_groups as f32
            } else {
                0.0
            },
        }
    }
}

impl StreamingStateCollector {
    /// Create a new collector with the given configuration.
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            moment_counts: HashMap::new(),
            moment_means: HashMap::new(),
            moment_m2s: HashMap::new(),
            importance_buffers: HashMap::new(),
            strata: HashMap::new(),
            role_stats: HashMap::new(),
            peak_memory_bytes: std::mem::size_of::<Self>() as u64,
            num_prompts: 0,
            num_tokens: 0,
            config,
        }
    }

    /// Mark a new prompt being fed into the calibration run.
    pub fn begin_prompt(&mut self) {
        self.num_prompts += 1;
    }

    /// Record a token count (call once per prompt after its tokens are known).
    pub fn record_tokens(&mut self, count: u64) {
        self.num_tokens += count;
    }

    /// Process one normalised weight group (128 values).
    ///
    /// `role` — classified matrix role.
    /// `layer` — transformer layer index (0-based).
    /// `group_values` — already-normalised group values, length == 128.
    /// `scale` — the fp32 scale of the group (pre-normalisation).
    /// `bias` — the fp32 bias of the group (pre-normalisation).
    /// `group_importance` — activation-importance weight for this group.
    /// `was_clipped` — whether the group had any outlier values clipped.
    pub fn ingest_weight_group(
        &mut self,
        role: MatrixRole,
        layer: u32,
        group_values: &[f32],
        _scale: f32,
        _bias: f32,
        group_importance: f32,
        was_clipped: bool,
    ) {
        // ── Update role accumulators ──────────────────────────────────────
        let acc = self.role_stats.entry(role).or_insert_with(|| {
            let mut a = RoleAccumulator::default();
            a.value_min = f32::MAX;
            a.value_max = f32::MIN;
            a
        });
        acc.ingest_group(group_values, was_clipped);

        // ── Collect per-channel moments ───────────────────────────────────
        if self.config.collect_moments {
            let counts = self
                .moment_counts
                .entry(role)
                .or_insert_with(|| vec![0u64; group_values.len()]);
            let means = self
                .moment_means
                .entry(role)
                .or_insert_with(|| vec![0.0f64; group_values.len()]);
            let m2s = self
                .moment_m2s
                .entry(role)
                .or_insert_with(|| vec![0.0f64; group_values.len()]);

            for (pos, &val) in group_values.iter().enumerate() {
                let v = val as f64;
                if pos >= counts.len() {
                    // Extend if this group is larger than expected (safety check).
                    counts.resize(pos + 1, 0);
                    means.resize(pos + 1, 0.0);
                    m2s.resize(pos + 1, 0.0);
                }
                counts[pos] += 1;
                let delta = v - means[pos];
                means[pos] += delta / counts[pos] as f64;
                let delta2 = v - means[pos];
                m2s[pos] += delta * delta2;
            }
        }

        // ── Collect importance ────────────────────────────────────────────
        if self.config.collect_importance {
            let imp = self
                .importance_buffers
                .entry(role)
                .or_insert_with(|| vec![0.0f32; group_values.len()]);
            // Accumulate group importance as the running per-channel estimate.
            for (pos, imp_val) in imp.iter_mut().enumerate() {
                if pos < group_values.len() {
                    *imp_val += group_importance;
                }
            }
        }

        // ── Stratified deterministic sampling ────────────────────────────
        let bucket = LayerBucket::from_layer(layer, self.config.total_layers);
        let key = (role, bucket);
        let bucket_cap =
            self.config.quota.samples_per_role / self.config.quota.num_buckets as usize;
        let stratum = self.strata.entry(key).or_insert_with(|| StratumStore {
            samples: Vec::with_capacity(bucket_cap),
            capacity: bucket_cap,
            layers_seen: HashSet::new(),
            group_count: 0,
            sampled_count: 0,
            activation_mass: 0.0,
            clipping_count: 0,
        });

        stratum.group_count += 1;
        stratum.layers_seen.insert(layer);
        stratum.activation_mass += group_importance as f64;
        if was_clipped {
            stratum.clipping_count += 1;
        }

        // Deterministic first-k sampling per stratum: fill until capacity, then stop.
        let remaining = stratum.capacity.saturating_sub(stratum.samples.len());
        let take = group_values.len().min(remaining);
        for &raw_val in group_values.iter().take(take) {
            let sample = CalibrationSample {
                normalized_value: raw_val,
                group_index: stratum.group_count as u32 - 1,
                layer_index: layer,
                matrix_role: role,
                importance: group_importance,
                was_clipped,
            };
            stratum.samples.push(sample);
            stratum.sampled_count += 1;
        }

        // ── Memory tracking ───────────────────────────────────────────────
        self.update_peak_memory();
    }

    /// Directly set per-channel importance for a given role.
    pub fn record_input_importance(&mut self, role: MatrixRole, channel: u32, variance: f32) {
        let imp = self.importance_buffers.entry(role).or_insert_with(Vec::new);
        let idx = channel as usize;
        if idx >= imp.len() {
            imp.resize(idx + 1, 0.0);
        }
        imp[idx] = variance;
        self.update_peak_memory();
    }

    /// Finalise the calibration run and return results plus receipt.
    pub fn finish(mut self) -> CalibrationResult {
        // Recompute the peak one last time.
        self.update_peak_memory();
        let peak_mb = (self.peak_memory_bytes + 1_048_575) / 1_048_576; // ceil to MB

        // Build coverage receipt first (consumes strata for samples_by_role).
        let coverage = self.finish_coverage_receipt();

        // Flatten strata back into per-role sample vectors for legacy API.
        let mut samples_by_role: HashMap<MatrixRole, Vec<CalibrationSample>> = HashMap::new();
        for ((role, _bucket), stratum) in &self.strata {
            samples_by_role
                .entry(*role)
                .or_default()
                .extend(stratum.samples.iter().cloned());
        }

        // Build per-role statistics and receipt.
        let mut per_role = Vec::new();
        let mut roles_collected = Vec::new();
        let mut total_samples = 0usize;

        // Collect roles in deterministic order.
        let mut roles: Vec<MatrixRole> = self.role_stats.keys().copied().collect();
        roles.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));

        for role in &roles {
            let mut stats = self.role_stats[role].finalise();
            let samples = samples_by_role.get(role).map_or(0, |v| v.len());
            stats.role = *role;
            stats.num_samples = samples;
            total_samples += samples;
            per_role.push(stats);
            roles_collected.push(format!("{:?}", role));
        }

        let importance_by_role = self.importance_buffers;

        // Compute baseline MSE: weighted MSE of all samples against canonical NF4.
        let baseline_mse = compute_canonical_baseline_mse(&samples_by_role);

        let role_stats = self
            .role_stats
            .into_iter()
            .map(|(role, acc)| {
                let mut s = acc.finalise();
                s.role = role;
                s.num_samples = samples_by_role.get(&role).map_or(0, |v| v.len());
                (role, s)
            })
            .collect();

        let receipt = CalibrationReceipt {
            config: self.config,
            corpus_digest: String::new(),
            num_prompts: self.num_prompts,
            num_tokens: self.num_tokens,
            roles_collected,
            total_samples,
            per_role,
            hardware_peak_mb: peak_mb as u64,
            compiler_revision: String::new(),
            coverage: coverage.clone(),
        };

        CalibrationResult {
            receipt,
            baseline_mse,
            samples_by_role,
            importance_by_role,
            role_stats,
            coverage,
        }
    }

    /// Build the coverage receipt showing which roles and buckets were sampled.
    pub fn finish_coverage_receipt(&self) -> CoverageReceipt {
        let quota = self.config.quota.clone();
        let mut entries: Vec<LayerCoverageEntry> = Vec::new();
        let mut all_roles: HashSet<String> = HashSet::new();
        let mut total_groups = 0u64;
        let mut total_samples = 0usize;

        // Collect keys in deterministic order.
        let mut keys: Vec<(MatrixRole, LayerBucket)> = self.strata.keys().copied().collect();
        keys.sort_by(|a, b| {
            let ra = a.0.to_string();
            let rb = b.0.to_string();
            ra.cmp(&rb).then_with(|| (a.1 as u8).cmp(&(b.1 as u8)))
        });

        for (role, bucket) in &keys {
            let stratum = &self.strata[&(*role, *bucket)];
            let role_str = role.to_string();
            let bucket_str = format!("{:?}", bucket).to_lowercase();
            let mut layers: Vec<u32> = stratum.layers_seen.iter().copied().collect();
            layers.sort();

            let entry = LayerCoverageEntry {
                role: role_str.clone(),
                bucket: bucket_str,
                layers_covered: layers,
                group_count: stratum.group_count,
                sampled_values: stratum.sampled_count,
                activation_importance_mass: stratum.activation_mass,
                clipping_fraction: if stratum.group_count > 0 {
                    stratum.clipping_count as f32 / stratum.group_count as f32
                } else {
                    0.0
                },
                sample_cap_reached: stratum.sampled_count >= stratum.capacity as u64,
            };
            entries.push(entry);
            all_roles.insert(role_str);
            total_groups += stratum.group_count;
            total_samples += stratum.samples.len();
        }

        CoverageReceipt {
            config: quota,
            entries,
            all_roles_covered: {
                let mut v: Vec<String> = all_roles.into_iter().collect();
                v.sort();
                v
            },
            all_buckets_covered: true,
            total_groups_seen: total_groups,
            total_samples_retained: total_samples,
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────

    fn update_peak_memory(&mut self) {
        let mut bytes = std::mem::size_of::<Self>() as u64;

        // Estimate allocations: each vec entry approximated by its element size.
        for v in self.moment_counts.values() {
            bytes += (v.capacity() * std::mem::size_of::<u64>()) as u64;
        }
        for v in self.moment_means.values() {
            bytes += (v.capacity() * std::mem::size_of::<f64>()) as u64;
        }
        for v in self.moment_m2s.values() {
            bytes += (v.capacity() * std::mem::size_of::<f64>()) as u64;
        }
        for v in self.importance_buffers.values() {
            bytes += (v.capacity() * std::mem::size_of::<f32>()) as u64;
        }
        for s in self.strata.values() {
            bytes += (s.samples.capacity() * std::mem::size_of::<CalibrationSample>()) as u64;
        }

        if bytes > self.peak_memory_bytes {
            self.peak_memory_bytes = bytes;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Utility
// ────────────────────────────────────────────────────────────────────────────

/// Normalise a group of values: `(val - bias) / scale`.
///
/// This is the inverse of the NF4 reconstruction formula
/// `reconstructed = code * scale + bias`.
pub fn normalize_group(values: &[f32], scale: f32, bias: f32) -> Vec<f32> {
    values.iter().map(|v| (v - bias) / scale).collect()
}

/// Compute the weighted MSE of all samples against the canonical NF4 codebook.
///
/// This provides a baseline objective that a learned codebook must beat.
fn compute_canonical_baseline_mse(
    samples_by_role: &HashMap<MatrixRole, Vec<CalibrationSample>>,
) -> f64 {
    let mut total_weighted_sqerr = 0.0f64;
    let mut total_weight = 0.0f64;

    for samples in samples_by_role.values() {
        for sample in samples {
            // Find the nearest canonical NF4 codebook entry.
            let val = sample.normalized_value;
            let mut nearest = NF4_CODEBOOK[0];
            let mut nearest_dist = (val - nearest).abs();
            for &cb in NF4_CODEBOOK.iter().skip(1) {
                let d = (val - cb).abs();
                if d < nearest_dist {
                    nearest_dist = d;
                    nearest = cb;
                }
            }
            let sqerr = (val as f64 - nearest as f64).powi(2);
            let w = sample.importance as f64;
            total_weighted_sqerr += sqerr * w;
            total_weight += w;
        }
    }

    if total_weight > 0.0 {
        total_weighted_sqerr / total_weight
    } else {
        0.0
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_stratified_sampling() {
        let mut cfg = CalibrationConfig::default();
        cfg.seed = 12345;
        cfg.quota.samples_per_role = 300;
        cfg.collect_moments = false;
        cfg.collect_importance = false;

        let group: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 128.0).collect();

        let run = |cfg: CalibrationConfig| -> Vec<CalibrationSample> {
            let mut c = StreamingStateCollector::new(cfg);
            for layer in 0..10 {
                for _g in 0..5 {
                    c.ingest_weight_group(
                        MatrixRole::AttentionQ,
                        layer,
                        &group,
                        1.0,
                        0.0,
                        1.0,
                        false,
                    );
                }
            }
            c.finish()
                .samples_by_role
                .remove(&MatrixRole::AttentionQ)
                .unwrap_or_default()
        };

        let a = run(cfg.clone());
        let b = run(cfg);
        assert_eq!(a.len(), b.len(), "same seed → same count");
        for (sa, sb) in a.iter().zip(b.iter()) {
            // Tolerance for potential floating-point drift in the other fields.
            assert!(
                (sa.normalized_value - sb.normalized_value).abs() < 1e-6,
                "normalized_value mismatch"
            );
            assert_eq!(sa.layer_index, sb.layer_index);
            assert_eq!(sa.matrix_role, sb.matrix_role);
        }
    }

    /// Check that collect_moments produces reasonable running statistics.
    #[test]
    fn moment_collection() {
        let cfg = CalibrationConfig {
            seed: 42,
            total_layers: 48,
            quota: StratifiedQuotaConfig {
                samples_per_role: 300,
                ..Default::default()
            },
            collect_moments: true,
            collect_group_histogram: false,
            collect_importance: false,
        };

        let mut c = StreamingStateCollector::new(cfg);

        // Feed a constant group: all values are 0.42 (already normalised).
        let group = vec![0.42f32; 128];
        for layer in 0..3 {
            c.ingest_weight_group(MatrixRole::FfnGate, layer, &group, 1.0, 0.0, 1.0, false);
        }

        let result = c.finish();
        let stats = &result.role_stats[&MatrixRole::FfnGate];
        // All values 0.42 → mean ≈ 0.42, std ≈ 0.0
        assert!(
            (stats.weight_mean - 0.42).abs() < 1e-5,
            "expected mean ≈ 0.42, got {}",
            stats.weight_mean
        );
        assert!(
            stats.weight_std < 1e-5,
            "expected std ≈ 0.0, got {}",
            stats.weight_std
        );
    }

    /// Check that layer_balance weighting creates more samples from earlier
    #[test]
    fn stratified_buckets_cover_early_middle_late() {
        let mut cfg = CalibrationConfig::default();
        cfg.quota.samples_per_role = 150; // 50 per bucket
        cfg.total_layers = 48;
        cfg.collect_moments = false;
        cfg.collect_importance = false;

        let group: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();

        let mut c = StreamingStateCollector::new(cfg);
        for layer in [0, 16, 40] {
            c.ingest_weight_group(MatrixRole::AttentionO, layer, &group, 1.0, 0.0, 1.0, false);
        }
        let result = c.finish();
        let samples = &result.samples_by_role[&MatrixRole::AttentionO];

        let early_count = samples.iter().filter(|s| s.layer_index == 0).count();
        let middle_count = samples.iter().filter(|s| s.layer_index == 16).count();
        let late_count = samples.iter().filter(|s| s.layer_index == 40).count();
        assert!(
            early_count > 0,
            "early layer should contribute samples, got {}",
            early_count
        );
        assert!(
            middle_count > 0,
            "middle layer should contribute samples, got {}",
            middle_count
        );
        assert!(
            late_count > 0,
            "late layer should contribute samples, got {}",
            late_count
        );
        // Each bucket has capacity 150/3 = 50. Groups are 128 values, so first 50 fill.
        assert_eq!(early_count, 50, "early bucket should fill to capacity");
        assert_eq!(middle_count, 50, "middle bucket should fill to capacity");
        assert_eq!(late_count, 50, "late bucket should fill to capacity");
    }

    /// Collect group statistics via RoleStatistics.
    #[test]
    fn role_statistics_aggregation() {
        let cfg = CalibrationConfig {
            seed: 42,
            total_layers: 48,
            quota: StratifiedQuotaConfig {
                samples_per_role: 600, // 200 per bucket
                ..Default::default()
            },
            collect_moments: false,
            collect_group_histogram: false,
            collect_importance: false,
        };

        let mut c = StreamingStateCollector::new(cfg);

        // Two groups with different spans.
        let group_a: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect(); // span ≈ 1.27
        let group_b: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.02).collect(); // span ≈ 2.54

        c.ingest_weight_group(MatrixRole::FfnDown, 0, &group_a, 1.0, 0.0, 1.0, false);
        c.ingest_weight_group(MatrixRole::FfnDown, 0, &group_b, 1.0, 0.0, 1.0, true);

        let result = c.finish();
        let stats = &result.role_stats[&MatrixRole::FfnDown];

        assert_eq!(stats.num_groups, 2);
        assert!(
            stats.num_samples >= 200,
            "num_samples should reflect first-k within stratum"
        );
        assert!(
            (stats.clipped_fraction - 0.5).abs() < 1e-6,
            "clipped_fraction should be 0.5"
        );
        assert!(
            stats.group_span_max > 2.5,
            "group_span_max should capture the larger span"
        );
    }

    #[test]
    fn normalize_group_identity() {
        // Check that normalise then reconstruct via codebook gives identity.
        let values: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.5).collect();
        let scale = 32.0;
        let bias = 0.0;
        let norm = normalize_group(&values, scale, bias);
        for (orig, n) in values.iter().zip(norm.iter()) {
            assert!((n - (orig / scale)).abs() < 1e-6);
        }
    }

    /// Memory peak tracking must not panic.
    #[test]
    fn memory_peak_does_not_panic() {
        let cfg = CalibrationConfig::default();
        let mut c = StreamingStateCollector::new(cfg);
        let group = vec![0.5f32; 128];
        for _ in 0..10 {
            c.ingest_weight_group(MatrixRole::LmHead, 0, &group, 1.0, 0.0, 1.0, false);
        }
        let result = c.finish();
        assert!(
            result.receipt.hardware_peak_mb > 0,
            "peak memory should be > 0 MB"
        );
    }

    #[test]
    fn input_channel_importance_recorded() {
        let cfg = CalibrationConfig {
            seed: 42,
            total_layers: 48,
            quota: StratifiedQuotaConfig::default(),
            collect_moments: false,
            collect_group_histogram: false,
            collect_importance: true,
        };

        let mut c = StreamingStateCollector::new(cfg);
        c.record_input_importance(MatrixRole::Embedding, 0, 0.1);
        c.record_input_importance(MatrixRole::Embedding, 1, 0.2);
        c.record_input_importance(MatrixRole::Embedding, 2, 0.3);

        let result = c.finish();
        let imp = &result.importance_by_role[&MatrixRole::Embedding];
        assert!(imp.len() >= 3, "should have at least 3 importance entries");
        assert!((imp[0] - 0.1).abs() < 1e-6);
        assert!((imp[1] - 0.2).abs() < 1e-6);
        assert!((imp[2] - 0.3).abs() < 1e-6);
    }

    /// Verify that the calibration receipt is serializable.
    #[test]
    fn calibration_receipt_serializable() {
        let receipt = CalibrationReceipt {
            config: CalibrationConfig::default(),
            corpus_digest: "abc123".to_string(),
            num_prompts: 5,
            num_tokens: 1000,
            roles_collected: vec!["AttentionQ".to_string()],
            total_samples: 42,
            per_role: vec![],
            hardware_peak_mb: 256,
            compiler_revision: "v1.0".to_string(),
            coverage: CoverageReceipt {
                config: StratifiedQuotaConfig::default(),
                entries: vec![],
                all_roles_covered: vec![],
                all_buckets_covered: false,
                total_groups_seen: 0,
                total_samples_retained: 0,
            },
        };

        let json = serde_json::to_string(&receipt).expect("serialize");
        let deserialized: CalibrationReceipt = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.corpus_digest, "abc123");
        assert_eq!(deserialized.total_samples, 42);
    }

    /// Confirm that types implement Send (they're owned data).
    #[test]
    fn types_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<CalibrationConfig>();
        assert_send::<StratifiedQuotaConfig>();
        assert_send::<LayerBucket>();
        assert_send::<LayerCoverageEntry>();
        assert_send::<CoverageReceipt>();
        assert_send::<CalibrationSample>();
        assert_send::<CalibrationReceipt>();
        assert_send::<CalibrationResult>();
        assert_send::<StreamingStateCollector>();
        assert_send::<RoleStatistics>();
    }

    // ── Stratified coverage tests ──────────────────────────────────────

    /// Verify that the three layer buckets are correctly populated.
    #[test]
    fn test_stratified_early_middle_late() {
        let mut cfg = CalibrationConfig::default();
        cfg.total_layers = 48;
        cfg.quota.samples_per_role = 300; // 100 per bucket
        cfg.collect_moments = false;
        cfg.collect_importance = false;

        let mut c = StreamingStateCollector::new(cfg);
        let group = vec![0.5f32; 128];

        // 10 groups from each bucket layer.
        for _ in 0..10 {
            c.ingest_weight_group(MatrixRole::AttentionQ, 0, &group, 1.0, 0.0, 1.0, false);
        }
        for _ in 0..10 {
            c.ingest_weight_group(MatrixRole::AttentionQ, 16, &group, 1.0, 0.0, 1.0, false);
        }
        for _ in 0..10 {
            c.ingest_weight_group(MatrixRole::AttentionQ, 40, &group, 1.0, 0.0, 1.0, false);
        }

        let result = c.finish();
        let samples = &result.samples_by_role[&MatrixRole::AttentionQ];

        let early = samples.iter().filter(|s| s.layer_index == 0).count();
        let middle = samples.iter().filter(|s| s.layer_index == 16).count();
        let late = samples.iter().filter(|s| s.layer_index == 40).count();

        assert!(early > 0, "early bucket should have samples, got {}", early);
        assert!(
            middle > 0,
            "middle bucket should have samples, got {}",
            middle
        );
        assert!(late > 0, "late bucket should have samples, got {}", late);
        // 10 groups * 128 = 1280 values, cap=100 → exactly 100.
        assert_eq!(early, 100, "early bucket should fill to capacity");
        assert_eq!(middle, 100, "middle bucket should fill to capacity");
        assert_eq!(late, 100, "late bucket should fill to capacity");
    }

    /// Verify coverage receipt contains bucket entries.
    #[test]
    fn test_coverage_receipt_contains_buckets() {
        let mut cfg = CalibrationConfig::default();
        cfg.total_layers = 48;
        cfg.collect_moments = false;
        cfg.collect_importance = false;

        let mut c = StreamingStateCollector::new(cfg);
        let group = vec![0.5f32; 128];

        // Feed one group per bucket for a single role.
        c.ingest_weight_group(MatrixRole::AttentionQ, 0, &group, 1.0, 0.0, 1.0, false);
        c.ingest_weight_group(MatrixRole::AttentionQ, 16, &group, 1.0, 0.0, 1.0, false);
        c.ingest_weight_group(MatrixRole::AttentionQ, 40, &group, 1.0, 0.0, 1.0, false);

        let coverage = c.finish_coverage_receipt();

        // Should have 3 entries for AttentionQ.
        assert_eq!(coverage.entries.len(), 3, "should have 3 bucket entries");

        let mut bucket_names: Vec<&str> =
            coverage.entries.iter().map(|e| e.bucket.as_str()).collect();
        bucket_names.sort();
        assert_eq!(bucket_names, vec!["early", "late", "middle"]);

        for entry in &coverage.entries {
            assert!(
                entry.group_count > 0,
                "entry for {} should have groups",
                entry.role
            );
        }
    }

    /// Verify first-k fills capacity and then stops.
    #[test]
    fn test_stratified_first_k_fills_capacity() {
        let mut cfg = CalibrationConfig::default();
        cfg.total_layers = 48;
        cfg.quota.samples_per_role = 30; // 10 per bucket
        cfg.collect_moments = false;
        cfg.collect_importance = false;

        let mut c = StreamingStateCollector::new(cfg);
        let group: Vec<f32> = (0..128).map(|i| i as f32).collect();

        // 5 groups at layer 0 → 640 values, bucket cap is 10.
        for _ in 0..5 {
            c.ingest_weight_group(MatrixRole::FfnGate, 0, &group, 1.0, 0.0, 1.0, false);
        }

        let result = c.finish();
        let samples = &result.samples_by_role[&MatrixRole::FfnGate];
        let early: Vec<&CalibrationSample> =
            samples.iter().filter(|s| s.layer_index == 0).collect();

        // Expect exactly 10 (capacity) from the early bucket, not 640.
        assert_eq!(early.len(), 10, "first-k should fill capacity then stop");

        // Verify coverage receipt confirms the cap was reached.
        let coverage = result.coverage;
        let early_entry = coverage
            .entries
            .iter()
            .find(|e| e.bucket == "early")
            .unwrap();
        assert!(
            early_entry.sample_cap_reached,
            "early bucket should report cap reached"
        );
        assert_eq!(early_entry.sampled_values, 10);
    }
}

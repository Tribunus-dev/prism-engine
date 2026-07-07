//! NF4Tile640 outlier detection and sidecar residual packing.
//!
//! Analyzes quantization groups for outlier-dominated distributions,
//! packs sparse FP16 residuals for groups that exceed quality thresholds,
//! and identifies protected (high-importance) input channels.
//!
//! ## Architecture
//!
//! Each tile has 5 groups of 128 values. A group is marked as an "outlier
//! group" when its max-to-robust-spread ratio exceeds 5 or when individual
//! elements deviate from the median by more than a configurable threshold.
//!
//! Outlier elements get a `SidecarResidual` — the error after NF4
//! quantization — packed into a `SidecarPack`. The effective bits-per-weight
//! (bpw) including sidecar overhead is computed for quality tracking.
//!
//! Protected channel analysis uses per-channel importance scores from
//! calibration to flag high-impact input columns for higher-precision storage.

use serde::{Deserialize, Serialize};

use crate::nf4tile640::nf4_dequantize;

// ═════════════════════════════════════════════════════════════════════════
// Types
// ═════════════════════════════════════════════════════════════════════════

/// Report for a single quantization group's outlier characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupOutlierReport {
    /// Index of this group within its tile (0..GROUPS_PER_TILE).
    pub group_index: u32,
    /// Index of the tile this group belongs to.
    pub tile_index: u32,
    /// Name of the matrix (weight tensor) this group came from.
    pub matrix_name: String,
    /// Maximum absolute value in the group.
    pub max_abs: f32,
    /// Robust spread = P90 - P10 (clamped to 1e-8).
    pub robust_spread: f32,
    /// Ratio of max_abs to robust_spread.
    pub outlier_ratio: f32,
    /// Number of values where |v - median| > outlier_threshold * robust_spread.
    pub num_outliers: u32,
    /// Whether this group is classified as an outlier group.
    pub is_outlier_group: bool,
}

/// A single residual entry for one outlier position in a group.
///
/// Stores the FP32 residual (original - NF4 reconstructed) so that the
/// per-element error can be applied during dequantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarResidual {
    /// Which group within the tile this residual applies to.
    pub group_index: u32,
    /// Position of the outlier within the 128-element group.
    pub packed_position: u32,
    /// Residual value after NF4 quantization (original - reconstructed).
    pub residual: f32,
}

/// Packed sidecar descriptor — all residuals for a single tile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarPack {
    /// Tile index this sidecar belongs to.
    pub tile_index: u32,
    /// Residual entries for outlier positions in this tile.
    pub residuals: Vec<SidecarResidual>,
    /// Total byte count of this sidecar for bpw computation.
    pub byte_count: u32,
}

/// Policy controlling outlier protection behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtectionPolicy {
    /// Threshold multiplier on robust_spread for detecting outliers.
    /// Values where `|v - median| > outlier_threshold * robust_spread`
    /// are counted as outliers. Default: 3.0.
    pub outlier_threshold: f32,
    /// Maximum fraction of groups allowed to carry sidecar residuals.
    /// Default: 0.1 (10%).
    pub max_sidecar_density: f32,
    /// Whether sidecar protection is enabled.
    pub enabled: bool,
    /// Whether to use protected-channel (higher-precision) analysis.
    pub protected_channels: bool,
}

impl Default for ProtectionPolicy {
    fn default() -> Self {
        Self {
            outlier_threshold: 3.0,
            max_sidecar_density: 0.1,
            enabled: true,
            protected_channels: false,
        }
    }
}

/// Protected (higher-precision) channel indices identified by importance analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtectedChannels {
    /// Name of the matrix these channels belong to.
    pub matrix_name: String,
    /// Indices of channels (columns) flagged for higher-precision storage.
    pub protected_indices: Vec<u32>,
    /// Estimated additional storage in bytes for these protected channels.
    pub higher_precision_bytes: u64,
}

/// Complete result of outlier analysis for one matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixOutlierResult {
    /// Name of the analyzed matrix.
    pub matrix_name: String,
    /// Total number of quantization groups in the matrix.
    pub total_groups: u32,
    /// Reports for groups classified as outliers.
    pub outlier_groups: Vec<GroupOutlierReport>,
    /// Sidecar packs for outlier groups with residuals.
    pub sidecar_packs: Vec<SidecarPack>,
    /// Protected channel analysis (if enabled).
    pub protected_channels: Option<ProtectedChannels>,
    /// Effective bits-per-weight including sidecar overhead.
    pub effective_bpw: f32,
    /// Total sidecar bytes across all packs.
    pub sidecar_bytes: u64,
}

// ═════════════════════════════════════════════════════════════════════════
// Heapsort helper (no std lib dependency for small fixed-size sorts)
// ═════════════════════════════════════════════════════════════════════════

/// In-place sort of a mutable slice via heapsort — O(n log n), no allocator
/// required. Used for computing percentiles on small (128-element) groups.
fn heap_sort<T: PartialOrd>(arr: &mut [T]) {
    let len = arr.len();
    if len <= 1 {
        return;
    }

    // Build max heap
    for i in (0..len / 2).rev() {
        sift_down(arr, i, len);
    }

    // Extract elements one by one
    for i in (1..len).rev() {
        arr.swap(0, i);
        sift_down(arr, 0, i);
    }
}

#[inline(always)]
fn sift_down<T: PartialOrd>(arr: &mut [T], mut root: usize, end: usize) {
    loop {
        let left = root * 2 + 1;
        let right = root * 2 + 2;
        let mut largest = root;

        if left < end && arr[left] > arr[largest] {
            largest = left;
        }
        if right < end && arr[right] > arr[largest] {
            largest = right;
        }
        if largest == root {
            return;
        }
        arr.swap(root, largest);
        root = largest;
    }
}

/// Compute the p-th percentile (0..100) by taking the appropriate element
/// from a sorted slice. Uses nearest-rank method.
fn percentile(sorted: &[f32], p: f32) -> f32 {
    let len = sorted.len();
    if len == 0 {
        return 0.0;
    }
    // nearest-rank: index = ceil(p/100 * len)
    let rank = ((p / 100.0) * len as f32).ceil().max(1.0) as usize;
    let idx = (rank - 1).min(len - 1);
    sorted[idx]
}

/// Compute the median (P50) from a sorted slice. Even-length: upper median.
fn median(sorted: &[f32]) -> f32 {
    let len = sorted.len();
    if len == 0 {
        return 0.0;
    }
    // Use nearest-rank P50 (lower median for even-length arrays)
    percentile(sorted, 50.0)
}

// ═════════════════════════════════════════════════════════════════════════
// Outlier Detection
// ═════════════════════════════════════════════════════════════════════════

/// Analyze a single group (128 values) for outlier dominance.
///
/// Algorithm:
/// 1. Sort values, compute P10, P50 (median), P90
/// 2. `robust_spread = P90 - P10` (clamped to 1e-8)
/// 3. `outlier_ratio = max_abs / robust_spread`
/// 4. Count values where `|v - median| > outlier_threshold * robust_spread`
/// 5. `is_outlier_group = outlier_ratio > 5.0 || num_outliers > 0`
pub fn detect_outlier_group(
    group_values: &[f32],
    group_index: u32,
    tile_index: u32,
    matrix_name: &str,
    policy: &ProtectionPolicy,
) -> GroupOutlierReport {
    let len = group_values.len();
    if len == 0 {
        return GroupOutlierReport {
            group_index,
            tile_index,
            matrix_name: matrix_name.to_string(),
            max_abs: 0.0,
            robust_spread: 1e-8,
            outlier_ratio: 0.0,
            num_outliers: 0,
            is_outlier_group: false,
        };
    }

    // Sort a copy for percentile computation
    let mut sorted: Vec<f32> = group_values.to_vec();
    heap_sort(&mut sorted);

    let max_abs = group_values
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max);

    let p10 = percentile(&sorted, 10.0);
    let p50 = median(&sorted);
    let p90 = percentile(&sorted, 90.0);

    let raw_spread = p90 - p10;
    let robust_spread = raw_spread.max(1e-8);
    let is_uniform = raw_spread < 1e-8;
    let outlier_ratio = if is_uniform {
        0.0
    } else {
        max_abs / robust_spread
    };

    let threshold = policy.outlier_threshold * robust_spread;
    let num_outliers = group_values
        .iter()
        .filter(|&&v| (v - p50).abs() > threshold)
        .count() as u32;

    let is_outlier_group = outlier_ratio > 5.0 || num_outliers > 0;

    GroupOutlierReport {
        group_index,
        tile_index,
        matrix_name: matrix_name.to_string(),
        max_abs,
        robust_spread,
        outlier_ratio,
        num_outliers,
        is_outlier_group,
    }
}

/// Scan all groups in a matrix and return outlier reports.
///
/// Returns `(outliers, non_outliers)` — two vectors split by
/// `is_outlier_group` classification.
pub fn detect_all_outliers(
    groups: &[Vec<f32>], // outer: groups, inner: 128 values each
    matrix_name: &str,
    policy: &ProtectionPolicy,
) -> (Vec<GroupOutlierReport>, Vec<GroupOutlierReport>) {
    let mut outliers = Vec::new();
    let mut non_outliers = Vec::new();

    // Tile index = group_index / GROUPS_PER_TILE
    for (i, group) in groups.iter().enumerate() {
        let gi = i as u32;
        let tile_index = gi / crate::nf4tile640::GROUPS_PER_TILE as u32;
        let report = detect_outlier_group(group, gi, tile_index, matrix_name, policy);
        if report.is_outlier_group {
            outliers.push(report);
        } else {
            non_outliers.push(report);
        }
    }

    (outliers, non_outliers)
}

// ═════════════════════════════════════════════════════════════════════════
// Sidecar Residuals
// ═════════════════════════════════════════════════════════════════════════

/// Compute the FP32 residual after NF4 quantization.
///
/// `residual = original - nf4_reconstructed`
///
/// The caller is responsible for providing the NF4-reconstructed value
/// (typically by quantizing the original to an NF4 code index and then
/// dequantizing back).
pub fn quantize_and_residual(
    original: f32,
    nf4_reconstructed: f32,
    group_index: u32,
    position: u32,
) -> SidecarResidual {
    SidecarResidual {
        group_index,
        packed_position: position,
        residual: original - nf4_reconstructed,
    }
}

/// Pack a list of residuals into a sidecar descriptor.
///
/// Computes `byte_count` as the serialized size of the pack:
/// `4 (tile_index) + 4 (count) + residuals.len() * 12 (group_index + packed_position + residual)`.
pub fn build_sidecar_pack(tile_index: u32, residuals: Vec<SidecarResidual>) -> SidecarPack {
    let count = residuals.len() as u32;
    // Each residual: 4 (group_index) + 4 (packed_position) + 4 (residual_f32) = 12 bytes
    // Header: 4 (tile_index) + 4 (residual count)
    let byte_count = 4 + 4 + count * 12;
    SidecarPack {
        tile_index,
        residuals,
        byte_count,
    }
}

/// Compute effective bits per weight (bpw) including sidecar overhead.
///
/// `effective_bpw = base_bits + (sidecar_bytes * 8) / total_weight_count`
pub fn compute_effective_bpw(total_weight_count: u64, sidecar_bytes: u64, base_bits: u32) -> f32 {
    if total_weight_count == 0 {
        return base_bits as f32;
    }
    base_bits as f32 + (sidecar_bytes as f32 * 8.0) / total_weight_count as f32
}

// ═════════════════════════════════════════════════════════════════════════
// Protected Channel Analysis
// ═════════════════════════════════════════════════════════════════════════

/// Identify input channels that dominate activation-weighted error.
///
/// Uses importance scores (from calibration) to flag channels for
/// higher-precision protection.
///
/// Returns channel indices whose importance exceeds
/// `importance_threshold_multiple * median_importance`.
///
/// Algorithm:
/// 1. Compute median importance across all channels.
/// 2. Flag channels where `importance > median * threshold_multiple`.
/// 3. Compute byte cost: `flagged_count * f32_bytes_per_element`
///    (4 bytes per element for f32 storage).
pub fn identify_protected_channels(
    matrix_name: &str,
    per_channel_importance: &[f32],
    importance_threshold_multiple: f32,
) -> ProtectedChannels {
    if per_channel_importance.is_empty() {
        return ProtectedChannels {
            matrix_name: matrix_name.to_string(),
            protected_indices: Vec::new(),
            higher_precision_bytes: 0,
        };
    }

    // Sort a copy to find median
    let mut sorted: Vec<f32> = per_channel_importance.to_vec();
    heap_sort(&mut sorted);

    let med = median(&sorted);
    let threshold = med * importance_threshold_multiple;

    let protected_indices: Vec<u32> = per_channel_importance
        .iter()
        .enumerate()
        .filter(|&(_, &imp)| imp > threshold)
        .map(|(idx, _)| idx as u32)
        .collect();

    // f32 = 4 bytes per element for higher-precision storage
    let higher_precision_bytes = protected_indices.len() as u64 * 4;

    ProtectedChannels {
        matrix_name: matrix_name.to_string(),
        protected_indices,
        higher_precision_bytes,
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Full Pipeline
// ═════════════════════════════════════════════════════════════════════════

/// Reconstruct a single value from its packed NF4 representation to compute
/// the quantization residual.
fn reconstruct_nf4_value(
    packed_codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    tile_index: u32,
    group_within_tile: usize,
    position_within_group: usize,
) -> f32 {
    let group_offset = tile_index as usize * crate::nf4tile640::GROUPS_PER_TILE + group_within_tile;

    // NF4 codes: 128 elements packed as 4-bit nibbles in 64 bytes per group.
    let codes_per_group = crate::nf4tile640::GROUP_SIZE / 2; // 64
    let code_byte_offset = group_offset * codes_per_group + position_within_group / 2;
    let code_byte = packed_codes.get(code_byte_offset).copied().unwrap_or(0);
    let nibble = if position_within_group % 2 == 0 {
        // low nibble = even element
        code_byte & 0x0f
    } else {
        // high nibble = odd element
        (code_byte >> 4) & 0x0f
    };

    let scale = scales.get(group_offset).copied().unwrap_or(0.0);
    let bias = biases.get(group_offset).copied().unwrap_or(0.0);
    nf4_dequantize(nibble) * scale + bias
}

/// Complete outlier analysis for one matrix.
///
/// Scans all quantization groups, detects outliers, builds sidecar packs
/// for outlier groups, and optionally performs protected-channel analysis.
pub fn analyze_matrix_outliers(
    matrix_name: &str,
    rows: u32,
    cols: u32,
    weights: &[f32],     // full row-major weight matrix
    groups: &[Vec<f32>], // pre-split groups (each 128 values)
    packed_codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    per_channel_importance: Option<&[f32]>,
    policy: &ProtectionPolicy,
) -> MatrixOutlierResult {
    let total_groups = groups.len() as u32;

    // Step 1: Detect outliers
    let (outlier_reports, _non_outlier_reports) = detect_all_outliers(groups, matrix_name, policy);

    // Step 2: Build sidecar packs for outlier groups
    let mut sidecar_packs = Vec::new();
    let groups_per_tile = crate::nf4tile640::GROUPS_PER_TILE;

    if policy.enabled {
        // Group outlier reports by tile_index
        use std::collections::BTreeMap;
        let mut tile_outliers: BTreeMap<u32, Vec<&GroupOutlierReport>> = BTreeMap::new();
        for report in &outlier_reports {
            tile_outliers
                .entry(report.tile_index)
                .or_default()
                .push(report);
        }

        for (tile_idx, tile_reports) in &tile_outliers {
            // Check density constraint
            let tile_group_count = groups_per_tile as f32;
            let allowed = (tile_group_count * policy.max_sidecar_density).ceil() as u32;
            if tile_reports.len() > allowed as usize {
                // Too many outlier groups in this tile — skip sidecar packing
                // for this tile to keep density under the limit.
                continue;
            }

            let mut residuals = Vec::new();
            for report in tile_reports {
                let gi = report.group_index;
                let group_within_tile = (gi % groups_per_tile as u32) as usize;

                // Iterate over each element in this group
                for pos in 0..crate::nf4tile640::GROUP_SIZE {
                    let reconstructed = reconstruct_nf4_value(
                        packed_codes,
                        scales,
                        biases,
                        *tile_idx,
                        group_within_tile,
                        pos,
                    );

                    // Find the original value: group_index * GROUP_SIZE + pos
                    let group_offset = gi as usize * crate::nf4tile640::GROUP_SIZE;
                    let original = weights.get(group_offset + pos).copied().unwrap_or(0.0);

                    let residual = quantize_and_residual(original, reconstructed, gi, pos as u32);

                    // Only store non-negligible residuals
                    if residual.residual.abs() > 1e-7 {
                        residuals.push(residual);
                    }
                }
            }

            if !residuals.is_empty() {
                sidecar_packs.push(build_sidecar_pack(*tile_idx, residuals));
            }
        }
    }

    // Step 3: Protected channel analysis
    let protected_channels = if policy.protected_channels {
        per_channel_importance.map(|imp| identify_protected_channels(matrix_name, imp, 5.0))
    } else {
        None
    };

    // Step 4: Compute sidecar byte count and effective bpw
    let sidecar_bytes: u64 = sidecar_packs.iter().map(|p| p.byte_count as u64).sum();
    let total_weights = rows as u64 * cols as u64;
    let effective_bpw = compute_effective_bpw(total_weights, sidecar_bytes, 4);

    MatrixOutlierResult {
        matrix_name: matrix_name.to_string(),
        total_groups,
        outlier_groups: outlier_reports,
        sidecar_packs,
        protected_channels,
        effective_bpw,
        sidecar_bytes,
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// A group of 128 values where one element is a clear outlier (10× others).
    fn make_outlier_group() -> Vec<f32> {
        let mut group = vec![1.0; 128];
        group[0] = 100.0; // strong outlier
        group
    }

    /// A uniform group with no outliers.
    fn make_uniform_group() -> Vec<f32> {
        vec![0.5; 128]
    }

    #[test]
    fn test_heap_sort() {
        let mut arr = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        heap_sort(&mut arr);
        assert_eq!(arr, [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]);
    }

    #[test]
    fn test_heap_sort_empty() {
        let mut arr: [f32; 0] = [];
        heap_sort(&mut arr);
    }

    #[test]
    fn test_heap_sort_single() {
        let mut arr = [42.0];
        heap_sort(&mut arr);
        assert_eq!(arr, [42.0]);
    }

    #[test]
    fn test_percentile_and_median() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!((percentile(&sorted, 10.0) - 1.0).abs() < 1e-6);
        assert!((percentile(&sorted, 50.0) - 5.0).abs() < 1e-6);
        assert!((percentile(&sorted, 90.0) - 9.0).abs() < 1e-6);
        assert!((median(&sorted) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_detect_outlier_group_detects_outlier() {
        let group = make_outlier_group();
        let policy = ProtectionPolicy::default();
        let report = detect_outlier_group(&group, 0, 0, "test_matrix", &policy);

        assert!(
            report.is_outlier_group,
            "group with extreme outlier should be flagged"
        );
        assert_eq!(report.group_index, 0);
        assert_eq!(report.tile_index, 0);
        assert!((report.max_abs - 100.0).abs() < 1e-6);
        assert!(report.num_outliers > 0);
    }

    #[test]
    fn test_detect_outlier_group_uniform_not_outlier() {
        let group = make_uniform_group();
        let policy = ProtectionPolicy::default();
        let report = detect_outlier_group(&group, 1, 0, "test", &policy);

        assert!(!report.is_outlier_group);
        assert_eq!(report.num_outliers, 0);
    }

    #[test]
    fn test_detect_outlier_group_empty() {
        let group: Vec<f32> = Vec::new();
        let policy = ProtectionPolicy::default();
        let report = detect_outlier_group(&group, 0, 0, "test", &policy);
        assert!(!report.is_outlier_group);
    }

    #[test]
    fn test_detect_all_outliers_splits_correctly() {
        let groups = vec![make_outlier_group(), make_uniform_group()];
        let policy = ProtectionPolicy::default();
        let (outliers, non_outliers) = detect_all_outliers(&groups, "test", &policy);

        assert_eq!(outliers.len(), 1);
        assert_eq!(non_outliers.len(), 1);
    }

    #[test]
    fn test_quantize_and_residual() {
        let residual = quantize_and_residual(3.14, 3.0, 0, 5);
        assert_eq!(residual.group_index, 0);
        assert_eq!(residual.packed_position, 5);
        assert!((residual.residual - 0.14).abs() < 1e-6);
    }

    #[test]
    fn test_build_sidecar_pack() {
        let residuals = vec![
            quantize_and_residual(1.0, 0.9, 0, 0),
            quantize_and_residual(2.0, 1.8, 0, 1),
        ];
        let pack = build_sidecar_pack(42, residuals);
        assert_eq!(pack.tile_index, 42);
        assert_eq!(pack.residuals.len(), 2);
        // header: 8 bytes, each residual: 12 bytes
        assert_eq!(pack.byte_count, 4 + 4 + 2 * 12);
    }

    #[test]
    fn test_compute_effective_bpw() {
        let bpw = compute_effective_bpw(640, 128, 4);
        // 4 + (128 * 8) / 640 = 4 + 1.6 = 5.6
        assert!((bpw - 5.6).abs() < 1e-6);
    }

    #[test]
    fn test_compute_effective_bpw_zero_weights() {
        let bpw = compute_effective_bpw(0, 100, 4);
        assert!((bpw - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_identify_protected_channels() {
        // 10 channels: most have importance 1.0, channels 0 and 9 have 10.0
        let imp: Vec<f32> = (0..10)
            .map(|i| if i == 0 || i == 9 { 10.0 } else { 1.0 })
            .collect();
        // median = 1.0, threshold = 5.0
        let result = identify_protected_channels("test", &imp, 5.0);
        assert_eq!(result.protected_indices, vec![0, 9]);
        assert_eq!(result.higher_precision_bytes, 8); // 2 channels * 4 bytes
    }

    #[test]
    fn test_identify_protected_channels_empty() {
        let result = identify_protected_channels("empty", &[], 5.0);
        assert!(result.protected_indices.is_empty());
        assert_eq!(result.higher_precision_bytes, 0);
    }

    #[test]
    fn test_identify_protected_channels_no_flagged() {
        let imp = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        // median = 1.0, threshold = 5.0, nothing exceeds
        let result = identify_protected_channels("test", &imp, 5.0);
        assert!(result.protected_indices.is_empty());
    }

    #[test]
    fn test_protection_policy_defaults() {
        let p = ProtectionPolicy::default();
        assert!((p.outlier_threshold - 3.0).abs() < 1e-6);
        assert!((p.max_sidecar_density - 0.1).abs() < 1e-6);
        assert!(p.enabled);
        assert!(!p.protected_channels);
    }

    #[test]
    fn test_nf4_reconstruct_value() {
        // Build a single tile of 640 values, pack it, then reconstruct one element.
        use crate::nf4tile640::nf4_quantize;
        use crate::nf4tile640::pack_nf4_tile;

        let values = [0.5_f32; 640];
        let (codes, scales, biases) = pack_nf4_tile(&values);

        // Reconstruct element in tile 0, group 0, position 0
        let recon = reconstruct_nf4_value(&codes, &scales, &biases, 0, 0, 0);
        // Should be close to the quantized-reconstructed value.
        // Note: pack_nf4_tile normalizes by scale before quantizing, so we must too.
        let code = nf4_quantize(0.5 / scales[0]);
        let expected = nf4_dequantize(code) * scales[0] + biases[0];
        assert!(
            (recon - expected).abs() < 1e-6,
            "recon={} expected={}",
            recon,
            expected
        );
    }

    #[test]
    fn test_analyze_matrix_outliers_simple() {
        let rows = 1u32;
        let cols = 640u32;
        let mut weights = vec![1.0_f32; 640];
        weights[0] = 100.0; // one outlier

        let groups: Vec<Vec<f32>> = weights.chunks(128).map(|c| c.to_vec()).collect();
        use crate::nf4tile640::pack_nf4_tile;
        let mut arr = [0.0_f32; 640];
        arr.copy_from_slice(&weights);
        let (codes, scales, biases) = pack_nf4_tile(&arr);

        let policy = ProtectionPolicy::default();
        let result = analyze_matrix_outliers(
            "test", rows, cols, &weights, &groups, &codes, &scales, &biases, None, &policy,
        );

        assert_eq!(result.matrix_name, "test");
        assert_eq!(result.total_groups, 5);
        assert!(result.outlier_groups.len() >= 1);
        assert!(result.sidecar_bytes > 0);
        assert!(result.effective_bpw > 4.0);
    }
}

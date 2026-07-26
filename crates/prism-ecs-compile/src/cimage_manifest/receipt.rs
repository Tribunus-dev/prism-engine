//! Compile receipt and per-stage timing profile — the post-emission evidence
//! written to `receipt.json` next to every finalized CImage manifest.
//!
//! This module owns the constitutional authority for the
//! [`CompileReceipt`] / [`StageProfile`] / [`SegmentReceipt`] /
//! [`TensorProvenance`] / [`IgnoredTensorClassification`] types and the
//! [`TensorDiff`] computation that drives the differential compile
//! path. The receipt is **post-emission evidence** derived from the
//! manifest; the manifest is the durable schema.
//!
//! The module does **not** own the manifest itself (see
//! [`super::header`]), the per-tensor table (see [`super::types`]),
//! the lease state machine (see [`super::lease`]), or the kernel
//! dispatch recipes (see [`super::kernel`]).

use serde::{Deserialize, Serialize};

use super::types::ShardHash;

// ── Native capability report ──────────────────────────────────────────────

/// Native dependency identity and capability report.
/// Populated at compile time from build constants and at runtime from FFI probes.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NativeCapabilityReport {
    pub mlx_core_version: String,
    pub mlx_c_version: String,
    pub mlx_rs_version: String,
    pub mlx_sys_version: String,
    pub compute_native_version: String,
    pub supports_quantized_matmul: bool,
    pub supports_dequantize: bool,
    pub supports_memory_telemetry: bool,
    pub supports_cache_control: bool,
    pub supports_external_array: bool,
    pub supports_multithreaded_execution: bool,
    pub metal_available: bool,
    pub accelerate_available: bool,
}

// ── Segment / tensor / ignored receipts ───────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentReceipt {
    pub id: String,
    pub filename: String,
    pub sha256: String,
    pub byte_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorProvenance {
    pub tensor_name: String,
    pub source_sha256: String,
    pub emitted_sha256: String,
    pub preserved_byte_for_byte: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IgnoredTensorClassification {
    pub name: String,
    pub classification: String,
}

// ── Stage timing profile ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StageProfile {
    pub source_discovery_ms: u64,
    pub source_hashing_ms: u64,
    pub header_parsing_ms: u64,
    pub architecture_normalization_ms: u64,
    pub binding_validation_ms: u64,
    pub layout_planning_ms: u64,
    pub payload_emission_ms: u64,
    pub segment_hashing_ms: u64,
    pub manifest_generation_ms: u64,
    pub verification_ms: u64,
    pub total_source_bytes: u64,
    pub total_emitted_bytes: u64,
    pub peak_rss_bytes: u64,
    pub peak_mlx_active_bytes: u64,
    pub peak_mlx_cache_bytes: u64,
}

impl StageProfile {
    /// Total wall-clock time spent in compile stages (in milliseconds).
    pub fn total_stage_ms(&self) -> u64 {
        self.source_discovery_ms
            .saturating_add(self.source_hashing_ms)
            .saturating_add(self.header_parsing_ms)
            .saturating_add(self.architecture_normalization_ms)
            .saturating_add(self.binding_validation_ms)
            .saturating_add(self.layout_planning_ms)
            .saturating_add(self.payload_emission_ms)
            .saturating_add(self.segment_hashing_ms)
            .saturating_add(self.manifest_generation_ms)
            .saturating_add(self.verification_ms)
    }
}

// ── Compile receipt ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompileReceipt {
    pub source_config_hash: String,
    pub source_shard_hashes: Vec<ShardHash>,
    pub compiler_version: String,
    pub runtime_abi: String,
    pub normalized_architecture_hash: String,
    pub execution_plan_hash: String,
    pub complete_image_hash: String,
    pub segment_hashes: Vec<SegmentReceipt>,
    pub tensor_count: usize,
    pub alias_count: usize,
    pub segment_count: usize,
    pub ignored_tensor_classifications: Vec<IgnoredTensorClassification>,
    pub total_source_bytes: u64,
    pub total_emitted_bytes: u64,
    pub elapsed_ms: u128,
    pub transformed_payloads: Vec<String>,
    pub byte_provenance: Vec<TensorProvenance>,
    pub structural_verification: bool,
    /// Native dependency identity captured at compile time.
    pub native_dependency_report: NativeCapabilityReport,
    /// Hardware assessment receipt from compile-time kernel profiling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hw_assessment: Option<serde_json::Value>,
    pub stage_profile: StageProfile,
}

// ── Compiled image (manifest + receipt bundle) ────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledImage {
    pub manifest: super::header::Manifest,
    pub receipt: CompileReceipt,
}

// ── Verification ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestVerification {
    pub manifest_hash_matches: bool,
    pub segment_hashes_match: bool,
    pub verified_segment_count: usize,
    pub total_bytes: u64,
}

// ── Tensor diff (differential compile) ────────────────────────────────────

/// Result of diffing current source tensors against a previous compilation
/// manifest.
#[derive(Default, Debug, Clone)]
pub struct TensorDiff {
    /// Tensor names whose hash matches the previous compile.
    pub unchanged: Vec<String>,
    /// Tensor names whose hash differs from the previous compile.
    pub changed: Vec<String>,
    /// Tensor names present in the source but not in the previous compile.
    pub new: Vec<String>,
    /// Tensor names present in the previous compile but absent from the source.
    pub removed: Vec<String>,
    /// Wall-clock milliseconds spent computing the diff.
    pub elapsed_ms: u128,
}

impl TensorDiff {
    /// Return true if the diff is empty (no changed / new / removed tensors).
    pub fn is_empty(&self) -> bool {
        self.changed.is_empty() && self.new.is_empty() && self.removed.is_empty()
    }

    /// Number of tensors that need to be re-emitted (changed + new).
    pub fn reemit_count(&self) -> usize {
        self.changed.len() + self.new.len()
    }
}

// ── Admission estimate ────────────────────────────────────────────────────

/// Admission-estimate for representation-aware memory budgeting.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct RepresentationAdmissionEstimate {
    pub virtual_mapped_bytes: u64,
    pub expected_resident_bytes: u64,
    pub persistent_materialized_bytes: u64,
    pub max_layer_window_bytes: u64,
    pub rope_bytes: u64,
    pub kv_budget_bytes: u64,
    pub mlx_workspace_bytes: u64,
    pub allocator_cache_bytes: u64,
    pub system_reserve_bytes: u64,
    /// Maximum single transient allocation during inference
    /// (attention workspace, output projection buffer, etc.).
    pub largest_transient_bytes: u64,
    /// Bytes that must be converted (dequantized, dtype-cast) at runtime.
    pub materialized_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_profile_total_sums_all_stage_fields() {
        let mut p = StageProfile::default();
        p.source_discovery_ms = 10;
        p.source_hashing_ms = 20;
        p.header_parsing_ms = 30;
        p.architecture_normalization_ms = 40;
        p.binding_validation_ms = 50;
        p.layout_planning_ms = 60;
        p.payload_emission_ms = 70;
        p.segment_hashing_ms = 80;
        p.manifest_generation_ms = 90;
        p.verification_ms = 100;
        // 10+20+30+40+50+60+70+80+90+100 = 550
        assert_eq!(p.total_stage_ms(), 550);
    }

    #[test]
    fn stage_profile_total_saturates_on_overflow() {
        let mut p = StageProfile::default();
        p.source_discovery_ms = u64::MAX;
        p.source_hashing_ms = 1;
        // u64::MAX + 1 saturates to u64::MAX
        assert_eq!(p.total_stage_ms(), u64::MAX);
    }

    #[test]
    fn tensor_diff_is_empty_when_no_deltas() {
        let diff = TensorDiff {
            unchanged: vec!["a".into()],
            changed: Vec::new(),
            new: Vec::new(),
            removed: Vec::new(),
            elapsed_ms: 5,
        };
        assert!(diff.is_empty());
        assert_eq!(diff.reemit_count(), 0);
    }

    #[test]
    fn tensor_diff_reemit_count_sums_changed_and_new() {
        let diff = TensorDiff {
            unchanged: vec!["a".into(), "b".into()],
            changed: vec!["c".into()],
            new: vec!["d".into(), "e".into()],
            removed: vec!["f".into()],
            elapsed_ms: 5,
        };
        assert!(!diff.is_empty());
        assert_eq!(diff.reemit_count(), 3);
    }

    #[test]
    fn compile_receipt_default_is_empty() {
        let r = CompileReceipt::default();
        assert_eq!(r.tensor_count, 0);
        assert_eq!(r.alias_count, 0);
        assert_eq!(r.segment_count, 0);
        assert!(!r.structural_verification);
        assert!(r.transformed_payloads.is_empty());
        assert!(r.byte_provenance.is_empty());
    }

    #[test]
    fn representation_admission_estimate_default_is_zero() {
        let e = RepresentationAdmissionEstimate::default();
        assert_eq!(e.virtual_mapped_bytes, 0);
        assert_eq!(e.expected_resident_bytes, 0);
        assert_eq!(e.materialized_bytes, 0);
    }

    #[test]
    fn native_capability_report_default_is_empty() {
        let r = NativeCapabilityReport::default();
        assert!(!r.metal_available);
        assert!(!r.accelerate_available);
        assert!(!r.supports_quantized_matmul);
        assert!(!r.supports_dequantize);
        assert!(r.mlx_core_version.is_empty());
    }
}

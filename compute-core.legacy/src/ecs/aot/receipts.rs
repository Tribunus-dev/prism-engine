//! Kernel receipts — compile, validation, and performance receipts
//! that accompany each kernel variant through its lifecycle.

use serde::{Deserialize, Serialize};

use super::parameters::{KernelFamily, KernelParameters};
use super::profile_id::{AppleSiliconProfileId, ProfileEvidenceStatus};

/// Compile receipt — records how and when a kernel variant was compiled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelCompileReceipt {
    pub receipt_id: String,
    pub variant_id: String,
    pub target_profile: AppleSiliconProfileId,
    pub kernel_family: KernelFamily,
    pub parameters: KernelParameters,
    pub metal_version: String,
    pub compiler_version: String,
    pub compiled_at: String,
    pub artifact_digest: String,
}

/// Validation receipt — records whether a kernel variant passes correctness checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelValidationReceipt {
    pub receipt_id: String,
    pub variant_id: String,
    pub target_profile: AppleSiliconProfileId,
    pub kernel_family: KernelFamily,
    pub parameters: KernelParameters,
    pub validation_passed: bool,
    pub held_out_shapes: Vec<HeldOutShapeResult>,
    pub max_nrmse: f64,
    pub max_cosine_distance: f64,
}

/// Result of validating against one held-out shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeldOutShapeResult {
    pub shape: Vec<u32>,
    pub nrmse: f64,
    pub cosine_similarity: f64,
    pub passed: bool,
}

/// Performance receipt — records measured throughput for a kernel variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelPerformanceReceipt {
    pub receipt_id: String,
    pub variant_id: String,
    pub target_profile: AppleSiliconProfileId,
    pub kernel_family: KernelFamily,
    pub codec_family: String,
    pub parameters: KernelParameters,
    pub tensor_shape: Vec<u32>,
    pub command_buffer_ms: f64,
    pub effective_bandwidth_gbps: f64,
    pub bandwidth_utilization: f64,
    pub tokens_per_second_estimate: Option<f64>,
    pub numeric_nrmse: f64,
    pub numeric_cosine: f64,
    pub validation_passed: bool,
    pub measured_or_predicted: ProfileEvidenceStatus,
}

/// Combined quality × performance score for sweep integration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QualityPerformanceScore {
    pub quality_passed: bool,
    pub numeric_score: f64,
    pub byte_savings_score: f64,
    pub throughput_score: f64,
    pub bandwidth_utilization_score: f64,
    pub final_score: f64,
}

impl QualityPerformanceScore {
    /// Compute the combined score.
    ///
    /// Rules:
    /// - If quality gates fail, final_score = rejected (0.0).
    /// - Otherwise: weighted sum of quality, byte savings, throughput, and bandwidth.
    pub fn compute(
        quality_passed: bool,
        numeric_nrmse: f64,
        byte_savings_ratio: f64,
        tokens_per_second: f64,
        bandwidth_utilization: f64,
    ) -> Self {
        const W_QUALITY: f64 = 0.40;
        const W_BYTES: f64 = 0.25;
        const W_PERF: f64 = 0.25;
        const W_BANDWIDTH: f64 = 0.10;

        if !quality_passed {
            return Self {
                quality_passed: false,
                numeric_score: 0.0,
                byte_savings_score: 0.0,
                throughput_score: 0.0,
                bandwidth_utilization_score: 0.0,
                final_score: 0.0,
            };
        }

        // Invert NRMSE so lower error → higher score. Cap at 1.0.
        let numeric_score = (1.0 - numeric_nrmse).max(0.0);

        // Byte savings ratio is already 0..1.
        let byte_savings_score = byte_savings_ratio.min(1.0);

        // Normalize throughput to a 0..1 scale (100 tok/s = 1.0).
        let throughput_score = (tokens_per_second / 100.0).min(1.0);

        // Bandwidth utilization is already 0..1.
        let bandwidth_utilization_score = bandwidth_utilization.min(1.0);

        let final_score = W_QUALITY * numeric_score
            + W_BYTES * byte_savings_score
            + W_PERF * throughput_score
            + W_BANDWIDTH * bandwidth_utilization_score;

        Self {
            quality_passed: true,
            numeric_score,
            byte_savings_score,
            throughput_score,
            bandwidth_utilization_score,
            final_score,
        }
    }
}

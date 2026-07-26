use serde::{Deserialize, Serialize};

/// Summary of a loaded CImage artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardCImageSummary {
    pub digest: String,
    pub path: String,
    pub artifact_kind: String,
    pub model_family: String,
    pub schema_version: i32,
    pub tensor_count: i32,
    pub receipt_count: i32,
    pub validation_status: String,
    pub compiler_policy_digest: Option<String>,
    pub hardware_profile: Option<String>,
    pub created_at: String,
}

/// Summary of a single tensor within a CImage artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTensorSummary {
    pub artifact_digest: String,
    pub tensor_key: String,
    pub tensor_class: String,
    pub codec: String,
    pub group_size: Option<i32>,
    pub effective_bpw: Option<f64>,
    pub logical_shape: Vec<u32>,
    pub payload_size: Option<i64>,
    pub promotion_status: String,
}

/// Summary of an admission receipt for a tensor quantization trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardAdmissionSummary {
    pub receipt_id: String,
    pub tensor_key: String,
    pub codec: String,
    pub group_size: i32,
    pub effective_bpw: Option<f64>,
    pub zero_fraction: Option<f64>,
    pub neg_fraction: Option<f64>,
    pub pos_fraction: Option<f64>,
    pub scale_mean: Option<f64>,
    pub scale_std: Option<f64>,
    pub scale_max: Option<f64>,
    pub operator_nrmse: Option<f64>,
    pub output_cosine: Option<f64>,
    pub activation_shift_l2: Option<f64>,
    pub deadzone_collapse: bool,
    pub rescue_required: bool,
    pub rescue_codec: Option<String>,
    pub promotion_status: String,
}

/// Summary of an execution receipt (Metal dispatch result).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardExecutionSummary {
    pub receipt_id: String,
    pub tensor_key: String,
    pub kernel_name: String,
    pub backend: String,
    pub command_buffer_ms: Option<f64>,
    pub bandwidth_gbps: Option<f64>,
    pub validation_passed: bool,
}

/// Summary of a sweep over quantization candidates for one tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSweepSummary {
    pub sweep_id: String,
    pub artifact_digest: String,
    pub tensor_key: String,
    pub candidate_count: i32,
    pub winner_candidate_id: Option<String>,
}

/// A single candidate within a quantization sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSweepCandidate {
    pub candidate_id: String,
    pub codec: String,
    pub group_size: i32,
    pub calibration_steps: i32,
    pub nrmse: f64,
    pub cosine: f64,
    pub bytes: i64,
    pub passed: bool,
}

/// A single entry in the evidence ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardEvidenceEntry {
    pub receipt_id: String,
    pub artifact_digest: String,
    pub scope: String,
    pub kind: String,
    pub validation_passed: bool,
    pub json: serde_json::Value,
}

/// Explanation of why a scope has or has not been promoted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardPromotionExplanation {
    pub scope_id: String,
    pub current_status: String,
    pub missing_receipts: Vec<String>,
    pub failed_gates: Vec<String>,
    pub recommendation: String,
}

/// Summary of a calibration run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardCalibrationSummary {
    pub calibration_id: String,
    pub stage_1_codec: String,
    pub stage_2_codec: Option<String>,
    pub loss_operator_nrmse: Option<f64>,
    pub loss_cosine: Option<f64>,
    pub execution_environment: String,
}

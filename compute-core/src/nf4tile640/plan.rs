use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationPlanEntry {
    pub tensor_name: String,
    pub source_tensor_digest: [u8; 32],
    pub profile_id: u32,
    pub group_importances: Vec<f32>,
    pub outlier_channels: Vec<usize>,
    pub verification_rmse: f32,
    pub gate_passed: bool,
    /// Activation-weighted MSE after optimization (from AW-LS fitting).
    /// None if not yet computed.
    pub aw_mse: Option<f64>,
    /// Per-channel activation second moments E[x_i^2], one per input channel.
    /// Collected during calibration pass. None if AW-LS not enabled.
    pub channel_second_moments: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationPlan {
    pub source_model_digest: String,
    pub quantizer_mode: String,
    pub target_strategy: String,
    pub entries: Vec<QuantizationPlanEntry>,
    pub profile_registry_ids: Vec<u32>,
    pub build_duration_secs: f64,
}

impl QuantizationPlan {
    pub fn compute_model_digest(entries: &[QuantizationPlanEntry]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for entry in entries {
            hasher.update(&entry.source_tensor_digest);
        }
        format!("{:x}", hasher.finalize())
    }
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

//! Architecture profiles used by Prism's native ROCm lowering.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RocmArchitecture {
    Mi300x,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocmTargetProfile {
    pub architecture: RocmArchitecture,
    pub gpu: String,
    pub wavefront_size: u32,
    pub hbm_bytes: Option<u64>,
    pub matrix_core: bool,
    pub multi_gpu: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelWeightPrecision {
    Bf16,
    Int8,
    Int4,
    TernaryMixed,
}

impl ModelWeightPrecision {
    pub fn bytes_per_parameter(self) -> f64 {
        match self {
            Self::Bf16 => 2.0,
            Self::Int8 => 1.0,
            Self::Int4 => 0.5,
            // Two-bit codes plus scales and lossless BF16 residuals on the
            // sensitive subset. This is deliberately conservative.
            Self::TernaryMixed => 0.325,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelFitRequest {
    pub parameters: u64,
    pub precision: ModelWeightPrecision,
    pub context_tokens: u32,
    pub kv_bytes_per_token: u64,
    pub workspace_bytes: u64,
    pub reserve_fraction: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelFitEstimate {
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    pub workspace_bytes: u64,
    pub available_bytes: u64,
    pub fits: bool,
}

impl RocmTargetProfile {
    pub fn mi300x() -> Self {
        Self {
            architecture: RocmArchitecture::Mi300x,
            gpu: "gfx942".into(),
            wavefront_size: 64,
            // Keep capacity explicit but optional: a VF or partition may expose less.
            hbm_bytes: None,
            matrix_core: true,
            multi_gpu: false,
        }
    }

    pub fn from_environment() -> Self {
        let mut profile = Self::mi300x();
        if let Ok(gpu) = std::env::var("PRISM_ROCM_GPU") {
            profile.gpu = gpu;
            profile.architecture = RocmArchitecture::Custom;
        }
        profile.multi_gpu = std::env::var("PRISM_ROCM_MULTI_GPU").as_deref() == Ok("1");
        profile
    }

    pub fn hipcc_args(&self) -> Vec<String> {
        vec![
            "--offload-arch".into(),
            self.gpu.clone(),
            "-O3".into(),
            "-ffast-math".into(),
        ]
    }

    pub fn estimate_model_fit(&self, request: ModelFitRequest) -> ModelFitEstimate {
        let hbm = self.hbm_bytes.unwrap_or(192 * 1024 * 1024 * 1024);
        let reserve = request.reserve_fraction.clamp(0.0, 0.9);
        let available = (hbm as f64 * (1.0 - reserve)) as u64;
        let weight_bytes =
            (request.parameters as f64 * request.precision.bytes_per_parameter()) as u64;
        let kv_bytes = request
            .kv_bytes_per_token
            .saturating_mul(request.context_tokens as u64);
        let fits = weight_bytes
            .saturating_add(kv_bytes)
            .saturating_add(request.workspace_bytes)
            <= available;
        ModelFitEstimate {
            weight_bytes,
            kv_bytes,
            workspace_bytes: request.workspace_bytes,
            available_bytes: available,
            fits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mi300x_profile_targets_cdna3() {
        let profile = RocmTargetProfile::mi300x();
        assert_eq!(profile.gpu, "gfx942");
        assert!(profile.matrix_core);
        assert_eq!(profile.wavefront_size, 64);
    }

    #[test]
    fn mi300x_fit_estimate_accounts_for_kv_and_workspace() {
        let profile = RocmTargetProfile::mi300x();
        let estimate = profile.estimate_model_fit(ModelFitRequest {
            parameters: 300_000_000_000,
            precision: ModelWeightPrecision::TernaryMixed,
            context_tokens: 32_000,
            kv_bytes_per_token: 160_000,
            workspace_bytes: 8 * 1024 * 1024 * 1024,
            reserve_fraction: 0.12,
        });
        assert!(estimate.fits);
        assert!(estimate.kv_bytes > 5_000_000_000);
    }

    #[test]
    fn mi300x_fit_rejects_oversized_bf16_model() {
        let profile = RocmTargetProfile::mi300x();
        let estimate = profile.estimate_model_fit(ModelFitRequest {
            parameters: 100_000_000_000,
            precision: ModelWeightPrecision::Bf16,
            context_tokens: 1,
            kv_bytes_per_token: 0,
            workspace_bytes: 8 * 1024 * 1024 * 1024,
            reserve_fraction: 0.12,
        });
        assert!(!estimate.fits);
    }
}

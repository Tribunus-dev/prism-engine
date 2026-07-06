use serde::{Deserialize, Serialize};

/// Profile identifier — unique within a CImage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProfileId(pub u32);

/// Well-known profile IDs.
pub const PROFILE_ID_CANONICAL_NF4_V1: ProfileId = ProfileId(0);
pub const PROFILE_ID_GEMMA_ATTENTION_V1: ProfileId = ProfileId(1);
pub const PROFILE_ID_GEMMA_FFN_V1: ProfileId = ProfileId(2);
pub const PROFILE_ID_GEMMA_BOUNDARY_V1: ProfileId = ProfileId(3);
pub const PROFILE_ID_TTS_CODEC_V1: ProfileId = ProfileId(4);

/// ABI version for profile descriptors.
pub const PROFILE_ABI_VERSION: u32 = 1;

/// Codebook kind: how the 16 reconstruction values were obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodebookKind {
    /// Fixed canonical NF4 quantiles of N(0,1).
    CanonicalNf4,
    /// Learned scalar codebook from model weights via Lloyd-Max.
    LearnedScalar,
    /// Reserved for future use.
    LearnedVector,
}

/// Clipping policy applied before scale/offset estimation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClippingPolicy {
    /// No clipping; use full max-abs range.
    None,
    /// Clip at the p-th percentile (0..100).
    Percentile(f32),
    /// MSE-optimal scalar clipping threshold.
    MseOptimal,
}

/// Bias policy: whether affine reconstruction is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BiasPolicy {
    /// No bias (forced to 0.0).
    None,
    /// Per-group affine bias enabled.
    Affine,
}

/// Sidecar policy for outlier protection.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SidecarPolicy {
    /// No sidecar.
    #[default]
    None,
    /// Sparse (index, fp16) residual pairs.
    SparseFp16Residual,
    /// Reserved: protected-channel higher-bit retention.
    ProtectedChannel,
}

/// Codebook descriptor — 16 reconstruction values plus metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CodebookDescriptor {
    /// Unique profile ID this codebook belongs to.
    pub profile_id: ProfileId,
    /// Profile display name (e.g. "gemma_attention_v1").
    pub name: String,
    /// ABI version (currently 1).
    pub abi_version: u32,
    /// How this codebook was derived.
    pub kind: CodebookKind,
    /// Codebook precision: the storage format of centroid values.
    pub precision: String, // "F32" | "F16"
    /// The 16 reconstruction values, sorted ascending.
    #[serde(default)]
    pub values: Vec<f32>,
    /// Clipping policy used during training.
    pub clipping_policy: ClippingPolicy,
    /// Bias policy.
    pub bias_policy: BiasPolicy,
    /// Sidecar policy.
    #[serde(default)]
    pub sidecar_policy: SidecarPolicy,
    /// Training objective string (e.g. "weighted_scalar_mse").
    #[serde(default)]
    pub training_objective: String,
    /// Number of Lloyd-Max iterations.
    #[serde(default)]
    pub training_iterations: u32,
    /// Deterministic seed used during training.
    #[serde(default)]
    pub training_seed: u64,
    /// SHA-256 of the calibration corpus used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_digest: Option<String>,
    /// SHA-256 of the source model checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_model_digest: Option<String>,
    /// Compiler revision string.
    #[serde(default)]
    pub compiler_revision: String,
}

impl CodebookDescriptor {
    /// Create a descriptor for the canonical NF4 profile.
    pub fn canonical_nf4() -> Self {
        Self {
            profile_id: PROFILE_ID_CANONICAL_NF4_V1,
            name: "canonical_nf4_v1".into(),
            abi_version: PROFILE_ABI_VERSION,
            kind: CodebookKind::CanonicalNf4,
            precision: "F32".into(),
            values: vec![
                -1.0, -0.6961928009986877, -0.5250731024742126,
                -0.3949175179004669, -0.28444138169288635,
                -0.18477340042591095, -0.09105004370212555,
                0.0, 0.07958029955625534,
                0.16093020141124725, 0.2461123028397569,
                0.3379152008295059, 0.44070979952812195,
                0.5626170039176941, 0.7229568362236023, 1.0,
            ],
            clipping_policy: ClippingPolicy::None,
            bias_policy: BiasPolicy::None,
            sidecar_policy: SidecarPolicy::None,
            training_objective: String::new(),
            training_iterations: 0,
            training_seed: 0,
            calibration_digest: None,
            source_model_digest: None,
            compiler_revision: String::new(),
        }
    }

    /// Validate the descriptor: must have exactly 16 values, sorted ascending.
    pub fn validate(&self) -> Result<(), String> {
        if self.values.len() != 16 {
            return Err(format!("codebook must have 16 values, got {}", self.values.len()));
        }
        for w in self.values.windows(2) {
            if w[0] > w[1] {
                return Err(format!("codebook not sorted: {} > {}", w[0], w[1]));
            }
        }
        Ok(())
    }
}

/// Complete quantizer profile that can be serialized into a CImage manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuantizerProfile {
    pub descriptor: CodebookDescriptor,
    /// Group size (must be 128).
    pub group_size: u32,
    /// Tile elements (must be 640).
    pub tile_elements: u32,
    /// Reserved for future use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_metadata: Option<serde_json::Value>,
}

impl QuantizerProfile {
    pub fn canonical_nf4() -> Self {
        Self {
            descriptor: CodebookDescriptor::canonical_nf4(),
            group_size: 128,
            tile_elements: 640,
            extra_metadata: None,
        }
    }
}

/// Profile registry — maps ProfileId to QuantizerProfile.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProfileRegistry {
    pub profiles: Vec<QuantizerProfile>,
}

impl ProfileRegistry {
    pub fn get(&self, id: ProfileId) -> Option<&QuantizerProfile> {
        self.profiles.iter().find(|p| p.descriptor.profile_id == id)
    }

    pub fn register(&mut self, profile: QuantizerProfile) {
        self.profiles.push(profile);
    }
}

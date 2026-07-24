//! Canonical export/decompilation of admitted Living CImage generations.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalModelFormat {
    SafeTensors,
    HuggingFace,
    Gguf,
    Onnx,
    Mlx,
    CoreMl,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportPrecisionPolicy {
    Preserve,
    DequantizeFp16,
    DequantizeBf16,
    NearestSupported,
    Explicit(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngramExportPolicy {
    PreserveAsSidecar,
    FoldIntoAdapters,
    DistillIntoWeights,
    ExportRuntimeExtension,
    RejectIfRequired,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportEquivalenceClass {
    Exact,
    RepresentationEquivalent,
    BehaviorallyEquivalent,
    DegradedButAdmitted,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelExportRequest {
    pub request_id: String,
    pub source_cimage_digest: String,
    pub source_generation: u64,
    pub source_generation_digest: String,
    pub target_format: CanonicalModelFormat,
    pub precision_policy: ExportPrecisionPolicy,
    pub fold_adapters: bool,
    pub engram_policy: EngramExportPolicy,
    pub require_behavioral_equivalence: bool,
    pub preserve_tokenizer_and_config: bool,
    #[serde(default)]
    pub request_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportedComponent {
    pub component: String,
    pub action: String,
    pub source_digest: String,
    pub target_digest: Option<String>,
    pub exact_preservation: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelExportValidation {
    pub measured: bool,
    pub validator_fingerprint: String,
    pub loader_roundtrip: bool,
    pub tensor_identity_valid: bool,
    pub tokenizer_config_valid: bool,
    pub forward_differential: f64,
    pub logit_divergence: f64,
    pub rollout_similarity: f64,
    pub agentic_success: f64,
    pub tool_call_correctness: f64,
    pub engram_dependent_task_score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelExportReceipt {
    pub request_digest: String,
    pub source_cimage_digest: String,
    pub source_generation: u64,
    pub target_format: CanonicalModelFormat,
    pub exported_artifact_digest: String,
    pub components: Vec<ExportedComponent>,
    pub external_sidecars: Vec<String>,
    pub omitted_components: Vec<String>,
    pub equivalence_class: ExportEquivalenceClass,
    pub admitted: bool,
    pub reasons: Vec<String>,
    pub receipt_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelExportPolicy {
    pub max_forward_differential: f64,
    pub max_logit_divergence: f64,
    pub min_rollout_similarity: f64,
    pub min_agentic_success: f64,
    pub min_tool_call_correctness: f64,
    pub min_engram_dependent_task_score: f64,
    pub require_loader_roundtrip: bool,
    pub require_tokenizer_config: bool,
}

impl Default for ModelExportPolicy {
    fn default() -> Self {
        Self {
            max_forward_differential: 0.02,
            max_logit_divergence: 0.02,
            min_rollout_similarity: 0.98,
            min_agentic_success: 0.98,
            min_tool_call_correctness: 0.99,
            min_engram_dependent_task_score: 0.95,
            require_loader_roundtrip: true,
            require_tokenizer_config: true,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ModelExportError {
    #[error("export request identity is incomplete")]
    MissingIdentity,
    #[error("export request digest mismatch")]
    DigestMismatch,
    #[error("export validation is not authoritative")]
    Unmeasured,
    #[error("required living component cannot be represented by target format")]
    UnsupportedComponent,
}

impl ModelExportRequest {
    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.request_digest.clear();
        digest_json(&canonical)
    }

    pub fn seal(mut self) -> Result<Self, ModelExportError> {
        self.request_digest = self.canonical_digest();
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<(), ModelExportError> {
        if self.request_id.is_empty() || self.source_cimage_digest.is_empty() || self.source_generation_digest.is_empty() {
            return Err(ModelExportError::MissingIdentity);
        }
        if !self.request_digest.is_empty() && self.request_digest != self.canonical_digest() {
            return Err(ModelExportError::DigestMismatch);
        }
        Ok(())
    }
}

pub fn validate_export(
    request: &ModelExportRequest,
    exported_artifact_digest: impl Into<String>,
    components: Vec<ExportedComponent>,
    external_sidecars: Vec<String>,
    omitted_components: Vec<String>,
    validation: &ModelExportValidation,
    policy: &ModelExportPolicy,
) -> Result<ModelExportReceipt, ModelExportError> {
    request.verify()?;
    if !validation.measured || validation.validator_fingerprint.is_empty() {
        return Err(ModelExportError::Unmeasured);
    }
    if matches!(request.engram_policy, EngramExportPolicy::RejectIfRequired)
        && components.iter().any(|component| component.component == "engram" && !component.exact_preservation)
    {
        return Err(ModelExportError::UnsupportedComponent);
    }
    let mut reasons = Vec::new();
    if policy.require_loader_roundtrip && !validation.loader_roundtrip { reasons.push("target loader roundtrip failed".into()); }
    if !validation.tensor_identity_valid { reasons.push("tensor identity validation failed".into()); }
    if policy.require_tokenizer_config && request.preserve_tokenizer_and_config && !validation.tokenizer_config_valid { reasons.push("tokenizer/config preservation failed".into()); }
    if validation.forward_differential > policy.max_forward_differential { reasons.push("forward differential exceeds gate".into()); }
    if validation.logit_divergence > policy.max_logit_divergence { reasons.push("logit divergence exceeds gate".into()); }
    if validation.rollout_similarity < policy.min_rollout_similarity { reasons.push("rollout similarity below gate".into()); }
    if validation.agentic_success < policy.min_agentic_success { reasons.push("agentic success below gate".into()); }
    if validation.tool_call_correctness < policy.min_tool_call_correctness { reasons.push("tool-call correctness below gate".into()); }
    if validation.engram_dependent_task_score < policy.min_engram_dependent_task_score { reasons.push("engram-dependent task score below gate".into()); }

    let exact = components.iter().all(|component| component.exact_preservation) && omitted_components.is_empty();
    let equivalence_class = if reasons.is_empty() && exact {
        ExportEquivalenceClass::Exact
    } else if reasons.is_empty() && request.require_behavioral_equivalence {
        ExportEquivalenceClass::BehaviorallyEquivalent
    } else if reasons.is_empty() {
        ExportEquivalenceClass::RepresentationEquivalent
    } else if validation.loader_roundtrip {
        ExportEquivalenceClass::DegradedButAdmitted
    } else {
        ExportEquivalenceClass::Unsupported
    };
    let admitted = reasons.is_empty() || matches!(equivalence_class, ExportEquivalenceClass::DegradedButAdmitted);
    let mut receipt = ModelExportReceipt {
        request_digest: request.request_digest.clone(),
        source_cimage_digest: request.source_cimage_digest.clone(),
        source_generation: request.source_generation,
        target_format: request.target_format.clone(),
        exported_artifact_digest: exported_artifact_digest.into(),
        components,
        external_sidecars,
        omitted_components,
        equivalence_class,
        admitted,
        reasons,
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = digest_json(&receipt);
    Ok(receipt)
}

fn digest_json<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("model export canonical serialization");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

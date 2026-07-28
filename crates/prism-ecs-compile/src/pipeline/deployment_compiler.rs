//! `pipeline::deployment_compiler` — deployment-time compilation surface.
//!
//! This file owns the canonical authority for the deployment-time
//! compiler types: the [`DeploymentRequest`], [`DeploymentResult`],
//! [`ServingProfile`], and the assembly intermediates used to hand off
//! from upstream compilation to lifecycle promotion. The full
//! hardware-gated compiler implementation is reserved for the prism
//! backend; the constitutional surface here is the typed contract that
//! engine callers depend on.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// A request to compile a model into a deployable cimage.
#[derive(Debug, Clone)]
pub struct DeploymentRequest {
    /// Path to the source model directory.
    pub model_path: PathBuf,
    /// Optional path for the produced cimage; defaults alongside the model.
    pub output_path: Option<PathBuf>,
    /// Target hardware identifier (e.g. "apple-m1").
    pub target: String,
    /// Precision / quantization mode (e.g. "nf4", "int8").
    pub precision: String,
    /// Whether multi-token prediction is enabled.
    pub mtp: bool,
    /// Maximum context length for the deployed model.
    pub max_context: Option<usize>,
    /// Optional admission-policy identifier.
    pub admission_policy: Option<String>,
}

impl Default for DeploymentRequest {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            output_path: None,
            target: "apple-m1".into(),
            precision: "nf4".into(),
            mtp: true,
            max_context: Some(8192),
            admission_policy: Some("fail-closed".into()),
        }
    }
}

/// Output from a successful deployment compilation.
#[derive(Debug, Clone)]
pub struct DeploymentResult {
    /// Path to the produced cimage.
    pub cimage_path: PathBuf,
    /// Identifier of the generated artifact.
    pub generation_id: String,
    /// Whether multi-token prediction was enabled.
    pub mtp_enabled: bool,
}

/// Metadata about a compiled model needed at serving time.
///
/// Carried in the cimage assembly and used by runtime loaders to
/// configure the serving environment without re-inspecting the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServingProfile {
    /// Human-readable model name.
    pub model_name: String,
    /// Short model tag for telemetry.
    pub model_tag: String,
    /// Architecture identifier (e.g. "gemma-4").
    pub architecture: String,
    /// Maximum context length in tokens.
    pub context_length: usize,
    /// Precision / quantization mode.
    pub precision: String,
    /// Whether multi-token prediction is enabled.
    pub mtp_enabled: bool,
}

/// A sealed cimage ready for lifecycle promotion.
/// Produced by `seal_and_validate()`, consumed by `promote_cimage()`.
#[derive(Debug, Clone)]
pub struct PromotableCimage {
    /// Fully-assembled cimage.
    pub assembly: CimageAssembly,
    /// Whether validation passed.
    pub validated: bool,
    /// Content-addressed digest of the assembly.
    pub digest: String,
}

/// Content-addressed tensor segment identifier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PhysicalSegmentId(pub String);

/// A compiled kernel artifact (Metal .metallib bytes + ABI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledKernelArtifact {
    /// Identifier of the implementation that produced this artifact.
    pub implementation_id: String,
    /// Compiled bytes (the .metallib on Apple platforms).
    pub compiled_bytes: Vec<u8>,
}

/// Stub execution-graph carrier for the constitutional surface.
///
/// The real `MTPExecutionGraph` lives in the engine and is hardware-
/// gated; the constitutional surface ships a thin placeholder that
/// keeps the type-level authority for downstream consumers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionGraphStub {
    /// MTP depth (0 = no multi-token prediction).
    pub mtp_depth: u32,
}

/// Stub memory-plan carrier for the constitutional surface.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryPlanStub {
    /// Total estimated memory in bytes.
    pub total_bytes: u64,
    /// Peak memory in bytes.
    pub peak_bytes: u64,
}

/// Stub runtime-state-plan carrier for the constitutional surface.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeStatePlanStub {
    /// KV-cache capacity in tokens.
    pub kv_cache_tokens: u64,
    /// Context length in tokens.
    pub context_length: u64,
}

/// A fully assembled deployable cimage ready for lifecycle promotion.
///
/// This is the intermediate representation between compiler output and
/// lifecycle promotion. The constitutional surface preserves the type
/// so engine callers can thread it through the deployment pipeline;
/// the engine's hardware-gated implementation populates it.
#[derive(Debug, Clone)]
pub struct CimageAssembly {
    /// Content-addressed tensor segments keyed by physical segment id.
    pub segments: BTreeMap<PhysicalSegmentId, Vec<u8>>,
    /// Compiled kernel artifacts (Metal .metallib bytes + ABI).
    pub kernel_artifacts: Vec<CompiledKernelArtifact>,
    /// Execution graph topology (MTP-aware for Gemma 4 in the engine).
    pub execution_graph: ExecutionGraphStub,
    /// Memory allocation plan.
    pub memory_plan: MemoryPlanStub,
    /// Runtime state plan (KV cache sizing, context length).
    pub runtime_state: RuntimeStatePlanStub,
    /// Serving metadata for runtime configuration.
    pub serving_profile: ServingProfile,
}

impl CimageAssembly {
    /// Compute a content-addressed digest over all segments and kernel
    /// artifacts. Uses SHA-256 of the concatenation; output is hex.
    pub fn compute_digest(&self) -> String {
        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        for (id, data) in &self.segments {
            hasher.update(id.0.as_bytes());
            hasher.update(data);
        }
        for artifact in &self.kernel_artifacts {
            hasher.update(artifact.implementation_id.as_bytes());
            hasher.update(&artifact.compiled_bytes);
        }
        format!("{:x}", hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_profile_default_construction() {
        let p = ServingProfile {
            model_name: "gemma-4-9b".into(),
            model_tag: "default".into(),
            architecture: "gemma-4".into(),
            context_length: 8192,
            precision: "nf4".into(),
            mtp_enabled: true,
        };
        assert_eq!(p.context_length, 8192);
        assert!(p.mtp_enabled);
    }

    #[test]
    fn deployment_request_default_is_fail_closed() {
        let r = DeploymentRequest::default();
        assert_eq!(r.target, "apple-m1");
        assert_eq!(r.precision, "nf4");
        assert!(r.mtp);
        assert_eq!(r.max_context, Some(8192));
        assert_eq!(r.admission_policy.as_deref(), Some("fail-closed"));
    }

    #[test]
    fn assembly_compute_digest_is_deterministic() {
        let mut segments: BTreeMap<PhysicalSegmentId, Vec<u8>> = BTreeMap::new();
        segments.insert(PhysicalSegmentId("a".into()), vec![1, 2, 3]);
        let assembly = CimageAssembly {
            segments,
            kernel_artifacts: vec![CompiledKernelArtifact {
                implementation_id: "k1".into(),
                compiled_bytes: vec![9, 9, 9],
            }],
            execution_graph: ExecutionGraphStub::default(),
            memory_plan: MemoryPlanStub::default(),
            runtime_state: RuntimeStatePlanStub::default(),
            serving_profile: ServingProfile {
                model_name: "m".into(),
                model_tag: "t".into(),
                architecture: "gemma-4".into(),
                context_length: 4096,
                precision: "nf4".into(),
                mtp_enabled: false,
            },
        };
        let d1 = assembly.compute_digest();
        let d2 = assembly.compute_digest();
        assert_eq!(d1, d2);
        assert!(!d1.is_empty());
    }

    #[test]
    fn assembly_compute_digest_differs_on_segment_change() {
        let mk = |seg_bytes: Vec<u8>| {
            let mut segments: BTreeMap<PhysicalSegmentId, Vec<u8>> = BTreeMap::new();
            segments.insert(PhysicalSegmentId("a".into()), seg_bytes);
            CimageAssembly {
                segments,
                kernel_artifacts: vec![],
                execution_graph: ExecutionGraphStub::default(),
                memory_plan: MemoryPlanStub::default(),
                runtime_state: RuntimeStatePlanStub::default(),
                serving_profile: ServingProfile {
                    model_name: "m".into(),
                    model_tag: "t".into(),
                    architecture: "gemma-4".into(),
                    context_length: 4096,
                    precision: "nf4".into(),
                    mtp_enabled: false,
                },
            }
        };
        let d1 = mk(vec![1, 2, 3]).compute_digest();
        let d2 = mk(vec![1, 2, 4]).compute_digest();
        assert_ne!(d1, d2);
    }
}

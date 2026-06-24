//! Core ML fixture qualification types for Track A.
//!
//! This module defines the data structures, enums, traits, and error types
//! used to represent Core ML model fixtures, execution policies, prediction
//! requests/results, qualification receipts, and the artifact executor trait.
//!
//! All types are gated behind the `mlx-backend` or `prism-backend` features,
//! matching the convention of sibling Core ML modules.

#![cfg(any(feature = "mlx-backend", feature = "prism-backend"))]

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

// ── Local type aliases for prism-engine types.rs references ───────────────

/// Unique identifier for a qualification receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReceiptId(pub Uuid);

impl ReceiptId {
    /// Create a new random receipt identifier.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ReceiptId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ReceiptId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Record of a materialization operation for Core ML fixture data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterializationReceipt {
    /// Bytes read during materialization.
    pub bytes_read: u64,
    /// Bytes written during materialization.
    pub bytes_written: u64,
    /// Duration of the materialization operation in microseconds.
    pub duration_us: u64,
    /// Reason or description of the materialization.
    pub reason: String,
}

/// How far a Core ML fixture has been qualified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationStatus {
    /// Not yet benchmarked or validated.
    Unqualified,
    /// Parity and latency verified on the target hardware.
    Qualified,
    /// Benchmark data is absent or inconclusive; pending qualification.
    BenchmarkUnresolved,
}

/// SHA-256 hex digest of a Core ML model output tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OutputDigest(pub String);

impl OutputDigest {
    /// Create a new output digest from a hex string.
    pub fn from_hex(hex: impl Into<String>) -> Self {
        Self(hex.into())
    }
}

impl fmt::Display for OutputDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// SHA-256 hex digest of a Core ML model artifact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArtifactDigest(pub String);

impl ArtifactDigest {
    /// Create a new artifact digest from a hex string.
    pub fn from_hex(hex: impl Into<String>) -> Self {
        Self(hex.into())
    }
}

impl fmt::Display for ArtifactDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ── Core ML execution policy ───────────────────────────────────────────────

/// Execution unit policy for Core ML model inference.
///
/// Maps to `MLComputeUnits` in the Apple Core ML runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i64)]
pub enum CoreMlExecutionPolicy {
    /// Let the system choose the default compute units.
    SystemDefault = 0,
    /// Prefer the Apple Neural Engine when available.
    PreferNeuralEngine = 1,
    /// Use both CPU and Neural Engine.
    CpuAndNeuralEngine = 2,
    /// Use all available compute units (CPU, GPU, Neural Engine).
    AllComputeUnits = 3,
}

impl CoreMlExecutionPolicy {
    /// Human-readable name for this policy.
    pub fn name(&self) -> &'static str {
        match self {
            Self::SystemDefault => "system_default",
            Self::PreferNeuralEngine => "prefer_neural_engine",
            Self::CpuAndNeuralEngine => "cpu_and_neural_engine",
            Self::AllComputeUnits => "all_compute_units",
        }
    }
}

// ── Named tensor types ─────────────────────────────────────────────────────

/// Named input tensor for a Core ML prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedTensorInput {
    /// Name of the input feature.
    pub name: String,
    /// Flattened float data.
    pub data: Vec<f32>,
    /// Shape of the tensor (e.g. `[1, 3, 224, 224]`).
    pub shape: Vec<usize>,
}

/// Named output tensor from a Core ML prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedTensorOutput {
    /// Name of the output feature.
    pub name: String,
    /// Flattened float data.
    pub data: Vec<f32>,
    /// Shape of the tensor (e.g. `[1, 1000]`).
    pub shape: Vec<usize>,
}

// ── Prediction request / result ────────────────────────────────────────────

/// A prediction request to a Core ML model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlPredictionRequest {
    /// Named input tensors.
    pub inputs: Vec<NamedTensorInput>,
    /// Execution policy override.
    pub execution_policy: CoreMlExecutionPolicy,
}

/// The result of a Core ML prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlPredictionResult {
    /// Named output tensors.
    pub outputs: Vec<NamedTensorOutput>,
    /// Measured provider latency in milliseconds.
    pub provider_latency_ms: f64,
}

// ── Fixture manifest ───────────────────────────────────────────────────────

/// Detailed manifest describing a Core ML test fixture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlFixtureManifest {
    /// Unique fixture identifier (e.g. UUID or descriptive slug).
    pub fixture_id: String,
    /// SHA-256 digest of the model file.
    pub model_digest: String,
    /// Compiler version used to produce this model (e.g. `"coremltools 7.2"`).
    pub compiler_version: String,
    /// Name of the model's primary input feature.
    pub input_name: String,
    /// Name of the model's primary output feature.
    pub output_name: String,
    /// Expected input tensor shape.
    pub input_shape: Vec<usize>,
    /// Expected output tensor shape.
    pub output_shape: Vec<usize>,
    /// Expected input tensor data for validation.
    pub expected_input: Vec<f32>,
    /// Expected output tensor data for validation.
    pub expected_output: Vec<f32>,
}

// ── Qualification receipt ──────────────────────────────────────────────────

/// Comprehensive qualification receipt for a Core ML fixture run.
///
/// This is the authoritative record of a qualification attempt, capturing
/// identification, execution parameters, hardware/environment metadata,
/// numerical validation metrics, and the materialization trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlQualificationReceipt {
    // ── Identity ──────────────────────────────────────────────────────────
    /// Unique receipt identifier.
    pub id: ReceiptId,
    /// Fixture identifier this receipt qualifies.
    pub fixture_id: String,
    /// Overall qualification status.
    pub status: QualificationStatus,

    // ── Artifact identification ───────────────────────────────────────────
    /// Digest of the model artifact tested.
    pub artifact_digest: ArtifactDigest,
    /// Digest of the model output produced during qualification.
    pub output_digest: OutputDigest,
    /// SHA-256 digest of the source model file.
    pub model_digest: String,

    // ── Software environment ──────────────────────────────────────────────
    /// Core ML compiler version used to produce the model.
    pub compiler_version: String,
    /// Hardware model identifier (e.g. `"Mac15,9"`, `"iPhone17,2"`).
    pub hardware_model: String,
    /// Operating system version (e.g. `"macOS 15.2"`, `"iOS 18.2"`).
    pub os_version: String,

    // ── Execution parameters ──────────────────────────────────────────────
    /// Execution policy used during qualification.
    pub execution_policy: CoreMlExecutionPolicy,

    // ── Latency measurements (ms) ─────────────────────────────────────────
    /// End-to-end provider latency in milliseconds.
    pub provider_latency_ms: f64,
    /// CPU-side latency contribution in milliseconds.
    pub cpu_latency_ms: f64,
    /// GPU-side latency contribution in milliseconds (zero if not used).
    pub gpu_latency_ms: f64,
    /// Apple Neural Engine latency contribution in milliseconds.
    pub ane_latency_ms: f64,

    // ── Numerical validation ──────────────────────────────────────────────
    /// Mean absolute error against the reference output.
    pub mean_absolute_error: f64,
    /// Maximum absolute error against the reference output.
    pub max_absolute_error: f64,
    /// Peak signal-to-noise ratio in dB.
    pub psnr: f64,
    /// Cosine similarity with the reference output.
    pub cosine_similarity: f64,

    // ── Tensor metadata ───────────────────────────────────────────────────
    /// Shape of the input tensor.
    pub input_shape: Vec<usize>,
    /// Shape of the output tensor.
    pub output_shape: Vec<usize>,
    /// Number of elements in the input tensor.
    pub input_element_count: usize,
    /// Number of elements in the output tensor.
    pub output_element_count: usize,

    // ── Materialization trail ─────────────────────────────────────────────
    /// Receipt describing how fixture data was materialized.
    pub materialization: MaterializationReceipt,

    // ── Qualification outcome ─────────────────────────────────────────────
    /// ISO 8601 timestamp of the qualification run.
    pub timestamp: String,
    /// Whether the qualification passed all acceptance criteria.
    pub passed: bool,
}

// ── Artifact handle / loaded artifact ──────────────────────────────────────

/// A handle referencing a Core ML model artifact on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlArtifactHandle {
    /// Filesystem path to the `.mlpackage` bundle.
    pub path: String,
    /// SHA-256 digest of the artifact contents.
    pub digest: [u8; 32],
}

/// A loaded Core ML model artifact ready for inference.
#[derive(Debug, Clone)]
pub struct LoadedCoreMlArtifact {
    /// The handle referencing the source artifact.
    pub handle: CoreMlArtifactHandle,
}

// ── Artifact executor trait ────────────────────────────────────────────────

/// Trait for executing predictions against a Core ML model artifact.
///
/// Implementations wrap the Core ML runtime (or a mock) and provide the
/// standard load/predict lifecycle used by the qualification harness.
pub trait CoreMlArtifactExecutor {
    /// The type of error returned by this executor.
    type Error: std::error::Error;

    /// Load a Core ML model from the given artifact handle.
    ///
    /// Returns a [`LoadedCoreMlArtifact`] on success, or an error if
    /// loading fails (e.g. file not found, invalid model, incompatible
    /// compute units).
    fn load(&self, handle: &CoreMlArtifactHandle) -> Result<LoadedCoreMlArtifact, Self::Error>;

    /// Run a prediction on a loaded artifact.
    ///
    /// Takes a reference to the loaded artifact and a [`CoreMlPredictionRequest`]
    /// describing the inputs and execution policy. Returns a
    /// [`CoreMlPredictionResult`] containing output tensors and latency
    /// measurements.
    fn predict(
        &self,
        artifact: &LoadedCoreMlArtifact,
        request: &CoreMlPredictionRequest,
    ) -> Result<CoreMlPredictionResult, Self::Error>;
}

// ── Bridge error enum ──────────────────────────────────────────────────────

/// Errors that can occur during Core ML Bridge operations.
#[derive(Debug, Clone)]
pub enum CoreMlBridgeError {
    /// The model file was not found at the specified path.
    ModelNotFound(String),
    /// The model could not be loaded by the Core ML runtime.
    ModelLoadFailed(String),
    /// Prediction execution failed.
    PredictionFailed(String),
    /// One or more input tensors are invalid (wrong name, shape, or type).
    InvalidInput(String),
    /// One or more output tensors are invalid or missing.
    InvalidOutput(String),
    /// The tensor shape does not match the model's expected shape.
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape provided or received.
        actual: Vec<usize>,
    },
    /// The specified execution policy is not supported on this hardware.
    UnsupportedPolicy(CoreMlExecutionPolicy),
    /// General execution failure.
    ExecutionFailed(String),
    /// Internal bridge error.
    BridgeError(String),
}

impl fmt::Display for CoreMlBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotFound(path) => {
                write!(f, "Core ML model not found at path: {path}")
            }
            Self::ModelLoadFailed(msg) => {
                write!(f, "Core ML model load failed: {msg}")
            }
            Self::PredictionFailed(msg) => {
                write!(f, "Core ML prediction failed: {msg}")
            }
            Self::InvalidInput(msg) => {
                write!(f, "invalid Core ML input: {msg}")
            }
            Self::InvalidOutput(msg) => {
                write!(f, "invalid Core ML output: {msg}")
            }
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "Core ML shape mismatch: expected {expected:?}, got {actual:?}"
                )
            }
            Self::UnsupportedPolicy(policy) => {
                write!(f, "unsupported Core ML execution policy: {policy:?}")
            }
            Self::ExecutionFailed(msg) => {
                write!(f, "Core ML execution failed: {msg}")
            }
            Self::BridgeError(msg) => {
                write!(f, "Core ML bridge error: {msg}")
            }
        }
    }
}

impl std::error::Error for CoreMlBridgeError {}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receipt_id_new_is_unique() {
        let a = ReceiptId::new();
        let b = ReceiptId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn test_receipt_id_display() {
        let id = ReceiptId::new();
        let s = id.to_string();
        assert!(!s.is_empty());
        assert_eq!(s.len(), 36); // UUID v4 format
    }

    #[test]
    fn test_execution_policy_name() {
        assert_eq!(
            CoreMlExecutionPolicy::SystemDefault.name(),
            "system_default"
        );
        assert_eq!(
            CoreMlExecutionPolicy::PreferNeuralEngine.name(),
            "prefer_neural_engine"
        );
        assert_eq!(
            CoreMlExecutionPolicy::CpuAndNeuralEngine.name(),
            "cpu_and_neural_engine"
        );
        assert_eq!(
            CoreMlExecutionPolicy::AllComputeUnits.name(),
            "all_compute_units"
        );
    }

    #[test]
    fn test_named_tensor_input() {
        let input = NamedTensorInput {
            name: "input".into(),
            data: vec![1.0, 2.0, 3.0],
            shape: vec![1, 3],
        };
        assert_eq!(input.name, "input");
        assert_eq!(input.data.len(), 3);
        assert_eq!(input.shape, vec![1, 3]);
    }

    #[test]
    fn test_named_tensor_output() {
        let output = NamedTensorOutput {
            name: "output".into(),
            data: vec![0.5, 0.8],
            shape: vec![1, 2],
        };
        assert_eq!(output.name, "output");
        assert_eq!(output.data.len(), 2);
    }

    #[test]
    fn test_prediction_request() {
        let req = CoreMlPredictionRequest {
            inputs: vec![NamedTensorInput {
                name: "x".into(),
                data: vec![0.0; 4],
                shape: vec![1, 4],
            }],
            execution_policy: CoreMlExecutionPolicy::AllComputeUnits,
        };
        assert_eq!(req.inputs.len(), 1);
        assert_eq!(req.execution_policy, CoreMlExecutionPolicy::AllComputeUnits);
    }

    #[test]
    fn test_prediction_result() {
        let result = CoreMlPredictionResult {
            outputs: vec![NamedTensorOutput {
                name: "y".into(),
                data: vec![1.0],
                shape: vec![1, 1],
            }],
            provider_latency_ms: 12.5,
        };
        assert_eq!(result.outputs.len(), 1);
        assert!((result.provider_latency_ms - 12.5).abs() < 1e-6);
    }

    #[test]
    fn test_artifact_handle() {
        let handle = CoreMlArtifactHandle {
            path: "/tmp/model.mlpackage".into(),
            digest: [0u8; 32],
        };
        assert_eq!(handle.path, "/tmp/model.mlpackage");
        assert_eq!(handle.digest.len(), 32);
    }

    #[test]
    fn test_loaded_artifact() {
        let loaded = LoadedCoreMlArtifact {
            handle: CoreMlArtifactHandle {
                path: "test.mlpackage".into(),
                digest: [1u8; 32],
            },
        };
        assert_eq!(loaded.handle.path, "test.mlpackage");
    }

    #[test]
    fn test_qualification_receipt_fields() {
        let receipt = CoreMlQualificationReceipt {
            id: ReceiptId::new(),
            fixture_id: "gemma-4-12b".into(),
            status: QualificationStatus::Qualified,
            artifact_digest: ArtifactDigest::from_hex("a".repeat(64)),
            output_digest: OutputDigest::from_hex("b".repeat(64)),
            model_digest: "c".repeat(64),
            compiler_version: "coremltools 7.2".into(),
            hardware_model: "Mac15,9".into(),
            os_version: "macOS 15.2".into(),
            execution_policy: CoreMlExecutionPolicy::AllComputeUnits,
            provider_latency_ms: 42.0,
            cpu_latency_ms: 5.0,
            gpu_latency_ms: 10.0,
            ane_latency_ms: 27.0,
            mean_absolute_error: 0.001,
            max_absolute_error: 0.005,
            psnr: 45.0,
            cosine_similarity: 0.9999,
            input_shape: vec![1, 3, 224, 224],
            output_shape: vec![1, 1000],
            input_element_count: 3 * 224 * 224,
            output_element_count: 1000,
            materialization: MaterializationReceipt {
                bytes_read: 4096,
                bytes_written: 4096,
                duration_us: 1500,
                reason: "fixture loaded from disk".into(),
            },
            timestamp: "2026-06-24T12:00:00Z".into(),
            passed: true,
        };

        assert_eq!(receipt.fixture_id, "gemma-4-12b");
        assert_eq!(receipt.status, QualificationStatus::Qualified);
        assert!(receipt.passed);
        assert_eq!(receipt.input_shape.len(), 4);
        assert_eq!(receipt.output_shape.len(), 2);
    }

    #[test]
    fn test_bridge_error_display() {
        let err = CoreMlBridgeError::ModelNotFound("/tmp/model.mlpackage".into());
        let msg = err.to_string();
        assert!(msg.contains("not found"));

        let err = CoreMlBridgeError::ShapeMismatch {
            expected: vec![1, 3],
            actual: vec![1, 4],
        };
        let msg = err.to_string();
        assert!(msg.contains("expected"));
        assert!(msg.contains("got"));

        let err = CoreMlBridgeError::PredictionFailed("oom".into());
        assert!(err.to_string().contains("failed"));
    }

    #[test]
    fn test_artifact_executor_trait_is_object_safe() {
        // Compile-time check: the trait must be usable as a trait object
        // if `Error` is boxed.  This test only verifies the trait compiles.
        fn _take_executor(_: &dyn CoreMlArtifactExecutor<Error = CoreMlBridgeError>) {}
        let _ = _take_executor;
    }
}

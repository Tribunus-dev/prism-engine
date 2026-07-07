//! ECS component types for the compilation pipeline.
//!
//! Each component represents one stage of a matrix's lifecycle through the
//! admission → binding → refinement compilation pipeline.  Components are
//! attached to entities representing individual weight matrices.

use crate::compute_image::compile::ternary::MatrixWeightBindingV1;
use crate::compilation::distill_core::OnPolicyRefinementResult;
use crate::quantization::contract::{CanonicalShape, RuntimeRepresentationClass};

// ---------------------------------------------------------------------------
// Phase and status
// ---------------------------------------------------------------------------

/// Compilation lifecycle phase for a matrix entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationPhase {
    /// Initial state — no validation has been performed.
    Pending,
    /// Source weights validated and shape inferred.
    SourceValidated,
    /// A quantization candidate passed weight-space screening.
    Admitted,
    /// Tensor binding prepared (MatrixWeightBindingV1).
    Bound,
    /// Cimage sealed successfully.
    Sealed,
    /// Compilation failed at some stage.
    Failed,
}

/// Compilation status attached to every matrix entity.
#[derive(Debug, Clone)]
pub struct CompilationStatus {
    pub phase: CompilationPhase,
    pub error: Option<String>,
    pub format: Option<RuntimeRepresentationClass>,
}

impl CompilationStatus {
    pub fn new() -> Self {
        Self {
            phase: CompilationPhase::Pending,
            error: None,
            format: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Source and shape
// ---------------------------------------------------------------------------

/// Raw source weights (f32 slice) for a matrix.
#[derive(Debug, Clone)]
pub struct SourceWeights(pub Vec<f32>);

/// Canonical tensor shape for this matrix.
#[derive(Debug, Clone, Copy)]
pub struct TensorShape(pub CanonicalShape);

// ---------------------------------------------------------------------------
// Admission output
// ---------------------------------------------------------------------------

/// Packed codes, tile scales/biases, and optional per-channel scale vector
/// from the winning quantization candidate.
#[derive(Debug, Clone)]
pub struct CodesData {
    pub codes: Vec<u8>,
    pub scales: Vec<f32>,
    pub biases: Vec<f32>,
    pub scale_vector: Option<Vec<f32>>,
}

/// Reconstructed weight matrix from the packed candidate.
#[derive(Debug, Clone)]
pub struct ReconstructedWeights(pub Vec<f32>);

// ---------------------------------------------------------------------------
// Binding output
// ---------------------------------------------------------------------------

/// Tensor binding in the execution graph (wraps MatrixWeightBindingV1).
#[derive(Debug, Clone)]
pub struct TensorBinding(pub MatrixWeightBindingV1);

// ---------------------------------------------------------------------------
// Refinement output
// ---------------------------------------------------------------------------

/// Result of on-policy refinement for a bound matrix.
#[derive(Debug, Clone)]
pub struct RefinementOutcome(pub OnPolicyRefinementResult);

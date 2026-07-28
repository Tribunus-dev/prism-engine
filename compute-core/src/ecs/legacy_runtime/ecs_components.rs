//! ECS component types for the compilation pipeline.
//!
//! Each component represents one stage of a matrix's lifecycle through the
//! admission → binding → refinement compilation pipeline.  Components are
//! attached to entities representing individual weight matrices.

use crate::ecs::canonical::generation::CimageGeneration;
use crate::ecs::legacy_cimage::generation_store::ContentStore;
use crate::ecs::cimage_runtime::context::CimageRuntimeContext;
use prism_ecs_compile::compilation::distill_core::OnPolicyRefinementResult;
use crate::ecs::legacy_compute_image_core::compile::ternary::MatrixWeightBindingV1;
use crate::ecs::runtime::world::{Entity, World};
use crate::ecs::runtime::world_txn::WorldTxn;
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

// ---------------------------------------------------------------------------
// Load generation into ECS world
// ---------------------------------------------------------------------------

/// Load a [`CimageGeneration`] from a [`ContentStore`] and attach the
/// resulting [`CimageRuntimeContext`] to a new ECS entity.
///
/// Resolves every tensor binding's physical segments through the content
/// store, decodes them into runtime tensor payloads, and spawns an entity
/// carrying the fully-loaded context. Returns `Err` describing the first
/// missing segment.
///
/// The returned entity owns the generation reference, resolved weight/scale
/// payload bytes, and a pre-built [`RuntimeTensorStore`] — ready for Metal
/// buffer creation or CPU fallback execution.
pub fn load_from_generation(
    world: &mut World,
    generation: CimageGeneration,
    store: &ContentStore,
) -> Result<Entity, String> {
    let context = CimageRuntimeContext::load_from_generation(generation, store)?;
    // Constitutional mutation seam: stage the spawn + insert on a
    // `WorldTxn` and commit atomically. Keeps the canonical authority
    // for generation-loading entity creation at a single point.
    let mut txn = WorldTxn::new();
    let token = txn.stage_spawn();
    txn.stage_insert_on(token, context);
    let mut spawned = txn.commit(world).map_err(|e| e.to_string())?;
    let entity = spawned
        .pop()
        .ok_or_else(|| "WorldTxn returned no entity for staged spawn".to_string())?;
    Ok(entity)
}

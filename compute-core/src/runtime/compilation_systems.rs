//! Compilation pipeline ECS systems.
//!
//! Each system is a stateless `fn run(&mut World)` that advances the
//! compilation lifecycle for all matching entities.  Systems are idempotent
//! in the sense that they only operate on entities in the expected phase.

use crate::compilation::distill_core::{
    on_policy_refine, RefinementConfig, TemperatureSchedule,
};
use crate::compute_image::compile::capability_registry::CapabilityRegistry;
use crate::compute_image::compile::ternary::MatrixWeightBindingV1;
use crate::quantization::admission::{
    candidate_plan, compute_weight_nrmse, pack_candidate, reconstruct_candidate,
};
use crate::quantization::contract::{CanonicalShape, 
    BackendKind, QuantizationHint, RuntimeRepresentationClass, TensorClass,
};
use crate::runtime::ecs_components::{
    CodesData, CompilationPhase, CompilationStatus, ReconstructedWeights,
    RefinementOutcome, SourceWeights, TensorBinding, TensorShape,
};
use crate::runtime::stage_graph::{StageConfig, StageGraph, StageQuantizationConfig};
pub use crate::compute_image::compile::ternary::ModelConfig;
use crate::runtime::world::{Entity, World};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Max allowed NRMSE per representation class for the simplified admission
/// gate used in the ECS pipeline (no activation-bank validation).
/// Falls back to hardcoded defaults if no stage config is present.
fn weight_screening_threshold(
    format: RuntimeRepresentationClass,
    stage_config: Option<&StageQuantizationConfig>,
) -> f64 {
    if let Some(cfg) = stage_config {
        if let Some(&threshold) = cfg.weight_nrmse_thresholds.get(&format) {
            return threshold;
        }
    }
    // Hardcoded defaults when no per-stage config is available.
    match format {
        RuntimeRepresentationClass::TernaryTile640Base => 0.02,
        RuntimeRepresentationClass::Nf4Tile640Base => 0.01,
        RuntimeRepresentationClass::Int8Tile640Base => 0.005,
        RuntimeRepresentationClass::RawF32 => f64::MAX,
    }
}

/// Code and metadata tile byte sizes per representation.
fn tile_byte_sizes(format: RuntimeRepresentationClass) -> (u32, u16) {
    match format {
        RuntimeRepresentationClass::TernaryTile640Base => (160, 4),
        RuntimeRepresentationClass::Nf4Tile640Base => (320, 8),
        RuntimeRepresentationClass::Int8Tile640Base => (640, 4),
        RuntimeRepresentationClass::RawF32 => (0, 0),
    }
}

// ===========================================================================
// System 1: Validate source weights
// ===========================================================================

/// Validate source weights and produce `CompilationPhase::SourceValidated`.
///
/// Checks that source weight data is present and non-empty.  Marks an entity
/// as `Failed` if validation fails.
pub fn validate_sources(world: &mut World) {
    let entities: Vec<Entity> = world.iter_entities_with::<SourceWeights>().collect();
    for entity in entities {
        // Check if source weights are present and non-empty while borrowing immutably
        let has_valid_weights = world.get::<SourceWeights>(entity)
            .map_or(false, |w| !w.0.is_empty());

        // Then mutably update status — all immutable borrows dropped
        if let Some(s) = world.get_mut::<CompilationStatus>(entity) {
            if has_valid_weights {
                s.phase = CompilationPhase::SourceValidated;
                s.error = None;
            } else {
                s.phase = CompilationPhase::Failed;
                s.error = Some("Empty source weights".into());
            }
        }
    }
}

// ===========================================================================
// System 2: Admit candidates
// ===========================================================================

/// Run the candidate admission pipeline on each validated or pending entity.
///
/// For each entity in `SourceValidated` or `Pending` phase, generates a
/// candidate plan, packs and reconstructs each eligible format, and selects
/// the first candidate whose weight-space NRMSE is within the per-format
/// threshold.  The winner's packed data is stored via `CodesData` and
/// `ReconstructedWeights`, and the status is advanced to `Admitted`.
///
/// If no candidate passes screening, the entity is marked `Failed`.
pub fn admit_candidates(world: &mut World) {
    let entities: Vec<Entity> = world.iter_entities_with::<SourceWeights>().collect();

    // Read per-stage quantization config (if present) for thresholds and format priority.
    let stage_config = world.get_resource::<StageConfigResource>().map(|r| r.0.clone());
    let stage_config_ref = stage_config.as_ref();

    for entity in entities {
        let _status = match world.get::<CompilationStatus>(entity) {
            Some(s) if s.phase == CompilationPhase::SourceValidated
                || s.phase == CompilationPhase::Pending =>
            {
                s.clone()
            }
            _ => continue,
        };

        // Clone data from world refs before any mutable access
        let owned_source = world.get::<SourceWeights>(entity).map(|w| w.0.clone());
        let owned_shape = world.get::<TensorShape>(entity).map(|s| s.0);
        let source = match owned_source { Some(s) => s, None => continue };
        let shape = match owned_shape { Some(s) => s, None => continue };
        let in_features = shape.in_features as usize;
        let out_features = shape.out_features as usize;

        let hint = QuantizationHint {
            tensor_class: TensorClass::Unknown,
            permit_int8_candidate: true,
        };

        let candidates = candidate_plan(in_features, out_features, &hint);

        // Use stage-level format ordering if available; otherwise fall back to candidate_plan order.
        let format_order: Vec<RuntimeRepresentationClass> = match &stage_config {
            Some(cfg) => cfg.permitted_formats.clone(),
            None => candidates.clone(),
        };

        for &format in &format_order {
            // Skip formats not in the candidate plan for this tensor shape.
            if !candidates.contains(&format) {
                continue;
            }

            // Skip formats that are not production-ready (but always allow RawF32).
            if format != RuntimeRepresentationClass::RawF32 {
                let blocked = world.get_resource::<CapabilityRegistry>().map_or(false, |r| {
                    !r.is_production_ready(format, 1, BackendKind::Metal)
                });
                if blocked {
                    continue;
                }
            }

            let (codes, scales, biases, scale_vector) =
                pack_candidate(&source, in_features, out_features, format, None);
            let reconstructed = reconstruct_candidate(
                format,
                &codes,
                &scales,
                &biases,
                in_features,
                out_features,
                scale_vector.as_deref(),
            );
            let weight_nrmse = compute_weight_nrmse(&source, &reconstructed);

            if weight_nrmse <= weight_screening_threshold(format, stage_config_ref) {
                // Store packed data and reconstruction.
                world.insert(
                    entity,
                    CodesData {
                        codes,
                        scales,
                        biases,
                        scale_vector,
                    },
                );
                world.insert(entity, ReconstructedWeights(reconstructed));

                // Advance status.
                if let Some(s) = world.get_mut::<CompilationStatus>(entity) {
                    s.phase = CompilationPhase::Admitted;
                    s.format = Some(format);
                    s.error = None;
                }
                break; // first passing candidate wins
            }
        }

        // If no CodesData was inserted, no candidate passed.
        if !world.has::<CodesData>(entity) {
            if let Some(s) = world.get_mut::<CompilationStatus>(entity) {
                s.phase = CompilationPhase::Failed;
                s.error = Some("No candidate format passed weight screening".into());
            }
        }
    }
}

// ===========================================================================
// System 3: Bind tensors
// ===========================================================================

/// Produce a `MatrixWeightBindingV1` for each admitted entity.
///
/// Computes tile geometry from the representation class and the entity's
/// tensor shape, then writes the binding into a `TensorBinding` component.
pub fn bind_tensors(world: &mut World) {
    let entities: Vec<Entity> = world.iter_entities_with::<SourceWeights>().collect();
    for entity in entities {
        let status = match world.get::<CompilationStatus>(entity) {
            Some(s) if s.phase == CompilationPhase::Admitted => s.clone(),
            _ => continue,
        };
        let format = match status.format {
            Some(f) => f,
            None => continue,
        };
        let in_features = world.get::<TensorShape>(entity).map_or(u32::MAX, |s| s.0.in_features);
        let out_features = world.get::<TensorShape>(entity).map_or(u32::MAX, |s| s.0.out_features);
        if in_features == u32::MAX || out_features == u32::MAX { continue; }
        let tiles = ((in_features as u64) + 639) / 640;
        let total_tiles = (out_features as u64) * tiles;
        let (code_tile_bytes, meta_tile_bytes) = tile_byte_sizes(format);

        let binding = MatrixWeightBindingV1 {
            binding_wire_version: 1,
            matrix_id: entity.0,
            tensor_id: [0u8; 16],
            representation: format as u8,
            representation_version: 1,
            kernel_abi_digest: [0u8; 32],
            in_features,
            out_features,
            reduction_tile_size: 640,
            tiles_per_output_channel: tiles as u32,
            tail_reduction_count: (in_features % 640) as u16,
            macro_layout: 1, // OutputChannelContiguous
            tail_handling: 1, // ActivationZeroPredicationV1
            code_segment: 41, // MatrixContract (placeholder)
            code_offset: 0,
            code_length: total_tiles * code_tile_bytes as u64,
            code_tile_stride_bytes: code_tile_bytes,
            metadata_segment: 41,
            metadata_offset: total_tiles * code_tile_bytes as u64,
            metadata_length: total_tiles * meta_tile_bytes as u64,
            metadata_tile_stride_bytes: meta_tile_bytes,
            sidecar_segment: 0,
            sidecar_offset: 0,
            sidecar_length: 0,
            sidecar_kind: 0,
            sidecar_element_format: 0,
            sidecar_count: 0,
            residual_segment: 0,
            residual_offset: 0,
            residual_length: 0,
            required_alignment_bytes: 64,
        };

        // Drop status so we can mutably access world below
        drop(status);

        world.insert(entity, TensorBinding(binding));

        if let Some(s) = world.get_mut::<CompilationStatus>(entity) {
            s.phase = CompilationPhase::Bound;
            s.error = None;
        }
    }
}


// ===========================================================================
// System 4: Refine tensors (on-policy distillation)
// ===========================================================================

/// Run on-policy refinement on bound entities.
///
/// For each entity in the `Bound` phase, attempts up to 8 rounds of
/// code-space refinement to improve reconstruction quality.  The refinement
/// modifies quantized codes, re-reconstructs, and accepts changes that
/// improve weight-space NRMSE.  Stores the result as `RefinementOutcome`.
pub fn refine_tensors(world: &mut World) {
    let entities: Vec<Entity> = world.iter_entities_with::<TensorBinding>().collect();
    let ref_config = RefinementConfig::default();
    let temp_schedule = TemperatureSchedule::default_r8();

    for entity in entities {
        let status = match world.get::<CompilationStatus>(entity) {
            Some(s) if s.phase == CompilationPhase::Bound => s.clone(),
            _ => continue,
        };

        // Clone data for the refinement closure
        let owned_source = match world.get::<SourceWeights>(entity).map(|w| w.0.clone()) {
            Some(s) => s,
            None => continue,
        };
        let owned_codes = match world.get::<CodesData>(entity) {
            Some(c) => c.clone(),
            None => continue,
        };
        let owned_recon = match world.get::<ReconstructedWeights>(entity).map(|w| w.0.clone()) {
            Some(r) => r,
            None => continue,
        };
        let owned_shape = match world.get::<TensorShape>(entity).map(|s| s.0) {
            Some(s) => s,
            None => continue,
        };
        let format = match status.format {
            Some(f) => f,
            None => continue,
        };

        let in_features = owned_shape.in_features as usize;
        let out_features = owned_shape.out_features as usize;

        // Compute initial loss (weight-space NRMSE)
        let initial_loss = crate::quantization::admission::compute_weight_nrmse(
            &owned_source, &owned_recon,
        );

        // Refinement closure — mutably captures cloned data
        let mut codes = owned_codes.codes.clone();
        let scales = owned_codes.scales.clone();
        let biases = owned_codes.biases.clone();
        let scale_vector = owned_codes.scale_vector.clone();
        let source = owned_source.clone();
        let format = format;

        let result = on_policy_refine(
            initial_loss,
            |round| {
                let temp = temp_schedule.temperature(round);

                // Perturb a small fraction of codes based on temperature
                let n_coords = codes.len();
                let n_flip = ((n_coords as f32) * 0.01 * temp).max(1.0) as usize;
                let n_flip = n_flip.min(n_coords / 10 + 1);

                // Flip some codes at random (seeded by round)
                use std::hash::{Hash, Hasher};
                let mut s = std::collections::hash_map::DefaultHasher::new();
                round.hash(&mut s);
                let seed = s.finish();
                // Simple LCG
                let mut rng = seed;
                for _ in 0..n_flip {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let idx = (rng >> 33) as usize % n_coords.max(1);
                    codes[idx] = codes[idx].wrapping_add(1);
                }

                // Reconstruct
                let reconstructed = crate::quantization::admission::reconstruct_candidate(
                    format,
                    &codes,
                    &scales,
                    &biases,
                    in_features,
                    out_features,
                    scale_vector.as_deref(),
                );

                let new_loss = crate::quantization::admission::compute_weight_nrmse(
                    &source, &reconstructed,
                );

                // Activation-parity check: teacher = source, student = reconstructed
                let rel_tol = 0.05;
                let acceptance = crate::compilation::distill_core::block_accept(
                    &source, &reconstructed, rel_tol,
                );

                (new_loss, acceptance)
            },
            &ref_config,
        );

        // If the final result improved, update the entity's codes
        if result.final_loss < initial_loss {
            if let Some(c) = world.get_mut::<CodesData>(entity) {
                c.codes = codes;
                c.scales = scales;
                c.biases = biases;
                c.scale_vector = scale_vector;
            }
        }

        // Store refinement outcome regardless
        world.insert(entity, RefinementOutcome(result));
    }
}

// ===========================================================================
// System 5: Seal cimage
// ===========================================================================

/// Seal the compiled tensors into a complete cimage binary.
///
/// Collects all entities in `Bound` phase, reads their `TensorBinding`,
/// `CodesData`, and `TensorShape` components, and calls `build_cimage` to
/// produce the sealed binary.  The result is stored as a `SealedCimage`
/// resource on the World.
pub fn seal_cimage(world: &mut World, model_config: ModelConfig) {
    use crate::compute_image::compile::ternary::{build_cimage, CompiledTensor};

    let entities: Vec<Entity> = world.iter_entities_with::<TensorBinding>().collect();
    let mut tensors = Vec::with_capacity(entities.len());

    for entity in &entities {
        let _status = match world.get::<CompilationStatus>(*entity) {
            Some(s) if s.phase == CompilationPhase::Bound => s,
            _ => continue,
        };
        let binding = match world.get::<TensorBinding>(*entity) {
            Some(b) => b.0.clone(),
            None => continue,
        };
        let codes_data = match world.get::<CodesData>(*entity) {
            Some(c) => c,
            None => continue,
        };

        let codes_bytes: Vec<u8> = codes_data.codes.clone();
        // Metadata is stored as serialized f32 scale values
        let meta_bytes: Vec<u8> = codes_data.scales.iter()
            .flat_map(|s| s.to_le_bytes())
            .chain(codes_data.biases.iter()
                .flat_map(|b| b.to_le_bytes()))
            .collect();

        tensors.push(CompiledTensor {
            binding,
            codes: codes_bytes,
            metadata: meta_bytes,
        });
    }

    if tensors.is_empty() {
        return;
    }

    match build_cimage(tensors, model_config) {
        Ok(cimage) => {
            world.insert_resource(SealedCimage(cimage));
            for entity in &entities {
                if let Some(s) = world.get_mut::<CompilationStatus>(*entity) {
                    if s.phase == CompilationPhase::Bound {
                        s.phase = CompilationPhase::Sealed;
                    }
                }
            }
        }
        Err(e) => {
            // Store error — caller can check via has_resource::<SealedCimageError>
            world.insert_resource(SealedCimageError(e));
        }
    }
}

/// Resource: the sealed cimage binary data (present on success).
#[derive(Debug, Clone)]
pub struct SealedCimage(pub Vec<u8>);

/// Resource: the sealing error message (present on failure).
#[derive(Debug, Clone)]
pub struct SealedCimageError(pub String);

/// Resource: per-stage quantization config injected by the orchestrator.
/// When absent, systems fall back to hardcoded default thresholds.
#[derive(Debug, Clone)]
pub struct StageConfigResource(pub StageQuantizationConfig);

/// Resource: the sealed cimage for one stage of a multi-stage compilation.
/// Produced by `compile_stage` and consumed by `compile_model`.
#[derive(Debug, Clone)]
pub struct StageCimageResource {
    pub stage_id: u32,
    pub cimage: Vec<u8>,
}

// ===========================================================================
// Orchestrator
// ===========================================================================

/// Input tensor descriptor for a compilation run.
#[derive(Debug, Clone)]
pub struct TensorInput {
    pub matrix_id: u32,
    pub weights: Vec<f32>,
    pub shape: CanonicalShape,
}

/// Output binding from the ECS compilation pipeline.
#[derive(Debug, Clone)]
pub struct CompiledBinding {
    pub matrix_id: u32,
    pub status: CompilationPhase,
    pub binding: Option<MatrixWeightBindingV1>,
    pub format: Option<RuntimeRepresentationClass>,
    pub errors: Vec<String>,
}

/// Run the full ECS compilation pipeline on a set of tensors.
pub fn compile_tensors(
    tensors: Vec<TensorInput>,
    registry: CapabilityRegistry,
) -> Vec<CompiledBinding> {
    let mut world = World::new();
    world.insert_resource(registry);
    let mut entity_for_input: Vec<(Entity, usize)> = Vec::with_capacity(tensors.len());
    for (i, tensor) in tensors.iter().enumerate() {
        let entity = world.spawn().unwrap();
        world.insert(entity, SourceWeights(tensor.weights.clone()));
        world.insert(entity, TensorShape(tensor.shape));
        world.insert(entity, CompilationStatus::new());
        entity_for_input.push((entity, i));
    }
    validate_sources(&mut world);
    admit_candidates(&mut world);
    bind_tensors(&mut world);
    let mut results: Vec<CompiledBinding> = Vec::with_capacity(tensors.len());
    results.resize_with(tensors.len(), || CompiledBinding {
        matrix_id: 0,
        status: CompilationPhase::Pending,
        binding: None,
        format: None,
        errors: vec![],
    });
    for (entity, idx) in entity_for_input {
        let status = world
            .get::<CompilationStatus>(entity)
            .cloned()
            .unwrap_or(CompilationStatus::new());
        let binding = world.get::<TensorBinding>(entity).map(|b| b.0.clone());
        let format = status.format;
        let errors = match &status.error {
            Some(e) => vec![e.clone()],
            None => vec![],
        };
        results[idx].matrix_id = tensors[idx].matrix_id;
        results[idx].status = status.phase;
        results[idx].binding = binding;
        results[idx].format = format;
        results[idx].errors = errors;
    }
    results
}



/// Compile one stage of a model into a sealed cimage.
///
/// Creates an ECS World, loads stage tensors as entities, inserts the
/// stage's quantization config, runs the full admission/binding/sealing
/// pipeline, and returns the sealed cimage wrapped in a
/// `StageCimageResource` along with per-tensor bindings.
pub fn compile_stage(
    tensors: Vec<TensorInput>,
    stage_config: StageConfig,
    model_config: ModelConfig,
    registry: CapabilityRegistry,
) -> (StageCimageResource, Vec<CompiledBinding>) {
    let mut world = World::new();
    world.insert_resource(registry);
    world.insert_resource(StageConfigResource(stage_config.quantization.clone()));

    let stage_id = stage_config.stage_id;
    let mut entity_for_input: Vec<(Entity, usize)> = Vec::with_capacity(tensors.len());
    for (i, tensor) in tensors.iter().enumerate() {
        let entity = world.spawn().unwrap();
        world.insert(entity, SourceWeights(tensor.weights.clone()));
        world.insert(entity, TensorShape(tensor.shape));
        world.insert(entity, CompilationStatus::new());
        entity_for_input.push((entity, i));
    }

    // Run the ECS compilation pipeline
    validate_sources(&mut world);
    admit_candidates(&mut world);
    bind_tensors(&mut world);
    refine_tensors(&mut world);
    seal_cimage(&mut world, model_config);

    // Collect per-tensor results
    let mut results: Vec<CompiledBinding> = Vec::with_capacity(tensors.len());
    results.resize_with(tensors.len(), || CompiledBinding {
        matrix_id: 0,
        status: CompilationPhase::Pending,
        binding: None,
        format: None,
        errors: vec![],
    });
    for (entity, idx) in entity_for_input {
        let status = world
            .get::<CompilationStatus>(entity)
            .cloned()
            .unwrap_or(CompilationStatus::new());
        let binding = world.get::<TensorBinding>(entity).map(|b| b.0.clone());
        let format = status.format;
        let errors = match &status.error {
            Some(e) => vec![e.clone()],
            None => vec![],
        };
        results[idx].matrix_id = tensors[idx].matrix_id;
        results[idx].status = status.phase;
        results[idx].binding = binding;
        results[idx].format = format;
        results[idx].errors = errors;
    }

    // Extract the sealed cimage
    let cimage = world
        .get_resource::<SealedCimage>()
        .map(|s| s.0.clone())
        .unwrap_or_default();

    (StageCimageResource { stage_id, cimage }, results)
}



/// Compile all stages of a model into a vector of sealed per-stage cimages.
///
/// Takes a `StageGraph` describing the model decomposition and a loader
/// function that returns `(tensors, model_config)` for a given stage.
/// Runs the full ECS compilation pipeline per stage and returns the
/// resulting sealed cimages.
///
/// # Type Parameters
///
/// - `F`: a loader closure `(&StageConfig) -> (Vec<TensorInput>, ModelConfig,
///        CapabilityRegistry)` that produces the input tensors, model config,
///        and capability registry for the given stage.
pub fn compile_model<F>(
    graph: &StageGraph,
    stage_loader: F,
) -> Vec<StageCimageResource>
where
    F: Fn(&StageConfig) -> (Vec<TensorInput>, ModelConfig, CapabilityRegistry),
{
    let mut outputs = Vec::with_capacity(graph.stages.len());
    for stage in &graph.stages {
        let (tensors, model_config, registry) = stage_loader(stage);
        let (stage_result, _bindings) = compile_stage(
            tensors,
            stage.clone(),
            model_config,
            registry,
        );
        outputs.push(stage_result);
    }
    outputs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_single_identity_tensor() {
        let tensor = TensorInput {
            matrix_id: 1,
            weights: vec![1.0, 0.0, 0.0, 1.0],
            shape: CanonicalShape { in_features: 2, out_features: 2, rank: 2 },
        };
        let registry = CapabilityRegistry::default_metal_v1();
        let results = compile_tensors(vec![tensor], registry);
        assert_eq!(results.len(), 1);
        // At minimum: should have a status and matrix_id
        assert_eq!(results[0].matrix_id, 1);
    }

    #[test]
    fn seal_e2e_rawf32() {
        // Full E2E: validate → admit → bind → seal for a small RawF32 matrix.
        // Verifies the SealedCimage resource is present and non-empty.
        let mut world = World::new();
        world.insert_resource(CapabilityRegistry::default_metal_v1());

        let entity = world.spawn().unwrap();
        world.insert(entity, SourceWeights(vec![127.0; 4]));
        world.insert(entity, TensorShape(CanonicalShape {
            in_features: 4,
            out_features: 1,
            rank: 2,
        }));
        world.insert(entity, CompilationStatus::new());

        validate_sources(&mut world);
        admit_candidates(&mut world);
        bind_tensors(&mut world);

        let model_config = ModelConfig {
            num_layers: 1,
            num_heads: 8,
            head_dim: 128,
            hidden_dim: 4096,
            intermediate_dim: 16384,
            vocab_size: 32000,
            quantization_schema: 1,
            draft_num_layers: 0,
        };
        seal_cimage(&mut world, model_config);

        let cimage = world.get_resource::<SealedCimage>()
            .expect("SealedCimage resource should be present")
            .clone();
        assert!(!cimage.0.is_empty(), "cimage should have content");
        assert!(cimage.0.len() > 256, "cimage should exceed header size");
    }
    #[test]
    fn compile_empty_weights_fails() {
        let tensor = TensorInput {
            matrix_id: 2,
            weights: vec![],
            shape: CanonicalShape { in_features: 0, out_features: 0, rank: 2 },
        };
        let registry = CapabilityRegistry::default_metal_v1();
        let results = compile_tensors(vec![tensor], registry);
        assert_eq!(results[0].status, CompilationPhase::Failed);
    }
}

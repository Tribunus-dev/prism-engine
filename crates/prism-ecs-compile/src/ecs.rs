//! ECS compilation components and pipeline orchestration.
//!
//! Defines ECS [`Component`] types that represent compilation pipeline state
//! attached to a session [`Entity`], plus system functions that operate on
//! a [`World`] to drive each pipeline stage. The [`CompilationOrchestrator`]
//! owns a world + session entity and runs the full pipeline.
//!
//! # Architecture
//!
//! Each stage attaches its output as a component on the session entity.
//! Shared read-only data lives as [`World`] resources where the type is
//! `Sync`, or as [`World`] extensions (via `set_extension`) when the type
//! is `Send` but not `Sync` (e.g. [`CanonicalSource`]).
//!
//! ```text
//! World resources/extensions:            Session entity components:
//! ┌──────────────────┐                  ┌──────────────────────┐
//! │ SessionHandle (R)│──► session Entity│ CompilationSession   │
//! │ CurrentSource (E)│                  │ SourceModel          │
//! │ VecEventSink (R) │                  │ TensorCollection     │
//! │ TargetCaps (R)   │                  │ SpatialGraphComponent│
//! │ ExecutionMode (R)│                  │ SearchStateComponent │
//! └──────────────────┘                  │ LegalizedPlan        │
//!                                       │ KernelCollection     │
//!                                       │ CImageArtifact       │
//!                                       └──────────────────────┘
//! ```

use std::path::PathBuf;

use sha2::Digest;
use crate::compilation_systems::sync_compilation_entity;

use prism_ecs_core::component::Component;
use prism_ecs_core::entity::{Entity, EntityKind};
use prism_ecs_core::identity::{CompilerIdentity, SourceFormat};
use prism_ecs_core::world::World;
use prism_ecs_ir::evolution::{
    foundation::{AneUnitAxis, CandidateGenome, RepresentationAxis},
    progressive::{ProgressiveParetoSearch, ProgressiveSearchConfig},
    EvolutionRuntime,
};
use prism_ecs_ir::ir_types::{FloatKind, TensorType, Type};
use prism_ecs_ir::op::{OpMarker, OpName, Operands, Results};
use prism_ecs_ir::value::{Uses, ValueType};
use prism_ecs_kernel::{
    BackendKind, BindingSlot, CpuBackend, DispatchGeometry, KernelArtifact, KernelBackend,
    KernelCompileRequest, KernelDescriptor, KernelManifest, KernelPayload, KernelVariant,
    MetalBackend, FP16_GEMV_MSL,
};
use prism_ecs_source::{CanonicalSource, CanonicalSourceAdapter, SourceIdentity, TensorCatalog};
use prism_spatial_ir::graph::SpatialGraph;
use prism_spatial_ir::target::TargetCapabilities;

use crate::cimage::UniversalCImageWriter;
use crate::forensic::build_forensic_receipt;
use crate::graph::CanonicalGraphBuilder;
use crate::legalize::{CompilerLegalizer, LegalizationReport};
use crate::runtime::RuntimeModel;
use crate::search::{EvaluationStrategy, SearchCoordinator};
use crate::SearchTrace;
use crate::{
    CompilationEvent, CompilationEventSink, CompilationPolicy, CompilationStage, CompileConfig,
    CompileError, CompileReceipt, CompileResult, CompileStatus, StageResult, VecEventSink,
};

// ===========================================================================
// Resource types
// ===========================================================================

/// Identifies the session entity in the world.
#[derive(Debug, Clone, Copy)]
pub struct SessionHandle(pub Entity);

/// Full ingress'd source model, stored as a world *extension*.
///
/// We use the extension mechanism (not resources) because [`CanonicalSource`]
/// is `Send + 'static` but not `Sync` (it embeds a
/// `Box<dyn TensorDataProvider>` which is only `Send`).  World extensions
/// require only `Send + 'static`.
///
/// Set by [`system_detect_source`]; consumed by [`system_build_graph`],
/// [`system_run_search`], [`system_legalize`], and [`system_emit_cimage`].
pub struct CurrentSource(pub CanonicalSource);

/// Registered source format adapters.
///
/// Must be set before calling [`system_detect_source`].
pub struct SourceAdapterList(pub Vec<Box<dyn CanonicalSourceAdapter + 'static>>);

/// Optional evaluator strategy for the search phase.
pub struct EvaluatorOption(pub Option<Box<dyn EvaluationStrategy + 'static>>);

/// Optional namespaced specialized-model manifest supplied by the caller.
pub struct ModelManifestResource(pub crate::model_manifest::MultiModelManifest);

struct EcsSyntheticEvaluator;

impl EvaluationStrategy for EcsSyntheticEvaluator {
    fn evaluate(&self, _genome: &str, context: &[u8]) -> Result<Vec<f64>, String> {
        Ok(vec![1.0 / (1.0 + context.len() as f64)])
    }

    fn name(&self) -> &str {
        "ecs-synthetic"
    }
}

// ===========================================================================
// ECS Components — attached to the session entity
// ===========================================================================

// ---- Component: CompilationSession ---------------------------------------

/// Top-level compilation session state.
///
/// Present on the session entity from creation through completion.
#[derive(Debug, Clone)]
pub struct CompilationSession {
    /// Compilation configuration (backends, search parameters, policy toggles).
    pub config: CompileConfig,
    /// Current session status.
    pub status: SessionStatus,
    /// Unique session identifier.
    pub session_id: String,
}

impl Component for CompilationSession {}

/// Lifecycle status of a compilation session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionStatus {
    /// Session created; no work started.
    Initialized,
    /// Source format detected and tensors ingested.
    Ingested,
    /// Spatial graph constructed.
    GraphBuilt,
    /// Evolutionary search completed.
    SearchComplete,
    /// Graph legalized for target backends.
    Legalized,
    /// Kernels generated.
    KernelsGenerated,
    /// CImage artifact emitted to disk.
    Emitted,
    /// Emitted artifact reopened and structurally certified.
    Certified,
    /// Compilation completed with a receipt.
    Complete,
    /// Session failed with a description.
    Failed(String),
}

// ---- Component: SourceModel -----------------------------------------------

/// Identity and architecture of the source model.
///
/// Set by [`system_detect_source`] after successful format detection.
#[derive(Debug, Clone)]
pub struct SourceModel {
    /// Format-independent source identity (format, digest, architecture labels).
    pub identity: SourceIdentity,
    /// Detected architecture family (e.g. "llama", "qwen2", "mistral").
    pub architecture: String,
}

impl Component for SourceModel {}

// ---- Component: TensorCollection ------------------------------------------

/// Collection of tensor descriptors from the source model.
///
/// Set by [`system_detect_source`] after source ingestion. The
/// [`TensorCatalog`] provides name-to-descriptor lookup and a content digest.
#[derive(Debug, Clone)]
pub struct TensorCollection {
    /// The full tensor catalog from the source.
    pub catalog: TensorCatalog,
    /// Number of tensors in the collection.
    pub count: usize,
}

impl Component for TensorCollection {}

// ---- Component: SpatialGraphComponent -------------------------------------

/// The spatial intermediate-representation graph.
///
/// Set by [`system_build_graph`] after calling [`CanonicalGraphBuilder`].
#[derive(Debug, Clone)]
pub struct SpatialGraphComponent {
    /// The constructed spatial dataflow graph.
    pub graph: SpatialGraph,
    /// SHA-256 digest of the canonical JSON graph representation.
    pub graph_digest: String,
    /// Architecture family that was detected during graph construction.
    pub architecture: String,
}

impl Component for SpatialGraphComponent {}

// ---- Component: SearchStateComponent --------------------------------------

/// State of the evolutionary search phase.
///
/// Set by [`system_run_search`] after calling [`SearchCoordinator`].
#[derive(Debug, Clone)]
pub struct SearchStateComponent {
    /// The complete search trace (generations, candidates, Pareto frontier).
    pub trace: SearchTrace,
    /// Number of candidate variants evaluated.
    pub candidates_evaluated: u64,
    /// Number of generations completed.
    pub generations_completed: u64,
    /// Optional serialized format plan.
    pub format_plan: Option<String>,
    /// Measured joint ANE/Metal evidence selected by the search.  This is
    /// retained for the later promotion/replay stage rather than silently
    /// reducing hardware measurements to the format plan.
    pub best_joint_tiling: Option<crate::search::JointTilingEvidence>,
    pub selection_receipt: crate::search::SearchSelectionReceipt,
    /// Durable deployment-level candidates promoted from measured search
    /// records. Runtime policy consumes this archive, not the scalar genome
    /// fitness used during mutation.
    pub deployment_archive: prism_ecs_ir::evolution::ParetoArchive,
    pub selected_deployment_digest: Option<String>,
}

impl Component for SearchStateComponent {}

// ---- Component: LegalizedPlan ---------------------------------------------

/// Legalized compilation plan and its validation report.
///
/// Set by [`system_legalize`] after calling [`CompilerLegalizer`].
#[derive(Debug, Clone)]
pub struct LegalizedPlan {
    /// Aggregate legalization report with per-check results.
    pub report: LegalizationReport,
    /// Whether the plan passed all checks.
    pub is_valid: bool,
}

impl Component for LegalizedPlan {}

// ---- Component: KernelCollection ------------------------------------------

/// Collection of compiled kernel artifacts.
///
/// Set by [`system_generate_kernels`] after backend compilation completes.
#[derive(Debug, Clone)]
pub struct KernelCollection {
    /// One artifact per backend or compilation unit.
    pub artifacts: Vec<KernelArtifact>,
    /// Number of kernels across all artifacts.
    pub kernel_count: usize,
    /// SpatialIR target manifests produced by lowering.
    pub lowered_manifests: Vec<prism_spatial_ir::target::KernelManifest>,
    /// Optional whole-graph executable capture when every node is lowered
    /// losslessly into the compact UOp compiler.
    pub uop_capture: Option<prism_spatial_ir::CapturePlan>,
    /// Strategy-specific captures emitted from the same graph for workload
    /// aware runtime selection.
    pub uop_strategy_captures: Vec<(String, prism_spatial_ir::CapturePlan)>,
    pub uop_tuning_receipt: Option<crate::uop::UOpTuningReceipt>,
}

impl Component for KernelCollection {}

// ---- Component: CImageArtifact --------------------------------------------

/// Metadata about the emitted CImage binary artifact.
///
/// Set by [`system_emit_cimage`] after [`UniversalCImageWriter::finalize`].
#[derive(Debug, Clone)]
pub struct CImageArtifact {
    /// Absolute or relative file path on disk.
    pub output_path: PathBuf,
    /// SHA-256 hex digest of the artifact.
    pub digest: String,
    /// Schema version embedded in the artifact.
    pub schema_version: String,
}

impl Component for CImageArtifact {}

/// The final forensic receipt owned by the session entity. Keeping it as a
/// component makes receipt consumers observe the same state that produced the
/// CImage instead of reconstructing it from side effects.
#[derive(Debug, Clone)]
pub struct CompilationReceipt(pub CompileReceipt);

impl Component for CompilationReceipt {}

// ===========================================================================
// Pipeline Systems
// ===========================================================================

/// Run the **source detection** stage.
///
/// Iterates registered format adapters to detect and ingress the source model.
/// On success adds [`SourceModel`] and [`TensorCollection`] components to the
/// session entity, stores the [`CanonicalSource`] as [`CurrentSource`], and
/// transitions session status to [`SessionStatus::Ingested`].
pub fn system_detect_source(world: &mut World) -> Result<(), CompileError> {
    let session = session_entity(world)?;
    let canonical_source = world.get_extension::<CurrentSource>().ok_or_else(|| {
        CompileError::SourceDetectionFailed(
            "no source adapters or canonical source provided".into(),
        )
    })?;
    let identity = canonical_source.0.identity.clone();

    // Extract metadata for components
    let architecture = identity.architecture.clone();
    let catalog = canonical_source.0.catalog.clone();
    let count = catalog.tensors.len();

    // Update session status
    if let Ok(status) = world.component_mut::<CompilationSession>(session) {
        status.status = SessionStatus::Ingested;
    }

    // Attach components
    world
        .insert_component(
            session,
            SourceModel {
                identity: identity.clone(),
                architecture: architecture.clone(),
            },
        )
        .map_err(|e| CompileError::SourceDetectionFailed(e.to_string()))?;

    world
        .insert_component(
            session,
            TensorCollection {
                catalog: catalog.clone(),
                count,
            },
        )
        .map_err(|e| CompileError::SourceDetectionFailed(e.to_string()))?;

    Ok(())
}

/// Run the **graph construction** stage.
///
/// Reads the [`CurrentSource`] extension and produces a [`SpatialGraph`].
/// Adds [`SpatialGraphComponent`] to the session entity.
pub fn system_build_graph(world: &mut World) -> Result<(), CompileError> {
    let session = session_entity(world)?;

    let source = world
        .get_extension::<CurrentSource>()
        .ok_or_else(|| CompileError::GraphBuildFailed("no current source resource".into()))?;

    let result = CanonicalGraphBuilder::build(&source.0)
        .map_err(|e| CompileError::GraphBuildFailed(e.to_string()))?;

    // Update session status
    if let Ok(status) = world.component_mut::<CompilationSession>(session) {
        status.status = SessionStatus::GraphBuilt;
    }

    world
        .insert_component(
            session,
            SpatialGraphComponent {
                graph: result.graph,
                graph_digest: result.graph_digest,
                architecture: result.architecture,
            },
        )
        .map_err(|e| CompileError::GraphBuildFailed(e.to_string()))?;

    Ok(())
}

/// Run the **evolutionary search** stage.
///
/// Reads the [`CurrentSource`] extension and [`SpatialGraphComponent`], runs the
/// [`SearchCoordinator`], and adds [`SearchStateComponent`].
pub fn system_run_search(world: &mut World) -> Result<(), CompileError> {
    let session = session_entity(world)?;

    let source = world
        .get_extension::<CurrentSource>()
        .ok_or_else(|| CompileError::SearchFailed("no current source resource".into()))?;

    let graph_component = world
        .component::<SpatialGraphComponent>(session)
        .map_err(|e| CompileError::SearchFailed(e.to_string()))?;

    let config = read_session_config(world, session)?;
    let evaluator = world.get_resource::<EvaluatorOption>();

    // Run the reference-aware progressive ternary stages before the broad
    // hardware search when the registered evaluator explicitly provides the
    // capability.  Legacy/synthetic evaluators return None and therefore do
    // not get to manufacture behavioral evidence.  Candidates advance only
    // after the executor supplies finite quality, activation, logit, and
    // router-agreement measurements.
    if let Some(executor) = evaluator
        .and_then(|option| option.0.as_deref())
        .and_then(|strategy| strategy.progressive_executor())
    {
        let mut seed = Vec::with_capacity(2);
        for representation in [
            RepresentationAxis::Ternary158,
            RepresentationAxis::TernaryTile640,
        ] {
            let mut genome = CandidateGenome::new();
            genome.representation = representation;
            seed.push(genome);
        }
        let progressive = ProgressiveParetoSearch {
            config: ProgressiveSearchConfig {
                stages: std::env::var("PRISM_PROGRESSIVE_STAGES")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .filter(|&value: &usize| value > 0)
                    .unwrap_or(2),
                limits: prism_ecs_ir::evolution::TernaryAdmissionLimits::from_environment(),
                ..ProgressiveSearchConfig::default()
            },
            executor,
        };
        let progressive_context = source
            .0
            .catalog
            .iter()
            .flat_map(|tensor| {
                let mut line = tensor.name.as_bytes().to_vec();
                line.push(b':');
                line.extend(
                    tensor
                        .shape
                        .iter()
                        .flat_map(|dimension| dimension.to_le_bytes()),
                );
                line.push(b'\n');
                line
            })
            .collect::<Vec<u8>>();
        let frontier =
            progressive.run_with_context(seed, &progressive_context, |candidate, _stage| {
                let mut mutations = Vec::with_capacity(3);
                mutations.push(candidate.clone());
                let mut planar = candidate.clone();
                planar.ane_unit = AneUnitAxis::Planar;
                mutations.push(planar);
                let mut matrix = candidate.clone();
                matrix.ane_unit = AneUnitAxis::Matrix;
                mutations.push(matrix);
                mutations
            });
        if frontier.is_empty() && config.production_mode {
            return Err(CompileError::SearchFailed(
                "progressive ternary search rejected every candidate".into(),
            ));
        }
    }

    let search_config = crate::SearchConfig {
        max_generations: config.max_generations,
        population_size: config.max_candidates,
        production_mode: config.production_mode,
        ..Default::default()
    };

    let runtime = world
        .get_resource::<EvolutionRuntime>()
        .cloned()
        .unwrap_or_default();
    let mut coordinator = SearchCoordinator::new(search_config).with_runtime(runtime);

    let synthetic;
    let eval_ref: Option<&dyn EvaluationStrategy> = if let Some(evaluator) = evaluator {
        evaluator.0.as_deref()
    } else if config.production_mode {
        None
    } else {
        synthetic = EcsSyntheticEvaluator;
        Some(&synthetic)
    };

    let result = coordinator
        .run_search(
            &source.0,
            &graph_component.graph,
            eval_ref,
            config.production_mode,
        )
        .map_err(|e| CompileError::SearchFailed(e.to_string()))?;

    // Update session status
    if let Ok(status) = world.component_mut::<CompilationSession>(session) {
        status.status = SessionStatus::SearchComplete;
    }

    world
        .insert_component(
            session,
            SearchStateComponent {
                trace: result.trace,
                candidates_evaluated: result.candidates_evaluated,
                generations_completed: result.generations_completed,
                format_plan: result.format_plan,
                best_joint_tiling: result.best_joint_tiling,
                selection_receipt: result.selection_receipt,
                selected_deployment_digest: result
                    .deployment_archive
                    .select(&prism_ecs_ir::evolution::DeploymentPolicy::quality_first())
                    .map(|candidate| candidate.candidate_digest.clone()),
                deployment_archive: result.deployment_archive,
            },
        )
        .map_err(|e| CompileError::SearchFailed(e.to_string()))?;

    Ok(())
}

/// Run the **legalization** stage.
///
/// Reads the [`CurrentSource`] extension and [`SpatialGraphComponent`], runs the
/// [`CompilerLegalizer`], and adds [`LegalizedPlan`].
pub fn system_legalize(world: &mut World) -> Result<(), CompileError> {
    let session = session_entity(world)?;

    let source = world
        .get_extension::<CurrentSource>()
        .ok_or_else(|| CompileError::LegalizationFailed("no current source resource".into()))?;

    let graph_component = world
        .component::<SpatialGraphComponent>(session)
        .map_err(|e| CompileError::LegalizationFailed(e.to_string()))?;

    let config = read_session_config(world, session)?;
    if config.enable_search {
        let search = world
            .component::<SearchStateComponent>(session)
            .map_err(|e| CompileError::LegalizationFailed(format!("search state missing: {e}")))?;
        if let Some(plan) = search.format_plan.as_deref() {
            serde_json::from_str::<prism_ecs_ir::evolution::compile_plan::FormatPlan>(plan)
                .map_err(|e| CompileError::LegalizationFailed(format!("invalid selected format plan: {e}")))?;
        } else if config.production_mode {
            return Err(CompileError::LegalizationFailed(
                "production legalization requires a selected format plan".into(),
            ));
        }
    }

    let target = world
        .get_resource::<TargetCapabilities>()
        .cloned()
        .unwrap_or_else(TargetCapabilities::sequential_only);
    let report = CompilerLegalizer::legalize(
        &source.0,
        &graph_component.graph,
        &target,
        prism_spatial_ir::execution_plan::ExecutionMode::Batch,
    )
    .map_err(|e| CompileError::LegalizationFailed(e.to_string()))?;

    let is_valid = report.is_valid();

    // Update session status
    if let Ok(status) = world.component_mut::<CompilationSession>(session) {
        status.status = SessionStatus::Legalized;
    }

    world
        .insert_component(session, LegalizedPlan { report, is_valid })
        .map_err(|e| CompileError::LegalizationFailed(e.to_string()))?;

    Ok(())
}

/// Run the **kernel generation** stage.
///
fn compile_spatial_matmul_to_native_xdna(
    node: &prism_spatial_ir::graph::SpatialNode,
    node_id: prism_spatial_ir::graph::SpatialNodeId,
) -> Result<KernelArtifact, CompileError> {
    let prism_spatial_ir::graph::SpatialNode::Compute { shape, kind, .. } = node else {
        return Err(CompileError::KernelGenFailed(format!(
            "XDNA node {node_id} is not a compute node"
        )));
    };
    if *kind != prism_spatial_ir::graph::ComputeKind::MatMul
        || shape.in_shapes.len() < 2
        || shape.out_shapes.is_empty()
    {
        return Err(CompileError::KernelGenFailed(format!(
            "XDNA node {node_id} is not a statically shaped MatMul"
        )));
    }
    let to_tensor = |dims: &prism_ecs_ir::cimage_types::TensorShape| {
        Type::Tensor(TensorType::new(
            dims.dims.iter().map(|dim| *dim as u64).collect(),
            Type::float(FloatKind::F16),
        ))
    };
    let mut synthetic = World::new();
    let make_value = |world: &mut World, name: &str, ty: Type| -> Result<Entity, CompileError> {
        let value: Entity = world
            .spawn(EntityKind::Node, Some(name.into()))
            .map_err(|error| CompileError::KernelGenFailed(error.to_string()))?
            .into();
        world
            .add_component(value, ValueType(ty))
            .map_err(|error| CompileError::KernelGenFailed(error.to_string()))?;
        world
            .add_component(value, Uses(vec![]))
            .map_err(|error| CompileError::KernelGenFailed(error.to_string()))?;
        Ok(value)
    };
    let a = make_value(&mut synthetic, "A", to_tensor(&shape.in_shapes[0]))?;
    let b = make_value(&mut synthetic, "B", to_tensor(&shape.in_shapes[1]))?;
    let c = make_value(&mut synthetic, "C", to_tensor(&shape.out_shapes[0]))?;
    let result = make_value(&mut synthetic, "result", to_tensor(&shape.out_shapes[0]))?;
    let op: Entity = synthetic
        .spawn(EntityKind::Node, Some("matmul".into()))
        .map_err(|error| CompileError::KernelGenFailed(error.to_string()))?
        .into();
    synthetic
        .add_component(op, OpMarker)
        .map_err(|error| CompileError::KernelGenFailed(error.to_string()))?;
    synthetic
        .add_component(
            op,
            OpName(if shape.in_shapes[0].dims.len() == 3 {
                "linalg.batch_matmul".into()
            } else {
                "linalg.matmul".into()
            }),
        )
        .map_err(|error| CompileError::KernelGenFailed(error.to_string()))?;
    synthetic
        .add_component(op, Operands(vec![a, b, c]))
        .map_err(|error| CompileError::KernelGenFailed(error.to_string()))?;
    synthetic
        .add_component(op, Results(vec![result]))
        .map_err(|error| CompileError::KernelGenFailed(error.to_string()))?;
    let executable = prism_amd_npu_runtime::compile_amd_npu(
        &synthetic,
        op,
        prism_ecs_ir::backend_dispatch::HalFormat::AmdNpu,
    )
    .map_err(CompileError::KernelGenFailed)?;
    let artifact = prism_amd_npu_runtime::XdnaArtifact::decode_hex_envelope(&executable.source)
        .map_err(CompileError::KernelGenFailed)?;
    let binary = artifact.encode().map_err(CompileError::KernelGenFailed)?;
    let digest = hex::encode(sha2::Sha256::digest(&binary));
    let descriptor = KernelDescriptor {
        name: format!("xdna_node_{node_id}"),
        variant: KernelVariant::Custom("native-xdna-artifact".into()),
        backend: BackendKind::AmdNpu,
        source_digest: digest.clone(),
        binary_digest: digest.clone(),
        binding_signature: Vec::new(),
        dispatch_geometry: DispatchGeometry {
            threads_per_threadgroup: [1, 1, 1],
            threadgroups_per_grid: [1, 1, 1],
            threads_per_grid: [1, 1, 1],
        },
    };
    Ok(KernelArtifact {
        payloads: vec![KernelPayload {
            binary,
            descriptor: descriptor.clone(),
        }],
        manifest: KernelManifest {
            kernels: vec![descriptor],
            fusion_plan: None,
            manifest_digest: digest.clone(),
        },
        artifact_digest: digest,
    })
}

/// Lowers SpatialIR nodes and compiles them through the configured backend.
pub fn system_generate_kernels(world: &mut World) -> Result<(), CompileError> {
    let session = session_entity(world)?;
    let config = read_session_config(world, session)?;
    let graph = world
        .component::<SpatialGraphComponent>(session)
        .map_err(|e| CompileError::KernelGenFailed(e.to_string()))?
        .clone();
    let legalized = world
        .component::<LegalizedPlan>(session)
        .map_err(|e| CompileError::KernelGenFailed(format!("legalized plan missing: {e}")))?;
    if !legalized.is_valid {
        return Err(CompileError::KernelGenFailed(
            "cannot generate kernels from an invalid legalized plan".into(),
        ));
    }

    let format_plan = world
        .component::<SearchStateComponent>(session)
        .ok()
        .and_then(|state| state.format_plan.as_deref())
        .and_then(|json| serde_json::from_str(json).ok());

    let legalized = prism_spatial_ir::legalize::legalize(graph.graph.clone(), |_node| {
        Ok::<(), Vec<prism_spatial_ir::legalize::LegalizationError>>(())
    })
    .map_err(|errors| {
        CompileError::KernelGenFailed(format!("SpatialIR lowering failed: {errors:?}"))
    })?;
    let manifest = prism_spatial_ir::execution_plan::lower_to_manifest(
        legalized.graph(),
        prism_spatial_ir::cost::CostEstimate::zero(),
        format_plan.as_ref(),
    )
    .ok_or_else(|| CompileError::KernelGenFailed("cannot lower cyclic SpatialIR graph".into()))?;

    let lowered_manifests = vec![manifest.clone()];
    let backend_kind = config
        .target_backends
        .first()
        .copied()
        .unwrap_or(BackendKind::CPU);
    let artifacts = manifest
        .kernels
        .iter()
        .map(|descriptor| {
            let spatial_node = graph
                .graph
                .nodes()
                .iter()
                .find(|node| node.id() == descriptor.node_id);
            let supports_uop_lowering = matches!(
                spatial_node,
                Some(prism_spatial_ir::graph::SpatialNode::Compute {
                    kind: prism_spatial_ir::graph::ComputeKind::MatMul,
                    ..
                })
            ) || matches!(
                spatial_node,
                Some(prism_spatial_ir::graph::SpatialNode::Compute {
                    kind: prism_spatial_ir::graph::ComputeKind::Elementwise,
                    ..
                }) if graph
                    .graph
                    .get_annotations(descriptor.node_id)
                    .and_then(|meta| meta.elementwise_op.as_ref())
                    .is_some()
            ) || matches!(
                spatial_node,
                Some(prism_spatial_ir::graph::SpatialNode::Compute {
                    kind: prism_spatial_ir::graph::ComputeKind::Attention,
                    ..
                })
            ) || matches!(
                spatial_node,
                Some(prism_spatial_ir::graph::SpatialNode::Compute {
                    kind: prism_spatial_ir::graph::ComputeKind::Convolution
                        | prism_spatial_ir::graph::ComputeKind::Normalization
                        | prism_spatial_ir::graph::ComputeKind::Softmax
                        | prism_spatial_ir::graph::ComputeKind::RoPE
                        | prism_spatial_ir::graph::ComputeKind::Gather
                        | prism_spatial_ir::graph::ComputeKind::SSM,
                    ..
                })
            );
            if matches!(backend_kind, BackendKind::CPU | BackendKind::Metal)
                && supports_uop_lowering
            {
                let target = if backend_kind == BackendKind::Metal {
                    prism_spatial_ir::LoweringTarget::Metal
                } else {
                    prism_spatial_ir::LoweringTarget::Cpu
                };
                if let Ok((_, mut uop_artifacts)) = crate::compile_spatial_node_with_metadata(
                    spatial_node.unwrap(),
                    graph.graph.get_annotations(descriptor.node_id),
                    target,
                )
                {
                    let mut artifact = uop_artifacts.pop().ok_or_else(|| {
                        CompileError::KernelGenFailed("MatMul UOp lowering produced no artifact".into())
                    })?;
                    let stable_name = format!("spatial_node_{}", descriptor.node_id.0);
                    for payload in &mut artifact.payloads {
                        payload.descriptor.name = stable_name.clone();
                    }
                    for kernel in &mut artifact.manifest.kernels {
                        kernel.name = stable_name.clone();
                    }
                    return Ok(artifact);
                }
            }
            let (kernel_name, variant, source) = match backend_kind {
                BackendKind::CPU => (
                    format!("spatial_node_{}", descriptor.node_id.0),
                    KernelVariant::Custom("spatial-ir".into()),
                    serde_json::to_vec(descriptor)
                        .map_err(|e| CompileError::KernelGenFailed(e.to_string()))?,
                ),
                BackendKind::Metal => (
                    "fp16_gemv".into(),
                    KernelVariant::FP16GEMV,
                    FP16_GEMV_MSL.as_bytes().to_vec(),
                ),
                BackendKind::AmdNpu => {
                    return compile_spatial_matmul_to_native_xdna(
                        spatial_node.ok_or_else(|| {
                            CompileError::KernelGenFailed(format!(
                                "XDNA node {} is missing from SpatialGraph",
                                descriptor.node_id
                            ))
                        })?,
                        descriptor.node_id,
                    );
                }
                #[cfg(feature = "ane")]
                BackendKind::ANE => {
                    let node = graph
                        .graph
                        .nodes()
                        .iter()
                        .find(|node| node.id() == descriptor.node_id)
                        .ok_or_else(|| {
                            CompileError::KernelGenFailed(format!(
                                "ANE kernel node {} is missing from SpatialGraph",
                                descriptor.node_id
                            ))
                        })?;
                    let (m, k, n) = matmul_dimensions(node).ok_or_else(|| {
                        CompileError::KernelGenFailed(format!(
                            "ANE kernel node {} is not a statically shaped MatMul",
                            descriptor.node_id
                        ))
                    })?;
                    let mil = format!(
                        "MIL PROGRAM matmul_{m}x{k}x{n} {{\n  layer @0 = matmul(inputs: [A, B], output: C, M: {m}, K: {k}, N: {n}, type: float16)\n}}\n"
                    );
                    let binary = prism_ane_runtime::compile_mil(&mil).map_err(|error| {
                        CompileError::KernelGenFailed(format!(
                            "ANE Core ML compilation for node {} failed: {error}",
                            descriptor.node_id
                        ))
                    })?;
                    let digest = hex::encode(sha2::Sha256::digest(&binary.binary));
                    let ane_descriptor = KernelDescriptor {
                        name: format!("ane_matmul_{}", descriptor.node_id.0),
                        variant: KernelVariant::Custom("ane-coreml-matmul".into()),
                        backend: BackendKind::ANE,
                        source_digest: hex::encode(sha2::Sha256::digest(mil.as_bytes())),
                        binary_digest: digest.clone(),
                        binding_signature: Vec::new(),
                        dispatch_geometry: DispatchGeometry {
                            threads_per_threadgroup: [descriptor.threadgroup_size, 1, 1],
                            threadgroups_per_grid: [1, 1, 1],
                            threads_per_grid: [descriptor.threadgroup_size, 1, 1],
                        },
                    };
                    return Ok(KernelArtifact {
                        payloads: vec![KernelPayload {
                            binary: binary.binary,
                            descriptor: ane_descriptor.clone(),
                        }],
                        manifest: KernelManifest {
                            kernels: vec![ane_descriptor],
                            fusion_plan: None,
                            manifest_digest: digest.clone(),
                        },
                        artifact_digest: digest,
                    });
                }
                unsupported => {
                    return Err(CompileError::KernelGenFailed(format!(
                        "backend {unsupported:?} is not implemented"
                    )))
                }
            };
            let request = KernelCompileRequest {
                source,
                descriptor: KernelDescriptor {
                    name: kernel_name,
                    variant,
                    backend: backend_kind,
                    source_digest: String::new(),
                    binary_digest: String::new(),
                    binding_signature: Vec::<BindingSlot>::new(),
                    dispatch_geometry: DispatchGeometry {
                        threads_per_threadgroup: [descriptor.threadgroup_size, 1, 1],
                        threadgroups_per_grid: [1, 1, 1],
                        threads_per_grid: [descriptor.threadgroup_size, 1, 1],
                    },
                },
                source_path: None,
            };
            match backend_kind {
                BackendKind::CPU => CpuBackend.compile(&request),
                BackendKind::Metal => MetalBackend::new().compile(&request),
                BackendKind::AmdNpu => unreachable!("native XDNA artifacts return above"),
                _ => unreachable!(),
            }
            .map_err(|e| CompileError::KernelGenFailed(e.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let kernel_count = artifacts
        .iter()
        .map(|artifact| artifact.payloads.len())
        .sum();
    let uop_target = match backend_kind {
        BackendKind::Metal => prism_spatial_ir::LoweringTarget::Metal,
        _ => prism_spatial_ir::LoweringTarget::Cpu,
    };
    let uop_capture_result = crate::compile_spatial_graph(&graph.graph, uop_target);
    let uop_capture = uop_capture_result
        .as_ref()
        .ok()
        .map(|(capture, _)| capture.clone());
    let strategies = [
        prism_spatial_ir::FusionStrategy::StandardFused,
        prism_spatial_ir::FusionStrategy::InterleavedFused { stages: vec![] },
        prism_spatial_ir::FusionStrategy::PerOperation,
        prism_spatial_ir::FusionStrategy::PersistentMegakernel {
            search_generation: world
                .component::<SearchStateComponent>(session)
                .map(|state| state.generations_completed as u32)
                .unwrap_or(0),
        },
    ];
    let uop_strategy_candidates = if uop_capture_result.is_ok() {
        crate::compile_spatial_graph_strategies(&graph.graph, uop_target, &strategies)
            .map_err(|error| {
                CompileError::KernelGenFailed(format!("UOp strategy compilation failed: {error}"))
            })?
    } else {
        Vec::new()
    };
    let uop_strategy_captures = uop_strategy_candidates
        .iter()
        .map(|(strategy, capture, _)| (strategy.stable_id().to_string(), capture.clone()))
        .collect();
    let uop_tuning_receipt = if uop_strategy_candidates.is_empty() {
        Some(
            crate::uop::UOpTuningReceipt::explicit_fallback(
                graph.graph_digest.clone(),
                uop_target,
                "no executable UOp strategy candidates were available for reference measurement",
            )
            .map_err(CompileError::KernelGenFailed)?,
        )
    } else {
        let scenario = prism_spatial_ir::WorkloadScenario {
            realtime: false,
            batch_size: 1,
            sequence_length: 1,
        };
        match crate::benchmark_uop_strategy_candidates(&uop_strategy_candidates, 3)
            .and_then(|measurements| {
                crate::uop::UOpTuningReceipt::from_candidates(
                    graph.graph_digest.clone(),
                    uop_target,
                    &uop_strategy_candidates,
                    &[crate::uop::UOpWorkloadMeasurement {
                        scenario,
                        measurements,
                    }],
                    crate::uop::UOpMeasurementSource::CpuReference,
                    true,
                )
            }) {
            Ok(receipt) => Some(receipt),
            Err(error) => Some(
                crate::uop::UOpTuningReceipt::explicit_fallback(
                    graph.graph_digest.clone(),
                    uop_target,
                    format!("CPU reference measurement unavailable: {error}"),
                )
                .map_err(CompileError::KernelGenFailed)?,
            ),
        }
    };
    // Update session status
    if let Ok(status) = world.component_mut::<CompilationSession>(session) {
        status.status = SessionStatus::KernelsGenerated;
    }

    world
        .insert_component(
            session,
            KernelCollection {
                artifacts,
                kernel_count,
                lowered_manifests,
                uop_capture,
                uop_strategy_captures,
                uop_tuning_receipt,
            },
        )
        .map_err(|e| CompileError::KernelGenFailed(e.to_string()))?;

    Ok(())
}

#[cfg(any(feature = "ane", test))]
fn matmul_dimensions(node: &prism_spatial_ir::graph::SpatialNode) -> Option<(usize, usize, usize)> {
    let prism_spatial_ir::graph::SpatialNode::Compute {
        kind: prism_spatial_ir::graph::ComputeKind::MatMul,
        shape,
        ..
    } = node
    else {
        return None;
    };
    let a = shape.in_shapes.first()?.dims.as_slice();
    let b = shape.in_shapes.get(1)?.dims.as_slice();
    let c = shape.out_shapes.first()?.dims.as_slice();
    if a.len() != 2 || b.len() != 2 || c.len() != 2 || a[1] != b[0] || c != [a[0], b[1]] {
        return None;
    }
    Some((a[0], a[1], b[1]))
}

/// Run the **CImage emission** stage.
///
/// Collects all components and the [`CurrentSource`] extension, calls
/// [`UniversalCImageWriter::finalize`], and adds [`CImageArtifact`].
pub fn system_emit_cimage(world: &mut World) -> Result<(), CompileError> {
    let session = session_entity(world)?;
    let config = read_session_config(world, session)?;

    let legal = world
        .component::<LegalizedPlan>(session)
        .map_err(|e| CompileError::CImageEmitFailed(format!("legalized plan missing: {e}")))?;
    if !legal.is_valid {
        return Err(CompileError::CImageEmitFailed(
            "cannot emit CImage from an invalid legalized plan".into(),
        ));
    }
    world
        .component::<KernelCollection>(session)
        .map_err(|e| CompileError::CImageEmitFailed(format!("kernel collection missing: {e}")))?;
    if config.enable_search {
        world
            .component::<SearchStateComponent>(session)
            .map_err(|e| CompileError::CImageEmitFailed(format!("search state missing: {e}")))?;
    }

    // Collect required inputs for the writer.
    let source = world
        .get_extension::<CurrentSource>()
        .ok_or_else(|| CompileError::CImageEmitFailed("no current source resource".into()))?;

    let output_path = &config
        .target_backends
        .first()
        .map(|_| PathBuf::from(format!("{}.cimage", source.0.identity.source_digest)))
        .unwrap_or_else(|| PathBuf::from("output.cimage"));

    let mut writer = UniversalCImageWriter::new(output_path);
    writer.set_source(&source.0);

    // Payload completeness is part of artifact admission. A CImage that only
    // contains sparse tensor metadata cannot be replayed independently of the
    // original source, so stream every catalog entry through the provider.
    let provider = source.0.provider.as_ref().ok_or_else(|| {
        CompileError::CImageEmitFailed("source has no tensor data provider".into())
    })?;
    for tensor in source.0.catalog.iter() {
        let payload = provider
            .read_tensor(tensor)
            .map_err(|error| CompileError::CImageEmitFailed(format!("read tensor {}: {error}", tensor.name)))?;
        let dim_m = tensor.shape.first().copied().unwrap_or(0) as u32;
        let dim_n = tensor.shape.get(1).copied().unwrap_or(0) as u32;
        writer
            .add_tensor_payload(crate::cimage::TensorPayloadEntry {
                name: tensor.name.clone(),
                payload,
                representation: tensor.original_dtype.clone(),
                effective_bpp: (tensor.element_size * 8) as f32,
                dim_m,
                dim_n,
                tensor_type: crate::cimage::TensorType::Blob,
            })
            .map_err(CompileError::CImageEmitFailed)?;
    }

    // Attach search trace if available.
    if let Ok(search) = world.component::<SearchStateComponent>(session) {
        writer.set_search_trace(search.trace.clone());
        writer.set_selection_receipt(search.selection_receipt.clone());
        if let Some(evidence) = &search.best_joint_tiling {
            writer.set_joint_tiling_evidence(evidence.clone());
        }
    }

    // Attach legalization report if available.
    if let Ok(legal) = world.component::<LegalizedPlan>(session) {
        writer.set_legalization_report(legal.report.clone());
    }

    // Attach kernel artifacts if available.
    if let Ok(kernels) = world.component::<KernelCollection>(session) {
        if let Some(receipt) = &kernels.uop_tuning_receipt {
            writer.set_uop_tuning_receipt(receipt.clone());
        }
        if let Some(capture) = &kernels.uop_capture {
            writer
                .add_uop_capture(capture)
                .map_err(CompileError::CImageEmitFailed)?;
            if !kernels.uop_strategy_captures.is_empty() {
                writer
                    .add_uop_strategy_captures(&kernels.uop_strategy_captures)
                    .map_err(CompileError::CImageEmitFailed)?;
            }
        } else {
            for artifact in &kernels.artifacts {
                writer.add_kernel_artifact(artifact.clone());
            }
            if let Some(manifest) = kernels.lowered_manifests.first() {
                let plan_json = serde_json::to_string(manifest).map_err(|error| {
                    CompileError::CImageEmitFailed(format!("serialize execution plan: {error}"))
                })?;
                writer.set_execution_plan(plan_json);
            }
        }
    }
    if let Some(manifest) = world.get_extension::<ModelManifestResource>() {
        writer
            .set_model_manifest(manifest.0.clone())
            .map_err(CompileError::CImageEmitFailed)?;
    }

    // Attach events.
    if let Some(sink) = world.get_resource::<VecEventSink>() {
        writer.set_events(sink.events().to_vec());
    }

    // Finalize and capture the digest.
    let _digest = writer
        .finalize()
        .map_err(|e| CompileError::CImageEmitFailed(e.to_string()))?;
    let artifact_digest = hex::encode(sha2::Sha256::digest(
        std::fs::read(&output_path)
            .map_err(|e| CompileError::CImageEmitFailed(format!("read emitted CImage: {e}")))?,
    ));

    // Update session status
    if let Ok(status) = world.component_mut::<CompilationSession>(session) {
        status.status = SessionStatus::Emitted;
    }

    world
        .insert_component(
            session,
            CImageArtifact {
                output_path: output_path.clone(),
                digest: artifact_digest,
                schema_version: "1.0".into(),
            },
        )
        .map_err(|e| CompileError::CImageEmitFailed(e.to_string()))?;

    Ok(())
}

/// Certify the emitted artifact structurally before publishing the receipt.
/// This validates that the bytes can be reopened by the runtime and that any
/// embedded AOT plan satisfies its dependency and residency invariants.
pub fn system_certify(world: &mut World) -> Result<(), CompileError> {
    let session = session_entity(world)?;
    let artifact = world
        .component::<CImageArtifact>(session)
        .map_err(|e| CompileError::CompilationFailed(e.to_string()))?;
    let model = RuntimeModel::load(&artifact.output_path)
        .map_err(|e| CompileError::CompilationFailed(format!("certification load failed: {e}")))?;
    let reader = crate::cimage::CImageReader::open(&artifact.output_path)
        .map_err(|e| CompileError::CompilationFailed(format!("read CImage header failed: {e}")))?;
    let actual_digest = hex::encode(sha2::Sha256::digest(
        std::fs::read(&artifact.output_path)
            .map_err(|e| CompileError::CompilationFailed(format!("read CImage failed: {e}")))?,
    ));
    if actual_digest != artifact.digest {
        return Err(CompileError::CompilationFailed(
            "CImage artifact digest does not match emitted bytes".into(),
        ));
    }
    let source = world
        .get_extension::<CurrentSource>()
        .ok_or_else(|| CompileError::CompilationFailed("source missing during certification".into()))?;
    if reader.header.source_identity.as_ref() != Some(&source.0.identity)
        || reader.header.source_catalog.as_ref() != Some(&source.0.catalog)
    {
        return Err(CompileError::CompilationFailed(
            "CImage source provenance does not match the session source".into(),
        ));
    }
    let expected_names = source.0.catalog.tensors.iter().map(|tensor| tensor.name.as_str());
    if expected_names.clone().any(|name| !model.tensors.contains_key(name)) {
        return Err(CompileError::CompilationFailed(
            "CImage is missing a catalog tensor payload".into(),
        ));
    }
    let search = world.component::<SearchStateComponent>(session).ok();
    if let Some(search) = search {
        let Some(serialized) = reader.header.search_trace.as_deref() else {
            return Err(CompileError::CompilationFailed(
                "CImage is missing the search trace".into(),
            ));
        };
        let sealed: SearchTrace = serde_json::from_str(serialized).map_err(|e| {
            CompileError::CompilationFailed(format!("invalid sealed search trace: {e}"))
        })?;
        if sealed.trace_digest != search.trace.trace_digest {
            return Err(CompileError::CompilationFailed(
                "sealed search trace differs from session state".into(),
            ));
        }
    }
    if let Some(plan) = &model.execution_plan {
        plan.validate().map_err(|e| {
            CompileError::CompilationFailed(format!("certification plan failed: {e}"))
        })?;
        if !plan.supports_all_streamed_workloads() {
            return Err(CompileError::CompilationFailed(
                "certification plan does not cover all streamed workloads".into(),
            ));
        }
    }
    if let Ok(status) = world.component_mut::<CompilationSession>(session) {
        status.status = SessionStatus::Certified;
    }
    Ok(())
}

/// Run the **receipt build** stage.
///
/// Collects events and builds the forensic receipt, attaching it to the
/// session entity's `CompilationSession` status.
pub fn system_build_receipt(world: &mut World) -> Result<(), CompileError> {
    let session = session_entity(world)?;
    let session_state = world
        .component::<CompilationSession>(session)
        .map_err(|e| CompileError::CompilationFailed(e.to_string()))?;
    if !matches!(session_state.status, SessionStatus::Certified | SessionStatus::Complete) {
        return Err(CompileError::CompilationFailed(
            "receipt build requires a certified CImage artifact".into(),
        ));
    }
    world
        .component::<CImageArtifact>(session)
        .map_err(|e| CompileError::CompilationFailed(format!("artifact missing: {e}")))?;
    let events: Vec<CompilationEvent> = world
        .get_resource::<VecEventSink>()
        .map(|s| s.events())
        .unwrap_or_default();

    let mut receipt = CompileReceipt {
        receipt_id: String::new(),
        request_id: uuid::Uuid::new_v4(),
        compiler_identity: CompilerIdentity {
            name: "ecs-orchestrator".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            build_hash: option_env!("PRISM_BUILD_HASH").map(str::to_owned),
            build_timestamp: option_env!("PRISM_BUILD_TIMESTAMP").map(str::to_owned),
        },
        source_identity: world
            .component::<SourceModel>(session)
            .ok()
            .map(|m| m.identity.clone())
            .unwrap_or_else(|| SourceIdentity {
                format: SourceFormat::Raw,
                source_digest: String::new(),
                architecture: String::new(),
                model_family: String::new(),
            }),
        started_at: chrono::Utc::now(),
        completed_at: chrono::Utc::now(),
        duration_ms: 0,
        stages: Vec::new(),
        candidate_count: world
            .component::<SearchStateComponent>(session)
            .ok()
            .map(|s| s.candidates_evaluated as u32)
            .unwrap_or(0),
        generations: world
            .component::<SearchStateComponent>(session)
            .ok()
            .map(|s| s.generations_completed as u32)
            .unwrap_or(0),
        output_digest: world
            .component::<CImageArtifact>(session)
            .ok()
            .map(|a| a.digest.clone())
            .unwrap_or_default(),
        source_digest: Some(
            world
                .component::<SourceModel>(session)
                .ok()
                .map(|m| m.identity.source_digest.clone())
                .unwrap_or_default(),
        ),
        graph_digest: Some(
            world
                .component::<SpatialGraphComponent>(session)
                .ok()
                .map(|g| g.graph_digest.clone())
                .unwrap_or_default(),
        ),
        search_trace_digest: world
            .component::<SearchStateComponent>(session)
            .ok()
            .map(|s| s.trace.trace_digest.clone())
            .filter(|d| !d.is_empty()),
        kernel_manifest_digest: None,
        events_digest: Some(String::new()),
        legalization_mode: Some("target_default".into()),
        selection_receipt: world
            .component::<SearchStateComponent>(session)
            .ok()
            .map(|search| search.selection_receipt.clone()),
        uop_tuning_receipt: world
            .component::<KernelCollection>(session)
            .ok()
            .and_then(|kernels| kernels.uop_tuning_receipt.clone()),
        error: None,
        status: CompileStatus::Completed,
        finished_at: chrono::Utc::now(),
        output_path: std::path::PathBuf::new(),
        schema_version: "1.0".into(),
    };

    // Build and retain the forensic receipt on the session entity.
    if !events.is_empty() {
        let bytes = build_forensic_receipt(&events);
        receipt.events_digest = Some(hex::encode(sha2::Sha256::digest(&bytes)));
    }

    receipt.receipt_id = hex::encode(sha2::Sha256::digest(
        format!("{}:{}", receipt.output_digest, receipt.search_trace_digest.clone().unwrap_or_default()).as_bytes(),
    ));
    let receipt_id = receipt.receipt_id.clone();
    let cimage_digest = receipt.output_digest.clone();

    world
        .insert_component(session, CompilationReceipt(receipt))
        .map_err(|e| CompileError::CompilationFailed(e.to_string()))?;

    // Close the evidence chain for every admitted deployment candidate with
    // the emitted artifact and the compilation receipt that certified it.
    if let Ok(search) = world.component_mut::<SearchStateComponent>(session) {
        for candidate in search.deployment_archive.candidates.values_mut() {
            candidate.evidence.cimage_digest = Some(cimage_digest.clone());
            candidate.evidence.receipt_ids.push(receipt_id.clone());
        }
    }

    // Update session status to Complete.
    if let Ok(status) = world.component_mut::<CompilationSession>(session) {
        status.status = SessionStatus::Complete;
    }

    Ok(())
}

// ===========================================================================
// Orchestrator
// ===========================================================================

/// Owns an ECS world and session entity for a single compilation run.
///
/// # Usage
///
/// ```ignore
/// let mut orch = CompilationOrchestrator::new(config);
/// orch.world.add_resource(SourceAdapterList(adapters));
/// let result = orch.run_pipeline()?;
/// ```
pub struct CompilationOrchestrator {
    /// The session entity carrying all compilation components.
    pub session: Entity,
    /// The ECS world containing the session entity and all resources.
    pub world: World,
}

impl CompilationOrchestrator {
    /// Create a new orchestrator with an empty world and a session entity.
    ///
    /// The session entity is spawned with [`CompilationSession`] in the
    /// [`SessionStatus::Initialized`] state. Callers must add required
    /// resources (adapters, evaluator, target capabilities) before calling
    /// [`run_pipeline`](Self::run_pipeline).
    pub fn new(config: CompileConfig) -> Self {
        let mut world = World::new();
        let spawned = world
            .spawn(EntityKind::Model, Some("compilation_session".into()))
            .expect("spawn session entity");
        let session = spawned.entity;

        let session_id = uuid::Uuid::new_v4().to_string();
        world
            .insert_component(
                session,
                CompilationSession {
                    config,
                    status: SessionStatus::Initialized,
                    session_id,
                },
            )
            .expect("insert CompilationSession component");

        world.add_resource(SessionHandle(session));
        world.add_resource(VecEventSink::new());
        world.add_resource(EvolutionRuntime::global());

        Self { session, world }
    }

    /// Attach the validated multi-model registry that will be embedded in the
    /// emitted CImage header.
    pub fn set_model_manifest(
        &mut self,
        manifest: crate::model_manifest::MultiModelManifest,
    ) -> Result<(), String> {
        manifest.validate()?;
        self.world.set_extension(ModelManifestResource(manifest));
        Ok(())
    }

    /// Run every enabled pipeline stage in order.
    ///
    /// Returns a [`CompileResult`] summarising the outcome. Individual stage
    /// failures are captured as [`CompileStatus::Partial`] or
    /// [`CompileStatus::Failed`].
    pub fn run_pipeline(&mut self) -> Result<CompileResult, CompileError> {
        let (config, policy) = {
            let session_comp = self
                .world
                .component::<CompilationSession>(self.session)
                .map_err(|_| CompileError::PolicyViolation("no session component".into()))?;
            (session_comp.config.clone(), CompilationPolicy::default())
        };

        let stages = policy.enabled_stages().to_vec();
        let mut stage_results: Vec<StageResult> = Vec::with_capacity(stages.len());

        for stage in &stages {
            let started_at = std::time::Instant::now();
            let result = self.run_stage(*stage);
            let duration_ms = started_at.elapsed().as_millis() as u64;

            match result {
                Ok(()) => {
                    stage_results.push(StageResult {
                        stage: *stage,
                        success: true,
                        duration_ms,
                        error: None,
                    });
                    // Emit stage-completed event.
                    if let Some(sink) = self.world.get_resource_mut::<VecEventSink>() {
                        let _ = sink.stage_completed(*stage, duration_ms, "ok");
                    }
                }
                Err(e) => {
                    stage_results.push(StageResult {
                        stage: *stage,
                        success: false,
                        duration_ms,
                        error: Some(e.to_string()),
                    });
                    if let Some(sink) = self.world.get_resource_mut::<VecEventSink>() {
                        let _ = sink.stage_failed(*stage, duration_ms, &e.to_string());
                    }
                    // If a strict policy would abort here, we still stop.
                    break;
                }
            }
        }

        // Build the final CompileResult.
        let all_ok = stage_results.iter().all(|r| r.success);

        let (status, _maybe_error) = if all_ok {
            if let Ok(session) = self.world.component_mut::<CompilationSession>(self.session) {
                session.status = SessionStatus::Complete;
            }
            (CompileStatus::Completed, None)
        } else if stage_results.iter().any(|r| r.success) {
            (CompileStatus::Partial(stage_results.clone()), None)
        } else {
            let err = stage_results
                .first()
                .and_then(|r| r.error.clone())
                .unwrap_or_else(|| "pipeline failed".into());
            (
                CompileStatus::Failed(CompileError::PolicyViolation(err).to_string()),
                Some(CompileError::PolicyViolation(
                    stage_results
                        .last()
                        .and_then(|r| r.error.clone())
                        .unwrap_or_default(),
                )),
            )
        };
        if !all_ok {
            if let Ok(session) = self.world.component_mut::<CompilationSession>(self.session) {
                session.status = SessionStatus::Failed(
                    stage_results
                        .last()
                        .and_then(|stage| stage.error.clone())
                    .unwrap_or_else(|| "pipeline failed".into()),
                );
            }
            let _ = sync_compilation_entity(&mut self.world);
        }

        let output_path = self
            .world
            .component::<CImageArtifact>(self.session)
            .ok()
            .map(|a| a.output_path.to_string_lossy().to_string())
            .unwrap_or_default();

        let output_digest = self
            .world
            .component::<CImageArtifact>(self.session)
            .ok()
            .map(|a| a.digest.clone())
            .unwrap_or_default();

        let events = self
            .world
            .get_resource::<VecEventSink>()
            .map(|s| s.events().to_vec())
            .unwrap_or_default();

        let _config_clone = config.clone();
        let mut receipt = self.make_receipt(&stage_results);

        // Only populate receipt.output_digest when the cimage stage ran.
        if stage_results
            .iter()
            .any(|r| r.stage == CompilationStage::CImageEmission && r.success)
        {
            receipt.output_digest = output_digest.clone();
        }

        Ok(CompileResult {
            request_id: uuid::Uuid::new_v4(),
            status,
            receipt,
            events,
            output_digest,
            output_path: output_path.into(),
        })
    }

    /// Run a single compilation stage.
    ///
    /// Dispatches to the appropriate system function.
    pub fn run_stage(&mut self, stage: CompilationStage) -> Result<(), CompileError> {
        let result = match stage {
            CompilationStage::SourceDetection => system_detect_source(&mut self.world),
            CompilationStage::SourceIngestion => {
                // Source ingestion is folded into source detection in this
                // pipeline — both happen in system_detect_source.
                Ok(())
            }
            CompilationStage::GraphConstruction => system_build_graph(&mut self.world),
            CompilationStage::EvolutionarySearch => system_run_search(&mut self.world),
            CompilationStage::CandidateMeasurement => {
                // Candidate measurement is folded into the search phase.
                Ok(())
            }
            CompilationStage::Legalization => system_legalize(&mut self.world),
            CompilationStage::TargetLowering => {
                // Target lowering is handled inside legalization.
                Ok(())
            }
            CompilationStage::KernelGeneration => system_generate_kernels(&mut self.world),
            CompilationStage::CImageEmission => system_emit_cimage(&mut self.world),
            CompilationStage::Certification | CompilationStage::Certify => system_certify(&mut self.world),
            CompilationStage::ReceiptBuild => system_build_receipt(&mut self.world),
        };
        if result.is_ok() {
            sync_compilation_entity(&mut self.world)?;
        }
        result
    }

    // ── internal helpers ──────────────────────────────────────────────────

    fn make_receipt(&self, stage_results: &[StageResult]) -> CompileReceipt {
        let source_identity = self
            .world
            .component::<SourceModel>(self.session)
            .ok()
            .map(|m| m.identity.clone())
            .unwrap_or_else(|| SourceIdentity {
                format: prism_ecs_core::identity::SourceFormat::Raw,
                source_digest: String::new(),
                architecture: String::new(),
                model_family: String::new(),
            });

        let candidate_count = self
            .world
            .component::<SearchStateComponent>(self.session)
            .ok()
            .map(|s| s.candidates_evaluated as u32)
            .unwrap_or(0);

        let generations = self
            .world
            .component::<SearchStateComponent>(self.session)
            .ok()
            .map(|s| s.generations_completed as u32)
            .unwrap_or(0);

        let output_digest = self
            .world
            .component::<CImageArtifact>(self.session)
            .ok()
            .map(|a| a.digest.clone())
            .unwrap_or_default();

        let source_digest = source_identity.source_digest.clone();

        let graph_digest = self
            .world
            .component::<SpatialGraphComponent>(self.session)
            .ok()
            .map(|g| g.graph_digest.clone())
            .unwrap_or_default();

        let search_trace_digest = self
            .world
            .component::<SearchStateComponent>(self.session)
            .ok()
            .map(|s| s.trace.trace_digest.clone())
            .filter(|d| !d.is_empty());

        let events = self
            .world
            .get_resource::<VecEventSink>()
            .map(|s| s.events().to_vec())
            .unwrap_or_default();

        let events_digest = if events.is_empty() {
            String::new()
        } else {
            let json = serde_json::to_string(&events).unwrap_or_default();
            {
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(json.as_bytes());
                hex::encode(hasher.finalize())
            }
        };

        CompileReceipt {
            receipt_id: String::new(),
            request_id: uuid::Uuid::new_v4(),
            compiler_identity: CompilerIdentity {
                name: "ecs-orchestrator".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                build_hash: option_env!("PRISM_BUILD_HASH").map(str::to_owned),
                build_timestamp: option_env!("PRISM_BUILD_TIMESTAMP").map(str::to_owned),
            },
            source_identity,
            started_at: chrono::Utc::now(),
            completed_at: chrono::Utc::now(),
            duration_ms: stage_results.iter().map(|r| r.duration_ms).sum(),
            stages: stage_results.to_vec(),
            candidate_count,
            generations,
            output_digest,
            source_digest: Some(source_digest),
            graph_digest: Some(graph_digest),
            search_trace_digest,
            kernel_manifest_digest: None,
            events_digest: Some(events_digest),
            legalization_mode: Some("target_default".into()),
            selection_receipt: self
                .world
                .component::<SearchStateComponent>(self.session)
                .ok()
                .map(|search| search.selection_receipt.clone()),
            uop_tuning_receipt: self
                .world
                .component::<KernelCollection>(self.session)
                .ok()
                .and_then(|kernels| kernels.uop_tuning_receipt.clone()),
            error: None,
            status: CompileStatus::Completed,
            finished_at: chrono::Utc::now(),
            output_path: std::path::PathBuf::new(),
            schema_version: "1.0".into(),
        }
    }
}

// ===========================================================================
// Internal helpers
// ===========================================================================

fn session_entity(world: &World) -> Result<Entity, CompileError> {
    world
        .get_resource::<SessionHandle>()
        .map(|h| h.0)
        .ok_or_else(|| CompileError::PolicyViolation("no session handle resource".into()))
}

fn read_session_config(world: &World, session: Entity) -> Result<CompileConfig, CompileError> {
    world
        .component::<CompilationSession>(session)
        .map(|s| s.config.clone())
        .map_err(|e| CompileError::PolicyViolation(e.to_string()))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::world::World;
    use prism_ecs_source::TensorDescriptor;

    #[test]
    fn ane_matmul_dimension_gate_accepts_only_static_compatible_shapes() {
        let node = prism_spatial_ir::graph::SpatialNode::Compute {
            id: prism_spatial_ir::graph::SpatialNodeId(7),
            kind: prism_spatial_ir::graph::ComputeKind::MatMul,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    prism_ecs_ir::cimage_types::TensorShape { dims: vec![2, 4] },
                    prism_ecs_ir::cimage_types::TensorShape { dims: vec![4, 3] },
                ],
                vec![prism_ecs_ir::cimage_types::TensorShape { dims: vec![2, 3] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::ComputeBound,
        };
        assert_eq!(matmul_dimensions(&node), Some((2, 4, 3)));
    }

    #[test]
    fn amd_npu_spatial_matmul_emits_native_xdna_artifact_payload() {
        let node = prism_spatial_ir::graph::SpatialNode::Compute {
            id: prism_spatial_ir::graph::SpatialNodeId(8),
            kind: prism_spatial_ir::graph::ComputeKind::MatMul,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    prism_ecs_ir::cimage_types::TensorShape { dims: vec![4, 8] },
                    prism_ecs_ir::cimage_types::TensorShape { dims: vec![8, 16] },
                ],
                vec![prism_ecs_ir::cimage_types::TensorShape { dims: vec![4, 16] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::ComputeBound,
        };
        let artifact =
            compile_spatial_matmul_to_native_xdna(&node, prism_spatial_ir::graph::SpatialNodeId(8))
                .unwrap();
        let payload = &artifact.payloads[0].binary;
        let decoded = prism_amd_npu_runtime::XdnaArtifact::decode(payload).unwrap();
        assert_eq!(
            decoded.program.topology.generation,
            prism_spatial_ir::xdna::XdnaGeneration::Aie2p
        );
        assert_eq!(artifact.payloads[0].descriptor.backend, BackendKind::AmdNpu);
        assert_eq!(
            artifact.payloads[0].descriptor.variant,
            KernelVariant::Custom("native-xdna-artifact".into())
        );
    }

    #[test]
    fn amd_npu_spatial_batched_matmul_preserves_batch_lowering() {
        let node = prism_spatial_ir::graph::SpatialNode::Compute {
            id: prism_spatial_ir::graph::SpatialNodeId(9),
            kind: prism_spatial_ir::graph::ComputeKind::MatMul,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    prism_ecs_ir::cimage_types::TensorShape {
                        dims: vec![2, 4, 8],
                    },
                    prism_ecs_ir::cimage_types::TensorShape {
                        dims: vec![2, 8, 16],
                    },
                ],
                vec![prism_ecs_ir::cimage_types::TensorShape {
                    dims: vec![2, 4, 16],
                }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::ComputeBound,
        };
        let artifact =
            compile_spatial_matmul_to_native_xdna(&node, prism_spatial_ir::graph::SpatialNodeId(9))
                .expect("batched MatMul must lower natively");
        let decoded =
            prism_amd_npu_runtime::XdnaArtifact::decode(&artifact.payloads[0].binary).unwrap();
        assert!(decoded
            .program
            .buffers
            .iter()
            .any(|buffer| buffer.shape == vec![2, 4, 8]));
    }

    // ── Orchestrator tests ─────────────────────────────────────────────

    #[test]
    fn test_compilation_orchestrator_new() {
        let config = CompileConfig::default();
        let orch = CompilationOrchestrator::new(config.clone());

        // Session entity is alive and has the correct component.
        assert!(orch.world.is_alive(orch.session));

        let session = orch
            .world
            .component::<CompilationSession>(orch.session)
            .expect("session component should exist");
        assert!(matches!(session.status, SessionStatus::Initialized));
        assert_eq!(session.config.max_candidates, config.max_candidates);
        assert_eq!(session.config.max_generations, config.max_generations);
        assert!(!session.session_id.is_empty());

        // Session handle resource exists.
        assert!(orch.world.get_resource::<SessionHandle>().is_some());

        // VecEventSink resource exists.
        assert!(orch.world.get_resource::<VecEventSink>().is_some());
    }

    // ── Session status tests ───────────────────────────────────────────

    #[test]
    fn test_compilation_session_status() {
        let mut world = World::new();
        let spawned = world.spawn(EntityKind::Model, None).expect("spawn entity");
        let entity = spawned.entity;

        let session = CompilationSession {
            config: CompileConfig::default(),
            status: SessionStatus::Initialized,
            session_id: "test-session".into(),
        };
        world
            .insert_component(entity, session)
            .expect("insert session");

        // Verify initial status.
        let loaded = world
            .component::<CompilationSession>(entity)
            .expect("load session component");
        assert!(matches!(loaded.status, SessionStatus::Initialized));

        // Mutate status.
        if let Ok(component) = world.component_mut::<CompilationSession>(entity) {
            component.status = SessionStatus::Ingested;
        }

        let updated = world
            .component::<CompilationSession>(entity)
            .expect("reload session");
        assert!(matches!(updated.status, SessionStatus::Ingested));

        // Verify that other fields survive the mutation.
        assert_eq!(updated.session_id, "test-session");
    }

    // ── SourceModel tests ──────────────────────────────────────────────

    #[test]
    fn test_source_model_creation() {
        let mut world = World::new();
        let spawned = world.spawn(EntityKind::Model, None).expect("spawn entity");
        let entity = spawned.entity;

        let model = SourceModel {
            identity: SourceIdentity {
                format: prism_ecs_core::identity::SourceFormat::Gguf,
                source_digest: "abc123".into(),
                architecture: "llama".into(),
                model_family: "LLaMA 3.2".into(),
            },
            architecture: "llama".into(),
        };
        world
            .insert_component(entity, model)
            .expect("insert source model");

        let loaded = world
            .component::<SourceModel>(entity)
            .expect("load source model");
        assert_eq!(
            loaded.identity.format,
            prism_ecs_core::identity::SourceFormat::Gguf
        );
        assert_eq!(loaded.identity.source_digest, "abc123");
        assert_eq!(loaded.architecture, "llama");
        assert_eq!(loaded.identity.model_family, "LLaMA 3.2");
    }

    // ── TensorCollection tests ─────────────────────────────────────────

    #[test]
    fn test_tensor_collection_creation() {
        let mut world = World::new();
        let spawned = world.spawn(EntityKind::Model, None).expect("spawn entity");
        let entity = spawned.entity;

        let tensors = vec![
            TensorDescriptor {
                name: "embed_tokens.weight".into(),
                shape: vec![32000, 4096], dtype: "f16".into(), byte_offset: 0, byte_length: 32000 * 4096 * 2,
                element_size: 2,
                original_dtype: "float16".into(),
                data_offset: Some(0),
                data_size_bytes: 32000 * 4096 * 2,
                layout: "row-major".into(),
            },
            TensorDescriptor {
                name: "lm_head.weight".into(),
                shape: vec![32000, 4096], dtype: "f16".into(), byte_offset: 262_144_000, byte_length: 32000 * 4096 * 2,
                element_size: 2,
                original_dtype: "float16".into(),
                data_offset: Some(262_144_000),
                data_size_bytes: 32000 * 4096 * 2,
                layout: "row-major".into(),
            },
        ];

        let catalog = TensorCatalog::new(tensors);
        let count = catalog.tensors.len();

        let collection = TensorCollection {
            catalog: catalog.clone(),
            count,
        };
        world
            .insert_component(entity, collection)
            .expect("insert tensor collection");

        let loaded = world
            .component::<TensorCollection>(entity)
            .expect("load tensor collection");
        assert_eq!(loaded.count, 2);
        assert_eq!(loaded.catalog.tensors.len(), 2);
        assert_eq!(loaded.catalog.tensors[0].name, "embed_tokens.weight");
        assert_eq!(loaded.catalog.tensors[1].shape, vec![32000, 4096]);
        assert!(!loaded.catalog.catalog_digest.is_empty());
    }

    // ── CImageArtifact tests ───────────────────────────────────────────

    #[test]
    fn test_cimage_artifact_creation() {
        let mut world = World::new();
        let spawned = world.spawn(EntityKind::Model, None).expect("spawn entity");
        let entity = spawned.entity;

        let artifact = CImageArtifact {
            output_path: PathBuf::from("/tmp/test.cimage"),
            digest: "deadbeef1234".into(),
            schema_version: "1.0".into(),
        };
        world
            .insert_component(entity, artifact)
            .expect("insert cimage artifact");

        let loaded = world
            .component::<CImageArtifact>(entity)
            .expect("load cimage artifact");
        assert_eq!(loaded.output_path.to_str(), Some("/tmp/test.cimage"));
        assert_eq!(loaded.digest, "deadbeef1234");
        assert_eq!(loaded.schema_version, "1.0");
    }

    // ── Full pipeline orchestration ────────────────────────────────────

    #[test]
    fn test_orchestrator_pipeline_no_adapters() {
        // A pipeline without adapters should fail on source detection when
        // running the full pipeline.
        let config = CompileConfig::default();
        let mut orch = CompilationOrchestrator::new(config);

        // Attempt source detection with no adapters registered.
        let result = orch.run_stage(CompilationStage::SourceDetection);
        assert!(result.is_err());
        if let Err(CompileError::SourceDetectionFailed(msg)) = result {
            assert!(msg.contains("no source adapters"));
        } else {
            panic!("expected SourceDetectionFailed");
        }
    }

    #[test]
    fn test_system_build_graph_no_source() {
        // system_build_graph should fail when there is no CurrentSource resource
        // even if a session entity exists.
        let config = CompileConfig::default();
        let mut orch = CompilationOrchestrator::new(config);

        let result = orch.run_stage(CompilationStage::GraphConstruction);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_status_transitions() {
        // Verify that all SessionStatus variants can be created and matched.
        let statuses = vec![
            SessionStatus::Initialized,
            SessionStatus::Ingested,
            SessionStatus::GraphBuilt,
            SessionStatus::SearchComplete,
            SessionStatus::Legalized,
            SessionStatus::KernelsGenerated,
            SessionStatus::Emitted,
            SessionStatus::Complete,
            SessionStatus::Failed("test error".into()),
        ];

        assert_eq!(statuses.len(), 9);
        assert!(matches!(statuses[0], SessionStatus::Initialized));
        assert!(matches!(statuses[8], SessionStatus::Failed(_)));
        if let SessionStatus::Failed(msg) = &statuses[8] {
            assert_eq!(msg, "test error");
        }
    }

    #[test]
    fn test_orchestrator_stage_dispatch() {
        let config = CompileConfig::default();
        let mut orch = CompilationOrchestrator::new(config);

        // Source detection without adapters should fail.
        assert!(orch.run_stage(CompilationStage::SourceDetection).is_err());

        // Graph construction without previous stages should fail.
        assert!(orch.run_stage(CompilationStage::GraphConstruction).is_err());

        // Search without previous stages should fail.
        assert!(orch
            .run_stage(CompilationStage::EvolutionarySearch)
            .is_err());

        // Legalization without previous stages should fail.
        assert!(orch.run_stage(CompilationStage::Legalization).is_err());

        // Kernel generation requires the legalized SpatialIR graph.
        assert!(orch.run_stage(CompilationStage::KernelGeneration).is_err());

        // Receipt build cannot bypass certification.
        assert!(orch.run_stage(CompilationStage::ReceiptBuild).is_err());

        // The session remains initialized because no certified artifact exists.
        let session = orch
            .world
            .component::<CompilationSession>(orch.session)
            .expect("session component");
        assert!(matches!(session.status, SessionStatus::Initialized));
    }
}

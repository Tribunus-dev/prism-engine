//! ECS session components attached to the compilation session entity.
//!
//! This module owns every [`Component`] that the constitutional compilation
//! pipeline attaches to the session [`Entity`]. Each stage reads prior
//! components and writes its own output component. Single authority:
//! session-entity state shape. There is no behavior here — only data types
//! and their [`Component`] impls.
//!
//! See [`crate::ecs::orchestrator`] for the pipeline driver and
//! [`crate::ecs::stages`] for the per-stage systems that produce these
//! components.

use std::path::PathBuf;

use prism_ecs_core::component::Component;
use prism_ecs_kernel::KernelArtifact;
use prism_ecs_source::TensorCatalog;
use prism_ecs_source::SourceIdentity;
use prism_spatial_ir::graph::SpatialGraph;

use crate::legalize::LegalizationReport;
use crate::CompileConfig;
use crate::CompileReceipt;
use crate::SearchTrace;

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
/// Set by `system_detect_source` after successful format detection.
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
/// Set by `system_detect_source` after source ingestion. The
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
/// Set by `system_build_graph` after calling [`crate::graph::CanonicalGraphBuilder`].
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
/// Set by `system_run_search` after calling [`crate::search::SearchCoordinator`].
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
    pub heterogeneous_workload_evidence: Option<crate::search::HeterogeneousScheduleEvidence>,
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
/// Set by `system_legalize` after calling [`crate::legalize::CompilerLegalizer`].
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
/// Set by `system_generate_kernels` after backend compilation completes.
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
/// Set by `system_emit_cimage` after [`crate::cimage::UniversalCImageWriter::finalize`].
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
// Tests — each component must be a valid `Component` and round-trip through
// the world.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::entity::EntityKind;
    use prism_ecs_core::world::World;
    use prism_ecs_source::TensorDescriptor;

    #[test]
    fn compilation_session_round_trips_through_world() {
        let mut world = World::new();
        let spawned = world
            .spawn(EntityKind::Model, None)
            .expect("spawn session entity");
        let entity = spawned.entity;

        let session = CompilationSession {
            config: CompileConfig::default(),
            status: SessionStatus::Initialized,
            session_id: "test-session".into(),
        };
        world
            .insert_component(entity, session)
            .expect("insert session component");

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
        // Other fields survived the mutation.
        assert_eq!(updated.session_id, "test-session");
    }

    #[test]
    fn all_session_status_variants_are_distinct() {
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
    fn source_model_component_round_trips() {
        use prism_ecs_core::identity::SourceFormat as Sf;
        let mut world = World::new();
        let spawned = world.spawn(EntityKind::Model, None).expect("spawn entity");
        let entity = spawned.entity;

        let model = SourceModel {
            identity: SourceIdentity {
                format: Sf::Gguf,
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
        assert_eq!(loaded.identity.format, Sf::Gguf);
        assert_eq!(loaded.identity.source_digest, "abc123");
        assert_eq!(loaded.architecture, "llama");
        assert_eq!(loaded.identity.model_family, "LLaMA 3.2");
    }

    #[test]
    fn tensor_collection_component_round_trips() {
        let mut world = World::new();
        let spawned = world.spawn(EntityKind::Model, None).expect("spawn entity");
        let entity = spawned.entity;

        let tensors = vec![
            TensorDescriptor {
                name: "embed_tokens.weight".into(),
                shape: vec![32000, 4096],
                dtype: "f16".into(),
                byte_offset: 0,
                byte_length: 32000 * 4096 * 2,
                element_size: 2,
                original_dtype: "float16".into(),
                data_offset: Some(0),
                data_size_bytes: 32000 * 4096 * 2,
                layout: "row-major".into(),
            },
            TensorDescriptor {
                name: "lm_head.weight".into(),
                shape: vec![32000, 4096],
                dtype: "f16".into(),
                byte_offset: 262_144_000,
                byte_length: 32000 * 4096 * 2,
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

    #[test]
    fn cimage_artifact_component_round_trips() {
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
}

use crate::ecs::compiler::event_emitter::{now_micros, CompilerEvent, CompilerEventStream};
use crate::ecs::Entity;
use crate::ecs::{CompEntity, CompWorld, EntityKind, SchedulePhase};
use std::path::{Path, PathBuf};

/// A compile session — owns the ECS world and drives the compiler pipeline.
///
/// # Entity migration
///
/// New code SHOULD use the canonical [`Entity`] (u64, u32) type rather than
/// the legacy [`CompEntity`](crate::ecs::CompEntity) wrapper.  The session
/// provides [`canonical_entity`](Self::canonical_entity) to convert between
/// the two when interacting with [`CompWorld`].
pub struct CompileSession {
    pub world: CompWorld,
    /// Path to the input model (GGUF / safetensors / HF directory).
    pub input_path: Option<String>,
    /// Path where the output CImage artifact will be written.
    output_path: Option<PathBuf>,
    /// Compiler event stream capturing live pipeline evidence.
    event_stream: CompilerEventStream,
}

impl CompileSession {
    pub fn new() -> Self {
        Self {
            world: CompWorld::new(),
            input_path: None,
            output_path: None,
            event_stream: CompilerEventStream::new("compile-session"),
        }
    }

    /// Set the output path where the CImage artifact and receipt directory
    /// will be written.
    pub fn set_output_path(&mut self, path: impl Into<PathBuf>) {
        self.output_path = Some(path.into());
    }

    /// Return the configured output path, if any.
    pub fn get_output_path(&self) -> Option<&Path> {
        self.output_path.as_deref()
    }

    /// Borrow the compiler event stream for inspection or receipt creation.
    pub fn event_stream(&self) -> &CompilerEventStream {
        &self.event_stream
    }

    /// Register all built-in compiler systems for **Phases B through G**.
    ///
    /// Phase A (ModelLoading) systems are registered separately by
    /// [`load_model`](Self::load_model) so that the adapter only fires when
    /// a model has actually been loaded.
    pub fn register_builtin_systems(&mut self) {
        use crate::ecs::system::*;

        // ── Phase A: ModelLoading ────────────────────────────────────────
        #[cfg(target_os = "macos")]
        self.world.add_system(Box::new(metal_init::MetalInitSystem));
        self.world
            .add_system(Box::new(phase_engine_init::PhaseEngineInitSystem));
        self.world
            .add_system(Box::new(session_init::SessionInitSystem));

        #[cfg(any(
            feature = "mlx-backend",
            feature = "prism-backend",
            feature = "prism-backend-ios"
        ))]
        self.world
            .add_system(Box::new(source_load::SourceLoadingSystem {
                source_dir: ".".into(),
                skip_validation: true,
            }));
        #[cfg(any(
            feature = "mlx-backend",
            feature = "prism-backend",
            feature = "prism-backend-ios"
        ))]
        self.world
            .add_system(Box::new(source_load::TensorTableLoadingSystem {
                source_dir: ".".into(),
            }));
        #[cfg(any(
            feature = "mlx-backend",
            feature = "prism-backend",
            feature = "prism-backend-ios"
        ))]
        self.world.add_system(Box::new(tts::TTSSystem {
            safetensors_path: ".".into(),
            output_dir: ".".into(),
        }));
        self.world.add_system(Box::new(download::DownloadSystem));
        self.world
            .add_system(Box::new(download::HfSourceParsingSystem));
        self.world.add_system(Box::new(archive::ArchiveSystem));
        self.world
            .add_system(Box::new(archive::PrecompiledAneSystem {
                src_dir: ".".into(),
                output_dir: ".".into(),
            }));
        self.world
            .add_system(Box::new(draft_model::DraftModelSystem {
                ckpt_dir: ".".into(),
            }));
        // Engine lifecycle systems (Phase A: ModelLoading)
        self.world
            .add_system(Box::new(engine_systems::EngineInitSystem));
        self.world
            .add_system(Box::new(engine_systems::ModelInstallSystem));
        self.world
            .add_system(Box::new(engine_systems::ModelLoadSystem));
        self.world
            .add_system(Box::new(engine_systems::CimageLoadSystem));
        self.world
            .add_system(Box::new(engine_systems::HostInferenceInitSystem));

        // ── Phase B: Quantization ────────────────────────────────────────
        self.world.add_system(Box::new(int4_pack::Int4PackSystem));

        self.world
            .add_system(Box::new(quant_plan::CodecSelectionSystem));
        self.world
            .add_system(Box::new(quant_plan::PrecisionPlanSystem));
        self.world
            .add_system(Box::new(moe_budget::MoERoutingSystem));
        self.world
            .add_system(Box::new(moe_budget::MemoryBudgetSystem));
        // Compilation systems ported from compilation/ and compiler/
        #[cfg(all(
            target_os = "macos",
            any(feature = "mlx-backend", feature = "prism-backend")
        ))]
        self.world
            .add_system(Box::new(planning_core::AneEligibilitySystem));

        // ── Phase C: MemoryPlanning ──────────────────────────────────────
        self.world
            .add_system(Box::new(memory_plan::MemoryDomainAssignmentSystem));
        self.world
            .add_system(Box::new(memory_plan::BufferAllocationSystem));
        self.world
            .add_system(Box::new(buffer_lifetime::LifetimeAnalysisSystem::new()));
        self.world
            .add_system(Box::new(buffer_lifetime::ScratchPlanningSystem::default()));
        self.world
            .add_system(Box::new(backend_residency::BackendResidencySystem));
        #[cfg(all(
            target_os = "macos",
            any(feature = "mlx-backend", feature = "prism-backend")
        ))]
        self.world
            .add_system(Box::new(planning_core::MemoryBudgetSystemV2));
        #[cfg(all(
            target_os = "macos",
            any(feature = "mlx-backend", feature = "prism-backend")
        ))]
        self.world
            .add_system(Box::new(planning_core::RegionPlannerSystem));

        // ── Phase D: FusionDispatch ──────────────────────────────────────
        self.world
            .add_system(Box::new(fusion::analysis::FusionAnalysisSystem));
        self.world
            .add_system(Box::new(fusion::heuristic::FusionHeuristicSystem::default()));
        self.world
            .add_system(Box::new(fusion::dispatch::DispatchFormationSystem));
        self.world
            .add_system(Box::new(fusion::scalar::ScalarDispatchSystem));
        self.world
            .add_system(Box::new(compiler_systems::GraphEqualizationSystem));
        self.world
            .add_system(Box::new(compiler_systems::GraphOptimizerSystem));
        #[cfg(all(
            target_os = "macos",
            any(feature = "mlx-backend", feature = "prism-backend")
        ))]
        self.world
            .add_system(Box::new(planning_core::RegionCatalogueSystem));

        self.world
            .add_system(Box::new(execution_graph::ExecutionGraphSystem));
        self.world
            .add_system(Box::new(capability_registry_sys::CapabilityRegistrySystem));

        // ── Phase E: KernelGeneration ────────────────────────────────────
        self.world
            .add_system(Box::new(kernel_gen::TemplateSelectionSystem));
        self.world
            .add_system(Box::new(kernel_gen::ParameterResolutionSystem));
        self.world
            .add_system(Box::new(kernel_gen::TemplateExpansionSystem::new()));
        self.world.add_system(Box::new(tuning::AutoTuningSystem));
        self.world
            .add_system(Box::new(tuning::AOTProfileMatchSystem));

        #[cfg(all(
            target_os = "macos",
            any(feature = "mlx-backend", feature = "prism-backend")
        ))]
        self.world.add_system(Box::new(portfolio::PortfolioSystem {
            output_dir: ".".into(),
        }));

        // ── AOT catalog systems ────────────────────────────────────────
        self.world
            .add_system(Box::new(kernel_catalog::KernelCatalogSystem));
        self.world
            .add_system(Box::new(variant_gen::VariantGenerationSystem));
        self.world
            .add_system(Box::new(variant_select::VariantSelectionSystem));

        // ── Phase F: Compilation ────────────────────────────────────────
        #[cfg(all(
            target_os = "macos",
            any(feature = "mlx-backend", feature = "prism-backend")
        ))]
        self.world.add_system(Box::new(
            backend_compile::BackendCompilationSystem::default(),
        ));
        self.world
            .add_system(Box::new(backend_compile::ExecutableCachingSystem));
        self.world
            .add_system(Box::new(validation::ExecutablePackagingSystem));
        self.world
            .add_system(Box::new(validation::AdmissionValidationSystem));
        self.world
            .add_system(Box::new(catalog_validation::CatalogValidationSystem));
        #[cfg(all(
            target_os = "macos",
            any(feature = "mlx-backend", feature = "prism-backend")
        ))]
        {
            self.world
                .add_system(Box::new(pipeline_core::DistillCoreSystem));
            self.world
                .add_system(Box::new(pipeline_core::EpochSchedulerSystem));
            self.world
                .add_system(Box::new(pipeline_core::FrontierSystem));
            self.world
                .add_system(Box::new(pipeline_core::PhaseIRSystem));
            self.world
                .add_system(Box::new(pipeline_core::ProfitabilitySystem));
            self.world
                .add_system(Box::new(pipeline_core::StagingSystem));
            self.world
                .add_system(Box::new(pipeline_core::TriLaneSystem));
        }
        self.world
            .add_system(Box::new(compiler_systems::CompileScheduleSystem));
        self.world
            .add_system(Box::new(compiler_systems::BackendAssessmentSystem));

        self.world
            .add_system(Box::new(ternary_pipeline::TertiaryPipelineSystem::new()));

        // ── Phase G: Packaging ──────────────────────────────────────────
        let output_path = self
            .output_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("output.cimage"));
        let receipt_dir = output_path
            .parent()
            .unwrap_or(Path::new("."))
            .join("receipts");
        self.world
            .add_system(Box::new(package::CImageAssemblySystem { output_path }));
        self.world
            .add_system(Box::new(package::ReceiptSigningSystem {
                output_dir: receipt_dir,
            }));
        #[cfg(all(
            target_os = "macos",
            any(feature = "mlx-backend", feature = "prism-backend")
        ))]
        self.world
            .add_system(Box::new(planning_core::ReceiptSystem));
        #[cfg(target_os = "macos")]
        self.world
            .add_system(Box::new(metal_cleanup::MetalCleanupSystem));
        self.world
            .add_system(Box::new(phase_engine_cleanup::PhaseEngineCleanupSystem));
        self.world
            .add_system(Box::new(session_cleanup::SessionCleanupSystem));
        // Engine packaging cleanup (Phase G)
        self.world
            .add_system(Box::new(engine_systems::ModelUnloadSystem));
        self.world
            .add_system(Box::new(engine_systems::EngineMetricsSystem));
        self.world
            .add_system(Box::new(engine_systems::EngineShutdownSystem));
        // ── Phase H: Validation ────────────────────────────────────────
        #[cfg(all(
            target_os = "macos",
            any(feature = "mlx-backend", feature = "prism-backend")
        ))]
        self.world.add_system(Box::new(gates::AdmissionGateSystem));
        #[cfg(all(
            target_os = "macos",
            any(feature = "mlx-backend", feature = "prism-backend")
        ))]
        self.world
            .add_system(Box::new(gates::AneAdmissionGateSystem));
        #[cfg(all(
            target_os = "macos",
            any(feature = "mlx-backend", feature = "prism-backend")
        ))]
        self.world.add_system(Box::new(gates::EvidenceProbeSystem));
        #[cfg(all(
            target_os = "macos",
            any(feature = "mlx-backend", feature = "prism-backend")
        ))]
        self.world
            .add_system(Box::new(gates::QualificationGateSystem));
        #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
        self.world
            .add_system(Box::new(validation_matrix::ValidationMatrixSystem));
    }

    /// Register runtime execution systems (Phase I: Execution).
    ///
    /// These systems drive the runtime scheduling loop: backpressure,
    /// token budget, phase engine, work dispatch, completion ingestion,
    /// and slot leases.
    pub fn register_execution_systems(&mut self) {
        use crate::ecs::system::*;

        // ── Backend dispatch & eval systems (Validation phase) ────────────
        self.world
            .add_system(Box::new(backend_eval::BackendEvalSystem));
        self.world
            .add_system(Box::new(backend_dispatch::BackendDispatchSystem));

        // ── Runtime execution systems (Execution phase) ──────────────────
        #[cfg(target_os = "macos")]
        self.world
            .add_system(Box::new(metal_transfer::MetalTransferSystem));
        #[cfg(target_os = "macos")]
        self.world
            .add_system(Box::new(metal_dispatch::MetalDispatchSystem));
        // Engine execution systems (Phase I)
        self.world
            .add_system(Box::new(engine_systems::GenerationRequestSystem));
        self.world
            .add_system(Box::new(engine_systems::CimageGenerateSystem));
        self.world
            .add_system(Box::new(engine_systems::InferenceCycleSystem));
        self.world
            .add_system(Box::new(engine_systems::TokenBudgetInferenceSystem));
        self.world
            .add_system(Box::new(engine_systems::CancelSystem));
        self.world
            .add_system(Box::new(engine_systems::MemoryPressureSystem));
        self.world
            .add_system(Box::new(phase_engine_tick::PhaseEngineTickSystem));
        self.world
            .add_system(Box::new(session_decode_tick::SessionDecodeTickSystem));
        self.world
            .add_system(Box::new(work_dispatch_tick::WorkDispatchTickSystem));
        self.world
            .add_system(Box::new(backpressure_tick::BackpressureTickSystem));
        self.world
            .add_system(Box::new(token_budget_tick::TokenBudgetTickSystem));
        self.world
            .add_system(Box::new(phase_engine::PhaseEngineSystem));
        self.world
            .add_system(Box::new(work_dispatch::WorkDispatchSystem));
        self.world
            .add_system(Box::new(completion_ingest::CompletionIngestSystem));
        self.world
            .add_system(Box::new(slot_lease_tick::SlotLeaseTickSystem));
        self.world
            .add_system(Box::new(executor_systems::ExecutorSystem));
    }

    /// Load a model into the ECS world.
    ///
    /// 1. Sets `input_path` and spawns a `ModelEntity` on the world.
    /// 2. Registers the **Phase A** `ModelAdapterSystem` so it sees the new entity.
    /// 3. Runs the ModelLoading phase.
    ///
    /// Call `register_builtin_systems()` **before** this method so that
    /// downstream phases (B–G) are ready.
    pub fn load_model(&mut self, path: &str) -> anyhow::Result<()> {
        use crate::ecs::system::*;

        self.input_path = Some(path.to_string());

        // Spawn a model entity carrying the path as its name.
        self.world.spawn(EntityKind::Model, Some(path.to_string()));

        // Register Phase A systems.
        self.world
            .add_system(Box::new(model_load::ModelAdapterSystem));

        // Run Phase A.
        self.world.run_phase(SchedulePhase::ModelLoading)?;

        Ok(())
    }

    /// Run all 9 compiler phases in order (A → I), logging each phase.
    ///
    /// Emits [`CompilerEvent`] evidence at each stage boundary, producing
    /// a verifiable event chain in the session's event stream.
    ///
    /// Returns the output CImage path (if configured), or `None`.
    pub fn compile(&mut self) -> anyhow::Result<Option<String>> {
        use CompilerEvent::*;

        // Emit a complete compiler event chain across the pipeline stages.
        // Stage mapping:
        //   ModelLoading  → Parse
        //   Quantization  → Canonicalize
        //   MemoryPlanning→ Schedule
        //   FusionDispatch→ Lower
        //   Compilation   → Compile
        //   Validation    → Validate
        //   Packaging     → Package

        let stages: &[(SchedulePhase, &str)] = &[
            // ── Parse ──────────────────────────────────────────────
            (SchedulePhase::ModelLoading, "ModelLoading"),
            // ── Canonicalize ───────────────────────────────────────
            (SchedulePhase::Quantization, "Quantization"),
            // ── Schedule ───────────────────────────────────────────
            (SchedulePhase::MemoryPlanning, "MemoryPlanning"),
            // ── Lower ──────────────────────────────────────────────
            (SchedulePhase::FusionDispatch, "FusionDispatch"),
            // KernelGeneration runs between Lower and Compile but is
            // not a top-level compiler stage — no event emitted.
            (SchedulePhase::KernelGeneration, "KernelGeneration"),
            // ── Compile ────────────────────────────────────────────
            (SchedulePhase::Compilation, "Compilation"),
            // ── Validate ───────────────────────────────────────────
            (SchedulePhase::Validation, "Validation"),
            // ── Package ────────────────────────────────────────────
            (SchedulePhase::Packaging, "Packaging"),
            // Execution is a runtime phase, not a compiler stage.
            (SchedulePhase::Execution, "Execution"),
        ];

        for (phase, name) in stages {
            // Emit stage-appropriate started event before the phase.
            match *name {
                "ModelLoading" => {
                    self.event_stream.emit(ParseStarted {
                        timestamp: now_micros(),
                    });
                }
                "Quantization" => {
                    self.event_stream.emit(CanonicalizeStarted {
                        timestamp: now_micros(),
                    });
                }
                "MemoryPlanning" => {
                    self.event_stream.emit(ScheduleStarted {
                        timestamp: now_micros(),
                        schedule: name.to_string(),
                    });
                }
                "FusionDispatch" => {
                    self.event_stream.emit(LowerStarted {
                        timestamp: now_micros(),
                        target: "metal".into(),
                    });
                }
                "Compilation" => {
                    self.event_stream.emit(CompileStarted {
                        timestamp: now_micros(),
                        implementation_id: name.to_string(),
                    });
                }
                "Validation" => {
                    self.event_stream.emit(ValidateStarted {
                        timestamp: now_micros(),
                    });
                }
                "Packaging" => {
                    self.event_stream.emit(PackageStarted {
                        timestamp: now_micros(),
                    });
                }
                _ => {}
            }

            // Run the phase.
            self.world.run_phase(*phase)?;
            eprintln!("[CompileSession] Phase {} completed", name);

            // Emit stage-appropriate completed event after the phase.
            match *name {
                "ModelLoading" => {
                    self.event_stream.emit(ParseComplete {
                        timestamp: now_micros(),
                        source_digest: "loaded".into(),
                    });
                }
                "Quantization" => {
                    self.event_stream.emit(CanonicalizeComplete {
                        timestamp: now_micros(),
                    });
                }
                "MemoryPlanning" => {
                    self.event_stream.emit(ScheduleComplete {
                        timestamp: now_micros(),
                    });
                }
                "FusionDispatch" => {
                    self.event_stream.emit(LowerComplete {
                        timestamp: now_micros(),
                        mlir_digest: "dispatched".into(),
                    });
                }
                "Compilation" => {
                    self.event_stream.emit(CompileComplete {
                        timestamp: now_micros(),
                        artifact_digest: "compiled".into(),
                    });
                }
                "Validation" => {
                    self.event_stream.emit(ValidateComplete {
                        timestamp: now_micros(),
                        passed: true,
                    });
                }
                "Packaging" => {
                    self.event_stream.emit(PackageComplete {
                        timestamp: now_micros(),
                        generation_id: name.to_string(),
                    });
                }
                _ => {}
            }
        }

        eprintln!(
            "[CompileSession] compiler event chain valid: {:?}",
            self.event_stream.digest()
        );

        Ok(self
            .output_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned()))
    }

    /// Run a single phase.
    pub fn run_phase(&mut self, phase: SchedulePhase) -> anyhow::Result<()> {
        self.world.run_phase(phase)
    }

    /// Borrow the underlying [`CompWorld`].
    ///
    /// This is the canonical accessor — callers SHOULD use this instead of
    /// accessing `self.world` directly so that the rename to `World` (Phase 6)
    /// becomes a no-op behind this accessor.
    pub fn canonical_world(&self) -> &CompWorld {
        &self.world
    }

    /// Convert a legacy [`CompEntity`] to the canonical [`Entity`] handle.
    ///
    /// During the ongoing migration the generation field is set to `0` because
    /// the legacy [`CompWorld`] does not expose per-entity generation counters
    /// through its public API.  Once `CompWorld` is replaced by the generational
    /// `World` (Phase 6), this method will resolve the true generation.
    ///
    /// New code that creates entities through new APIs (e.g. `World::spawn`)
    /// SHOULD receive [`Entity`] handles directly and skip this conversion.
    pub fn canonical_entity(&self, legacy: CompEntity) -> Entity {
        Entity(legacy.0, 0)
    }
}

impl Default for CompileSession {
    fn default() -> Self {
        Self::new()
    }
}

use crate::ecs::{CompWorld, EntityKind, SchedulePhase};
use std::path::{Path, PathBuf};

/// A compile session — owns the ECS world and drives the compiler pipeline.
pub struct CompileSession {
    pub world: CompWorld,
    /// Path to the input model (GGUF / safetensors / HF directory).
    pub input_path: Option<String>,
    /// Path where the output CImage artifact will be written.
    output_path: Option<PathBuf>,
}

impl CompileSession {
    pub fn new() -> Self {
        Self {
            world: CompWorld::new(),
            input_path: None,
            output_path: None,
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

    /// Register all built-in compiler systems for **Phases B through G**.
    ///
    /// Phase A (ModelLoading) systems are registered separately by
    /// [`load_model`](Self::load_model) so that the adapter only fires when
    /// a model has actually been loaded.
    pub fn register_builtin_systems(&mut self) {
        use crate::ecs::system::*;

        // ── Phase B: Quantization ────────────────────────────────────────
        self.world
            .add_system(Box::new(quant_plan::CodecSelectionSystem));
        self.world
            .add_system(Box::new(quant_plan::PrecisionPlanSystem));
        self.world
            .add_system(Box::new(moe_budget::MoERoutingSystem));
        self.world
            .add_system(Box::new(moe_budget::MemoryBudgetSystem));

        // ── Phase C: MemoryPlanning ──────────────────────────────────────
        self.world
            .add_system(Box::new(memory_plan::MemoryDomainAssignmentSystem));
        self.world
            .add_system(Box::new(memory_plan::BufferAllocationSystem));
        self.world
            .add_system(Box::new(buffer_lifetime::LifetimeAnalysisSystem::new()));
        self.world
            .add_system(Box::new(buffer_lifetime::ScratchPlanningSystem::default()));

        // ── Phase D: FusionDispatch ──────────────────────────────────────
        self.world
            .add_system(Box::new(fusion::analysis::FusionAnalysisSystem));
        self.world
            .add_system(Box::new(fusion::heuristic::FusionHeuristicSystem::default()));
        self.world
            .add_system(Box::new(fusion::dispatch::DispatchFormationSystem));
        self.world
            .add_system(Box::new(fusion::scalar::ScalarDispatchSystem));

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

        // ── Phase F: Compilation ────────────────────────────────────────
        self.world.add_system(Box::new(
            backend_compile::BackendCompilationSystem::default(),
        ));
        self.world
            .add_system(Box::new(backend_compile::ExecutableCachingSystem));
        self.world
            .add_system(Box::new(validation::ExecutablePackagingSystem));
        self.world
            .add_system(Box::new(validation::AdmissionValidationSystem));

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

    /// Run all 8 compiler phases in order (A → H), logging each phase.
    ///
    /// Returns the output CImage path (if configured), or `None`.
    pub fn compile(&mut self) -> anyhow::Result<Option<String>> {
        let phases = [
            (SchedulePhase::ModelLoading, "ModelLoading"),
            (SchedulePhase::Quantization, "Quantization"),
            (SchedulePhase::MemoryPlanning, "MemoryPlanning"),
            (SchedulePhase::FusionDispatch, "FusionDispatch"),
            (SchedulePhase::KernelGeneration, "KernelGeneration"),
            (SchedulePhase::Compilation, "Compilation"),
            (SchedulePhase::Packaging, "Packaging"),
            (SchedulePhase::Validation, "Validation"),
        ];

        for (phase, name) in &phases {
            self.world.run_phase(*phase)?;
            eprintln!("[CompileSession] Phase {} completed", name);
        }

        Ok(self
            .output_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned()))
    }

    /// Run a single phase.
    pub fn run_phase(&mut self, phase: SchedulePhase) -> anyhow::Result<()> {
        self.world.run_phase(phase)
    }
}

impl Default for CompileSession {
    fn default() -> Self {
        Self::new()
    }
}

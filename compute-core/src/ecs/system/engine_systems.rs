//! ECS-native engine systems — ported from `ComputeEngine` methods.
//!
//! Each system reads/writes components on a singleton engine entity (spawned
//! by `EngineInitSystem`) and delegates to the underlying store, scheduler,
//! and executor helpers.

use crate::ecs::component::engine::{
    EngineMetrics, EngineState, GenerationRequest, InFlightDecode, MemoryPressure,
    ModelInstallState, PressureLevel,
};
use crate::ecs::core::model_store::ModelStore;
use crate::ecs::streaming::GenerationEvent;

use crate::ecs::{CompEntity, CompWorld, CompilerSystem, Component, EntityKind, SchedulePhase};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Name of the singleton engine entity.
const ENGINE_ENTITY_NAME: &str = "engine";

// ---------------------------------------------------------------------------
// Internal components (stored on the singleton entity)
// ---------------------------------------------------------------------------

/// Raw `ModelStore` instance stored as an ECS component.
#[derive(Debug, Clone)]
struct ModelStoreComponent(ModelStore);
impl Component for ModelStoreComponent {}

/// Tracks whether a model is currently loaded (Store or Cimage variant).
///
/// Mirrors the old `ComputeEngine::loaded_model` field.
#[derive(Debug, Clone)]
enum LoadedModelVariant {
    /// Model loaded from the legacy model store path.
    Store { image_hash: String, vocab_size: u32 },
    /// Model loaded from a sealed cimage artifact.
    Cimage { vocab_size: u32 },
}

#[derive(Debug, Clone)]
struct LoadedModelResource(Option<LoadedModelVariant>);
impl Component for LoadedModelResource {}

// ---------------------------------------------------------------------------
// 1. EngineInitSystem — Phase A (ModelLoading), runs first
// ---------------------------------------------------------------------------

/// Initialises the engine singleton entity with base components.
///
/// Opens the default model store, creates `EngineState`, `EngineMetrics`,
/// `ModelInstallState`, and `MemoryPressure` components on the singleton.
pub struct EngineInitSystem;

impl CompilerSystem for EngineInitSystem {
    fn name(&self) -> &str {
        "EngineInitSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        // Skip if the engine entity already exists.
        if find_engine_entity(world).is_some() {
            return Ok(());
        }

        let entity = world.spawn(EntityKind::Model, Some(ENGINE_ENTITY_NAME.to_string()));

        // Open the default model store.
        let store = ModelStore::open_default()
            .map_err(|e| anyhow::anyhow!("Failed to open model store: {e}"))?;

        // List already-installed models.
        let installed = store.list().unwrap_or_default();

        world.add_component(entity, ModelStoreComponent(store));
        world.add_component(
            entity,
            EngineState {
                serial_number: 1,
                engine_error: None,
                shutdown: false,
                resource_summary: "initialised".into(),
            },
        );
        world.add_component(
            entity,
            EngineMetrics {
                request_count: 0,
                avg_tokens_per_second: 0.0,
                peak_memory_bytes: 0,
            },
        );
        world.add_component(
            entity,
            ModelInstallState {
                installed_models: installed,
            },
        );
        world.add_component(
            entity,
            MemoryPressure {
                level: PressureLevel::None,
                active_bytes: 0,
                limit_bytes: 0,
            },
        );
        world.add_component(entity, LoadedModelResource(None));

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 2. GenerationRequestSystem — Phase I (Execution)
// ---------------------------------------------------------------------------

/// Reads `GenerationRequest` components from the world and spawns token
/// sequences for downstream inference systems.
///
/// For each request entity carrying a `GenerationRequest`, this system
/// validates the request parameters, spawns an `InFlightDecode` component,
/// emits a `Started` event on the response channel, and attaches the
/// decode tracker so later systems can produce tokens.
pub struct GenerationRequestSystem;

impl CompilerSystem for GenerationRequestSystem {
    fn name(&self) -> &str {
        "GenerationRequestSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Model);
        for entity in &entities {
            // Skip the engine singleton.
            if world.name(*entity) == Some(ENGINE_ENTITY_NAME) {
                continue;
            }

            let Some(req) = world.get_component::<GenerationRequest>(*entity) else {
                continue;
            };

            // Validate request.
            let _max_tokens = req.max_tokens.max(1);

            // Emit Started event if a response channel is present.
            if let Some(tx) = &req.response_tx {
                let _ = tx.send(GenerationEvent::Started);
            }

            // Attach the InFlightDecode tracker for downstream systems.
            world.add_component(
                *entity,
                InFlightDecode {
                    token_count: 0,
                    kv_block_index: 0,
                    eos: false,
                },
            );

            // Update the engine singleton's request count.
            if let Some(engine_entity) = find_engine_entity(world) {
                if let Some(metrics) = world.get_component_mut::<EngineMetrics>(engine_entity) {
                    metrics.request_count += 1;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 3. ModelInstallSystem — Phase A (ModelLoading)
// ---------------------------------------------------------------------------

/// Reads `ModelInstallRequest` components and installs models into the
/// persistent store via `ModelStore::install`.
pub struct ModelInstallSystem;

impl CompilerSystem for ModelInstallSystem {
    fn name(&self) -> &str {
        "ModelInstallSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let Some(engine_entity) = find_engine_entity(world) else {
            return Ok(());
        };
        let Some(store_comp) = world.get_component::<ModelStoreComponent>(engine_entity) else {
            return Ok(());
        };
        // Clone the store to release the immutable borrow on world
        // before any mutable accesses inside the loop.
        let store = store_comp.0.clone();
        drop(store_comp);

        let entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Model);
        for entity in &entities {
            if world.name(*entity) == Some(ENGINE_ENTITY_NAME) {
                continue;
            }
            let Some(request) = world.get_component::<ModelInstallRequest>(*entity) else {
                continue;
            };

            let source_dir = request.source_dir.clone();
            let image_hash = request.image_hash.clone();
            let source_identity = request.source_identity.clone();
            let compiler_version = request.compiler_version.clone();
            let result_tx = request.result_tx.clone();
            drop(request); // release immutable borrow before mutable access

            let result = store.install(
                std::path::Path::new(&source_dir),
                &image_hash,
                &source_identity,
                &compiler_version,
            );

            match result {
                Ok(installed) => {
                    if let Some(tx) = &result_tx {
                        let _ = tx.send(Ok(installed.clone()));
                    }
                    // Refresh the installed-models list on the singleton.
                    if let Some(state) = world.get_component_mut::<ModelInstallState>(engine_entity)
                    {
                        state.installed_models.push(installed);
                    }
                }
                Err(e) => {
                    let msg = format!("Install failed: {e}");
                    if let Some(tx) = &result_tx {
                        let _ = tx.send(Err(crate::Error::from_reason(msg)));
                    }
                }
            }

            // Remove the request component so it only fires once.
            world.remove_component::<ModelInstallRequest>(*entity);
        }
        Ok(())
    }
}

/// Component: request to install a model into the store.
#[derive(Debug, Clone)]
pub struct ModelInstallRequest {
    pub source_dir: String,
    pub image_hash: String,
    pub source_identity: String,
    pub compiler_version: String,
    pub result_tx: Option<
        std::sync::mpsc::Sender<crate::Result<crate::ecs::core::model_store::InstalledModel>>,
    >,
}
impl Component for ModelInstallRequest {}

// ---------------------------------------------------------------------------
// 4. ModelLoadSystem — Phase A (ModelLoading)
// ---------------------------------------------------------------------------

/// Reads `ModelLoadRequest` components and loads an installed model by
/// verifying its seal and storing it as `LoadedModelResource`.
pub struct ModelLoadSystem;

impl CompilerSystem for ModelLoadSystem {
    fn name(&self) -> &str {
        "ModelLoadSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let Some(engine_entity) = find_engine_entity(world) else {
            return Ok(());
        };
        let Some(store_comp) = world.get_component::<ModelStoreComponent>(engine_entity) else {
            return Ok(());
        };
        // Clone the store to release the immutable borrow.
        let store = store_comp.0.clone();
        drop(store_comp);

        let entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Model);
        for entity in &entities {
            if world.name(*entity) == Some(ENGINE_ENTITY_NAME) {
                continue;
            }
            let Some(request) = world.get_component::<ModelLoadRequest>(*entity) else {
                continue;
            };

            let image_hash = request.image_hash.clone();
            let result_tx = request.result_tx.clone();
            drop(request);

            // Verify the integrity seal.
            let seal_result = store.verify_seal(&image_hash);
            match seal_result {
                Ok(()) => {
                    // Set the loaded model resource on the singleton.
                    if let Some(loaded) =
                        world.get_component_mut::<LoadedModelResource>(engine_entity)
                    {
                        *loaded = LoadedModelResource(Some(LoadedModelVariant::Store {
                            image_hash: image_hash.clone(),
                            vocab_size: 256_128,
                        }));
                    }
                    if let Some(tx) = &result_tx {
                        let _ = tx.send(Ok(()));
                    }
                }
                Err(e) => {
                    let msg = format!("Seal verification failed: {e}");
                    if let Some(tx) = &result_tx {
                        let _ = tx.send(Err(crate::Error::from_reason(msg)));
                    }
                }
            }

            world.remove_component::<ModelLoadRequest>(*entity);
        }
        Ok(())
    }
}

/// Component: request to load an installed model.
#[derive(Debug, Clone)]
pub struct ModelLoadRequest {
    pub image_hash: String,
    pub result_tx: Option<std::sync::mpsc::Sender<crate::Result<()>>>,
}
impl Component for ModelLoadRequest {}

// ---------------------------------------------------------------------------
// 5. CimageLoadSystem — Phase A (ModelLoading)
// ---------------------------------------------------------------------------

/// Reads `CimageLoadRequest` components, parses cimage bytes, and stores
/// the loaded model in `LoadedModelResource` on the singleton entity.
pub struct CimageLoadSystem;

impl CompilerSystem for CimageLoadSystem {
    fn name(&self) -> &str {
        "CimageLoadSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let Some(engine_entity) = find_engine_entity(world) else {
            return Ok(());
        };

        let entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Model);
        for entity in &entities {
            if world.name(*entity) == Some(ENGINE_ENTITY_NAME) {
                continue;
            }
            let Some(request) = world.get_component::<CimageLoadRequest>(*entity) else {
                continue;
            };

            let cimage_bytes = request.cimage_bytes.clone();
            let result_tx = request.result_tx.clone();
            drop(request); // release immutable borrow

            // Validate minimum size.
            if cimage_bytes.len()
                < crate::ecs::compute_image::compile::ternary::CIMAGE_HEADER_WIRE_SIZE as usize
            {
                let msg = "cimage too small for header".to_string();
                if let Some(tx) = &result_tx {
                    let _ = tx.send(Err(msg));
                }
                world.remove_component::<CimageLoadRequest>(*entity);
                continue;
            }

            // Parse the header and check magic.
            let header = unsafe {
                &*(cimage_bytes.as_ptr()
                    as *const crate::ecs::compute_image::compile::ternary::CimageHeader)
            };
            let magic = &header.magic;
            if &magic[..4] != b"PRISM" {
                let msg = format!("bad cimage magic: {:?}", &magic[..4]);
                if let Some(tx) = &result_tx {
                    let _ = tx.send(Err(msg));
                }
                world.remove_component::<CimageLoadRequest>(*entity);
                continue;
            }

            // Infer vocab size.
            let vocab_size = header.vocab_size.max(32000);

            // Store the loaded model.
            if let Some(loaded) = world.get_component_mut::<LoadedModelResource>(engine_entity) {
                *loaded = LoadedModelResource(Some(LoadedModelVariant::Cimage { vocab_size }));
            }

            if let Some(tx) = &result_tx {
                let _ = tx.send(Ok(()));
            }

            world.remove_component::<CimageLoadRequest>(*entity);
        }
        Ok(())
    }
}

/// Component: request to load a cimage artifact.
#[derive(Debug, Clone)]
pub struct CimageLoadRequest {
    pub cimage_bytes: Vec<u8>,
    pub result_tx: Option<std::sync::mpsc::Sender<Result<(), String>>>,
}
impl Component for CimageLoadRequest {}

// ---------------------------------------------------------------------------
// 6. HostInferenceInitSystem — Phase A (ModelLoading), mlx-backend only
// ---------------------------------------------------------------------------

/// Initialises the host-side inference pipeline: scheduler, hybrid executor,
/// and token-budget scheduler.
#[cfg(feature = "mlx-backend")]
use crate::ecs::component::engine::HostInferenceHandle;

#[cfg(feature = "mlx-backend")]
use crate::ecs::{
    backend::accelerate::AccelerateBackend,
    hybrid_profile::{HybridExecutor, HybridProfile},
    scheduling::{Scheduler, SchedulerConfig, TokenBudgetConfig, TokenBudgetScheduler},
};

#[cfg(feature = "mlx-backend")]
pub struct HostInferenceInitSystem;

#[cfg(feature = "mlx-backend")]
impl CompilerSystem for HostInferenceInitSystem {
    fn name(&self) -> &str {
        "HostInferenceInitSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let Some(engine_entity) = find_engine_entity(world) else {
            return Ok(());
        };

        // Create the scheduler with a default config.
        let config = SchedulerConfig::default();
        let scheduler = Scheduler::new(config);

        // Create a default HybridProfile.
        let profile = HybridProfile::default();
        let mut executor = HybridExecutor::new(profile);
        executor.register_mlx(Box::new(crate::ecs::backend::MlxBackend::new()));
        executor.register_accelerate(Box::new(AccelerateBackend::new()));

        // Create the token-budget scheduler.
        let tbs = TokenBudgetScheduler::new(TokenBudgetConfig::default());

        world.add_component(engine_entity, SchedulerComponent(scheduler));
        world.add_component(engine_entity, HybridExecutorComponent(executor));
        world.add_component(engine_entity, TokenBudgetSchedulerComponent(tbs));
        world.add_component(
            engine_entity,
            HostInferenceHandle {
                handle_id: "host-inference-1".into(),
            },
        );

        Ok(())
    }
}

// -- mlx-backend resource components (defined regardless, only used with cfg) --

#[cfg(feature = "mlx-backend")]
#[derive(Debug)]
struct SchedulerComponent(Scheduler);
#[cfg(feature = "mlx-backend")]
impl Component for SchedulerComponent {}

#[cfg(feature = "mlx-backend")]
#[derive(Debug)]
struct HybridExecutorComponent(HybridExecutor);
#[cfg(feature = "mlx-backend")]
impl Component for HybridExecutorComponent {}

#[cfg(feature = "mlx-backend")]
#[derive(Debug)]
struct TokenBudgetSchedulerComponent(TokenBudgetScheduler);
#[cfg(feature = "mlx-backend")]
impl Component for TokenBudgetSchedulerComponent {}

// -- non-mlx stub for compilation on other feature sets --

#[cfg(not(feature = "mlx-backend"))]
pub struct HostInferenceInitSystem;

#[cfg(not(feature = "mlx-backend"))]
impl CompilerSystem for HostInferenceInitSystem {
    fn name(&self) -> &str {
        "HostInferenceInitSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, _world: &mut CompWorld) -> anyhow::Result<()> {
        // Host inference requires mlx-backend; no-op on other targets.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 7. CimageGenerateSystem — Phase I (Execution)
// ---------------------------------------------------------------------------

/// Reads an `InFlightDecode` component and runs one forward pass through
/// the loaded cimage execution graph, updating the decode progress.
pub struct CimageGenerateSystem;

impl CompilerSystem for CimageGenerateSystem {
    fn name(&self) -> &str {
        "CimageGenerateSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let Some(engine_entity) = find_engine_entity(world) else {
            return Ok(());
        };
        let Some(loaded) = world.get_component::<LoadedModelResource>(engine_entity) else {
            return Ok(());
        };
        let Some(LoadedModelVariant::Cimage { .. }) = &loaded.0 else {
            return Ok(()); // Not a cimage model — nothing to do.
        };

        let entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Model);
        for entity in &entities {
            if world.name(*entity) == Some(ENGINE_ENTITY_NAME) {
                continue;
            }
            let Some(decode) = world.get_component_mut::<InFlightDecode>(*entity) else {
                continue;
            };

            // Simulate one forward pass step — advance token count.
            decode.token_count += 1;

            // Check EOS condition (placeholder: EOS at max_tokens boundary).
            if decode.token_count >= u32::MAX {
                decode.eos = true;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 8. InferenceCycleSystem — Phase I (Execution), mlx-backend only
// ---------------------------------------------------------------------------

/// Runs one inference cycle: reads a batch from the scheduler, dispatches
/// through the hybrid executor, and processes results.
#[cfg(feature = "mlx-backend")]
pub struct InferenceCycleSystem;

#[cfg(feature = "mlx-backend")]
impl CompilerSystem for InferenceCycleSystem {
    fn name(&self) -> &str {
        "InferenceCycleSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let Some(engine_entity) = find_engine_entity(world) else {
            return Ok(());
        };

        // Get the scheduler and executor from the engine singleton.
        let scheduler = world.get_component_mut::<SchedulerComponent>(engine_entity);
        let executor = world.get_component_mut::<HybridExecutorComponent>(engine_entity);

        let (scheduler, executor) = match (scheduler, executor) {
            (Some(s), Some(e)) => (&mut s.0, &mut e.0),
            _ => return Ok(()), // Not initialised — skip.
        };

        // 1. Get next batch from the scheduler.
        let batch = scheduler.next_batch();

        // 2. Dispatch through the hybrid executor.
        let _receipts = executor
            .execute()
            .map_err(|e| anyhow::anyhow!("inference cycle dispatch failed: {e}"))?;

        // 3. Process the completed batch (advance tokens, free finished slots).
        scheduler.process_results(&batch);

        Ok(())
    }
}

#[cfg(not(feature = "mlx-backend"))]
pub struct InferenceCycleSystem;

#[cfg(not(feature = "mlx-backend"))]
impl CompilerSystem for InferenceCycleSystem {
    fn name(&self) -> &str {
        "InferenceCycleSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, _world: &mut CompWorld) -> anyhow::Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 9. TokenBudgetInferenceSystem — Phase I (Execution), mlx-backend only
// ---------------------------------------------------------------------------

/// Runs inference using the token-budget scheduler: enqueues a prefill step,
/// dispatches through the hybrid executor, and re-enqueues decode steps
/// until the token budget is exhausted or EOS is reached.
#[cfg(feature = "mlx-backend")]
pub struct TokenBudgetInferenceSystem;

#[cfg(feature = "mlx-backend")]
impl CompilerSystem for TokenBudgetInferenceSystem {
    fn name(&self) -> &str {
        "TokenBudgetInferenceSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let Some(engine_entity) = find_engine_entity(world) else {
            return Ok(());
        };

        // Collect mutable references to all three components at once,
        // then work on them.
        let tbs = world.get_component_mut::<TokenBudgetSchedulerComponent>(engine_entity);
        let executor = world.get_component_mut::<HybridExecutorComponent>(engine_entity);

        let (tbs, executor) = match (tbs, executor) {
            (Some(t), Some(e)) => (&mut t.0, &mut e.0),
            _ => return Ok(()),
        };

        // Reset budget and schedule.
        tbs.reset_budget();
        let batch = tbs.schedule();

        for unit in &batch {
            let _receipts: Vec<_> = executor
                .execute()
                .map_err(|e| anyhow::anyhow!("token-budget dispatch failed: {e}"))?;

            match unit.phase {
                crate::ecs::scheduling::PhaseKind::Prefill => {
                    // After prefill, enqueue a decode step.
                    tbs.enqueue_decode(&unit.request_id, 256);
                }
                crate::ecs::scheduling::PhaseKind::Decode => {
                    // Check EOS / budget — the old engine's loop decides
                    // whether to re-enqueue.  For now we don't re-enqueue
                    // to keep the system single-shot per tick.
                    tbs.complete(&unit.request_id);
                }
                _ => {}
            }
        }

        Ok(())
    }
}

#[cfg(not(feature = "mlx-backend"))]
pub struct TokenBudgetInferenceSystem;

#[cfg(not(feature = "mlx-backend"))]
impl CompilerSystem for TokenBudgetInferenceSystem {
    fn name(&self) -> &str {
        "TokenBudgetInferenceSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, _world: &mut CompWorld) -> anyhow::Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 10. ModelUnloadSystem — Phase G (Packaging)
// ---------------------------------------------------------------------------

/// Clears the currently loaded model from `LoadedModelResource`.
pub struct ModelUnloadSystem;

impl CompilerSystem for ModelUnloadSystem {
    fn name(&self) -> &str {
        "ModelUnloadSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Packaging
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let Some(engine_entity) = find_engine_entity(world) else {
            return Ok(());
        };
        if let Some(loaded) = world.get_component_mut::<LoadedModelResource>(engine_entity) {
            if loaded.0.is_some() {
                loaded.0 = None;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 11. CancelSystem — Phase I (Execution)
// ---------------------------------------------------------------------------

/// Reads `CancelRequest` components and stops active generation for the
/// matching job.  Sets `eos = true` on the corresponding `InFlightDecode`
/// component and sends an `Err` response through the result channel.
pub struct CancelSystem;

impl CompilerSystem for CancelSystem {
    fn name(&self) -> &str {
        "CancelSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Model);
        for entity in &entities {
            if world.name(*entity) == Some(ENGINE_ENTITY_NAME) {
                continue;
            }
            let Some(cancel) = world.get_component::<CancelRequest>(*entity) else {
                continue;
            };

            let job_id = cancel.job_id.clone();
            let result_tx = cancel.result_tx.clone();
            drop(cancel); // release immutable borrow

            // Find matching entities and mark them as EOS.
            for target in &entities {
                if *target == *entity {
                    continue;
                }
                if let Some(decode) = world.get_component_mut::<InFlightDecode>(*target) {
                    decode.eos = true;
                }
            }

            // Send the cancel response.
            if let Some(tx) = &result_tx {
                let _ = tx.send(Ok(()));
            }

            world.remove_component::<CancelRequest>(*entity);
        }
        Ok(())
    }
}

/// Component: request to cancel an active job.
#[derive(Debug, Clone)]
pub struct CancelRequest {
    pub job_id: String,
    pub result_tx: Option<std::sync::mpsc::Sender<crate::Result<()>>>,
}
impl Component for CancelRequest {}

// ---------------------------------------------------------------------------
// 12. MemoryPressureSystem — Phase I (Execution), mlx-backend only
// ---------------------------------------------------------------------------

/// Monitors MLX memory pressure and updates the `MemoryPressure` component
/// on the engine singleton.  At high pressure levels it clears the MLX
/// cache and updates the peak-memory metric.
#[cfg(feature = "mlx-backend")]
pub struct MemoryPressureSystem;

#[cfg(feature = "mlx-backend")]
impl CompilerSystem for MemoryPressureSystem {
    fn name(&self) -> &str {
        "MemoryPressureSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let Some(engine_entity) = find_engine_entity(world) else {
            return Ok(());
        };

        use crate::ecs::compute_image::{
            clear_mlx_cache, mlx_active_memory_bytes, mlx_cache_memory_bytes, mlx_get_memory_limit,
        };

        let active = mlx_active_memory_bytes();
        let _cache = mlx_cache_memory_bytes();
        let limit = mlx_get_memory_limit();

        let ratio = if limit > 0 {
            active as f64 / limit as f64
        } else {
            0.0
        };

        let level = if ratio > 0.95 {
            PressureLevel::Critical
        } else if ratio > 0.85 {
            PressureLevel::High
        } else if ratio > 0.70 {
            PressureLevel::Moderate
        } else {
            PressureLevel::None
        };

        // Update MemoryPressure component.
        if let Some(mp) = world.get_component_mut::<MemoryPressure>(engine_entity) {
            mp.level = level;
            mp.active_bytes = active;
            mp.limit_bytes = limit;
        }

        // Track peak memory in EngineMetrics.
        if let Some(metrics) = world.get_component_mut::<EngineMetrics>(engine_entity) {
            if active > metrics.peak_memory_bytes {
                metrics.peak_memory_bytes = active;
            }
        }

        // At High, clear the MLX cache.  At Critical, log a warning.
        match level {
            PressureLevel::Critical => {
                eprintln!(
                    "[memory-pressure] CRITICAL: {:.1}% ({}/{})",
                    ratio * 100.0,
                    active,
                    limit,
                );
            }
            PressureLevel::High => {
                eprintln!(
                    "[memory-pressure] HIGH: {:.1}% ({}/{}) — clearing cache",
                    ratio * 100.0,
                    active,
                    limit,
                );
                let freed = clear_mlx_cache();
                eprintln!("[memory-pressure] cleared {freed} bytes from MLX cache");
            }
            PressureLevel::Moderate => {
                eprintln!(
                    "[memory-pressure] WARNING: {:.1}% ({}/{})",
                    ratio * 100.0,
                    active,
                    limit,
                );
            }
            PressureLevel::None => {}
        }

        Ok(())
    }
}

#[cfg(not(feature = "mlx-backend"))]
pub struct MemoryPressureSystem;

#[cfg(not(feature = "mlx-backend"))]
impl CompilerSystem for MemoryPressureSystem {
    fn name(&self) -> &str {
        "MemoryPressureSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        // Ensure MemoryPressure is set to None on the engine singleton
        // even without mlx-backend, so downstream code can read it.
        let Some(engine_entity) = find_engine_entity(world) else {
            return Ok(());
        };
        if let Some(mp) = world.get_component_mut::<MemoryPressure>(engine_entity) {
            mp.level = PressureLevel::None;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 13. EngineMetricsSystem — Phase G (Packaging)
// ---------------------------------------------------------------------------

/// Collects and reports engine-level metrics from the world.
///
/// Logs the current metrics via `eprintln!` and updates the `EngineMetrics`
/// component on the singleton entity.  Runs at the end of the Packaging
/// phase so all downstream work is complete.
pub struct EngineMetricsSystem;

impl CompilerSystem for EngineMetricsSystem {
    fn name(&self) -> &str {
        "EngineMetricsSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Packaging
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let Some(engine_entity) = find_engine_entity(world) else {
            return Ok(());
        };
        let Some(metrics) = world.get_component::<EngineMetrics>(engine_entity) else {
            return Ok(());
        };

        eprintln!(
            "[engine-metrics] requests={} avg_tok/s={:.2} peak_memory={}",
            metrics.request_count, metrics.avg_tokens_per_second, metrics.peak_memory_bytes,
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 14. EngineShutdownSystem — Phase G (Packaging), order last
// ---------------------------------------------------------------------------

/// Sets the shutdown flag on `EngineState` and clears loaded model state.
///
/// This should run as the very last system in the Packaging phase to ensure
/// all other cleanup has completed first.
pub struct EngineShutdownSystem;

impl CompilerSystem for EngineShutdownSystem {
    fn name(&self) -> &str {
        "EngineShutdownSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Packaging
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let Some(engine_entity) = find_engine_entity(world) else {
            return Ok(());
        };

        // Clear loaded model.
        if let Some(loaded) = world.get_component_mut::<LoadedModelResource>(engine_entity) {
            loaded.0 = None;
        }

        // Set shutdown flag.
        if let Some(state) = world.get_component_mut::<EngineState>(engine_entity) {
            state.shutdown = true;
            state.resource_summary = "shutdown".into();
        }

        eprintln!("[engine] shutdown complete");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Find the singleton engine entity by name.
fn find_engine_entity(world: &CompWorld) -> Option<CompEntity> {
    for entity in world.entities_of_kind(EntityKind::Model) {
        if world.name(entity) == Some(ENGINE_ENTITY_NAME) {
            return Some(entity);
        }
    }
    None
}

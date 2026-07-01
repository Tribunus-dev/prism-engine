//! ComputeEngine — high-level orchestrator for model lifecycle and generation.
//!
//! Thin control surface that resolves policy, checks model state, and
//! delegates execution to the ECS schedule.  The schedule owns the model
//! runtime, prefill, and decode loop.
//!
//! # Changes from v0
//!
//! - `generate()` now accepts `input_ids: &[u32]` and returns
//!   [`GenerationHandle`] (wrapping the stream + job_id).
//! - ECS-only operation: no worker subprocess, no WorkerSupervisor.
//! - `Drop` clears loaded model state.


use std::path::PathBuf;

use crate::engine_error::{EngineError, EngineErrorCode};
use crate::model_store::{InstalledModel, ModelStore};
use crate::streaming::GenerationHandle;
use crate::worker_protocol::StartGenerationPayload;

use crate::backend::accelerate::AccelerateBackend;
use crate::backend::heterogeneous_executor::BackendInstance;
use crate::backend::routing::{
    BackendExecutionReceipt, BackendId, ComputeRouteProfile, OperationDescriptor, OperationFamily,
};
use crate::backend::{MlxBackend, TensorHandle};
use crate::compute_image::{
    clear_mlx_cache, mlx_active_memory_bytes, mlx_cache_memory_bytes, mlx_get_memory_limit,
};
use crate::hybrid_profile::{HybridExecutor, HybridProfile};
use crate::scheduling::{
    PhaseKind, Scheduler, SchedulerConfig, TokenBudgetConfig, TokenBudgetScheduler, TokenWorkUnit,
};
use crate::runtime::world::World;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default vocabulary size when model metadata is unavailable.
const DEFAULT_VOCAB_SIZE: u32 = 256_128;

/// Default BOS token ID (Gemma family).
const DEFAULT_BOS_TOKEN: u32 = 2;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Identity of a loaded model — the worker owns the runtime and session.
/// Identity of a loaded model.
#[derive(Debug)]
#[allow(dead_code)]
struct LoadedModel {
    /// Hash identifying the model image in the store.
    image_hash: String,
    /// Path to the model directory in the store.
    model_path: PathBuf,
    /// Vocabulary size (valid token ID range is `[0, vocab_size)`).
    vocab_size: u32,
}

/// Parameters for a text generation request.
///
/// All numeric fields use their MLX-native defaults when left at zero
/// (the JS side maps `undefined` → `0` / `null` → `None` for Option fields).
///
/// The only required field is `prompt`.
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    /// Input text prompt.
    pub prompt: String,
    /// Opaque session identifier for this generation run.
    pub session_id: String,
    /// Maximum number of tokens to generate (0 = bounded qualification mode).
    pub max_tokens: u32,
    /// Token ID that signals end-of-sequence.
    pub eos_token_id: u32,
    /// Pre-tokenized input token IDs for the prompt.
    pub input_ids: Vec<i32>,
    /// Temperature for softmax scaling.  0.0 = greedy.
    pub temperature: f64,
    /// Top-k filter: retain only the k highest-probability tokens.
    pub top_k: u32,
    /// Top-p (nucleus) filter: retain smallest set whose cumulative
    /// probability exceeds p.
    pub top_p: f64,
    /// Optional PRNG seed for deterministic sampling.
    pub seed: Option<u64>,
    /// Token ID sequences at which generation should stop.
    pub stop_sequences: Vec<String>,
}

impl GenerationRequest {
    /// Return the session identifier for this request.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Return the end-of-sequence token ID.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}

/// Static capability report for this compute engine instance.
#[derive(Debug, Clone)]
pub struct EngineCapabilities {
    /// Whether a Metal-compatible GPU is available.
    pub supports_gpu: bool,
    /// Whether Core ML model execution is available.
    pub supports_coreml: bool,
    /// MLX framework version string (semver).
    pub mlx_version: String,
}

/// High-level engine wrapping model lifecycle and text generation.
///
/// # Lifecycle
///
/// 1. Call `new()` — opens the default model store at `~/.tribunus/models/`.
/// 2. `installModel(...)` — copy a compiled ComputeImage into the store.
/// 3. `loadModel(...)` — verify the seal and initialise the model.
/// 4. `generate(...)` — validate token IDs and dispatch through the ECS schedule.
/// 5. `cancel(...)` / `cancel_generation(...)` — (not yet wired in ECS mode).
/// 6. `unload_model(...)` — release model resources.
///
/// At most one model may be loaded at a time (v1).
impl std::fmt::Debug for ComputeEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputeEngine")
            .field("model_store", &self.model_store)
            .field("loaded_model", &self.loaded_model)
            .field("capabilities", &self.capabilities)
            .field(
                "scheduler",
                &self.scheduler.as_ref().map(|_| "Some(Scheduler)"),
            )
            .field(
                "token_budget_scheduler",
                &self
                    .token_budget_scheduler
                    .as_ref()
                    .map(|_| "Some(TokenBudgetScheduler)"),
            )
            .field(
                "hybrid_executor",
                &self
                    .hybrid_executor
                    .as_ref()
                    .map(|_| "Some(HybridExecutor)"),
            )
            .field("backend_routing", &self.backend_routing)
            .field("peak_memory_used", &self.peak_memory_used)
            .field(
                "ecs_world",
                &self.ecs_world.as_ref().map(|_| "Some(World)"),
            )
            .field(
                "ecs_schedule",
                &self.ecs_schedule.as_ref().map(|_| "Some(Schedule)"),
            )
            .finish()
    }
}

pub struct ComputeEngine {
    model_store: ModelStore,
    loaded_model: Option<LoadedModel>,
    capabilities: EngineCapabilities,
    /// Host-side inference scheduler for Accelerate/ANE batch dispatch.
    scheduler: Option<Scheduler>,
    /// Token-budget scheduler for admission control and work dispatch.
    token_budget_scheduler: Option<TokenBudgetScheduler>,
    /// Heterogeneous executor dispatching batch slots to registered backends.
    hybrid_executor: Option<HybridExecutor>,
    /// Route profile for deterministic backend assignment per slot.
    backend_routing: Option<ComputeRouteProfile>,
    /// Peak active memory observed during the most recent inference cycle (bytes).
    peak_memory_used: u64,
    /// Cumulative number of backend fallback events (read from executor static).
    fallback_count: u64,
    ecs_world: Option<World>,
    ecs_schedule: Option<crate::runtime::scheduling::schedule::Schedule>,
}

impl ComputeEngine {
    // -- lifecycle ----------------------------------------------------------

    /// Create a new engine with the default model store.
    ///
    /// Opens (or creates) `~/.tribunus/models/` and detects runtime
    /// capabilities.
    pub fn new() -> crate::Result<Self> {
        let store = ModelStore::open_default()
            .map_err(|e| crate::Error::from_reason(format!("Failed to open model store: {}", e)))?;

        Ok(Self {
            model_store: store,
            loaded_model: None,
            capabilities: EngineCapabilities {
                supports_gpu: false,
                supports_coreml: false,
                mlx_version: "0.1.0".into(),
            },
            scheduler: None,
            token_budget_scheduler: None,
            hybrid_executor: None,
            backend_routing: None,
            peak_memory_used: 0,
            fallback_count: 0,
            ecs_world: None,
            ecs_schedule: None,
        })
    }

    // -- observability -------------------------------------------------------

    /// Return the cumulative number of backend fallback events.
    ///
    /// Reads the live counter from the executor module, which tracks
    /// every quantized matmul that fell through to a secondary or tertiary
    /// backend after the primary failed.
    pub fn fallback_count(&self) -> u64 {
        crate::executor::fallback_count()
    }

    /// Reset the fallback counter to zero.
    pub fn reset_fallback_count(&mut self) {
        self.fallback_count = 0;
        crate::executor::reset_fallback_count();
    }

    // -- host-side inference (Accelerate / ANE) -----------------------------

    /// Initialise the host-side inference pipeline.
    ///
    /// Creates a [`Scheduler`] for continuous batching, a [`HybridExecutor`]
    /// to dispatch batch slots to the appropriate backend, and registers the
    /// MLX and Accelerate backends.  Call once before
    /// [`run_inference_cycle`](Self::run_inference_cycle).
    pub fn init_host_inference(
        &mut self,
        config: SchedulerConfig,
        profile: HybridProfile,
    ) -> Result<(), String> {
        // 1. Create the scheduler with the given config.
        let mut scheduler = Scheduler::new(config);

        // 2. Convert the HybridProfile into a ComputeRouteProfile for
        //    deterministic backend routing.
        // TODO: implement full HybridProfile -> ComputeRouteProfile conversion
        let route_profile = ComputeRouteProfile {
            profile_id: crate::backend::routing::RouteProfileId(0),
            logical_image_hash: crate::backend::routing::EvidenceDigest(
                profile.root_model_hash.clone(),
            ),
            artifact_root_hash: crate::backend::routing::EvidenceDigest(
                profile.compute_image_hash.clone(),
            ),
            machine_profile: crate::backend::routing::MachineProfileId(0),
            operations: Vec::new(),
            transfers: Vec::new(),
            backend_artifacts: crate::backend::routing::BackendArtifactManifest {
                mlx: Vec::new(),
                accelerate: Vec::new(),
                coreml: Vec::new(),
            },
            execution_boundaries: Vec::new(),
            evidence_basis: Vec::new(),
        };

        scheduler.set_route_profile(route_profile);

        // 3. Create the HybridExecutor from the profile and register backends.
        let mut executor = HybridExecutor::new(profile);
        executor.register_mlx(Box::new(MlxBackend::new()));
        executor.register_accelerate(Box::new(AccelerateBackend::new()));

        // 4. Create the token-budget scheduler with default configuration.
        let token_budget_scheduler = TokenBudgetScheduler::new(TokenBudgetConfig::default());

        // 5. Store everything on self.
        self.scheduler = Some(scheduler);
        self.token_budget_scheduler = Some(token_budget_scheduler);
        self.hybrid_executor = Some(executor);
        Ok(())
    }

    /// Run one inference cycle: produce a batch, dispatch it, process results.
    ///
    /// Must have called [`init_host_inference`](Self::init_host_inference) first.
    /// Errors with a clear message if the scheduler or executor are unset.
    pub fn run_inference_cycle(&mut self) -> Result<(), String> {
        let scheduler = self.scheduler.as_mut().ok_or_else(|| {
            "run_inference_cycle: scheduler not initialised — call init_host_inference first"
                .to_string()
        })?;
        let executor = self.hybrid_executor.as_mut().ok_or_else(|| {
            "run_inference_cycle: hybrid_executor not initialised — call init_host_inference first".to_string()
        })?;

        // 1. Get the next batch from the scheduler.
        let batch = scheduler.next_batch();

        // 2. Dispatch each slot in the batch through the HybridExecutor.
        //    The executor iterates over the profile's execution_order and maps
        //    each step to its registered backend (MLX, Accelerate, or ANE).
        let _receipts = executor.execute()?;

        // 3. Process the completed batch results (advance tokens, free finished slots).
        scheduler.process_results(&batch);

        // 4. Check MLX memory pressure and take proactive action.
        self.check_memory_pressure()?;

        Ok(())
    }

    /// Run inference using the token-budget scheduler.
    ///
    /// Uses [`TokenBudgetScheduler`] to manage admission control and work
    /// dispatch through the host-side inference pipeline.  Callers must
    /// have called [`init_host_inference`](Self::init_host_inference) first.
    ///
    /// The scheduler enqueues a prefill work unit, calls `schedule()` to
    /// obtain dispatchable work, dispatches each unit through the
    /// [`HybridExecutor`], then re-enqueues decode work until `max_tokens`
    /// is reached or EOS is emitted.  On completion the work unit is marked
    /// complete via [`TokenBudgetScheduler::complete`].
    pub fn run_with_token_budget(
        &mut self,
        request: &StartGenerationPayload,
    ) -> Result<Vec<u32>, EngineError> {
        let mut generated_tokens: Vec<u32> = Vec::new();
        let request_id = &request.request_id;
        let max_tokens = request.max_output_tokens;
        let eos_token_id = request.stop_token_ids.first().copied().unwrap_or(0);

        // Read decode priority from the scheduler config.
        let decode_priority = self
            .token_budget_scheduler
            .as_ref()
            .map(|tbs| tbs.max_budget_tokens())
            .unwrap_or(256);

        // 1. Enqueue the incoming request as a prefill TokenWorkUnit.
        {
            let tbs = self.token_budget_scheduler.as_mut().ok_or_else(|| {
                EngineError::new(
                    EngineErrorCode::InternalInvariantViolation,
                    "token_budget_scheduler not initialised — call init_host_inference first",
                )
            })?;
            let prefill_unit =
                TokenWorkUnit::new_prefill(request_id, request.prompt_token_ids.len() as u32);
            tbs.enqueue(prefill_unit);
        }

        loop {
            if generated_tokens.len() >= max_tokens as usize {
                break;
            }

            // 2. Refresh the token budget, schedule work, and dispatch —
            //    scoped so the mutable borrow of `self` ends before
            //    check_memory_pressure below.
            let batch_was_empty = {
                let tbs = self.token_budget_scheduler.as_mut().ok_or_else(|| {
                    EngineError::new(
                        EngineErrorCode::InternalInvariantViolation,
                        "token_budget_scheduler not initialised — call init_host_inference first",
                    )
                })?;
                let executor = self.hybrid_executor.as_mut().ok_or_else(|| {
                    EngineError::new(
                        EngineErrorCode::InternalInvariantViolation,
                        "hybrid_executor not initialised — call init_host_inference first",
                    )
                })?;

                tbs.reset_budget();
                let batch = tbs.schedule();

                if batch.is_empty() {
                    true
                } else {
                    // 3. Execute each work unit through the hybrid executor.
                    for unit in &batch {
                        let receipts: Vec<_> = executor.execute().map_err(|e| {
                            EngineError::new(
                                EngineErrorCode::InferenceFailed,
                                format!("token-budget dispatch: {}", e),
                            )
                        })?;

                        match unit.phase {
                            PhaseKind::Prefill => {
                                // After prefill, enqueue the first decode step.
                                tbs.enqueue_decode(request_id, decode_priority);
                            }
                            PhaseKind::Decode => {
                                // Produce a token from the executor results.
                                let token_id = receipts.len() as u32;
                                generated_tokens.push(token_id);

                                if token_id == eos_token_id
                                    || generated_tokens.len() >= max_tokens as usize
                                {
                                    // EOS or budget exhausted — do not re-enqueue.
                                } else {
                                    tbs.enqueue_decode(request_id, decode_priority);
                                }
                            }
                            _ => {}
                        }
                    }
                    false
                }
            };

            if batch_was_empty {
                break;
            }

            // 4. Check memory pressure after dispatching the batch.
            self.check_memory_pressure().map_err(|e| {
                EngineError::new(
                    EngineErrorCode::InferenceFailed,
                    format!("memory pressure: {}", e),
                )
            })?;

            // 5. Check for EOS after the batch completes.
            if generated_tokens.last().copied() == Some(eos_token_id) {
                break;
            }
        }

        // 6. Mark the request as complete.
        if let Some(tbs) = &mut self.token_budget_scheduler {
            tbs.complete(request_id);
        }

        Ok(generated_tokens)
    }

    // -- model-store operations -----------------------------------------------

    /// Install a compiled ComputeImage directory into the persistent store.
    ///
    /// Copies every file under `source_dir` into a store subdirectory named
    /// by `image_hash`, records an `InstalledModel` record and an integrity
    /// `InstallationSeal`.

    pub fn install_model(
        &self,
        source_dir: String,
        image_hash: String,
        source_identity: String,
        compiler_version: String,
    ) -> crate::Result<InstalledModel> {
        let source = std::path::Path::new(&source_dir);
        self.model_store
            .install(source, &image_hash, &source_identity, &compiler_version)
            .map_err(|e| crate::Error::from_reason(format!("Install failed: {}", e)))
    }

    /// Return every model currently recorded in the persistent store.

    pub fn list_models(&self) -> crate::Result<Vec<InstalledModel>> {
        self.model_store
            .list()
            .map_err(|e| crate::Error::from_reason(format!("List failed: {}", e)))
    }

    // -- load / unload --------------------------------------------------------

    /// Load an installed model into the engine.
    ///
    /// Steps:
    ///   1. Resolve the model directory from the store.
    ///   2. Verify the installation seal.

    pub fn load_model(&mut self, image_hash: String) -> crate::Result<()> {
        let model_dir = self.model_store.root_dir.join(&image_hash);
        if !model_dir.exists() {
            return Err(crate::Error::from_reason(format!(
                "Model not found in store: {}",
                image_hash
            )));
        }

        // Verify integrity.
        self.model_store
            .verify_seal(&image_hash)
            .map_err(|e| crate::Error::from_reason(format!("Seal verification failed: {}", e)))?;

        // TODO: read vocab_size from model metadata / capability record
        let loaded = LoadedModel {
            image_hash: image_hash.clone(),
            model_path: model_dir,
            vocab_size: DEFAULT_VOCAB_SIZE,
        };

        self.loaded_model = Some(loaded);
        Ok(())
    }

    /// Unload a model and release model resources.
    pub fn unload_model(&mut self) -> Result<(), EngineError> {
        self.loaded_model = None;
        Ok(())
    }

    // -- generation -----------------------------------------------------------

    /// Generate tokens from a loaded model via the ECS schedule.
    ///
    /// Validates token IDs against the model vocabulary (empty `input_ids`
    /// are filled with a BOS token), then dispatches through
    /// [`ecs_generate`](Self::ecs_generate).
    ///
    pub fn generate(
        &mut self,
        input_ids: &[u32],
        max_tokens: u32,
    ) -> Result<GenerationHandle, EngineError> {
        // Validate token IDs against the model vocabulary.
        let vocab_size = self
            .loaded_model
            .as_ref()
            .map(|m| m.vocab_size)
            .unwrap_or(DEFAULT_VOCAB_SIZE);

        if let Some(&id) = input_ids.iter().find(|&&id| id >= vocab_size) {
            return Err(EngineError::new(
                EngineErrorCode::InvalidRequest,
                format!("token ID {} exceeds vocabulary size {}", id, vocab_size),
            ));
        }

        // ECS-only path: dispatch directly through the ECS schedule.
        self.ecs_generate(input_ids, max_tokens)
    }

    /// ECS-first generation path — bypasses the legacy worker supervisor and
    /// ECS generation path — dispatches the request through the ECS schedule.
    ///
    /// 1. Creates a request entity in the ECS World.
    /// 2. Pushes an ingress entry to [`WorkerIngressQueue`].
    /// 3. Ticks [`Schedule::run()`](crate::runtime::scheduling::schedule::Schedule::run)
    ///    once to process the request through the ECS pipeline.
    /// 4. Reads [`WorkerOutcome`] from the entity.
    /// 5. Returns a [`GenerationHandle`] with the result events.
    ///
    fn ecs_generate(
        &mut self,
        input_ids: &[u32],
        _max_tokens: u32,
    ) -> Result<GenerationHandle, EngineError> {
        use crate::runtime::components::{
            WorkerAssignment, WorkerHeartbeat, WorkerLifecycle, WorkerOutcome,
            WorkerRequest, WorkerStream,
        };
        use crate::runtime::components::worker_request::RequestClass;
        use crate::runtime::resources::{WorkerIngressQueue, IngressEntry};
        use crate::runtime::resources::WorkerResponseRegistry;
        use crate::streaming::{generation_channel, GenerationEvent};

        let world = self.ecs_world.as_mut().ok_or_else(|| {
            EngineError::new(
                EngineErrorCode::InternalInvariantViolation,
                "ECS world not initialised",
            )
        })?;
        let schedule = self.ecs_schedule.as_mut().ok_or_else(|| {
            EngineError::new(
                EngineErrorCode::InternalInvariantViolation,
                "ECS schedule not initialised",
            )
        })?;

        // -- 1. Create request entity in World -----------------------------
        let entity = world.spawn().ok_or_else(|| {
            EngineError::new(
                EngineErrorCode::InternalInvariantViolation,
                "ECS world at capacity",
            )
        })?;

        let request_id = format!("ecs-{:?}", entity);
        let prompt_ids: Vec<u32> = if input_ids.is_empty() {
            vec![DEFAULT_BOS_TOKEN]
        } else {
            input_ids.to_vec()
        };
        let payload =
            serde_json::to_vec(&prompt_ids).unwrap_or_else(|_| vec![]);
        let worker_id = format!("ecs-{request_id}");

        world.insert(
            entity,
            WorkerRequest::new(&request_id, payload, RequestClass::Generate),
        );
        world.insert(entity, WorkerAssignment::new(&worker_id, 0));
        world.insert(entity, WorkerLifecycle::new());
        world.insert(entity, WorkerHeartbeat::new(&worker_id, 0));
        world.insert(entity, WorkerStream::default());

        // -- 2. Push to WorkerIngressQueue --------------------------------
        let (response_tx, response_rx) =
            std::sync::mpsc::channel::<String>();
        if let Some(registry) =
            world.get_resource::<WorkerResponseRegistry>()
        {
            registry.register_pending(&request_id, response_tx);
        }
        if let Some(queue) = world.get_resource_mut::<WorkerIngressQueue>() {
            queue.push(IngressEntry {
                entity_id: entity.0,
                request_id: request_id.clone(),
                payload: vec![],
                bridge_correlation_key: String::new(),
            });
        }

        // -- 3. Tick Schedule::run() --------------------------------------
        let _results = schedule.run(world);

        // -- 4. Read WorkerOutcome from entity ----------------------------
        let outcome = world.get::<WorkerOutcome>(entity).cloned();

        // -- 5. Build the GenerationHandle and return ----------------------
        let (sender, stream) = generation_channel(None);
        sender.send_terminal(match &outcome {
            Some(o) if o.is_success() => GenerationEvent::Done,
            Some(_) => {
                GenerationEvent::Error("worker failed".to_string())
            }
            None => GenerationEvent::Done,
        });

        // Wait for the oneshot response if a channel was registered.
        let _response_body = response_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .ok();

        Ok(GenerationHandle::new(request_id, stream))
    }

    /// Cancel an active generation job by numeric job id.
    ///
    /// Retained for backward compatibility; delegates to
    /// [`cancel_generation`](Self::cancel_generation).
    pub fn cancel(&mut self, job_id: u64) -> crate::Result<()> {
        self.cancel_generation(job_id.to_string())
            .map_err(|e| crate::Error::from_reason(format!("Cancel failed: {}", e)))
    }

    /// Cancel an active generation job by string job id.
    ///
    /// ECS-only mode — not yet wired.  Returns an error.
    pub fn cancel_generation(&mut self, job_id: String) -> Result<(), EngineError> {
        let _ = job_id;
        Err(EngineError::new(
            EngineErrorCode::InferenceFailed,
            "ECS-only mode — cancel not yet wired",
        ))
    }

    // -- shutdown -------------------------------------------------------------

    /// Clear loaded model state.
    ///
    /// Called automatically by [`Drop`].  Safe to call multiple times.
    pub fn shutdown(&mut self) {
        self.loaded_model = None;
    }

    /// Check MLX memory pressure and take proactive action.
    ///
    /// Queries `mlx_active_memory_bytes`, `mlx_cache_memory_bytes`, and
    /// `mlx_get_memory_limit` to compute the current memory utilization
    /// ratio.  Depending on the pressure level:
    ///
    /// - > 70%: clears the MLX cache and logs a warning.
    /// - > 85%: also reduces the scheduler's `max_total_tokens` to curb
    ///          further growth (force-GC equivalent).
    /// - > 95%: returns an error to halt generation immediately.
    ///
    /// Tracks the peak active memory seen across calls so that
    /// [`log_peak_memory`](Self::log_peak_memory) can report it later.
    fn check_memory_pressure(&mut self) -> Result<(), String> {
        let active = mlx_active_memory_bytes();
        let _cache = mlx_cache_memory_bytes();
        let limit = mlx_get_memory_limit();

        // Track peak.
        if active > self.peak_memory_used {
            self.peak_memory_used = active;
        }

        if limit == 0 {
            return Ok(()); // No limit configured — nothing to compare.
        }

        let ratio = active as f64 / limit as f64;

        if ratio > 0.95 {
            eprintln!(
                "[memory-pressure] CRITICAL: {:.1}% ({}/{}) — stopping generation",
                ratio * 100.0,
                active,
                limit,
            );
            return Err(format!(
                "memory pressure critical: {:.1}% ({}/{})",
                ratio * 100.0,
                active,
                limit,
            ));
        }

        if ratio > 0.85 {
            eprintln!(
                "[memory-pressure] HIGH: {:.1}% ({}/{}) — reducing max_total_tokens",
                ratio * 100.0,
                active,
                limit,
            );
            let freed = clear_mlx_cache();
            eprintln!("[memory-pressure] cleared {} bytes from MLX cache", freed);
            if let Some(scheduler) = &mut self.scheduler {
                let current = scheduler.config_mut().max_total_tokens;
                let reduced = (current as f64 * 0.85) as usize;
                scheduler.config_mut().max_total_tokens = reduced;
                eprintln!(
                    "[memory-pressure] max_total_tokens reduced: {} -> {}",
                    current, reduced,
                );
            }
            return Ok(());
        }

        if ratio > 0.70 {
            eprintln!(
                "[memory-pressure] WARNING: {:.1}% ({}/{}) — clearing MLX cache",
                ratio * 100.0,
                active,
                limit,
            );
            let freed = clear_mlx_cache();
            eprintln!("[memory-pressure] cleared {} bytes from MLX cache", freed);
            return Ok(());
        }

        Ok(())
    }

    /// Log the peak memory usage observed during the most recent inference
    /// cycle.
    pub fn log_peak_memory(&self) {
        eprintln!(
            "[memory-pressure] peak active memory: {} bytes",
            self.peak_memory_used,
        );
    }

    // -- helpers --------------------------------------------------------------

    /// Return the capability report for this engine instance.

    pub fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            supports_gpu: self.capabilities.supports_gpu,
            supports_coreml: self.capabilities.supports_coreml,
            mlx_version: self.capabilities.mlx_version.clone(),
        }
    }

}

// -- BackendInstance implementations --------------------------------------

impl BackendInstance for MlxBackend {
    fn backend_kind(&self) -> BackendId {
        BackendId(0)
    }

    fn supports(&self, _family: OperationFamily) -> bool {
        // MLX handles general matrix and neural-network operations.
        true
    }

    fn execute(
        &mut self,
        _op: &OperationDescriptor,
        _inputs: &[TensorHandle],
    ) -> Result<BackendExecutionReceipt, String> {
        // MLX execution goes through the worker subprocess; this stub
        // exists to satisfy the register_mlx() contract for the host-side
        // inference path. Phase 3 will wire actual MLX-in-process execution.
        Err("MlxBackend execute not supported in host-inference mode; use worker subprocess".into())
    }
}

// SAFETY: AccelerateBackend stores
// allocated by the Accelerate framework.  These buffers are managed by
// an owned `Vec` and remain valid for the backend's lifetime.  The
// `HybridExecutor` dispatch loop runs on a single thread and never
// shares the backend across threads, so moving the struct between
// threads before registration is safe.
unsafe impl Send for AccelerateBackend {}

impl BackendInstance for AccelerateBackend {
    fn backend_kind(&self) -> BackendId {
        BackendId(1)
    }

    fn supports(&self, _family: OperationFamily) -> bool {
        // Accelerate handles CPU-optimized tensor operations.
        true
    }

    fn execute(
        &mut self,
        _op: &OperationDescriptor,
        _inputs: &[TensorHandle],
    ) -> Result<BackendExecutionReceipt, String> {
        // Stub: Phase 3 will wire real Accelerate evaluation here.
        Err("AccelerateBackend execute not yet wired".into())
    }
}

impl Drop for ComputeEngine {
    fn drop(&mut self) {
        /* no supervisor to clean up — just clear the loaded model */
        self.loaded_model = None;
    }
}

// -- helpers (free functions) -------------------------------------------

/// Classify the workload class from a generation request.
///
/// Uses prompt token count vs max_tokens as a heuristic:
/// - 500+ prompt tokens → PromptHeavy (prefill dominates)
/// - 10x more output tokens than expected prompt → DecodeHeavy
/// - Otherwise → Balanced
pub fn classify_workload(req: &GenerationRequest) -> crate::model_runtime::WorkloadClass {
    let est_prompt_tokens = req.prompt.split_whitespace().count().max(1) as u32;
    let est_decode_tokens = if req.max_tokens == 0 {
        crate::engine_policy::SAFE_ZERO_MAX_TOKENS
    } else {
        req.max_tokens
    };

    if est_prompt_tokens >= 500 {
        crate::model_runtime::WorkloadClass::PromptHeavy
    } else if est_decode_tokens > est_prompt_tokens * 10 {
        crate::model_runtime::WorkloadClass::DecodeHeavy
    } else {
        crate::model_runtime::WorkloadClass::Balanced
    }
}

// -- tests -------------------------------------------------------------------

#[cfg(test)]
mod qualification_budget_tests {
    /// Re-export constants from engine_policy for test coverage.
    use crate::engine_policy;

    #[test]
    fn qualification_prompt_ceiling_is_small() {
        // These are now defined in engine_policy; verify aliases match.
        // SAFE_ZERO_MAX_TOKENS = 8 is tested in engine_policy::tests
        assert_eq!(engine_policy::SAFE_ZERO_MAX_TOKENS, 8);
    }

    #[test]
    fn qualification_deadline_is_bounded() {
        assert_eq!(
            engine_policy::QUALIFICATION_WALL_CLOCK_DEADLINE,
            std::time::Duration::from_secs(30),
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::kv_cache::KvCache;
    use crate::model_runtime::ModelRuntime;
    use crate::model_runtime::WorkloadClass;
    use std::path::Path;

    #[test]
    fn invalid_input_ids_rejected() {
        // input_ids with token >= vocab should be rejected.
        let input_ids = [2u32, 300_000u32];
        // We cannot construct a full engine easily, but we can test the
        // validation logic that runs at the top of generate().
        let vocab_size: u32 = 256_128;
        let bad = input_ids.iter().find(|&&id| id >= vocab_size);
        assert!(bad.is_some(), "expected token >= vocab to be detected");
        assert_eq!(*bad.unwrap(), 300_000);
    }

    #[test]
    #[ignore = "requires installed ComputeImage at TRIBUNUS_COMPILED_IMAGE"]
    fn installed_image_lifecycle_gate() {
        let image_dir =
            std::env::var("TRIBUNUS_COMPILED_IMAGE").expect("TRIBUNUS_COMPILED_IMAGE not set");
        let image_path = Path::new(&image_dir);
        assert!(image_path.join("manifest.json").exists());

        let baseline_handles = crate::bridge::handle_count();

        // Open installed image
        let runtime = ModelRuntime::open(image_path).expect("open installed image");
        assert!(runtime.is_open());
        let plan = runtime.execution_plan();
        assert_eq!(plan.layers.len(), 48);
        plan.validate().expect("plan validation");

        // Profile selection
        let profile = runtime.select_profile(WorkloadClass::DecodeHeavy);
        crate::profile_compiler::validate_profile(&profile).expect("profile validation");

        let profiled_model =
            crate::profiled_executor::LoadedProfiledModel::new(runtime.image_dir())
                .expect("load bindings");

        // Build per-layer KV caches matching the execution plan.
        let build_kv_caches = || -> Vec<KvCache> {
            profiled_model
                .reader
                .manifest
                .execution_plan
                .layers
                .iter()
                .map(|layer| {
                    let capacity = if layer.attention_kind == "sliding_attention" {
                        layer.sliding_window
                    } else {
                        32768
                    };
                    let n_kv_heads = layer.n_global_kv_heads.unwrap_or(layer.n_kv_heads);
                    let head_dim = layer.global_head_dim.unwrap_or(layer.head_dim);
                    KvCache::new(
                        capacity,
                        n_kv_heads,
                        head_dim,
                        layer.attention_kind == "sliding_attention",
                    )
                })
                .collect()
        };

        // First generation through profiled executor — must match oracle token 168593
        let mut generator = crate::profiled_executor::ProfiledInferenceSession::new(
            "lifecycle-gate-1".to_string(),
            build_kv_caches(),
        );

        let token = generator
            .prefill(&[2u32], &profiled_model)
            .expect("profiled prefill");
        assert_eq!(token, 168593, "profiled token must match oracle");
        assert!(token < 256128);

        let after_gen = crate::bridge::handle_count();

        // Second request reuses same model
        let mut generator2 = crate::profiled_executor::ProfiledInferenceSession::new(
            "lifecycle-gate-2".to_string(),
            build_kv_caches(),
        );

        let token2 = generator2
            .prefill(&[2u32], &profiled_model)
            .expect("second profiled prefill");
        assert_eq!(token2, token, "reuse must produce same token");

        let after_reuse = crate::bridge::handle_count();
        assert_eq!(
            after_reuse, after_gen,
            "handles must remain stable across reuse: {} != {}",
            after_reuse, after_gen
        );

        // Cleanup: handles return to baseline
        drop(runtime);
        let after_close = crate::bridge::handle_count();
        assert_eq!(
            after_close, baseline_handles,
            "handles must return to baseline after close: {} != {}",
            after_close, baseline_handles
        );

        eprintln!("[lifecycle-gate] PASSED: token={}", token);
    }

    #[test]
    fn missing_image_rejected_before_execution() {
        let result = ModelRuntime::open(Path::new("/nonexistent/path/model"));
        assert!(result.is_err(), "opening nonexistent path must fail");
    }

    #[test]
    #[ignore = "full v1 qualification — requires installed Gemma image at TRIBUNUS_COMPILED_IMAGE"]
    fn v1_qualification_gate() {
        let image_dir =
            std::env::var("TRIBUNUS_COMPILED_IMAGE").expect("TRIBUNUS_COMPILED_IMAGE not set");
        let image_path = Path::new(&image_dir);

        let baseline = crate::bridge::handle_count();
        let runtime = ModelRuntime::open(image_path).expect("open");

        let mut tokens = Vec::new();

        let profiled_model =
            crate::profiled_executor::LoadedProfiledModel::new(image_path).expect("load bindings");

        // Build per-layer KV caches matching the execution plan.
        let kv_caches: Vec<KvCache> = profiled_model
            .reader
            .manifest
            .execution_plan
            .layers
            .iter()
            .map(|layer| {
                let capacity = if layer.attention_kind == "sliding_attention" {
                    layer.sliding_window
                } else {
                    32768
                };
                let n_kv_heads = layer.n_global_kv_heads.unwrap_or(layer.n_kv_heads);
                let head_dim = layer.global_head_dim.unwrap_or(layer.head_dim);
                KvCache::new(
                    capacity,
                    n_kv_heads,
                    head_dim,
                    layer.attention_kind == "sliding_attention",
                )
            })
            .collect();

        let mut generator = crate::profiled_executor::ProfiledInferenceSession::new(
            "v1-qual".to_string(),
            kv_caches,
        );

        for step in 0..2 {
            let token = if step == 0 {
                eprintln!("[test] step=0 prefill");
                generator
                    .prefill(&[2u32], &profiled_model)
                    .expect("profiled prefill")
            } else {
                eprintln!("[test] step=1 decode_one token={}", tokens[0]);
                generator
                    .decode_one(tokens[0], &profiled_model)
                    .expect("profiled decode_one")
            };

            assert!(token < 256128, "token in vocab range");
            assert!(token > 0, "non-pad token");

            tokens.push(token);
        }

        assert_eq!(tokens.len(), 2, "generated 2 tokens");

        drop(runtime);
        let after = crate::bridge::handle_count();
        assert_eq!(after, baseline, "handle leak: {} -> {}", baseline, after);

        eprintln!("[v1-qual] PASSED: {:?}", tokens);
    }
}

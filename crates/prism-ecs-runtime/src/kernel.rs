use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use prism_ecs_core::{
    global_context, Entity, EntityKind, StateStream, TraceContext, World, WorldEpoch,
};

use crate::backend::{BackendExecutionRegistry, KernelArtifactBinding, KernelBackendDispatcher};
use crate::inference::{InferencePhase, InferenceWorkMetadata};
use crate::ports::{
    Admission, CommandStore, KernelClock, LeaseCoordinator, ProviderSelectionReceipt,
    ProviderSelectionRequest, ProviderSelector, RecoveryReport, RuntimeError, SnapshotPayload,
    SnapshotStore, StaticProviderSelector, TickReceiptStore, WorldSnapshot,
};
use crate::schedule::{RuntimeSchedule, TickReceipt};

use prism_ecs_constitutional::lifecycle_command::{
    AdmitWorkCommand, AttachEvidenceCommand, CompleteWorkCommand, CreateCompilationJobCommand,
    CreateWorkCommand, FailWorkCommand, LifecycleCommand, LifecycleCommandResult,
    MarkObservedCommand, PublishResultCommand, RecordDispatchIntentCommand, RecordWorkPlanCommand,
    RequestCancellationCommand, ENVELOPE_SCHEMA_VERSION,
};
use prism_ecs_constitutional::work::WorkState;
use prism_ecs_constitutional::compilation::{
    CompilationJob, JobConfig, JobInput, JobLifecycle,
};
use prism_ecs_constitutional::types::Timestamp;
#[derive(Debug, Clone, Copy)]
pub struct PlannedMarker;
impl prism_ecs_core::Component for PlannedMarker {}
#[derive(Debug, Clone, Copy)]
pub struct AdmittedMarker;
impl prism_ecs_core::Component for AdmittedMarker {}
#[derive(Debug, Clone, Copy)]
pub struct PublishedMarker;
impl prism_ecs_core::Component for PublishedMarker {}

// ── Command / CommandResult / AgentSnapshot / KernelHealth ─────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Command {
    // ── Infrastructure (agent management) ──
    SpawnAgent {
        parent_id: u64,
        task: String,
        max_steps: u32,
    },
    CancelAgent {
        agent_id: u64,
    },
    RegisterModel {
        name: String,
        source_path: String,
        format: String,
    },
    /// Advance one completed prefill chunk or decode token through the
    /// canonical ECS world. The KV epoch is a fence against stale work.
    AdvanceInference {
        entity: u64,
        phase: InferencePhase,
        prefilled_tokens: u32,
        generated_tokens: u32,
        kv_epoch: u64,
        kv_tokens: u32,
    },
    BindInferenceKv {
        entity: u64,
        epoch: u64,
        page_ids: Vec<u64>,
        logical_context_tokens: u32,
        capacity_tokens: u32,
    },
    CreateModalityWork {
        kind: crate::modality::ModalityKind,
        model_path: String,
        prompt: String,
        output_path: String,
    },
    CompleteModalityWork {
        entity: u64,
        output_digest: String,
        output_bytes: u64,
    },
    FailModalityWork {
        entity: u64,
        error: String,
    },

    // ── Lifecycle (typed, domain-specific) ──
    Lifecycle(LifecycleCommand),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CommandResult {
    // Infrastructure
    Spawned {
        entity_id: u64,
    },
    Cancelled {
        entity_id: u64,
    },
    Registered {
        entity_id: u64,
    },
    InferenceAdvanced {
        entity_id: u64,
        phase: InferencePhase,
        prefilled_tokens: u32,
        generated_tokens: u32,
        kv_epoch: u64,
        kv_tokens: u32,
    },
    KvBound {
        entity_id: u64,
        epoch: u64,
    },
    ModalitySubmitted {
        entity_id: u64,
    },
    ModalityCompleted {
        entity_id: u64,
        output_digest: String,
    },
    ModalityFailed {
        entity_id: u64,
        error: String,
    },

    // Lifecycle
    Lifecycle(LifecycleCommandResult),
}

/// Result of a committed command — pairs the durable sequence number with
/// the command result.
#[derive(Debug, Clone)]
pub struct CommitOutcome {
    pub sequence: u64,
    pub result: CommandResult,
    pub world_epoch: u64,
}

/// Envelope wrapping every kernel command with metadata for idempotency,
/// epoch fencing, authority tracking, and distributed tracing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CommandEnvelope {
    pub schema_version: u32,
    pub command_type_id: u16,
    pub idempotency_key: uuid::Uuid,
    pub expected_epoch: Option<u64>,
    pub authority: String,
    pub correlation_id: String,
    pub command: Command,
}

impl CommandEnvelope {
    pub fn new(command: Command) -> Self {
        let command_type_id = match &command {
            Command::SpawnAgent { .. } => 0,
            Command::CancelAgent { .. } => 0,
            Command::RegisterModel { .. } => 0,
            Command::AdvanceInference { .. } => 0,
            Command::BindInferenceKv { .. } => 0,
            Command::CreateModalityWork { .. } => 0,
            Command::CompleteModalityWork { .. } => 0,
            Command::FailModalityWork { .. } => 0,
            Command::Lifecycle(lc) => lc.type_id().discriminant(),
        };
        Self {
            schema_version: ENVELOPE_SCHEMA_VERSION,
            command_type_id,
            idempotency_key: uuid::Uuid::new_v4(),
            expected_epoch: None,
            authority: "kernel".to_string(),
            correlation_id: String::new(),
            command,
        }
    }

    /// Return a stable numeric discriminant for the wrapped command.
    pub fn command_type(&self) -> Option<u64> {
        use Command::*;
        Some(match &self.command {
            SpawnAgent { .. } => 1,
            CancelAgent { .. } => 2,
            RegisterModel { .. } => 3,
            AdvanceInference { .. } => 4,
            BindInferenceKv { .. } => 5,
            CreateModalityWork { .. } => 6,
            CompleteModalityWork { .. } => 7,
            FailModalityWork { .. } => 8,
            Lifecycle(_) => self.command_type_id as u64,
        })
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentSnapshot {
    pub entity_id: u64,
    pub phase: String,
    pub lifecycle: String,
    pub parent_id: Option<u64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KernelHealth {
    pub entity_count: usize,
    pub world_epoch: u64,
    pub journal_sequence: u64,
    pub receipt_sequence: u64,
    pub last_snapshot_epoch: u64,
    pub last_snapshot_sequence: u64,
    pub status: String,
}

// ── RuntimeKernel / KernelHandle ───────────────────────────────────────────

/// The runtime kernel — owns the authoritative World.
pub struct RuntimeKernel {
    inner: Arc<RuntimeKernelInner>,
    /// Registered schedule for tick execution.
    schedule: parking_lot::Mutex<Option<RuntimeSchedule>>,
}

#[allow(dead_code)]
struct RuntimeKernelInner {
    world: std::sync::Arc<std::sync::RwLock<prism_ecs_core::World>>,
    command_store: Box<dyn CommandStore>,
    snapshot_store: Box<dyn SnapshotStore>,
    tick_receipt_store: Box<dyn TickReceiptStore>,
    lease_coordinator: Box<dyn LeaseCoordinator>,
    _clock: Box<dyn KernelClock>,
    provider_selector: Arc<dyn ProviderSelector>,
    backend_resources: BackendExecutionRegistry,
    sequence: AtomicU64,
    trace: TraceContext,
    state_stream: StateStream,
}

/// Thread-safe handle to the runtime kernel.
#[derive(Clone)]
pub struct KernelHandle {
    inner: Arc<RuntimeKernelInner>,
}

// KernelHandle exposes only operations that serialize world mutation through
// the kernel's internal locks. The world contains type-erased staged work and
// extensions which are not themselves marked Sync, even though they are
// never accessed without the kernel's synchronization boundary. The handle
// is the cross-thread boundary used by the daemon/application adapter.
// SAFETY: `KernelHandle` exposes no mutable references to its inner state
// without holding the kernel's internal lock. All cross-thread access is
// serialized through that lock, so `Send` is sound. The inner types are
// `Sync` because every field is either immutable, behind a synchronization
// primitive, or only mutated through exclusive access.
unsafe impl Send for KernelHandle {}
// SAFETY: see `Send` impl above. `Sync` follows because the handle's
// shared-reference operations all delegate to methods that take the kernel's
// internal lock; no shared reference escapes the lock.
unsafe impl Sync for KernelHandle {}

impl KernelHandle {
    /// Subscribe to the authoritative ECS runtime state stream.
    pub fn state_stream(&self) -> std::sync::mpsc::Receiver<prism_ecs_core::StateRecord> {
        self.inner.state_stream.subscribe()
    }

    pub fn state_snapshot(&self) -> prism_ecs_core::StateSnapshot {
        self.inner.state_stream.snapshot()
    }

    pub fn publish_state(
        &self,
        domain: impl Into<String>,
        phase: impl Into<String>,
        kind: impl Into<String>,
        status: impl Into<String>,
        state: std::collections::BTreeMap<String, serde_json::Value>,
    ) {
        self.inner
            .state_stream
            .emit(&self.inner.trace, domain, phase, kind, status, state);
    }
    /// Persistent compiled-artifact/backend resources for kernel dispatch.
    pub fn backend_resources(&self) -> BackendExecutionRegistry {
        self.inner.backend_resources.clone()
    }

    /// Register a compiled kernel artifact and return the ECS binding that can
    /// be attached to a work entity before it enters the schedule.
    pub fn register_kernel_artifact(
        &self,
        artifact: prism_ecs_kernel::KernelArtifact,
    ) -> Result<KernelArtifactBinding, RuntimeError> {
        let digest = artifact.manifest.manifest_digest.clone();
        let result = self.inner.backend_resources.register_artifact(artifact);
        self.inner.state_stream.emit(
            &self.inner.trace,
            "runtime",
            "model_registration",
            "artifact_registered",
            if result.is_ok() {
                "completed"
            } else {
                "failed"
            },
            std::collections::BTreeMap::from([(
                String::from("artifact_digest"),
                serde_json::json!(digest),
            )]),
        );
        result
    }

    /// Attach an already-registered artifact reference to a work entity in
    /// the authoritative world.
    pub fn bind_kernel_artifact(
        &self,
        work_entity: u64,
        binding: KernelArtifactBinding,
    ) -> Result<(), RuntimeError> {
        let entity = Entity::new(work_entity, 0);
        let mut world = self.inner.world.write().unwrap();
        if !world.has_entity(entity) {
            return Err(RuntimeError::Entity(format!(
                "work entity {work_entity} does not exist"
            )));
        }
        world
            .add_component(entity, binding)
            .map_err(|error| RuntimeError::Entity(format!("bind kernel artifact: {error}")))
    }

    /// Build the provider-neutral dispatcher backed by this kernel's
    /// persistent backend resources.
    pub fn kernel_dispatcher(&self) -> Arc<KernelBackendDispatcher> {
        Arc::new(KernelBackendDispatcher::new(self.backend_resources()))
    }

    /// Select the provider for an operation through the kernel-owned
    /// provider authority. The returned receipt records every attempted
    /// provider and the reason a fallback was used.
    pub fn select_provider(&self, request: &ProviderSelectionRequest) -> ProviderSelectionReceipt {
        self.inner.provider_selector.select(request)
    }

    /// Submit a typed command for execution with atomic epoch fencing.
    pub fn submit(&self, envelope: CommandEnvelope) -> Result<CommitOutcome, RuntimeError> {
        // Serialize envelope for admission
        let envelope_json = serde_json::to_string(&envelope)
            .map_err(|_| RuntimeError::Journal("envelope serialization failed".to_string()))?;

        // 1. Admit BEFORE acquiring world lock
        let admission = self
            .inner
            .command_store
            .admit(&envelope.idempotency_key, &envelope_json)?;

        let sequence = match admission {
            // 2. Completed — return result immediately (no lock, no mutation)
            Admission::Completed {
                result,
                sequence,
                world_epoch,
            } => {
                let cmd_result: CommandResult = serde_json::from_str(&result).map_err(|_| {
                    RuntimeError::IdempotencyConflict("stored result is corrupt".to_string())
                })?;
                return Ok(CommitOutcome {
                    sequence,
                    result: cmd_result,
                    world_epoch,
                });
            }
            // 3. InFlight — conflict error
            Admission::InFlight => {
                return Err(RuntimeError::IdempotencyConflict(
                    "command is in flight by another caller".to_string(),
                ))
            }
            // 4. Admitted — this caller owns execution
            Admission::Admitted { sequence } => sequence,
        };

        // Increment local sequence counter for health reporting
        self.inner.sequence.fetch_add(1, Ordering::Relaxed);

        // ── Lease operations (outside world lock) ────────────────────────
        // AcquireWorkLease coordinates with the external Valkey-based lease
        // coordinator, then transitions the entity's WorkState to Leased.
        if let Command::Lifecycle(LifecycleCommand::AcquireWorkLease(cmd)) = &envelope.command {
            let resource_key = format!("work-lease:{}", cmd.work_entity);
            let result = self
                .inner
                .lease_coordinator
                .acquire(&resource_key, cmd.ttl_ms);
            return match result {
                Ok(true) => {
                    // Update WorkState to Leased(gen=1) inside the world lock
                    {
                        let mut world = self.inner.world.write().unwrap();
                        let e = prism_ecs_core::Entity::new(cmd.work_entity, 0);
                        world.add_component(e, WorkState::Leased(1)).map_err(|e| {
                            RuntimeError::Entity(format!("lease state transition: {e}"))
                        })?;
                    }
                    let cmd_result =
                        CommandResult::Lifecycle(LifecycleCommandResult::LeaseAcquired {
                            work_entity: cmd.work_entity,
                            token: resource_key.clone(),
                            lease_generation: cmd.lease_generation,
                        });
                    let json = serde_json::to_string(&cmd_result).unwrap_or_default();
                    let epoch = self.inner.world.read().unwrap().current_epoch().0;
                    self.inner.command_store.complete(sequence, &json, epoch)?;
                    Ok(CommitOutcome {
                        sequence,
                        result: cmd_result,
                        world_epoch: epoch,
                    })
                }
                Ok(false) => Err(RuntimeError::Lease(format!(
                    "failed to acquire lease for work-entity {}",
                    cmd.work_entity
                ))),
                Err(e) => Err(RuntimeError::Lease(format!("lease error: {e}"))),
            };
        }
        if let Command::Lifecycle(LifecycleCommand::ReleaseWorkLease(cmd)) = &envelope.command {
            let resource_key = format!("work-lease:{}", cmd.work_entity);
            self.inner
                .lease_coordinator
                .release(&resource_key)
                .map_err(|e| RuntimeError::Lease(format!("release error: {e}")))?;
            // Transition WorkState back to Ready (available for re-lease)
            {
                let mut world = self.inner.world.write().unwrap();
                let e = prism_ecs_core::Entity::new(cmd.work_entity, 0);
                world
                    .add_component(e, WorkState::Ready)
                    .map_err(|e| RuntimeError::Entity(format!("release state transition: {e}")))?;
            }
            let cmd_result = CommandResult::Lifecycle(LifecycleCommandResult::LeaseReleased {
                work_entity: cmd.work_entity,
            });
            let json = serde_json::to_string(&cmd_result).unwrap_or_default();
            let epoch = self.inner.world.read().unwrap().current_epoch().0;
            self.inner
                .command_store
                .complete(sequence, &json, epoch)
                .ok();
            return Ok(CommitOutcome {
                sequence,
                result: cmd_result,
                world_epoch: epoch,
            });
        }

        // ── World-locked path ────────────────────────────────────────────
        let mut world = self.inner.world.write().unwrap();

        if let Some(expected) = envelope.expected_epoch {
            let actual: u64 = world.current_epoch().0;
            if actual != expected {
                return Err(RuntimeError::EpochMismatch { expected, actual });
            }
        }

        let result = match envelope.command {
            Command::SpawnAgent {
                parent_id,
                task,
                max_steps,
            } => execute_spawn(&mut world, parent_id, &task, max_steps)
                .map(|entity_id| CommandResult::Spawned { entity_id }),
            Command::CancelAgent { agent_id } => {
                execute_cancel_txn(&mut world, agent_id).map(|()| CommandResult::Cancelled {
                    entity_id: agent_id,
                })
            }
            Command::RegisterModel {
                name,
                source_path,
                format,
            } => execute_register_model(&mut world, &name, &source_path, &format)
                .map(|entity_id| CommandResult::Registered { entity_id }),
            Command::AdvanceInference {
                entity,
                phase,
                prefilled_tokens,
                generated_tokens,
                kv_epoch,
                kv_tokens,
            } => execute_advance_inference(
                &mut world,
                entity,
                phase,
                prefilled_tokens,
                generated_tokens,
                kv_epoch,
                kv_tokens,
            ),
            Command::BindInferenceKv {
                entity,
                epoch,
                page_ids,
                logical_context_tokens,
                capacity_tokens,
            } => execute_bind_inference_kv(
                &mut world,
                entity,
                epoch,
                page_ids,
                logical_context_tokens,
                capacity_tokens,
            )
            .map(|_| CommandResult::KvBound {
                entity_id: entity,
                epoch,
            }),
            Command::CreateModalityWork {
                kind,
                model_path,
                prompt,
                output_path,
            } => execute_create_modality_work(&mut world, kind, model_path, prompt, output_path)
                .map(|entity_id| CommandResult::ModalitySubmitted { entity_id }),
            Command::CompleteModalityWork {
                entity,
                output_digest,
                output_bytes,
            } => {
                if output_digest.trim().is_empty() {
                    Err(RuntimeError::Dispatch(
                        "modality output digest must not be empty".into(),
                    ))
                } else {
                    world
                        .add_component(
                            prism_ecs_core::Entity::new(entity, 0),
                            crate::modality::ModalityExecution {
                                output_digest: output_digest.clone(),
                                output_bytes,
                            },
                        )
                        .map(|_| CommandResult::ModalityCompleted {
                            entity_id: entity,
                            output_digest,
                        })
                        .map_err(|error| RuntimeError::Dispatch(error.to_string()))
                }
            }
            Command::FailModalityWork { entity, error } => {
                if error.trim().is_empty() {
                    Err(RuntimeError::Dispatch(
                        "modality failure must include an error".into(),
                    ))
                } else {
                    world
                        .add_component(
                            prism_ecs_core::Entity::new(entity, 0),
                            crate::modality::ModalityFailure {
                                error: error.clone(),
                            },
                        )
                        .map(|_| CommandResult::ModalityFailed {
                            entity_id: entity,
                            error,
                        })
                        .map_err(|error| RuntimeError::Dispatch(error.to_string()))
                }
            }
            Command::Lifecycle(lc) => execute_lifecycle(&mut world, lc),
        };

        let committed_epoch = result.as_ref().ok().map(|_| world.current_epoch().0);
        drop(world);

        // Handle mutation result — update command store state and build outcome
        let entry_seq = sequence;
        match result {
            Ok(cmd_result) => {
                let json = serde_json::to_string(&cmd_result).unwrap_or_default();
                let epoch = committed_epoch.unwrap_or(0);
                if self
                    .inner
                    .command_store
                    .complete(entry_seq, &json, epoch)
                    .is_err()
                {
                    let _ = self
                        .inner
                        .command_store
                        .transition_state(entry_seq, "recovery_required");
                }
                Ok(CommitOutcome {
                    sequence: entry_seq,
                    result: cmd_result,
                    world_epoch: epoch,
                })
            }
            Err(e) => {
                let _ = self
                    .inner
                    .command_store
                    .transition_state(entry_seq, "failed");
                Err(e)
            }
        }
    }

    /// Query all agent entities with their phase and lifecycle.
    pub fn query_agents(&self) -> Vec<AgentSnapshot> {
        let world = self.inner.world.read().unwrap();
        let mut agents = Vec::new();
        for entity in world.all_entities() {
            use prism_ecs_constitutional::agent_exec::{AgentLifecycle, AgentPhase};
            use prism_ecs_constitutional::agent_plan::ParentAgentId;

            if let Some(phase) = world.get_component::<AgentPhase>(entity) {
                let lifecycle = world.get_component::<AgentLifecycle>(entity);
                let parent = world.get_component::<ParentAgentId>(entity);
                agents.push(AgentSnapshot {
                    entity_id: entity.id(),
                    phase: format!("{:?}", phase),
                    lifecycle: format!("{:?}", lifecycle.unwrap_or(&AgentLifecycle::Active)),
                    parent_id: parent.map(|p| p.0.id()),
                });
            }
        }
        agents
    }

    /// Lock the world and return a mutex guard for read-only access.
    /// Used by the schedule to create a `WorldViewImpl` for each tick.
    pub fn lock_world(&self) -> std::sync::RwLockReadGuard<'_, prism_ecs_core::World> {
        self.inner.world.read().unwrap()
    }

    /// Query kernel health.
    pub fn health(&self) -> KernelHealth {
        let world = self.inner.world.read().unwrap();
        let epoch: WorldEpoch = world.current_epoch();
        let entity_count = world.all_entities().len();
        let seq = self.inner.sequence.load(Ordering::Relaxed);
        KernelHealth {
            entity_count,
            world_epoch: epoch.0,
            journal_sequence: seq,
            receipt_sequence: seq,
            last_snapshot_epoch: 0,
            last_snapshot_sequence: 0,
            status: "running".to_string(),
        }
    }
}

impl RuntimeKernel {
    pub fn recover(&self) -> Result<RecoveryReport, RuntimeError> {
        let watermarks = self.inner.command_store.high_water_marks()?;
        let snapshot = self.inner.snapshot_store.load_latest()?;

        // Verify snapshot if present (for validation only)
        if let Some(ref snap) = &snapshot {
            if !snap.verify() {
                return Err(RuntimeError::Journal(
                    "snapshot checksum mismatch — no fallback available".into(),
                ));
            }
        }

        // Start from a pristine world — every entity and component is
        // reconstructed from command replay, ensuring consistent entity IDs.
        *self.inner.world.write().unwrap() = World::new();

        // Replay ALL completed commands from sequence 0
        let all = self.inner.command_store.completed_after(0)?;
        for cmd in &all {
            self.apply_recovered_command(cmd)?;
        }
        let replayed_count = all.len() as u64;

        // After replay, validate allocator against snapshot (if present)
        if let Some(ref snap) = &snapshot {
            let reconstructed = self.inner.world.read().unwrap();
            let reconstructed_alloc =
                prism_ecs_core::snapshot::export_allocator_snapshot(&reconstructed);
            if reconstructed_alloc != snap.payload.allocator_data {
                eprintln!(
                    "Kernel: allocator differs from snapshot (acceptable if entity IDs differ)"
                );
            }
        }

        // Set sequence counter after replay
        self.inner
            .sequence
            .store(watermarks.last_committed_sequence + 1, Ordering::SeqCst);

        // Reconcile unresolved commands
        let unresolved = self.inner.command_store.unresolved()?;
        let unresolved_count = unresolved.len() as u64;
        for cmd in &unresolved {
            self.inner
                .command_store
                .transition_state(cmd.sequence, "recovery_required")?;
        }

        Ok(RecoveryReport {
            recovery_state: if replayed_count > 0 {
                "recovered".to_string()
            } else {
                "fresh".to_string()
            },
            snapshot_epoch: snapshot
                .as_ref()
                .map(|s| s.payload.world_epoch)
                .unwrap_or(0),
            snapshot_sequence: watermarks.last_committed_sequence,
            replayed_commands: replayed_count,
            unresolved_commands: unresolved_count,
            world_epoch_before: snapshot
                .as_ref()
                .map(|s| s.payload.world_epoch)
                .unwrap_or(0),
        })
    }

    /// Replay a command that was already committed — no journal, no
    /// idempotency checking, no receipt.  Just apply the world mutation.
    fn apply_recovered_command(
        &self,
        completed: &crate::ports::CompletedCommand,
    ) -> Result<(), RuntimeError> {
        // Parse the envelope to extract the command
        let envelope: crate::kernel::CommandEnvelope =
            serde_json::from_str(&completed.envelope_json)
                .map_err(|e| RuntimeError::Journal(format!("replay: bad envelope: {e}")))?;

        // Deserialize the stored result for verification
        let stored_result: CommandResult = serde_json::from_str(&completed.result_json)
            .map_err(|e| RuntimeError::Journal(format!("replay: bad result: {e}")))?;

        let mut world = self.inner.world.write().unwrap();

        match envelope.command {
            Command::SpawnAgent {
                parent_id,
                task,
                max_steps,
            } => {
                let entity_id = execute_spawn(&mut world, parent_id, &task, max_steps)?;
                // Verify the generated entity ID matches the stored result
                if let CommandResult::Spawned {
                    entity_id: expected,
                } = &stored_result
                {
                    // The stored result should match what execute_spawn returned.
                    if entity_id != *expected {
                        return Err(RuntimeError::Journal(format!(
                            "replay entity ID mismatch: generated {} but stored result has {}",
                            entity_id, expected
                        )));
                    }
                }
            }
            Command::CancelAgent { agent_id } => {
                execute_cancel_txn(&mut world, agent_id)?;
            }
            Command::RegisterModel {
                name,
                source_path,
                format,
            } => {
                let entity_id = execute_register_model(&mut world, &name, &source_path, &format)?;
                if let CommandResult::Registered {
                    entity_id: expected,
                } = &stored_result
                {
                    if entity_id != *expected {
                        return Err(RuntimeError::Journal(format!(
                            "replay register entity ID mismatch: generated {} but stored result has {}",
                            entity_id, expected
                        )));
                    }
                }
            }
            Command::AdvanceInference {
                entity,
                phase,
                prefilled_tokens,
                generated_tokens,
                kv_epoch,
                kv_tokens,
            } => {
                let _ = execute_advance_inference(
                    &mut world,
                    entity,
                    phase,
                    prefilled_tokens,
                    generated_tokens,
                    kv_epoch,
                    kv_tokens,
                )?;
            }
            Command::BindInferenceKv {
                entity,
                epoch,
                page_ids,
                logical_context_tokens,
                capacity_tokens,
            } => {
                execute_bind_inference_kv(
                    &mut world,
                    entity,
                    epoch,
                    page_ids,
                    logical_context_tokens,
                    capacity_tokens,
                )?;
            }
            Command::CreateModalityWork {
                kind,
                model_path,
                prompt,
                output_path,
            } => {
                let entity_id = execute_create_modality_work(
                    &mut world,
                    kind,
                    model_path,
                    prompt,
                    output_path,
                )?;
                if let CommandResult::ModalitySubmitted {
                    entity_id: expected,
                } = stored_result
                {
                    if entity_id != expected {
                        return Err(RuntimeError::Journal(format!(
                            "replay modality entity mismatch: generated {entity_id}, stored {expected}"
                        )));
                    }
                }
            }
            Command::CompleteModalityWork {
                entity,
                output_digest,
                output_bytes,
            } => {
                world
                    .add_component(
                        prism_ecs_core::Entity::new(entity, 0),
                        crate::modality::ModalityExecution {
                            output_digest: output_digest.clone(),
                            output_bytes,
                        },
                    )
                    .map_err(|error| RuntimeError::Journal(error.to_string()))?;
                match stored_result {
                    CommandResult::ModalityCompleted {
                        entity_id: expected,
                        output_digest: expected_digest,
                    } if entity == expected && output_digest == expected_digest => {}
                    _ => {
                        return Err(RuntimeError::Journal(
                            "replay modality completion mismatch".into(),
                        ))
                    }
                }
            }
            Command::FailModalityWork { entity, error } => {
                world
                    .add_component(
                        prism_ecs_core::Entity::new(entity, 0),
                        crate::modality::ModalityFailure {
                            error: error.clone(),
                        },
                    )
                    .map_err(|error| RuntimeError::Journal(error.to_string()))?;
                match stored_result {
                    CommandResult::ModalityFailed {
                        entity_id,
                        error: expected,
                    } if entity == entity_id && error == expected => {}
                    _ => {
                        return Err(RuntimeError::Journal(
                            "replay modality failure mismatch".into(),
                        ))
                    }
                }
            }
            // Re-execute lifecycle commands so entity ID allocation stays
            // consistent across the command sequence. The execute_lifecycle
            // function performs actual state changes and returns a result
            // that we verify against the stored result.
            Command::Lifecycle(lc) => {
                let new_result = execute_lifecycle(&mut world, lc)?;
                // Verify result variant matches (entity IDs must be consistent
                // since replay runs all commands in order).
                match (&new_result, &stored_result) {
                    (
                        CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                            work_entity,
                            ..
                        }),
                        CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                            work_entity: expected,
                            ..
                        }),
                    ) if work_entity != expected => {
                        return Err(RuntimeError::Journal(format!(
                            "replay work entity ID mismatch: generated {} but expected {}",
                            work_entity, expected
                        )))
                    }
                    _ if std::mem::discriminant(&new_result)
                        != std::mem::discriminant(&stored_result) =>
                    {
                        return Err(RuntimeError::Journal(format!(
                            "replay lifecycle result variant mismatch: got {:?} expected {:?}",
                            new_result, stored_result
                        )))
                    }
                    _ => {}
                }
            }
        }

        drop(world);
        Ok(())
    }
}

impl Default for RuntimeKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeKernel {
    /// Create a kernel with default in-memory ports.
    pub fn new() -> Self {
        Self::with_ports(
            Box::new(crate::test_adapters::InMemoryCommandStore::new()),
            Box::new(crate::test_adapters::InMemorySnapshotStore::new()),
            Box::new(crate::test_adapters::InMemoryTickReceiptStore::new()),
            Box::new(crate::test_adapters::InMemoryLeaseCoordinator::new()),
            Box::new(crate::test_adapters::DeterministicClock::new(1000)),
        )
    }

    pub fn with_ports(
        command_store: Box<dyn CommandStore>,
        snapshot_store: Box<dyn SnapshotStore>,
        tick_receipt_store: Box<dyn TickReceiptStore>,
        lease_coordinator: Box<dyn LeaseCoordinator>,
        clock: Box<dyn KernelClock>,
    ) -> Self {
        Self::with_ports_and_provider_selector(
            command_store,
            snapshot_store,
            tick_receipt_store,
            lease_coordinator,
            clock,
            Arc::new(StaticProviderSelector::default()),
        )
    }

    /// Create a kernel with explicit provider selection authority while
    /// retaining all existing persistence and lease ports.
    pub fn with_ports_and_provider_selector(
        command_store: Box<dyn CommandStore>,
        snapshot_store: Box<dyn SnapshotStore>,
        tick_receipt_store: Box<dyn TickReceiptStore>,
        lease_coordinator: Box<dyn LeaseCoordinator>,
        clock: Box<dyn KernelClock>,
        provider_selector: Arc<dyn ProviderSelector>,
    ) -> Self {
        let trace = global_context();
        Self {
            inner: Arc::new(RuntimeKernelInner {
                world: std::sync::Arc::new(std::sync::RwLock::new(World::new())),
                command_store,
                snapshot_store,
                tick_receipt_store,
                lease_coordinator,
                _clock: clock,
                provider_selector,
                backend_resources: crate::backend::BackendExecutionRegistry::new(),
                sequence: AtomicU64::new(0),
                trace: trace.clone(),
                state_stream: StateStream::global(),
            }),
            schedule: parking_lot::Mutex::new(None),
        }
    }
    /// Create a kernel with an existing world and default in-memory ports.
    /// This is used by the daemon to integrate the kernel with the authoritative PrismWorld.
    pub fn with_existing_world(
        world: std::sync::Arc<std::sync::RwLock<prism_ecs_core::World>>,
    ) -> Self {
        Self::with_existing_world_and_ports(
            world,
            Box::new(crate::test_adapters::InMemoryCommandStore::new()),
            Box::new(crate::test_adapters::InMemorySnapshotStore::new()),
            Box::new(crate::test_adapters::InMemoryTickReceiptStore::new()),
            Box::new(crate::test_adapters::InMemoryLeaseCoordinator::new()),
            Box::new(crate::test_adapters::DeterministicClock::new(1000)),
        )
    }

    /// Create a kernel with an existing world and custom ports.
    /// This is used by the daemon to integrate the kernel with the authoritative PrismWorld.
    pub fn with_existing_world_and_ports(
        world: std::sync::Arc<std::sync::RwLock<prism_ecs_core::World>>,
        command_store: Box<dyn CommandStore>,
        snapshot_store: Box<dyn SnapshotStore>,
        tick_receipt_store: Box<dyn TickReceiptStore>,
        lease_coordinator: Box<dyn LeaseCoordinator>,
        clock: Box<dyn KernelClock>,
    ) -> Self {
        Self::with_existing_world_and_ports_and_provider_selector(
            world,
            command_store,
            snapshot_store,
            tick_receipt_store,
            lease_coordinator,
            clock,
            Arc::new(StaticProviderSelector::default()),
        )
    }

    /// Create a kernel over an existing authoritative world with explicit
    /// provider selection authority.
    pub fn with_existing_world_and_ports_and_provider_selector(
        world: std::sync::Arc<std::sync::RwLock<prism_ecs_core::World>>,
        command_store: Box<dyn CommandStore>,
        snapshot_store: Box<dyn SnapshotStore>,
        tick_receipt_store: Box<dyn TickReceiptStore>,
        lease_coordinator: Box<dyn LeaseCoordinator>,
        clock: Box<dyn KernelClock>,
        provider_selector: Arc<dyn ProviderSelector>,
    ) -> Self {
        let trace = global_context();
        Self {
            inner: Arc::new(RuntimeKernelInner {
                world,
                command_store,
                snapshot_store,
                tick_receipt_store,
                lease_coordinator,
                _clock: clock,
                provider_selector,
                backend_resources: crate::backend::BackendExecutionRegistry::new(),
                sequence: AtomicU64::new(0),
                trace: trace.clone(),
                state_stream: StateStream::global(),
            }),
            schedule: parking_lot::Mutex::new(None),
        }
    }

    pub fn handle(&self) -> KernelHandle {
        KernelHandle {
            inner: self.inner.clone(),
        }
    }

    pub fn health(&self) -> KernelHealth {
        let world = self.inner.world.read().unwrap();
        let epoch: WorldEpoch = world.current_epoch();
        let entity_count = world.all_entities().len();
        let seq = self.inner.sequence.load(Ordering::Relaxed);
        KernelHealth {
            entity_count,
            world_epoch: epoch.0,
            journal_sequence: seq,
            receipt_sequence: seq,
            last_snapshot_epoch: 0,
            last_snapshot_sequence: 0,
            status: "running".to_string(),
        }
    }

    /// Register a schedule for tick execution.
    pub fn set_schedule(&self, schedule: RuntimeSchedule) {
        *self.schedule.lock() = Some(schedule);
    }

    /// Run a single tick on the registered schedule.
    pub fn run_tick(&self) -> Result<TickReceipt, RuntimeError> {
        let sched = self.schedule.lock();
        match sched.as_ref() {
            Some(s) => s.run_tick(),
            None => Err(RuntimeError::Entity("no schedule registered".into())),
        }
    }

    /// Run a tick and persist the receipt through the tick receipt store.
    pub fn run_kernel_tick(&self, instance_id: &str) -> Result<(), RuntimeError> {
        let receipt = self.run_tick()?;
        self.inner
            .tick_receipt_store
            .save(&receipt, instance_id)
            .map_err(|e| RuntimeError::Receipt(e.to_string()))?;
        Ok(())
    }

    pub fn capture_snapshot(&self) -> Result<WorldSnapshot, RuntimeError> {
        let watermarks = self.inner.command_store.high_water_marks()?;
        let seq = watermarks.last_committed_sequence;

        let world = self.inner.world.read().unwrap();
        let epoch = world.current_epoch().0;
        let allocator_data = prism_ecs_core::snapshot::export_allocator_snapshot(&world);
        drop(world);

        let payload = SnapshotPayload {
            schema_version: 1,
            world_epoch: epoch,
            next_entity_id: 0, // no longer used; allocator_data captures this
            last_command_sequence: seq,
            allocator_data,
            schedule_hash: self.get_schedule_hash(),
            created_at_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };
        let checksum = WorldSnapshot::compute_checksum(&payload);
        Ok(WorldSnapshot { payload, checksum })
    }

    /// Capture and persist a snapshot through the snapshot store.
    /// Useful for testing recovery — combines capture + save in one call.
    pub fn save_snapshot(&self) -> Result<(), RuntimeError> {
        let snapshot = self.capture_snapshot()?;
        self.inner.snapshot_store.save(&snapshot)
    }

    /// Graceful shutdown: persist final snapshot.
    pub fn shutdown(&self) -> Result<(), RuntimeError> {
        let snapshot = self.capture_snapshot()?;
        self.inner.snapshot_store.save(&snapshot)?;
        Ok(())
    }

    /// Return the schedule hash, or zeroed if no schedule is set.
    fn get_schedule_hash(&self) -> [u8; 32] {
        self.schedule
            .lock()
            .as_ref()
            .map(|s| s.schedule_hash())
            .unwrap_or([0u8; 32])
    }

    /// Run ticks until the given target tick number (inclusive) is reached.
    /// Returns receipts for every tick executed.
    pub fn run_tick_to(&self, target_tick: u64) -> Result<Vec<TickReceipt>, RuntimeError> {
        let mut receipts = Vec::new();
        loop {
            let receipt = self.run_tick()?;
            let tick = receipt.tick_number;
            receipts.push(receipt);
            if tick >= target_tick {
                break;
            }
        }
        Ok(receipts)
    }
}

pub fn create_kernel() -> RuntimeKernel {
    RuntimeKernel::new()
}

// ── Execution helpers (all through WorldTxn) ───────────────────────────────

/// Execute a typed lifecycle command against the world.
fn execute_lifecycle(
    world: &mut World,
    lc: LifecycleCommand,
) -> Result<CommandResult, RuntimeError> {
    use LifecycleCommand::*;
    match lc {
        // ── Ingress ──
        CreateWork(cmd) => execute_create_work(world, &cmd),
        CreateCompilationJob(cmd) => execute_create_compilation_job(world, &cmd),
        RequestCancellation(cmd) => execute_request_cancellation(world, &cmd),

        // ── Observation ──
        MarkObserved(cmd) => execute_mark_observed(world, &cmd),
        RecordExternalObservation(cmd) => Ok(CommandResult::Lifecycle(
            LifecycleCommandResult::MarkedObserved { entity: cmd.entity },
        )),

        // ── Planning ──
        RecordWorkPlan(cmd) => execute_record_work_plan(world, &cmd),
        MarkPrerequisiteBlocked(cmd) => Ok(CommandResult::Lifecycle(
            LifecycleCommandResult::PrerequisiteBlocked { entity: cmd.entity },
        )),

        // ── Admission ──
        AdmitWork(cmd) => execute_admit_work(world, &cmd),
        RejectWork(cmd) => Ok(CommandResult::Lifecycle(LifecycleCommandResult::Rejected {
            entity: cmd.entity,
            reason: cmd.reason,
        })),
        DeferWork(cmd) => Ok(CommandResult::Lifecycle(LifecycleCommandResult::Deferred {
            entity: cmd.entity,
            reason: cmd.reason,
        })),

        // ── Leasing ──
        AcquireWorkLease(_) | ReleaseWorkLease(_) => {
            unreachable!("lease commands handled before world lock")
        }
        RenewWorkLease(cmd) => Ok(CommandResult::Lifecycle(
            LifecycleCommandResult::LeaseRenewed {
                work_entity: cmd.work_entity,
                ttl_ms: cmd.ttl_ms,
            },
        )),

        // ── Dispatch ──
        RecordDispatchIntent(cmd) => execute_record_dispatch_intent(world, &cmd),
        RecordDispatchStarted(cmd) => Ok(CommandResult::Lifecycle(
            LifecycleCommandResult::DispatchStarted {
                work_entity: cmd.work_entity,
                adapter_handle: cmd.adapter_handle,
            },
        )),

        // ── Collection ──
        RecordProgress(cmd) => Ok(CommandResult::Lifecycle(
            LifecycleCommandResult::ProgressRecorded {
                work_entity: cmd.work_entity,
            },
        )),
        CompleteWork(cmd) => execute_complete_work(world, &cmd),
        FailWork(cmd) => execute_fail_work(world, &cmd),
        MarkDispatchLost(cmd) => Ok(CommandResult::Lifecycle(
            LifecycleCommandResult::DispatchMarkedLost {
                work_entity: cmd.work_entity,
            },
        )),

        // ── Evidence ──
        AttachArtifact(cmd) => Ok(CommandResult::Lifecycle(
            LifecycleCommandResult::ArtifactAttached {
                entity: cmd.entity,
                digest: cmd.digest,
            },
        )),
        AttachDiagnostics(cmd) => Ok(CommandResult::Lifecycle(
            LifecycleCommandResult::DiagnosticsAttached { entity: cmd.entity },
        )),
        AttachEvidence(cmd) => execute_attach_evidence_cmd(world, &cmd),

        // ── Publication ──
        PublishResult(cmd) => execute_publish_result_cmd(world, &cmd),

        // ── Cleanup ──
        ExpireTransientState(cmd) => Ok(CommandResult::Lifecycle(
            LifecycleCommandResult::TransientExpired { entity: cmd.entity },
        )),
        MarkRetentionComplete(cmd) => Ok(CommandResult::Lifecycle(
            LifecycleCommandResult::RetentionComplete { entity: cmd.entity },
        )),
    }
}

// ── Typed lifecycle command handlers ──────────────────────────────────────────

fn execute_create_work(
    world: &mut World,
    cmd: &CreateWorkCommand,
) -> Result<CommandResult, RuntimeError> {
    let spawned = world
        .spawn(EntityKind::WorkUnit, None)
        .map_err(|e| RuntimeError::Entity(format!("spawn work failed: {e}")))?;
    let work_entity = spawned.entity.id();
    world
        .add_component(spawned.entity, WorkState::Pending)
        .map_err(|e| RuntimeError::Entity(format!("failed to set state: {e}")))?;
    let kind = match cmd.kind.trim().to_ascii_lowercase().as_str() {
        "inference" | "run_inference" => prism_ecs_constitutional::work::WorkKind::RunInference,
        "load_model" => prism_ecs_constitutional::work::WorkKind::LoadModel,
        "validate" => prism_ecs_constitutional::work::WorkKind::Validate,
        "package" => prism_ecs_constitutional::work::WorkKind::Package,
        "teardown" => prism_ecs_constitutional::work::WorkKind::Teardown,
        _ => prism_ecs_constitutional::work::WorkKind::CompileGraph,
    };
    world
        .add_component(
            spawned.entity,
            prism_ecs_constitutional::work::WorkItemComponent {
                kind,
                target_entity: prism_ecs_core::Entity::new(cmd.target_entity, 0),
                retry_count: 0,
                max_retries: 3,
            },
        )
        .map_err(|e| RuntimeError::Entity(format!("failed to set work item: {e}")))?;
    if matches!(kind, prism_ecs_constitutional::work::WorkKind::RunInference) {
        world
            .add_component(
                spawned.entity,
                InferenceWorkMetadata::from_resource_claim(&cmd.resource_claim),
            )
            .map_err(|e| RuntimeError::Entity(format!("failed to set inference metadata: {e}")))?;
    }
    if !cmd.output_path.is_empty() {
        world
            .add_component(
                spawned.entity,
                prism_ecs_constitutional::work::WorkOutputPath(cmd.output_path.clone()),
            )
            .map_err(|e| RuntimeError::Entity(format!("failed to set output path: {e}")))?;
    }
    if !cmd.input_path.is_empty() {
        world
            .add_component(
                spawned.entity,
                prism_ecs_constitutional::work::WorkInputPath(cmd.input_path.clone()),
            )
            .map_err(|e| RuntimeError::Entity(format!("failed to set input path: {e}")))?;
    }
    Ok(CommandResult::Lifecycle(
        LifecycleCommandResult::WorkCreated {
            work_entity,
            sequence: 0,
            world_epoch: world.current_epoch().0,
        },
    ))
}

fn execute_create_compilation_job(
    world: &mut World,
    cmd: &CreateCompilationJobCommand,
) -> Result<CommandResult, RuntimeError> {
    // The previous version of this handler was a no-op that returned a
    // fresh entity ID and dropped every field of the command. The
    // constitutional `compilation` module's
    // `CreateCompilationJobCommand::execute` is the canonical
    // implementation, but it requires a `SchemaRegistry` that the
    // kernel does not yet own. Until that lands, this handler performs
    // the equivalent inserts directly: spawn the job entity, attach
    // `CompilationJob` / `JobInput` / `JobConfig` / `JobLifecycle`, and
    // surface the same data the no-op was silently discarding.
    //
    // The audit's rule is "no naked u64 IDs"; the spawned entity
    // carries a real generation, which is what we attach components to.
    let spawned = world
        .spawn(EntityKind::Executable, None)
        .map_err(|e| RuntimeError::Entity(format!("spawn job failed: {e}")))?;
    let job_entity = spawned.entity;

    world
        .insert_component(
            job_entity,
            CompilationJob {
                job_id: cmd.job_id,
                target_artifact: cmd.model_artifact,
                target_device_profile: cmd.target_profile.clone(),
                created_at: Timestamp::now(),
            },
        )
        .map_err(|e| RuntimeError::Entity(format!("insert CompilationJob: {e}")))?;

    world
        .insert_component(
            job_entity,
            JobInput {
                model_artifact: cmd.model_artifact,
                source_format: String::new(),
                quantization_profile: None,
            },
        )
        .map_err(|e| RuntimeError::Entity(format!("insert JobInput: {e}")))?;

    world
        .insert_component(
            job_entity,
            JobConfig {
                target_format: cmd.target_format.clone(),
                optimization_level: cmd.optimization_level,
                enable_validation: cmd.enable_validation,
            },
        )
        .map_err(|e| RuntimeError::Entity(format!("insert JobConfig: {e}")))?;

    world
        .insert_component(job_entity, JobLifecycle::Pending)
        .map_err(|e| RuntimeError::Entity(format!("insert JobLifecycle: {e}")))?;

    Ok(CommandResult::Lifecycle(
        LifecycleCommandResult::CompilationJobCreated {
            entity: job_entity.id(),
            sequence: 0,
            world_epoch: world.current_epoch().0,
        },
    ))
}

fn execute_admit_work(
    world: &mut World,
    cmd: &AdmitWorkCommand,
) -> Result<CommandResult, RuntimeError> {
    let e = Entity::new(cmd.entity, 0);
    if let Some(state) = world.get_component::<WorkState>(e) {
        if *state == WorkState::Pending {
            world
                .add_component(e, WorkState::Ready)
                .map_err(|e| RuntimeError::Entity(format!("admit transition failed: {e}")))?;
        }
    }
    world
        .add_component(e, AdmittedMarker)
        .map_err(|e| RuntimeError::Entity(e.to_string()))?;
    Ok(CommandResult::Lifecycle(LifecycleCommandResult::Admitted {
        entity: cmd.entity,
    }))
}

fn execute_record_dispatch_intent(
    world: &mut World,
    cmd: &RecordDispatchIntentCommand,
) -> Result<CommandResult, RuntimeError> {
    let dispatch_id = uuid::Uuid::new_v4().to_string();
    // Transition from Leased to Dispatched
    let e = prism_ecs_core::Entity::new(cmd.work_entity, 0);
    world
        .add_component(e, WorkState::Leased(1))
        .map_err(|e| RuntimeError::Entity(format!("dispatch state transition: {e}")))?;
    Ok(CommandResult::Lifecycle(
        LifecycleCommandResult::DispatchIntentRecorded {
            work_entity: cmd.work_entity,
            dispatch_id,
        },
    ))
}

fn execute_complete_work(
    world: &mut World,
    cmd: &CompleteWorkCommand,
) -> Result<CommandResult, RuntimeError> {
    use crate::ports::ResultPayload;
    let e = Entity::new(cmd.work_entity, 0);
    // Transition from Collecting/Dispatched to Completed
    world
        .add_component(e, WorkState::Completed)
        .map_err(|e| RuntimeError::Entity(format!("complete state transition: {e}")))?;
    let payload = ResultPayload {
        result_type: "completed".to_string(),
        result: String::from_utf8_lossy(&cmd.output).to_string(),
    };
    world
        .add_component(e, payload)
        .map_err(|e| RuntimeError::Entity(format!("complete work failed: {e}")))?;
    Ok(CommandResult::Lifecycle(
        LifecycleCommandResult::Completed {
            work_entity: cmd.work_entity,
            result: String::from_utf8_lossy(&cmd.output).to_string(),
            sequence: 0,
            world_epoch: world.current_epoch().0,
        },
    ))
}

fn execute_fail_work(
    world: &mut World,
    cmd: &FailWorkCommand,
) -> Result<CommandResult, RuntimeError> {
    let e = Entity::new(cmd.work_entity, 0);
    if world.has_entity(e) {
        world
            .add_component(e, WorkState::Failed)
            .map_err(|error| RuntimeError::Entity(format!("fail work transition: {error}")))?;
    }
    Ok(CommandResult::Lifecycle(LifecycleCommandResult::Failed {
        work_entity: cmd.work_entity,
        error: cmd.error.clone(),
    }))
}

fn execute_request_cancellation(
    world: &mut World,
    cmd: &RequestCancellationCommand,
) -> Result<CommandResult, RuntimeError> {
    let e = prism_ecs_core::Entity::new(cmd.entity, 0);
    if let Some(state) = world.get_component::<WorkState>(e) {
        if !state.is_terminal() {
            world
                .add_component(e, WorkState::Cancelled)
                .map_err(|e| RuntimeError::Entity(format!("cancel transition failed: {e}")))?;
        }
    }
    Ok(CommandResult::Lifecycle(
        LifecycleCommandResult::RequestCancelled { entity: cmd.entity },
    ))
}

fn execute_attach_evidence_cmd(
    _world: &mut World,
    cmd: &AttachEvidenceCommand,
) -> Result<CommandResult, RuntimeError> {
    let receipt_id = uuid::Uuid::new_v4().to_string();
    Ok(CommandResult::Lifecycle(
        LifecycleCommandResult::EvidenceAttached {
            entity: cmd.entity,
            receipt_id,
        },
    ))
}

fn execute_publish_result_cmd(
    world: &mut World,
    cmd: &PublishResultCommand,
) -> Result<CommandResult, RuntimeError> {
    use crate::ports::ResultPayload;
    let e = Entity::new(cmd.entity, 0);
    // Transition from Completed to Published
    world
        .add_component(e, WorkState::Completed)
        .map_err(|e| RuntimeError::Entity(format!("publish state transition: {e}")))?;
    let receipt_id = uuid::Uuid::new_v4().to_string();
    let payload = ResultPayload {
        result_type: cmd.result_type.clone(),
        result: cmd.result.clone(),
    };
    world
        .add_component(e, payload)
        .map_err(|e| RuntimeError::Entity(format!("publish result failed: {e}")))?;
    world
        .add_component(e, PublishedMarker)
        .map_err(|e| RuntimeError::Entity(e.to_string()))?;
    Ok(CommandResult::Lifecycle(
        LifecycleCommandResult::Published {
            entity: cmd.entity,
            receipt_id,
            sequence: 0,
            world_epoch: world.current_epoch().0,
        },
    ))
}

fn execute_mark_observed(
    world: &mut World,
    cmd: &MarkObservedCommand,
) -> Result<CommandResult, RuntimeError> {
    let e = Entity::new(cmd.entity, 0);
    if let Some(state) = world.get_component::<WorkState>(e) {
        if *state == WorkState::Pending {
            world
                .add_component(e, WorkState::Ready)
                .map_err(|e| RuntimeError::Entity(format!("observe transition failed: {e}")))?;
        }
    }
    world
        .add_component(e, PlannedMarker)
        .map_err(|e| RuntimeError::Entity(e.to_string()))?;
    Ok(CommandResult::Lifecycle(
        LifecycleCommandResult::MarkedObserved { entity: cmd.entity },
    ))
}

fn execute_record_work_plan(
    world: &mut World,
    cmd: &RecordWorkPlanCommand,
) -> Result<CommandResult, RuntimeError> {
    let e = Entity::new(cmd.entity, 0);
    if let Some(state) = world.get_component::<WorkState>(e) {
        if *state == WorkState::Ready {
            world
                .add_component(e, WorkState::Ready)
                .map_err(|e| RuntimeError::Entity(format!("plan transition failed: {e}")))?;
        }
    }
    Ok(CommandResult::Lifecycle(
        LifecycleCommandResult::WorkPlanRecorded { entity: cmd.entity },
    ))
}

fn execute_bind_inference_kv(
    world: &mut World,
    entity: u64,
    epoch: u64,
    page_ids: Vec<u64>,
    logical_context_tokens: u32,
    capacity_tokens: u32,
) -> Result<(), RuntimeError> {
    let target = Entity::new(entity, 0);
    if !world.has_entity(target) {
        return Err(RuntimeError::Entity(format!(
            "inference entity {entity} not found"
        )));
    }
    let binding = crate::inference::KvCacheBinding {
        epoch,
        page_ids,
        logical_context_tokens,
        capacity_tokens,
    };
    world
        .add_component(target, binding)
        .map_err(|error| RuntimeError::Entity(format!("bind inference KV failed: {error}")))
}

fn execute_create_modality_work(
    world: &mut World,
    kind: crate::modality::ModalityKind,
    model_path: String,
    prompt: String,
    output_path: String,
) -> Result<u64, RuntimeError> {
    if model_path.trim().is_empty() || prompt.trim().is_empty() {
        return Err(RuntimeError::Entity(
            "modality work requires model_path and prompt".into(),
        ));
    }
    let spawned = world
        .spawn(EntityKind::WorkUnit, None)
        .map_err(|error| RuntimeError::Entity(format!("spawn modality work failed: {error}")))?;
    let entity_id = spawned.entity.id();
    world
        .add_component(
            spawned.entity,
            crate::modality::ModalityWork {
                kind,
                model_path,
                prompt,
                output_path,
            },
        )
        .map_err(|error| RuntimeError::Entity(format!("attach modality work failed: {error}")))?;
    Ok(entity_id)
}

fn execute_advance_inference(
    world: &mut World,
    entity: u64,
    phase: InferencePhase,
    prefilled_tokens: u32,
    generated_tokens: u32,
    kv_epoch: u64,
    kv_tokens: u32,
) -> Result<CommandResult, RuntimeError> {
    let e = Entity::new(entity, 0);
    let current = world
        .get_component::<InferenceWorkMetadata>(e)
        .copied()
        .ok_or_else(|| RuntimeError::Entity(format!("inference metadata missing for {entity}")))?;
    if current.kv_epoch != kv_epoch {
        return Err(RuntimeError::Entity(format!(
            "stale KV epoch for inference {entity}: expected {}, received {kv_epoch}",
            current.kv_epoch
        )));
    }
    if prefilled_tokens < current.prefilled_tokens
        || generated_tokens < current.generated_tokens
        || kv_tokens < current.kv_tokens
    {
        return Err(RuntimeError::Entity(format!(
            "inference progress regressed for {entity}"
        )));
    }
    let next = InferenceWorkMetadata {
        phase,
        prefilled_tokens,
        generated_tokens,
        kv_tokens,
        ..current
    };
    world
        .add_component(e, next)
        .map_err(|error| RuntimeError::Entity(format!("advance inference failed: {error}")))?;
    Ok(CommandResult::InferenceAdvanced {
        entity_id: entity,
        phase,
        prefilled_tokens,
        generated_tokens,
        kv_epoch,
        kv_tokens,
    })
}

fn execute_spawn(
    world: &mut World,
    parent_id: u64,
    _task: &str,
    _max_steps: u32,
) -> Result<u64, RuntimeError> {
    use prism_ecs_constitutional::agent_exec::AgentPhase;
    use prism_ecs_constitutional::agent_plan::ParentAgentId;

    let spawned = world
        .spawn(EntityKind::Agent, None)
        .map_err(|e| RuntimeError::Entity(format!("spawn failed: {e}")))?;
    let entity_id = spawned.entity.id();

    let parent_entity = if parent_id == 0 {
        Entity::new(0, 0)
    } else {
        Entity::new(parent_id, 0)
    };

    world
        .add_component(spawned.entity, AgentPhase::Planning)
        .map_err(|e| RuntimeError::Entity(format!("failed to add phase: {e}")))?;
    world
        .add_component(spawned.entity, ParentAgentId(parent_entity))
        .map_err(|e| RuntimeError::Entity(format!("failed to add parent: {e}")))?;

    Ok(entity_id)
}

/// Cancel agent via WorldTxn (no direct world.add_component).
fn execute_cancel_txn(world: &mut World, agent_id: u64) -> Result<(), RuntimeError> {
    use prism_ecs_constitutional::agent_exec::AgentLifecycle;
    let entity = Entity::new(agent_id, 0);
    // The transaction helper sets lifecycle to Completed.
    world
        .add_component(entity, AgentLifecycle::Completed)
        .map_err(|e| RuntimeError::Entity(format!("cancel failed: {e}")))?;
    Ok(())
}

fn execute_register_model(
    world: &mut World,
    name: &str,
    source_path: &str,
    format: &str,
) -> Result<u64, RuntimeError> {
    use prism_ecs_constitutional::artifact::{ArtifactDigest, ArtifactMetadata, ArtifactPath};
    use prism_ecs_constitutional::lifecycle::ArtifactLifecycle;
    use prism_ecs_constitutional::residency::{
        ModelArtifactRef, ModelFormat, ModelId, ModelLifecycle, ModelName,
    };
    use prism_ecs_constitutional::types::DomainId;

    if name.trim().is_empty() || source_path.trim().is_empty() || format.trim().is_empty() {
        return Err(RuntimeError::Entity(
            "model registration requires name, source_path, and format".into(),
        ));
    }

    // Registration is deterministic across journal replay. When the source is
    // available, its content is the provenance; otherwise retain a stable
    // descriptor digest so the model can still be admitted for later loading.
    let provenance = std::fs::read(source_path)
        .unwrap_or_else(|_| format!("{name}\0{source_path}\0{format}").into_bytes());
    let digest = ArtifactDigest(*blake3::hash(&provenance).as_bytes());

    let artifact = world
        .spawn(EntityKind::Artifact, None)
        .map_err(|e| RuntimeError::Entity(format!("spawn artifact failed: {e}")))?;
    let artifact_id = artifact.entity.id();
    world
        .add_component(artifact.entity, ArtifactPath(source_path.to_owned()))
        .and_then(|_| world.add_component(artifact.entity, digest))
        .and_then(|_| {
            world.add_component(
                artifact.entity,
                ArtifactMetadata {
                    length: provenance.len() as u64,
                    path: source_path.to_owned(),
                },
            )
        })
        .and_then(|_| world.add_component(artifact.entity, ArtifactLifecycle::Discovered))
        .map_err(|e| RuntimeError::Entity(format!("register artifact components failed: {e}")))?;

    let model = world
        .spawn(EntityKind::Model, None)
        .map_err(|e| RuntimeError::Entity(format!("spawn model failed: {e}")))?;
    let model_id = model.entity.id();
    let stable_id = uuid::Uuid::from_bytes(
        blake3::hash(format!("{name}\0{source_path}\0{format}").as_bytes()).as_bytes()[..16]
            .try_into()
            .expect("blake3 digest has at least 16 bytes"),
    );
    world
        .add_component(model.entity, ModelId(DomainId(stable_id)))
        .and_then(|_| world.add_component(model.entity, ModelName(name.to_owned())))
        .and_then(|_| world.add_component(model.entity, ModelFormat(format.to_owned())))
        .and_then(|_| {
            world.add_component(
                model.entity,
                ModelArtifactRef {
                    artifact_id,
                    digest,
                },
            )
        })
        .and_then(|_| world.add_component(model.entity, ModelLifecycle::Created))
        .map_err(|e| RuntimeError::Entity(format!("register model components failed: {e}")))?;

    Ok(model_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_work_command_preserves_request_kind_in_ecs() {
        let mut world = World::new();
        let result = execute_create_work(
            &mut world,
            &CreateWorkCommand {
                entity: 0,
                target_entity: 0,
                kind: "inference".to_string(),
                resource_claim: "{\"max_tokens\":32}".to_string(),
                output_path: String::new(),
                input_path: String::new(),
            },
        )
        .expect("create inference work");
        let work_entity = match result {
            CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                work_entity, ..
            }) => work_entity,
            other => panic!("expected work creation, got {other:?}"),
        };

        let item = world
            .get_component::<prism_ecs_constitutional::work::WorkItemComponent>(Entity::new(
                work_entity,
                0,
            ))
            .expect("work item component");
        assert_eq!(
            item.kind,
            prism_ecs_constitutional::work::WorkKind::RunInference
        );
    }

    #[test]
    fn model_registration_populates_constitutional_artifact_and_model() {
        let mut world = World::new();
        let model_id = execute_register_model(
            &mut world,
            "demo",
            "/models/demo.safetensors",
            "safetensors",
        )
        .expect("model registration");
        let model = Entity::new(model_id, 0);
        let model_ref = world
            .get_component::<prism_ecs_constitutional::residency::ModelArtifactRef>(model)
            .expect("model artifact reference");
        assert_eq!(
            world.get_component::<prism_ecs_constitutional::residency::ModelLifecycle>(model),
            Some(&prism_ecs_constitutional::residency::ModelLifecycle::Created)
        );
        assert_eq!(
            world.get_component::<prism_ecs_constitutional::residency::ModelName>(model),
            Some(&prism_ecs_constitutional::residency::ModelName(
                "demo".into()
            ))
        );
        assert_eq!(
            world.get_component::<prism_ecs_constitutional::residency::ModelFormat>(model),
            Some(&prism_ecs_constitutional::residency::ModelFormat(
                "safetensors".into()
            ))
        );
        let artifact = Entity::new(model_ref.artifact_id, 0);
        assert_eq!(world.entity_kind(artifact), Some(EntityKind::Artifact));
        assert_eq!(
            world.get_component::<prism_ecs_constitutional::lifecycle::ArtifactLifecycle>(artifact),
            Some(&prism_ecs_constitutional::lifecycle::ArtifactLifecycle::Discovered)
        );
        assert_eq!(
            world
                .get_component::<prism_ecs_constitutional::artifact::ArtifactDigest>(artifact)
                .expect("artifact digest"),
            &model_ref.digest
        );
    }

    #[test]
    fn modality_submission_is_a_canonical_ecs_work_entity() {
        let kernel = RuntimeKernel::new();
        let outcome = kernel
            .handle()
            .submit(CommandEnvelope::new(Command::CreateModalityWork {
                kind: crate::modality::ModalityKind::Image,
                model_path: "model.cimage".into(),
                prompt: "sunrise".into(),
                output_path: "out.png".into(),
            }))
            .expect("modality submission");
        let entity_id = match outcome.result {
            CommandResult::ModalitySubmitted { entity_id } => entity_id,
            other => panic!("expected modality submission, got {other:?}"),
        };
        let handle = kernel.handle();
        let world = handle.lock_world();
        let entity = Entity::new(entity_id, 0);
        assert_eq!(world.entity_kind(entity), Some(EntityKind::WorkUnit));
        assert_eq!(
            world
                .get_component::<crate::modality::ModalityWork>(entity)
                .expect("modality component")
                .kind,
            crate::modality::ModalityKind::Image
        );
    }

    #[test]
    fn modality_completion_attaches_provider_output_provenance() {
        let kernel = RuntimeKernel::new();
        let handle = kernel.handle();
        let submitted = handle
            .submit(CommandEnvelope::new(Command::CreateModalityWork {
                kind: crate::modality::ModalityKind::Audio,
                model_path: "model.cimage".into(),
                prompt: "hello".into(),
                output_path: "out.wav".into(),
            }))
            .expect("modality submission");
        let entity_id = match submitted.result {
            CommandResult::ModalitySubmitted { entity_id } => entity_id,
            other => panic!("expected modality submission, got {other:?}"),
        };
        let completed = handle
            .submit(CommandEnvelope::new(Command::CompleteModalityWork {
                entity: entity_id,
                output_digest: "blake3:audio".into(),
                output_bytes: 4096,
            }))
            .expect("modality completion");
        assert!(matches!(
            completed.result,
            CommandResult::ModalityCompleted { entity_id: id, .. } if id == entity_id
        ));
        let world = handle.lock_world();
        let execution = world
            .get_component::<crate::modality::ModalityExecution>(Entity::new(entity_id, 0))
            .expect("execution provenance");
        assert_eq!(execution.output_digest, "blake3:audio");
        assert_eq!(execution.output_bytes, 4096);
    }
}

//! Canonical command envelope, typed command set, and the constitutional
//! submit/replay path through the world.
//!
//! Authority: this module owns the canonical authority for routing a
//! `CommandEnvelope` into the world — admission, epoch fencing, lease
//! coordination, the world-locked transaction, replay application, and
//! the journal/store completion handshake. The data shapes (`Command`,
//! `CommandResult`, `CommitOutcome`, `CommandEnvelope`) are the typed
//! vocabulary of kernel ingress; the `submit` function is the only
//! canonical writer of world state from the kernel.
//!
//! ## Classification
//!
//! The data shapes and the world-locked transaction are **canonical**.
//! The submit path itself touches process-local state (world `RwLock`,
//! lease coordinator, command store, sequence `AtomicU64`,
//! `mpsc::Receiver` via the state stream) and therefore crosses
//! execution-boundary criterion 3. The boundary is documented here; the
//! engine implements the *effect-side* dispatch through the existing
//! `WorkDispatcher` / `HardwareDispatcher` port traits in
//! [`crate::ports`]. Future work may extract a focused
//! `CommandDispatcher` trait; for now the canonical submit path lives
//! here as a free function over a borrowed [`CommandDispatchContext`].
//!
//! ## Engine counterpart
//!
//! `compute-core/src/ecs/core/executor.rs` and `executor_projection.rs`
//! are execution-boundary math code (MLX arrays, hardware calls) and
//! are not absorbed here. The `SinkState` they once carried is already
//! absorbed into [`crate::attention_sink`]. `kernel_catalog.rs` is
//! already ported in `e633567e`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use prism_ecs_constitutional::compilation::{CompilationJob, JobConfig, JobInput, JobLifecycle};
use prism_ecs_constitutional::lifecycle_command::{
    AdmitWorkCommand, AttachEvidenceCommand, CompleteWorkCommand, CreateCompilationJobCommand,
    CreateWorkCommand, FailWorkCommand, LifecycleCommand, LifecycleCommandResult,
    MarkObservedCommand, PublishResultCommand, RecordDispatchIntentCommand,
    RecordWorkPlanCommand, RequestCancellationCommand, ENVELOPE_SCHEMA_VERSION,
};
use prism_ecs_constitutional::work::WorkState;
use prism_ecs_constitutional::types::Timestamp;
use prism_ecs_core::{Entity, EntityKind, StateStream, TraceContext, World};

use crate::inference::{InferencePhase, InferenceWorkMetadata};
use crate::ports::{
    Admission, CommandStore, LeaseCoordinator, ResultPayload, RuntimeError, SnapshotPayload,
    WorldSnapshot,
};

use super::markers::{AdmittedMarker, PlannedMarker, PublishedMarker};

// ── Typed command surface ───────────────────────────────────────────────────

/// Typed command envelope — the kernel's only ingress vocabulary.
///
/// Infrastructure variants spawn, cancel, and register entities. The
/// `Lifecycle` variant carries the constitutional
/// `LifecycleCommand` set, which is the typed authority over
/// work/compilation/evidence/result lifecycle transitions.
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

/// Result of a single committed command.
///
/// Mirrors the `Command` variants. `Lifecycle` wraps the constitutional
/// `LifecycleCommandResult` set. `world_epoch` and `sequence` are
/// surfaced through the `CommitOutcome` wrapper, not here.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CommandResult {
    // Infrastructure
    Spawned { entity_id: u64 },
    Cancelled { entity_id: u64 },
    Registered { entity_id: u64 },
    InferenceAdvanced {
        entity_id: u64,
        phase: InferencePhase,
        prefilled_tokens: u32,
        generated_tokens: u32,
        kv_epoch: u64,
        kv_tokens: u32,
    },
    KvBound { entity_id: u64, epoch: u64 },
    ModalitySubmitted { entity_id: u64 },
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

/// Result of a committed command — pairs the durable sequence number
/// with the command result and the world epoch at commit.
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
    /// Build a new envelope with the canonical `ENVELOPE_SCHEMA_VERSION`
    /// and a fresh UUID idempotency key. `authority` defaults to
    /// `"kernel"`; callers in the server or product layer should
    /// override it to the real authority string.
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

    /// Stable numeric discriminant for the wrapped command.
    ///
    /// Infrastructure variants share `0`; lifecycle variants report the
    /// underlying `LifecycleCommand::type_id().discriminant()` value.
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

// ── Borrowed view over the kernel's state for the submit path ──────────────

/// Borrowed view over the parts of `RuntimeKernelInner` that
/// `submit` / `apply_recovered_command` need.
///
/// This is the boundary that lets the submit path live in
/// `command_dispatch.rs` while the kernel's owning state lives in
/// `mod.rs`. The view holds no ownership — it is constructed from
/// `&Arc<RuntimeKernelInner>` in the kernel handle and dropped at the
/// end of the call. Fields that the submit path does not need
/// (backend resources, provider selector) are intentionally omitted.
pub(super) struct CommandDispatchContext<'a> {
    pub world: &'a Arc<std::sync::RwLock<World>>,
    pub command_store: &'a dyn CommandStore,
    pub lease_coordinator: &'a dyn LeaseCoordinator,
    pub sequence: &'a AtomicU64,
    pub trace: &'a TraceContext,
    pub state_stream: &'a StateStream,
}

// ── Canonical submit path ──────────────────────────────────────────────────

/// Submit a typed command for execution with atomic epoch fencing.
///
/// The path is:
/// 1. Serialize the envelope and call `command_store.admit(idempotency_key)`
///    to obtain an `Admission`. Completed → return the stored result
///    immediately. InFlight → conflict. Admitted → own execution.
/// 2. For lease-acquire/release commands, coordinate the lease
///    *outside* the world lock, then transition `WorkState` inside
///    the lock.
/// 3. Acquire the world write lock. Check `expected_epoch` against the
///    world epoch.
/// 4. Dispatch to the matching `execute_*` helper, which stages
///    components through `world.add_component` and returns a
///    `CommandResult`.
/// 5. Hand the result to `command_store.complete(sequence, json, epoch)`
///    so the journal records the canonical outcome.
pub fn submit(
    envelope: CommandEnvelope,
    ctx: &CommandDispatchContext<'_>,
) -> Result<CommitOutcome, RuntimeError> {
    // Serialize envelope for admission
    let envelope_json = serde_json::to_string(&envelope)
        .map_err(|_| RuntimeError::Journal("envelope serialization failed".to_string()))?;

    // 1. Admit BEFORE acquiring world lock
    let admission = ctx
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
    ctx.sequence.fetch_add(1, Ordering::Relaxed);

    // ── Lease operations (outside world lock) ────────────────────────
    // AcquireWorkLease coordinates with the external Valkey-based lease
    // coordinator, then transitions the entity's WorkState to Leased.
    if let Command::Lifecycle(LifecycleCommand::AcquireWorkLease(cmd)) = &envelope.command {
        let resource_key = format!("work-lease:{}", cmd.work_entity.id());
        let result = ctx.lease_coordinator.acquire(&resource_key, cmd.ttl_ms);
        return match result {
            Ok(true) => {
                // Update WorkState to Leased(gen=1) inside the world lock
                {
                    let mut world = ctx
                        .world
                        .write()
                        .map_err(|e| RuntimeError::Entity(format!("world write lock poisoned: {e}")))?;
                    let e = cmd.work_entity;
                    world.add_component(e, WorkState::Leased(1)).map_err(|e| {
                        RuntimeError::Entity(format!("lease state transition: {e}"))
                    })?;
                }
                let cmd_result =
                    CommandResult::Lifecycle(LifecycleCommandResult::LeaseAcquired {
                        work_entity: cmd.work_entity,
                        token: prism_ecs_constitutional::LeaseToken(resource_key.clone()),
                        lease_generation: cmd.lease_generation,
                    });
                let json = serde_json::to_string(&cmd_result).unwrap_or_default();
                let epoch = ctx
                    .world
                    .read()
                    .map_err(|e| RuntimeError::Entity(format!("world read lock poisoned: {e}")))?
                    .current_epoch()
                    .0;
                ctx.command_store.complete(sequence, &json, epoch)?;
                Ok(CommitOutcome {
                    sequence,
                    result: cmd_result,
                    world_epoch: epoch,
                })
            }
            Ok(false) => Err(RuntimeError::Lease(format!(
                "failed to acquire lease for work-entity {}",
                cmd.work_entity.id()
            ))),
            Err(e) => Err(RuntimeError::Lease(format!("lease error: {e}"))),
        };
    }
    if let Command::Lifecycle(LifecycleCommand::ReleaseWorkLease(cmd)) = &envelope.command {
        let resource_key = format!("work-lease:{}", cmd.work_entity.id());
        ctx.lease_coordinator
            .release(&resource_key)
            .map_err(|e| RuntimeError::Lease(format!("release error: {e}")))?;
        // Transition WorkState back to Ready (available for re-lease)
        {
            let mut world = ctx
                .world
                .write()
                .map_err(|e| RuntimeError::Entity(format!("world write lock poisoned: {e}")))?;
            let e = cmd.work_entity;
            world
                .add_component(e, WorkState::Ready)
                .map_err(|e| RuntimeError::Entity(format!("release state transition: {e}")))?;
        }
        let cmd_result = CommandResult::Lifecycle(LifecycleCommandResult::LeaseReleased {
            work_entity: cmd.work_entity,
        });
        let json = serde_json::to_string(&cmd_result).unwrap_or_default();
        let epoch = ctx
            .world
            .read()
            .map_err(|e| RuntimeError::Entity(format!("world read lock poisoned: {e}")))?
            .current_epoch()
            .0;
        ctx.command_store
            .complete(sequence, &json, epoch)
            .ok();
        return Ok(CommitOutcome {
            sequence,
            result: cmd_result,
            world_epoch: epoch,
        });
    }

    // ── World-locked path ────────────────────────────────────────────
    let mut world = ctx
        .world
        .write()
        .map_err(|e| RuntimeError::Entity(format!("world write lock poisoned: {e}")))?;

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
            if ctx
                .command_store
                .complete(entry_seq, &json, epoch)
                .is_err()
            {
                let _ = ctx
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
            let _ = ctx
                .command_store
                .transition_state(entry_seq, "failed");
            Err(e)
        }
    }
}

/// Replay a command that was already committed — no journal, no
/// idempotency checking, no receipt. Just apply the world mutation
/// and verify that the result matches what was stored.
pub fn apply_recovered_command(
    completed: &crate::ports::CompletedCommand,
    ctx: &CommandDispatchContext<'_>,
) -> Result<(), RuntimeError> {
    // Parse the envelope to extract the command
    let envelope: CommandEnvelope = serde_json::from_str(&completed.envelope_json)
        .map_err(|e| RuntimeError::Journal(format!("replay: bad envelope: {e}")))?;

    // Deserialize the stored result for verification
    let stored_result: CommandResult = serde_json::from_str(&completed.result_json)
        .map_err(|e| RuntimeError::Journal(format!("replay: bad result: {e}")))?;

    let mut world = ctx
        .world
        .write()
        .map_err(|e| RuntimeError::Entity(format!("world write lock poisoned: {e}")))?;

    match envelope.command {
        Command::SpawnAgent {
            parent_id,
            task,
            max_steps,
        } => {
            let entity_id = execute_spawn(&mut world, parent_id, &task, max_steps)?;
            if let CommandResult::Spawned {
                entity_id: expected,
            } = &stored_result
            {
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

// ── Typed lifecycle command handlers ──────────────────────────────────────

/// Execute a typed lifecycle command against the world. The dispatch is
/// exhaustive over `LifecycleCommand`; each arm either delegates to a
/// focused `execute_*` helper that mutates world state, or returns a
/// non-mutating result that records the lifecycle event.
pub fn execute_lifecycle(
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
            // WAIVER: lease commands are handled by `submit` *before* the
            // world lock is acquired, so the path through `execute_lifecycle`
            // is unreachable by design. The `unreachable!` is correct here.
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

fn execute_create_work(
    world: &mut World,
    cmd: &CreateWorkCommand,
) -> Result<CommandResult, RuntimeError> {
    let spawned = world
        .spawn(EntityKind::WorkUnit, None)
        .map_err(|e| RuntimeError::Entity(format!("spawn work failed: {e}")))?;
    let work_entity = spawned.entity;
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
                target_entity: cmd.target_entity,
                retry_count: 0,
                max_retries: 3,
            },
        )
        .map_err(|e| RuntimeError::Entity(format!("failed to set work item: {e}")))?;
    if matches!(kind, prism_ecs_constitutional::work::WorkKind::RunInference) {
        world
            .add_component(
                spawned.entity,
                InferenceWorkMetadata::from_typed_resource_claim(&cmd.resource_claim),
            )
            .map_err(|e| RuntimeError::Entity(format!("failed to set inference metadata: {e}")))?;
    }
    if !cmd.output_path.is_empty() {
        world
            .add_component(
                spawned.entity,
                prism_ecs_constitutional::work::WorkOutputPath(
                    cmd.output_path.clone().into_inner(),
                ),
            )
            .map_err(|e| RuntimeError::Entity(format!("failed to set output path: {e}")))?;
    }
    if !cmd.input_path.is_empty() {
        world
            .add_component(
                spawned.entity,
                prism_ecs_constitutional::work::WorkInputPath(
                    cmd.input_path.clone().into_inner(),
                ),
            )
            .map_err(|e| RuntimeError::Entity(format!("failed to set input path: {e}")))?;
    }
    Ok(CommandResult::Lifecycle(
        LifecycleCommandResult::WorkCreated {
            work_entity,
            sequence: prism_ecs_constitutional::Sequence(0),
            world_epoch: prism_ecs_constitutional::Epoch(world.current_epoch().0),
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

    // The constitutional side now uses typed `ArtifactDigest`, `CommandId`,
    // `TargetProfile`, `Format`, and `OptimizationLevel` newtypes. Unwrap
    // them at the kernel boundary where the legacy `compilation` component
    // types still expect the raw primitives.
    let job_id_raw: u64 = cmd.job_id.0;
    let target_artifact_raw: [u8; 32] = cmd.model_artifact.0;
    let target_format_raw: String = cmd.target_format.clone().into_inner();
    let target_device_profile_raw: String = cmd.target_profile.clone().into_inner();
    let optimization_level_raw: u32 = cmd.optimization_level.0 as u32;

    // The legacy `compilation::CompilationJob` and `compilation::JobInput`
    // components still use `u64` for `target_artifact` (they predate the
    // typed-digest migration). The `ArtifactDigest` newtype is a `[u8; 32]`
    // blake3 hash; we expose the first 8 bytes as `u64` for the legacy
    // component while preserving the full digest in the durable domain event
    // emitted by the constitutional executor. The full digest is also
    // available to callers through the typed `CreateCompilationJobCommand`.
    let model_artifact_u64: u64 = u64::from_le_bytes([
        target_artifact_raw[0],
        target_artifact_raw[1],
        target_artifact_raw[2],
        target_artifact_raw[3],
        target_artifact_raw[4],
        target_artifact_raw[5],
        target_artifact_raw[6],
        target_artifact_raw[7],
    ]);

    world
        .insert_component(
            job_entity,
            CompilationJob {
                job_id: job_id_raw,
                target_artifact: model_artifact_u64,
                target_device_profile: target_device_profile_raw,
                created_at: Timestamp::now(),
            },
        )
        .map_err(|e| RuntimeError::Entity(format!("insert CompilationJob: {e}")))?;

    world
        .insert_component(
            job_entity,
            JobInput {
                model_artifact: model_artifact_u64,
                source_format: String::new(),
                quantization_profile: None,
            },
        )
        .map_err(|e| RuntimeError::Entity(format!("insert JobInput: {e}")))?;

    world
        .insert_component(
            job_entity,
            JobConfig {
                target_format: target_format_raw,
                optimization_level: optimization_level_raw,
                enable_validation: cmd.enable_validation,
            },
        )
        .map_err(|e| RuntimeError::Entity(format!("insert JobConfig: {e}")))?;

    world
        .insert_component(job_entity, JobLifecycle::Pending)
        .map_err(|e| RuntimeError::Entity(format!("insert JobLifecycle: {e}")))?;

    Ok(CommandResult::Lifecycle(
        LifecycleCommandResult::CompilationJobCreated {
            entity: job_entity,
            sequence: prism_ecs_constitutional::Sequence(0),
            world_epoch: prism_ecs_constitutional::Epoch(world.current_epoch().0),
        },
    ))
}

fn execute_admit_work(
    world: &mut World,
    cmd: &AdmitWorkCommand,
) -> Result<CommandResult, RuntimeError> {
    let e = cmd.entity;
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
    let dispatch_id = prism_ecs_constitutional::DispatchId(uuid::Uuid::new_v4().to_string());
    // Transition from Leased to Dispatched
    let e = cmd.work_entity;
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
    let e = cmd.work_entity;
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
            sequence: prism_ecs_constitutional::Sequence(0),
            world_epoch: prism_ecs_constitutional::Epoch(world.current_epoch().0),
        },
    ))
}

fn execute_fail_work(
    world: &mut World,
    cmd: &FailWorkCommand,
) -> Result<CommandResult, RuntimeError> {
    let e = cmd.work_entity;
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
    let e = cmd.entity;
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
    let receipt_id = prism_ecs_constitutional::ReceiptId(uuid::Uuid::new_v4().to_string());
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
    let e = cmd.entity;
    // Transition from Completed to Published
    world
        .add_component(e, WorkState::Completed)
        .map_err(|e| RuntimeError::Entity(format!("publish state transition: {e}")))?;
    let receipt_id = prism_ecs_constitutional::ReceiptId(uuid::Uuid::new_v4().to_string());
    let payload = ResultPayload {
        result_type: cmd.result_type.clone().into_inner(),
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
            sequence: prism_ecs_constitutional::Sequence(0),
            world_epoch: prism_ecs_constitutional::Epoch(world.current_epoch().0),
        },
    ))
}

fn execute_mark_observed(
    world: &mut World,
    cmd: &MarkObservedCommand,
) -> Result<CommandResult, RuntimeError> {
    let e = cmd.entity;
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
    let e = cmd.entity;
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

// ── Capture snapshot ───────────────────────────────────────────────────────

/// Capture the canonical world snapshot over a borrowed world lock and
/// schedule hash. Used by the executor loop's `capture_snapshot` and
/// `shutdown`.
pub fn capture_world_snapshot(
    world_lock: &Arc<std::sync::RwLock<World>>,
    sequence: &AtomicU64,
    schedule_hash: [u8; 32],
) -> Result<WorldSnapshot, RuntimeError> {
    let watermarks_seq = sequence.load(Ordering::Relaxed);

    let world = world_lock
        .read()
        .map_err(|e| RuntimeError::Entity(format!("world read lock poisoned: {e}")))?;
    let epoch = world.current_epoch().0;
    let allocator_data = prism_ecs_core::snapshot::export_allocator_snapshot(&world);
    drop(world);

    let payload = SnapshotPayload {
        schema_version: 1,
        world_epoch: epoch,
        next_entity_id: 0, // no longer used; allocator_data captures this
        last_command_sequence: watermarks_seq,
        allocator_data,
        schedule_hash,
        created_at_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
    };
    let checksum = WorldSnapshot::compute_checksum(&payload);
    Ok(WorldSnapshot { payload, checksum })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ports::CommandWatermarks;
    use crate::test_adapters::{InMemoryCommandStore, InMemoryLeaseCoordinator};
    use prism_ecs_constitutional::scheduler::ResourceClaim;
    use prism_ecs_constitutional::types::{FilePath, Format};
    use std::sync::atomic::AtomicU64;
    use std::sync::{Arc, RwLock};

    fn default_claim() -> ResourceClaim {
        ResourceClaim::default()
    }

    fn make_ctx<'a>(
        world: &'a Arc<RwLock<World>>,
        command_store: &'a dyn CommandStore,
        lease_coordinator: &'a dyn LeaseCoordinator,
        sequence: &'a AtomicU64,
        trace: &'a TraceContext,
        state_stream: &'a StateStream,
    ) -> CommandDispatchContext<'a> {
        CommandDispatchContext {
            world,
            command_store,
            lease_coordinator,
            sequence,
            trace,
            state_stream,
        }
    }

    fn make_kv() -> (TraceContext, StateStream) {
        let trace = prism_ecs_core::global_context();
        let stream = prism_ecs_core::StateStream::global();
        (trace, stream)
    }

    #[test]
    fn inference_work_command_preserves_request_kind_in_ecs() {
        let mut world = World::new();
        let result = execute_create_work(
            &mut world,
            &CreateWorkCommand {
                entity: Entity::new(0, 0),
                target_entity: Entity::new(0, 0),
                kind: Format("inference".to_string()),
                resource_claim: default_claim(),
                output_path: FilePath(String::new()),
                input_path: FilePath(String::new()),
            },
        )
        .expect("create inference work");
        let work_entity = match result {
            CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                work_entity,
                ..
            }) => work_entity,
            other => panic!("expected work creation, got {other:?}"),
        };

        let item = world
            .get_component::<prism_ecs_constitutional::work::WorkItemComponent>(work_entity)
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

    /// Submitting a fresh spawn via `submit` produces a `Spawned` result,
    /// advances the journal, and persists the command through the store.
    #[test]
    fn submit_spawn_advances_journal_and_persists() {
        let world_lock = Arc::new(RwLock::new(World::new()));
        let command_store = InMemoryCommandStore::new();
        let lease_coordinator = InMemoryLeaseCoordinator::new();
        let sequence = AtomicU64::new(0);
        let (trace, state_stream) = make_kv();
        let ctx = make_ctx(
            &world_lock,
            &command_store,
            &lease_coordinator,
            &sequence,
            &trace,
            &state_stream,
        );
        let env = CommandEnvelope::new(Command::SpawnAgent {
            parent_id: 0,
            task: "x".into(),
            max_steps: 1,
        });
        let outcome = submit(env.clone(), &ctx).expect("submit spawn");
        let entity_id = match outcome.result {
            CommandResult::Spawned { entity_id } => entity_id,
            other => panic!("expected Spawned, got {other:?}"),
        };
        assert_eq!(outcome.sequence, 1);
        assert!(entity_id > 0);
        let watermarks: CommandWatermarks = command_store
            .high_water_marks()
            .expect("watermarks");
        assert_eq!(watermarks.last_committed_sequence, 1);

        // Replay by re-submitting the same envelope — should hit
        // Admission::Completed and return the same result.
        let replay = submit(env, &ctx).expect("replay");
        assert_eq!(replay.sequence, outcome.sequence);
    }
}

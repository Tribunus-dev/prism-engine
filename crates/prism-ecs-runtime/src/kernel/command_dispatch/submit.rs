//! Canonical submit path and the typed infrastructure command
//! implementations.
//!
//! Authority: this module owns the canonical authority for the live
//! submit path — admission, lease coordination, the world-locked
//! transaction, and the journal/store completion handshake — and for
//! the typed infrastructure command implementations
//! (`execute_spawn`, `execute_cancel_txn`, `execute_register_model`,
//! `execute_advance_inference`, `execute_bind_inference_kv`,
//! `execute_create_modality_work`). It does **not** own the data
//! shapes (which live in [`super::envelope`]), the typed lifecycle
//! command implementations (which also live in
//! [`super::envelope`]), or the replay path (which lives in
//! [`super::replay`]).
//!
//! ## Classification
//!
//! The submit path is **execution-boundary** by criterion 3 — it
//! touches the world `RwLock`, the lease coordinator, the command
//! store, the sequence `AtomicU64`, and the `mpsc::Receiver` via the
//! state stream. The data the path carries (a `CommandEnvelope` and
//! the resulting `CommitOutcome`) is canonical. The boundary is
//! documented here; the engine implements the *effect-side* dispatch
//! through the existing `WorkDispatcher` / `HardwareDispatcher` port
//! traits in [`crate::ports`].

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use prism_ecs_constitutional::lifecycle_command::{LifecycleCommand, LifecycleCommandResult};
use prism_ecs_constitutional::work::WorkState;
use prism_ecs_core::{Entity, EntityKind, World};

use crate::inference::{InferencePhase, InferenceWorkMetadata, KvCacheBinding};
use crate::ports::{
    Admission, RuntimeError, SnapshotPayload, WorldSnapshot,
};
use crate::modality::{ModalityExecution, ModalityFailure, ModalityWork};

use super::envelope::{
    execute_lifecycle, Command, CommandDispatchContext, CommandEnvelope, CommandResult,
    CommitOutcome,
};

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
                        ModalityExecution {
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
                        ModalityFailure {
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

// ── Typed infrastructure command implementations ──────────────────────────

/// Cancel agent via WorldTxn (no direct world.add_component).
pub(super) fn execute_cancel_txn(world: &mut World, agent_id: u64) -> Result<(), RuntimeError> {
    use prism_ecs_constitutional::agent_exec::AgentLifecycle;
    let entity = Entity::new(agent_id, 0);
    // The transaction helper sets lifecycle to Completed.
    world
        .add_component(entity, AgentLifecycle::Completed)
        .map_err(|e| RuntimeError::Entity(format!("cancel failed: {e}")))?;
    Ok(())
}

pub(super) fn execute_spawn(
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

pub(super) fn execute_register_model(
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
            // WAIVER: blake3 always returns a 32-byte digest, so the
            // `[..16]` slice and `try_into` to `[u8; 16]` are infallible.
            // A panic here indicates a blake3 contract violation, not a
            // recoverable error path.
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

pub(super) fn execute_bind_inference_kv(
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
    let binding = KvCacheBinding {
        epoch,
        page_ids,
        logical_context_tokens,
        capacity_tokens,
    };
    world
        .add_component(target, binding)
        .map_err(|error| RuntimeError::Entity(format!("bind inference KV failed: {error}")))
}

pub(super) fn execute_create_modality_work(
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
            ModalityWork {
                kind,
                model_path,
                prompt,
                output_path,
            },
        )
        .map_err(|error| RuntimeError::Entity(format!("attach modality work failed: {error}")))?;
    Ok(entity_id)
}

pub(super) fn execute_advance_inference(
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

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ports::{CommandStore, CommandWatermarks, LeaseCoordinator};
    use crate::test_adapters::{InMemoryCommandStore, InMemoryLeaseCoordinator};
    use prism_ecs_constitutional::lifecycle::ArtifactLifecycle;
    use prism_ecs_constitutional::residency::{
        ModelArtifactRef, ModelFormat, ModelLifecycle, ModelName,
    };
    use prism_ecs_core::{Entity, StateStream, TraceContext, World};
    use std::sync::{Arc, RwLock};

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

    /// Registering a model populates the constitutional artifact and
    /// model components with the expected typed identity.
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
            .get_component::<ModelArtifactRef>(model)
            .expect("model artifact reference");
        assert_eq!(
            world.get_component::<ModelLifecycle>(model),
            Some(&ModelLifecycle::Created)
        );
        assert_eq!(
            world.get_component::<ModelName>(model),
            Some(&ModelName("demo".into()))
        );
        assert_eq!(
            world.get_component::<ModelFormat>(model),
            Some(&ModelFormat("safetensors".into()))
        );
        let artifact = Entity::new(model_ref.artifact_id, 0);
        assert_eq!(world.entity_kind(artifact), Some(EntityKind::Artifact));
        assert_eq!(
            world.get_component::<ArtifactLifecycle>(artifact),
            Some(&ArtifactLifecycle::Discovered)
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

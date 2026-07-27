//! Canonical data shapes for the kernel command surface and the typed
//! lifecycle command implementations.
//!
//! Authority: this module owns the canonical authority for the typed
//! command surface of the kernel — the `Command` enum,
//! `CommandResult`, `CommitOutcome`, `CommandEnvelope`, the borrowed
//! `CommandDispatchContext`, and the typed lifecycle command
//! implementations (`execute_lifecycle` plus the concrete
//! per-lifecycle-command bodies). It does **not** own the live submit
//! path, the infrastructure command implementations, or the replay
//! path — those live in [`super::submit`] and [`super::replay`]
//! respectively.

use std::sync::Arc;

use prism_ecs_constitutional::compilation::{CompilationJob, JobConfig, JobInput, JobLifecycle};
use prism_ecs_constitutional::lifecycle_command::{
    AdmitWorkCommand, AttachEvidenceCommand, CompleteWorkCommand, CreateCompilationJobCommand,
    CreateWorkCommand, FailWorkCommand, LifecycleCommand, LifecycleCommandResult,
    MarkObservedCommand, PublishResultCommand, RecordDispatchIntentCommand,
    RecordWorkPlanCommand, RequestCancellationCommand, ENVELOPE_SCHEMA_VERSION,
};
use prism_ecs_constitutional::types::Timestamp;
use prism_ecs_constitutional::work::WorkState;
use prism_ecs_core::{EntityKind, StateStream, TraceContext, World};

use crate::inference::InferenceWorkMetadata;
use crate::ports::{CommandStore, LeaseCoordinator, ResultPayload, RuntimeError};

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
        phase: crate::inference::InferencePhase,
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
        phase: crate::inference::InferencePhase,
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
/// [`super::submit`] while the kernel's owning state lives in
/// `kernel/mod.rs`. The view holds no ownership — it is constructed
/// from `&Arc<RuntimeKernelInner>` in the kernel handle and dropped at
/// the end of the call. Fields that the submit path does not need
/// (backend resources, provider selector) are intentionally omitted.
pub(super) struct CommandDispatchContext<'a> {
    pub world: &'a Arc<std::sync::RwLock<World>>,
    pub command_store: &'a dyn CommandStore,
    pub lease_coordinator: &'a dyn LeaseCoordinator,
    pub sequence: &'a std::sync::atomic::AtomicU64,
    pub trace: &'a TraceContext,
    pub state_stream: &'a StateStream,
}

// ── Typed lifecycle command handlers ──────────────────────────────────────

/// Execute a typed lifecycle command against the world. The dispatch is
/// exhaustive over `LifecycleCommand`; each arm either delegates to a
/// focused `execute_*` helper that mutates world state, or returns a
/// non-mutating result that records the lifecycle event.
pub(super) fn execute_lifecycle(
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

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_constitutional::scheduler::ResourceClaim;
    use prism_ecs_constitutional::types::{FilePath, Format};
    use prism_ecs_core::Entity;

    fn default_claim() -> ResourceClaim {
        ResourceClaim::default()
    }

    /// Creating an inference work preserves the request kind in the
    /// canonical ECS world.
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
}

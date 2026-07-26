//! Runtime-facing lifecycle command adapter.
//!
//! This aggregate preserves the runtime protocol while delegating semantic
//! state transitions to the constitutional work components. It intentionally
//! does not introduce legacy `WorkState` variants.
use crate::artifact::ArtifactDigest;
use crate::scheduler::ResourceClaim;
use crate::types::{
    AdapterHandle, CommandId, Config, DispatchId, Epoch, FilePath, Format, Generation,
    LeaseToken, OptimizationLevel, ReceiptId, RejectionReason, Sequence, TargetProfile,
};
use prism_ecs_kernel::BackendKind;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleCommand {
    CreateWork(CreateWorkCommand),
    CreateCompilationJob(CreateCompilationJobCommand),
    RequestCancellation(RequestCancellationCommand),
    MarkObserved(MarkObservedCommand),
    RecordExternalObservation(RecordExternalObservationCommand),
    RecordWorkPlan(RecordWorkPlanCommand),
    MarkPrerequisiteBlocked(MarkPrerequisiteBlockedCommand),
    AdmitWork(AdmitWorkCommand),
    RejectWork(RejectWorkCommand),
    DeferWork(DeferWorkCommand),
    AcquireWorkLease(AcquireWorkLeaseCommand),
    ReleaseWorkLease(ReleaseWorkLeaseCommand),
    RenewWorkLease(RenewWorkLeaseCommand),
    RecordDispatchIntent(RecordDispatchIntentCommand),
    RecordDispatchStarted(RecordDispatchStartedCommand),
    RecordProgress(RecordProgressCommand),
    CompleteWork(CompleteWorkCommand),
    FailWork(FailWorkCommand),
    MarkDispatchLost(MarkDispatchLostCommand),
    AttachArtifact(AttachArtifactCommand),
    AttachDiagnostics(AttachDiagnosticsCommand),
    AttachEvidence(AttachEvidenceCommand),
    PublishResult(PublishResultCommand),
    ExpireTransientState(ExpireTransientCommand),
    MarkRetentionComplete(MarkRetentionCompleteCommand),
}
impl LifecycleCommand {
    pub fn type_id(&self) -> LifecycleTypeId {
        LifecycleTypeId(self as *const _ as usize as u16)
    }
}
#[derive(Debug, Clone, Copy)]
pub struct LifecycleTypeId(pub u16);
impl LifecycleTypeId {
    pub fn discriminant(&self) -> u16 {
        self.0
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleCommandResult {
    WorkCreated {
        work_entity: prism_ecs_core::Entity,
        sequence: Sequence,
        world_epoch: Epoch,
    },
    CompilationJobCreated {
        entity: prism_ecs_core::Entity,
        sequence: Sequence,
        world_epoch: Epoch,
    },
    RequestCancelled {
        entity: prism_ecs_core::Entity,
    },
    MarkedObserved {
        entity: prism_ecs_core::Entity,
    },
    PrerequisiteBlocked {
        entity: prism_ecs_core::Entity,
    },
    Admitted {
        entity: prism_ecs_core::Entity,
    },
    Rejected {
        entity: prism_ecs_core::Entity,
        reason: RejectionReason,
    },
    Deferred {
        entity: prism_ecs_core::Entity,
        reason: RejectionReason,
    },
    LeaseAcquired {
        work_entity: prism_ecs_core::Entity,
        lease_generation: Generation,
        token: LeaseToken,
    },
    LeaseReleased {
        work_entity: prism_ecs_core::Entity,
    },
    LeaseRenewed {
        work_entity: prism_ecs_core::Entity,
        ttl_ms: u64,
    },
    DispatchIntentRecorded {
        work_entity: prism_ecs_core::Entity,
        dispatch_id: DispatchId,
    },
    DispatchStarted {
        work_entity: prism_ecs_core::Entity,
        adapter_handle: AdapterHandle,
    },
    ProgressRecorded {
        work_entity: prism_ecs_core::Entity,
    },
    Completed {
        work_entity: prism_ecs_core::Entity,
        result: String,
        sequence: Sequence,
        world_epoch: Epoch,
    },
    Failed {
        work_entity: prism_ecs_core::Entity,
        error: RejectionReason,
    },
    DispatchMarkedLost {
        work_entity: prism_ecs_core::Entity,
    },
    ArtifactAttached {
        entity: prism_ecs_core::Entity,
        digest: ArtifactDigest,
    },
    DiagnosticsAttached {
        entity: prism_ecs_core::Entity,
    },
    EvidenceAttached {
        entity: prism_ecs_core::Entity,
        receipt_id: ReceiptId,
    },
    Published {
        entity: prism_ecs_core::Entity,
        receipt_id: ReceiptId,
        sequence: Sequence,
        world_epoch: Epoch,
    },
    WorkPlanRecorded {
        entity: prism_ecs_core::Entity,
    },
    TransientExpired {
        entity: prism_ecs_core::Entity,
    },
    RetentionComplete {
        entity: prism_ecs_core::Entity,
    },
}
macro_rules! cmd {($($n:ident{$($f:ident:$t:ty),* $(,)?}),* $(,)?)=>{$(#[derive(Debug,Clone,Serialize,Deserialize)] pub struct $n {$(pub $f:$t),*})*}}
cmd! {
    CreateWorkCommand{
        entity:prism_ecs_core::Entity,
        target_entity:prism_ecs_core::Entity,
        kind:Format,
        input_path:FilePath,
        output_path:FilePath,
        resource_claim:ResourceClaim,
    },
    CreateCompilationJobCommand{
        entity:prism_ecs_core::Entity,
        model_artifact:ArtifactDigest,
        target_profile:TargetProfile,
        job_id:CommandId,
        target_format:Format,
        optimization_level:OptimizationLevel,
        enable_validation:bool,
    },
    RequestCancellationCommand{
        entity:prism_ecs_core::Entity,
        reason:RejectionReason,
    },
    MarkObservedCommand{
        entity:prism_ecs_core::Entity,
        observed_epoch:Epoch,
    },
    RecordExternalObservationCommand{
        entity:prism_ecs_core::Entity,
    },
    RecordWorkPlanCommand{
        entity:prism_ecs_core::Entity,
        backend:BackendKind,
        output_format:Format,
        resource_estimate_bytes:u64,
        timeout_ms:u64,
    },
    MarkPrerequisiteBlockedCommand{
        entity:prism_ecs_core::Entity,
    },
    AdmitWorkCommand{
        entity:prism_ecs_core::Entity,
    },
    RejectWorkCommand{
        entity:prism_ecs_core::Entity,
        reason:RejectionReason,
    },
    DeferWorkCommand{
        entity:prism_ecs_core::Entity,
        reason:RejectionReason,
    },
    AcquireWorkLeaseCommand{
        work_entity:prism_ecs_core::Entity,
        lease_generation:Generation,
        ttl_ms:u64,
    },
    ReleaseWorkLeaseCommand{
        work_entity:prism_ecs_core::Entity,
    },
    RenewWorkLeaseCommand{
        work_entity:prism_ecs_core::Entity,
        ttl_ms:u64,
    },
    RecordDispatchIntentCommand{
        work_entity:prism_ecs_core::Entity,
        backend:BackendKind,
        config:Config,
        deadline_ms:u64,
    },
    RecordDispatchStartedCommand{
        work_entity:prism_ecs_core::Entity,
        adapter_handle:AdapterHandle,
    },
    RecordProgressCommand{
        work_entity:prism_ecs_core::Entity,
    },
    CompleteWorkCommand{
        work_entity:prism_ecs_core::Entity,
        lease_generation:Generation,
        output:Vec<u8>,
        output_path:FilePath,
    },
    FailWorkCommand{
        work_entity:prism_ecs_core::Entity,
        error:RejectionReason,
        lease_generation:Generation,
        retryable:bool,
    },
    MarkDispatchLostCommand{
        work_entity:prism_ecs_core::Entity,
    },
    AttachArtifactCommand{
        entity:prism_ecs_core::Entity,
        digest:ArtifactDigest,
    },
    AttachDiagnosticsCommand{
        entity:prism_ecs_core::Entity,
    },
    AttachEvidenceCommand{
        entity:prism_ecs_core::Entity,
        digest:ArtifactDigest,
    },
    PublishResultCommand{
        entity:prism_ecs_core::Entity,
        result_type:Format,
        result:String, // free-form payload body
    },
    ExpireTransientCommand{
        entity:prism_ecs_core::Entity,
    },
    MarkRetentionCompleteCommand{
        entity:prism_ecs_core::Entity,
    },
}
pub const ENVELOPE_SCHEMA_VERSION: u32 = 1;

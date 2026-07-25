//! Runtime-facing lifecycle command adapter.
//!
//! This aggregate preserves the runtime protocol while delegating semantic
//! state transitions to the constitutional work components. It intentionally
//! does not introduce legacy `WorkState` variants.
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
        work_entity: u64,
        sequence: u64,
        world_epoch: u64,
    },
    CompilationJobCreated {
        entity: u64,
        sequence: u64,
        world_epoch: u64,
    },
    RequestCancelled {
        entity: u64,
    },
    MarkedObserved {
        entity: u64,
    },
    PrerequisiteBlocked {
        entity: u64,
    },
    Admitted {
        entity: u64,
    },
    Rejected {
        entity: u64,
        reason: String,
    },
    Deferred {
        entity: u64,
        reason: String,
    },
    LeaseAcquired {
        work_entity: u64,
        lease_generation: u32,
        token: String,
    },
    LeaseReleased {
        work_entity: u64,
    },
    LeaseRenewed {
        work_entity: u64,
        ttl_ms: u64,
    },
    DispatchIntentRecorded {
        work_entity: u64,
        dispatch_id: String,
    },
    DispatchStarted {
        work_entity: u64,
        adapter_handle: String,
    },
    ProgressRecorded {
        work_entity: u64,
    },
    Completed {
        work_entity: u64,
        result: String,
        sequence: u64,
        world_epoch: u64,
    },
    Failed {
        work_entity: u64,
        error: String,
    },
    DispatchMarkedLost {
        work_entity: u64,
    },
    ArtifactAttached {
        entity: u64,
        digest: String,
    },
    DiagnosticsAttached {
        entity: u64,
    },
    EvidenceAttached {
        entity: u64,
        receipt_id: String,
    },
    Published {
        entity: u64,
        receipt_id: String,
        sequence: u64,
        world_epoch: u64,
    },
    WorkPlanRecorded {
        entity: u64,
    },
    TransientExpired {
        entity: u64,
    },
    RetentionComplete {
        entity: u64,
    },
}
macro_rules! cmd {($($n:ident{$($f:ident:$t:ty),*}),* $(,)?)=>{$(#[derive(Debug,Clone,Serialize,Deserialize)] pub struct $n {$(pub $f:$t),*})*}}
cmd! {
 CreateWorkCommand{entity:u64,target_entity:u64,kind:String,input_path:String,output_path:String,resource_claim:String}, CreateCompilationJobCommand{entity:u64,model_artifact:u64,target_profile:String,job_id:u64,target_format:String,optimization_level:u32,enable_validation:bool}, RequestCancellationCommand{entity:u64, reason:String}, MarkObservedCommand{entity:u64, observed_epoch:u64}, RecordExternalObservationCommand{entity:u64}, RecordWorkPlanCommand{entity:u64,backend:String,output_format:String,resource_estimate_bytes:u64,timeout_ms:u64}, MarkPrerequisiteBlockedCommand{entity:u64}, AdmitWorkCommand{entity:u64}, RejectWorkCommand{entity:u64, reason:String}, DeferWorkCommand{entity:u64, reason:String}, AcquireWorkLeaseCommand{work_entity:u64, lease_generation:u32,ttl_ms:u64}, ReleaseWorkLeaseCommand{work_entity:u64}, RenewWorkLeaseCommand{work_entity:u64, ttl_ms:u64}, RecordDispatchIntentCommand{work_entity:u64,backend:String,config:String,deadline_ms:u64}, RecordDispatchStartedCommand{work_entity:u64, adapter_handle:String}, RecordProgressCommand{work_entity:u64}, CompleteWorkCommand{work_entity:u64, lease_generation:u32, output:Vec<u8>,output_path:String}, FailWorkCommand{work_entity:u64,error:String,lease_generation:u32,retryable:bool}, MarkDispatchLostCommand{work_entity:u64}, AttachArtifactCommand{entity:u64,digest:String}, AttachDiagnosticsCommand{entity:u64}, AttachEvidenceCommand{entity:u64,digest:String}, PublishResultCommand{entity:u64,result_type:String,result:String}, ExpireTransientCommand{entity:u64}, MarkRetentionCompleteCommand{entity:u64}
}
pub const ENVELOPE_SCHEMA_VERSION: u32 = 1;

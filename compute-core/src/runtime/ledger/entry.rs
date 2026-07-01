use serde::{Deserialize, Serialize};

use crate::runtime::components::worker_health::{TerminalStatus, WorkerErrorCategory};
use crate::runtime::components::worker_lifecycle::WorkerRequestPhase;
use crate::runtime::scheduling::metadata::{Stage, SystemId};
use crate::runtime::world::Entity;

use super::digest::ReceiptDigest;

pub const TRANSITION_RECEIPT_SCHEMA_VERSION: u16 = 1;
pub const DEFAULT_TRANSITION_LEDGER_CAPACITY: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeterministicReceiptPayload {
    pub schema_version: u16,
    pub receipt_sequence: u64,
    pub scheduler_epoch: u64,
    pub microcycle: u32,
    pub stage: Stage,
    pub command_count: u32,
    pub commands: Vec<SemanticStampedCommand>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitionReceipt {
    pub payload: DeterministicReceiptPayload,
    pub deterministic_digest: ReceiptDigest,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_at_ns: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticStampedCommand {
    pub stage: Stage,
    pub system_id: SystemId,
    pub entity: Entity,
    pub entity_generation: Option<u64>,
    pub sequence: u64,
    pub command: SemanticCommandPayload,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum SemanticCommandPayload {
    EntitySpawned {
        entity_kind: String,
    },
    EntityDespawned {
        reason: String,
    },
    WorkerRequestPhaseTransitioned {
        from: WorkerRequestPhase,
        to: WorkerRequestPhase,
        cause: String,
    },
    WorkerAssigned {
        worker_id: String,
        assignment_generation: u64,
        request_class: String,
    },
    WorkerHeartbeatObserved {
        worker_id: String,
        assignment_generation: u64,
        sequence: u64,
    },
    WorkerStreamAdvanced {
        worker_id: String,
        token_count: u32,
        tail_length: u16,
        stream_closed: bool,
    },
    WorkerOutcomeRecorded {
        worker_id: String,
        assignment_generation: u64,
        terminal_status: TerminalStatus,
        error_category: Option<WorkerErrorCategory>,
    },
    WorkerWatchdogTriggered {
        worker_id: String,
        assignment_generation: u64,
        reason: String,
    },
    LegacyWorkerOperation {
        operation: String,
        outcome: String,
    },
}

use crate::linux::backend::{BackendKind, DeviceId};
use crate::linux::queue::QueueClass;
use crate::linux::submission::{ValidationMode, SubmissionStatus, SubmissionKind};
use crate::linux::memory::BufferId;

pub type SubmissionId = u64;

#[derive(Debug, Clone)]
pub struct BackendErrorReceipt {
    pub error_message: String,
}

#[derive(Debug, Clone)]
pub struct DeviceSubmissionReceipt {
    pub submission_id: crate::linux::submission::SubmissionHandle,
    pub backend: BackendKind,
    pub device_id: DeviceId,
    pub queue_class: QueueClass,
    pub submitted_at_unix_ms: u64,
    pub completed_at_unix_ms: Option<u64>,
    pub operation_kind: SubmissionKind,
    pub input_buffer_ids: Vec<BufferId>,
    pub output_buffer_ids: Vec<BufferId>,
    pub status: SubmissionStatus,
    pub bytes_transferred: u64,
    pub host_wait_ns: Option<u64>,
    pub device_elapsed_ns: Option<u64>,
    pub validation_mode: ValidationMode,
    pub output_hash: Option<u64>,
    pub cpu_reference_hash: Option<u64>,
    pub error: Option<BackendErrorReceipt>,
}


#[derive(Debug, Clone)]
pub struct ConformanceCase {
    pub name: &'static str,
    pub submission: crate::linux::submission::Submission,
    pub expected_output_hash: u64,
    pub expected_status: crate::linux::submission::SubmissionStatus,
}

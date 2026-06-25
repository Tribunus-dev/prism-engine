use crate::linux::memory::BufferId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOperation {
    SumU32,
    XorU64,
    MinU32,
    MaxU32,
}

#[derive(Debug, Clone)]
pub enum Submission {
    Fill {
        destination: BufferId,
        value: u32,
        element_count: u64,
    },
    Copy {
        source: BufferId,
        destination: BufferId,
        size_bytes: u64,
    },
    Reduction {
        source: BufferId,
        destination: BufferId,
        element_count: u64,
        operation: ReductionOperation,
    },
    DeterministicHash {
        source: BufferId,
        destination: BufferId,
        element_count: u64,
        seed: u64,
    },
    ScanPreparation {
        source: BufferId,
        destination: BufferId,
        element_count: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationMode {
    NotValidated,
    DeviceOnly,
    CpuReferenceCompared,
}

use crate::linux::backend::RuntimeResourceId;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SubmissionHandle {
    pub id: RuntimeResourceId,
    pub queue_id: RuntimeResourceId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmissionStatus {
    Pending,
    Running,
    Complete,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmissionKind {
    Fill,
    Copy,
    Reduction,
    DeterministicHash,
    ScanPreparation,
}

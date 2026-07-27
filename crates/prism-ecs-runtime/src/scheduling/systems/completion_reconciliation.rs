//! Completion reconciliation system (constitutional home).
//!
//! Placeholder for the engine's `completion_bridge.rs` (496 LOC).
//! The full algorithm migrates in step 28. The engine file is the
//! legacy duplicate and is deleted in step 58.
//!
//! The completion-reconciliation system converts a kernel-side
//! completion value into a staged, authoritative state transition.
//! The kernel's completion is non-authoritative until the
//! reconciliation system stages it through `ConstitutionalWorldTxn`.

use std::collections::BTreeMap;

use crate::scheduling::state::lane_work::{WorkCompletion, WorkId, ExecutionLane};

/// Reconciliation result: the staged transition.
#[derive(Debug, Clone)]
pub struct ReconciledCompletion {
    pub work_id: WorkId,
    pub success: bool,
    pub epoch: u64,
}

/// Reconcile a kernel completion into a staged transition.
/// Placeholder: returns a successful reconciliation. The full
/// algorithm (status mapping, error categorization, batched
/// reconciliation) arrives with step 28.
pub fn reconcile(completion: &WorkCompletion) -> ReconciledCompletion {
    ReconciledCompletion {
        work_id: completion.work_id,
        success: completion.success,
        epoch: 0,
    }
}

/// Per-work-id completion buffer. Uses BTreeMap for stable order.
#[derive(Debug, Default)]
pub struct CompletionBuffer {
    pending: BTreeMap<WorkId, WorkCompletion>,
}

impl CompletionBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, completion: WorkCompletion) {
        self.pending.insert(completion.work_id, completion);
    }

    pub fn drain(&mut self) -> Vec<WorkCompletion> {
        let values: Vec<WorkCompletion> = self.pending.values().cloned().collect();
        self.pending.clear();
        values
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::state::lane_work::{
        BackendExecutionTiming, BackendStatus, ExecutionLane, NumericalStatus, PhaseId,
        SlotLeaseId, TimestampQuality, VariantId,
    };

    fn make_completion(id: u64, success: bool) -> WorkCompletion {
        WorkCompletion {
            work_id: WorkId(id),
            phase_id: PhaseId(0),
            variant_id: VariantId(0),
            lane: ExecutionLane::MlxGpu,
            success,
            output_slot: SlotLeaseId(0),
            backend_status: if success {
                BackendStatus::Completed
            } else {
                BackendStatus::Failed("test".into())
            },
            numerical_status: NumericalStatus::Ok,
            timing: BackendExecutionTiming {
                submit_ns: 0,
                backend_start_ns: 0,
                backend_end_ns: 0,
                completion_callback_ns: 0,
                timestamp_quality: TimestampQuality::BackendCallback,
            },
        }
    }

    #[test]
    fn reconcile_preserves_work_id_and_success() {
        // Architectural invariant: the reconciliation result carries
        // the same work_id and success flag as the input completion.
        let c = make_completion(42, true);
        let r = reconcile(&c);
        assert_eq!(r.work_id, c.work_id);
        assert!(r.success);
    }

    #[test]
    fn buffer_push_and_drain() {
        // Architectural invariant: a completion buffer accumulates
        // completions and drains them in stable BTreeMap order.
        let mut buf = CompletionBuffer::new();
        buf.push(make_completion(1, true));
        buf.push(make_completion(2, false));
        assert_eq!(buf.len(), 2);
        let drained = buf.drain();
        assert_eq!(drained.len(), 2);
        assert!(buf.is_empty());
        // Drained order is sorted by work_id (BTreeMap order).
        assert_eq!(drained[0].work_id.0, 1);
        assert_eq!(drained[1].work_id.0, 2);
    }

    #[test]
    fn buffer_dedupes_by_work_id() {
        // Architectural invariant: a second push for the same
        // work_id REPLACES the first (BTreeMap insert semantics).
        // The completion buffer holds the most recent completion
        // for each work_id.
        let mut buf = CompletionBuffer::new();
        buf.push(make_completion(1, true));
        buf.push(make_completion(1, false));
        assert_eq!(buf.len(), 1);
        let drained = buf.drain();
        assert_eq!(drained.len(), 1);
        assert!(!drained[0].success);
    }
}

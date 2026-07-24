// ── Prism LLM Inference — Inference Scheduler ────────────────────────────
//
// Schedules prefill, decode, and auxiliary work for inference sessions.
// Generates monotonic DispatchId values, creates LaneDispatch records with
// the appropriate execution lane and inference phase, and maintains an
// ordered dispatch history for observability.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use crate::runtime::manifest::{ExecutionLane, InferencePhase, SessionId};
use crate::runtime::server_types::{
    CompletionFenceId, DispatchId, LaneDispatch, SlowConsumerAction, StreamBackpressurePolicy,
};

/// A dispatched lane execution paired with the session that requested it.
// Fields are written by the scheduler but read externally only for
// observability — suppress the dead-code lint.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DispatchRecord {
    dispatch: LaneDispatch,
    session_id: SessionId,
    kv_metadata: Option<KvDispatchMetadata>,
    model_id: Option<String>,
}

/// Manages scheduling of inference work across execution lanes.
///
/// Produces monotonic dispatch identifiers and records each scheduled
/// lane dispatch in submission order. Consumers call the appropriate
/// `schedule_*` method for their inference phase and receive a
/// `DispatchId` that identifies the unit of work.
pub struct InferenceScheduler {
    counter: AtomicU64,
    dispatches: Mutex<Vec<DispatchRecord>>,
    modality_receipts: Mutex<Vec<ModalityDispatchReceipt>>,
    observation_sink: Mutex<Option<Arc<dyn Fn(ModalityDispatchReceipt) + Send + Sync>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModalityDispatchReceipt {
    pub dispatch_id: DispatchId,
    pub session_id: SessionId,
    pub model_id: String,
    pub modality: String,
    pub status: String,
    pub output_digest: Option<String>,
    pub output_units: u64,
}

/// Identity and ownership metadata attached to an inference dispatch.
///
/// The page list is deliberately carried alongside the epoch rather than
/// reconstructed from token ranges. This lets completion and stale-result
/// fencing prove that a dispatch used the exact KV pages owned by the
/// session at submission time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvDispatchMetadata {
    pub model_id: String,
    pub epoch_id: crate::runtime::server_types::KvEpochId,
    pub page_ids: Vec<crate::runtime::server_types::KvPageId>,
    pub absolute_decode_position: u32,
}

impl InferenceScheduler {
    /// Creates a new, empty scheduler.
    ///
    /// The dispatch counter starts at zero and the dispatch history is
    /// initialised as empty.
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
            dispatches: Mutex::new(Vec::new()),
            modality_receipts: Mutex::new(Vec::new()),
            observation_sink: Mutex::new(None),
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────

    /// Atomically allocates the next dispatch identifier.
    fn next_dispatch_id(&self) -> DispatchId {
        DispatchId(self.counter.fetch_add(1, Ordering::SeqCst))
    }

    /// Appends a dispatched lane to the ordered history and returns its
    /// dispatch identifier.
    fn record(&self, session_id: SessionId, dispatch: LaneDispatch) -> DispatchId {
        self.record_with_metadata(session_id, dispatch, None)
    }

    fn record_with_metadata(
        &self,
        session_id: SessionId,
        dispatch: LaneDispatch,
        kv_metadata: Option<KvDispatchMetadata>,
    ) -> DispatchId {
        let id = dispatch.dispatch_id;
        let mut log = self.dispatches.lock().expect("scheduler lock poisoned");
        log.push(DispatchRecord {
            dispatch,
            session_id,
            kv_metadata,
            model_id: None,
        });
        id
    }

    /// Admit a non-text modality into the same ECS dispatch history used by
    /// inference. The provider remains responsible for producing bytes, but
    /// model identity and lane/phase admission are owned by ECS.
    pub fn schedule_modality(
        &self,
        model_id: impl Into<String>,
    ) -> Result<(SessionId, DispatchId), String> {
        let model_id = model_id.into();
        if model_id.trim().is_empty() {
            return Err("modality model_id must not be empty".into());
        }
        let session_id = SessionId(uuid::Uuid::new_v4());
        let id = self.next_dispatch_id();
        let dispatch = LaneDispatch {
            dispatch_id: id,
            lane: ExecutionLane::Metal,
            phase: InferencePhase::AuxiliaryInference,
            input_allocations: Vec::new(),
            output_allocations: Vec::new(),
            required_epoch: None,
            dependencies: Vec::new(),
            completion_fence: CompletionFenceId(0),
        };
        let mut log = self
            .dispatches
            .lock()
            .map_err(|_| "scheduler lock poisoned".to_string())?;
        log.push(DispatchRecord {
            dispatch,
            session_id,
            kv_metadata: None,
            model_id: Some(model_id),
        });
        Ok((session_id, id))
    }

    pub fn model_id(&self, dispatch_id: DispatchId) -> Option<String> {
        self.dispatches
            .lock()
            .ok()?
            .iter()
            .find(|record| record.dispatch.dispatch_id == dispatch_id)
            .and_then(|record| record.model_id.clone())
    }

    pub fn complete_modality(
        &self,
        dispatch_id: DispatchId,
        modality: impl Into<String>,
        output_digest: Option<String>,
        output_units: u64,
    ) -> Result<(), String> {
        let model_id = self
            .model_id(dispatch_id)
            .ok_or_else(|| format!("unknown modality dispatch {dispatch_id:?}"))?;
        let receipt = ModalityDispatchReceipt {
            dispatch_id,
            session_id: self
                .dispatches
                .lock()
                .map_err(|_| "scheduler lock poisoned".to_string())?
                .iter()
                .find(|record| record.dispatch.dispatch_id == dispatch_id)
                .map(|record| record.session_id)
                .ok_or_else(|| format!("unknown modality dispatch {dispatch_id:?}"))?,
            model_id,
            modality: modality.into(),
            status: "completed".into(),
            output_digest,
            output_units,
        };
        self.modality_receipts
            .lock()
            .map_err(|_| "modality receipt lock poisoned".to_string())?
            .push(receipt.clone());
        if let Some(sink) = self
            .observation_sink
            .lock()
            .map_err(|_| "observation sink lock poisoned".to_string())?
            .clone()
        {
            sink(receipt);
        }
        Ok(())
    }

    /// Attach the daemon's workflow/event projection to committed runtime
    /// observations. The sink is called only after the scheduler has stored
    /// the receipt, so workflow publication cannot become execution authority.
    pub fn set_observation_sink<F>(&self, sink: F) -> Result<(), String>
    where
        F: Fn(ModalityDispatchReceipt) + Send + Sync + 'static,
    {
        *self
            .observation_sink
            .lock()
            .map_err(|_| "observation sink lock poisoned".to_string())? = Some(Arc::new(sink));
        Ok(())
    }

    pub fn modality_receipt(&self, dispatch_id: DispatchId) -> Option<ModalityDispatchReceipt> {
        self.modality_receipts
            .lock()
            .ok()?
            .iter()
            .find(|receipt| receipt.dispatch_id == dispatch_id)
            .cloned()
    }

    /// Returns the KV ownership metadata captured at dispatch submission.
    pub fn kv_metadata(&self, dispatch_id: DispatchId) -> Option<KvDispatchMetadata> {
        self.dispatches
            .lock()
            .ok()?
            .iter()
            .find(|record| record.dispatch.dispatch_id == dispatch_id)
            .and_then(|record| record.kv_metadata.clone())
    }

    // ── Public scheduling API ────────────────────────────────────────

    /// Schedules prompt-prefill work for the given session.
    ///
    /// Creates a `Metal`-lane dispatch with phase `PromptPrefill`.
    /// `prompt_length` is accepted for future capacity planning and is
    /// recorded in the dispatch history.
    pub fn schedule_prefill(
        &self,
        session_id: &SessionId,
        _prompt_length: u32,
    ) -> Result<DispatchId, String> {
        self.schedule_prefill_with_metadata(session_id, _prompt_length, None)
    }

    pub fn schedule_prefill_with_metadata(
        &self,
        session_id: &SessionId,
        _prompt_length: u32,
        metadata: Option<KvDispatchMetadata>,
    ) -> Result<DispatchId, String> {
        let id = self.next_dispatch_id();
        let dispatch = LaneDispatch {
            dispatch_id: id,
            lane: ExecutionLane::Metal,
            phase: InferencePhase::PromptPrefill,
            input_allocations: Vec::new(),
            output_allocations: Vec::new(),
            required_epoch: metadata.as_ref().map(|value| value.epoch_id),
            dependencies: Vec::new(),
            completion_fence: CompletionFenceId(0),
        };
        Ok(self.record_with_metadata(*session_id, dispatch, metadata))
    }

    /// Schedules decode work for the given session.
    ///
    /// Creates a `Metal`-lane dispatch with phase `Decode`.
    pub fn schedule_decode(&self, session_id: &SessionId) -> Result<DispatchId, String> {
        self.schedule_decode_with_metadata(session_id, None)
    }

    pub fn schedule_decode_with_metadata(
        &self,
        session_id: &SessionId,
        metadata: Option<KvDispatchMetadata>,
    ) -> Result<DispatchId, String> {
        let id = self.next_dispatch_id();
        let dispatch = LaneDispatch {
            dispatch_id: id,
            lane: ExecutionLane::Metal,
            phase: InferencePhase::Decode,
            input_allocations: Vec::new(),
            output_allocations: Vec::new(),
            required_epoch: metadata.as_ref().map(|value| value.epoch_id),
            dependencies: Vec::new(),
            completion_fence: CompletionFenceId(0),
        };
        Ok(self.record_with_metadata(*session_id, dispatch, metadata))
    }

    /// Schedules auxiliary inference work for the given session and island.
    ///
    /// Creates a `CoreMlAne`-lane dispatch with phase `AuxiliaryInference`.
    /// `island_id` is accepted for future routing decisions and is recorded
    /// in the dispatch history.
    pub fn schedule_auxiliary(
        &self,
        session_id: &SessionId,
        _island_id: &str,
    ) -> Result<DispatchId, String> {
        let id = self.next_dispatch_id();
        let dispatch = LaneDispatch {
            dispatch_id: id,
            lane: ExecutionLane::CoreMlAne,
            phase: InferencePhase::AuxiliaryInference,
            input_allocations: Vec::new(),
            output_allocations: Vec::new(),
            required_epoch: None,
            dependencies: Vec::new(),
            completion_fence: CompletionFenceId(0),
        };
        Ok(self.record(*session_id, dispatch))
    }

    /// Returns the current backpressure policy for streaming output.
    ///
    /// Defaults to buffering up to 1024 events or 4096 tokens, with a
    /// 30-second consumer timeout and `PauseGeneration` on overflow.
    pub fn get_backpressure(&self) -> StreamBackpressurePolicy {
        StreamBackpressurePolicy {
            max_buffered_events: 1024,
            max_buffered_tokens: 4096,
            slow_consumer_timeout_secs: 30.0,
            action_on_overflow: SlowConsumerAction::PauseGeneration,
        }
    }
}

impl Default for InferenceScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_schedule_prefill_returns_unique_dispatch_id() {
        let scheduler = InferenceScheduler::new();
        let sid = SessionId(uuid::Uuid::new_v4());

        let id1 = scheduler.schedule_prefill(&sid, 128).unwrap();
        let id2 = scheduler.schedule_prefill(&sid, 256).unwrap();

        assert_ne!(id1, id2, "each prefill dispatch must have a unique id");
    }

    #[test]
    fn test_schedule_decode_returns_dispatch_id() {
        let scheduler = InferenceScheduler::new();
        let sid = SessionId(uuid::Uuid::new_v4());

        let id = scheduler.schedule_decode(&sid).unwrap();
        assert_eq!(id, DispatchId(0), "first decode dispatch should be id 0");
    }

    #[test]
    fn dispatch_preserves_model_epoch_and_page_ownership() {
        let scheduler = InferenceScheduler::new();
        let sid = SessionId(uuid::Uuid::new_v4());
        let metadata = KvDispatchMetadata {
            model_id: "vision/encoder".into(),
            epoch_id: crate::runtime::server_types::KvEpochId(9),
            page_ids: vec![
                crate::runtime::server_types::KvPageId(11),
                crate::runtime::server_types::KvPageId(12),
                crate::runtime::server_types::KvPageId(19),
            ],
            absolute_decode_position: 4096,
        };
        let dispatch = scheduler
            .schedule_decode_with_metadata(&sid, Some(metadata.clone()))
            .unwrap();
        assert_eq!(scheduler.kv_metadata(dispatch), Some(metadata));
    }

    #[test]
    fn modality_dispatch_preserves_model_namespace() {
        let scheduler = InferenceScheduler::new();
        let (_session, dispatch) = scheduler.schedule_modality("video/decoder").unwrap();
        assert_eq!(
            scheduler.model_id(dispatch).as_deref(),
            Some("video/decoder")
        );
        scheduler
            .complete_modality(dispatch, "video", None, 8)
            .unwrap();
        assert_eq!(
            scheduler.modality_receipt(dispatch).unwrap().status,
            "completed"
        );
    }

    #[test]
    fn modality_observation_sink_receives_ecs_session_identity() {
        let scheduler = InferenceScheduler::new();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let sink_observed = observed.clone();
        scheduler
            .set_observation_sink(move |receipt| {
                sink_observed.lock().unwrap().push(receipt);
            })
            .unwrap();
        let (session, dispatch) = scheduler.schedule_modality("audio/decoder").unwrap();
        scheduler
            .complete_modality(dispatch, "audio", Some("digest".into()), 24000)
            .unwrap();
        let receipt = observed.lock().unwrap().pop().unwrap();
        assert_eq!(receipt.session_id, session);
        assert_eq!(receipt.model_id, "audio/decoder");
        assert_eq!(receipt.output_units, 24000);
    }

    #[test]
    fn test_schedule_auxiliary_returns_dispatch_id() {
        let scheduler = InferenceScheduler::new();
        let sid = SessionId(uuid::Uuid::new_v4());

        let id = scheduler.schedule_auxiliary(&sid, "vit-encoder").unwrap();
        assert_eq!(id, DispatchId(0), "first auxiliary dispatch should be id 0");
    }

    #[test]
    fn test_counter_is_monotonic_across_phases() {
        let scheduler = InferenceScheduler::new();
        let sid = SessionId(uuid::Uuid::new_v4());

        let prefill = scheduler.schedule_prefill(&sid, 128).unwrap();
        let decode = scheduler.schedule_decode(&sid).unwrap();
        let aux = scheduler.schedule_auxiliary(&sid, "encoder").unwrap();

        assert_eq!(prefill, DispatchId(0));
        assert_eq!(decode, DispatchId(1));
        assert_eq!(aux, DispatchId(2));
    }

    #[test]
    fn test_multiple_sessions_get_independent_dispatches() {
        let scheduler = Arc::new(InferenceScheduler::new());
        let sid_a = SessionId(uuid::Uuid::new_v4());
        let sid_b = SessionId(uuid::Uuid::new_v4());

        let a1 = scheduler.schedule_decode(&sid_a).unwrap();
        let b1 = scheduler.schedule_decode(&sid_b).unwrap();
        let a2 = scheduler.schedule_decode(&sid_a).unwrap();

        assert_eq!(a1, DispatchId(0));
        assert_eq!(b1, DispatchId(1));
        assert_eq!(a2, DispatchId(2));
    }

    #[test]
    fn test_get_backpressure_returns_default_policy() {
        let scheduler = InferenceScheduler::new();
        let policy = scheduler.get_backpressure();

        assert_eq!(policy.max_buffered_events, 1024);
        assert_eq!(policy.max_buffered_tokens, 4096);
        assert_eq!(policy.slow_consumer_timeout_secs, 30.0);
        assert_eq!(
            policy.action_on_overflow,
            SlowConsumerAction::PauseGeneration
        );
    }

    #[test]
    fn test_new_is_default() {
        let a = InferenceScheduler::new();
        let b = InferenceScheduler::default();

        let sid = SessionId(uuid::Uuid::new_v4());
        assert_eq!(
            a.schedule_decode(&sid).unwrap(),
            b.schedule_decode(&sid).unwrap(),
            "new and default should behave identically"
        );
    }

    #[test]
    fn test_concurrent_scheduling_safety() {
        let scheduler = Arc::new(InferenceScheduler::new());
        let sid = SessionId(uuid::Uuid::new_v4());

        let mut handles = Vec::new();
        for _ in 0..10 {
            let sched = Arc::clone(&scheduler);
            let s = sid;
            handles.push(std::thread::spawn(move || {
                sched.schedule_decode(&s).unwrap()
            }));
        }

        let mut ids: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        ids.sort();
        ids.dedup();
        assert_eq!(
            ids.len(),
            10,
            "all ten concurrent dispatches must be unique"
        );
    }
}

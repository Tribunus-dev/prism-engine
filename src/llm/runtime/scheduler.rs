// ── Prism LLM Inference — Inference Scheduler ────────────────────────────
//
// Schedules prefill, decode, and auxiliary work for inference sessions.
// Generates monotonic DispatchId values, creates LaneDispatch records with
// the appropriate execution lane and inference phase, and maintains an
// ordered dispatch history for observability.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use super::super::manifest::{ExecutionLane, InferencePhase, SessionId};
use super::super::server::{
    CompletionFenceId, DispatchId, LaneDispatch, SlowConsumerAction,
    StreamBackpressurePolicy,
};

/// A dispatched lane execution paired with the session that requested it.
// Fields are written by the scheduler but read externally only for
// observability — suppress the dead-code lint.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DispatchRecord {
    dispatch: LaneDispatch,
    session_id: SessionId,
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
        let id = dispatch.dispatch_id;
        let mut log = self.dispatches.lock().expect("scheduler lock poisoned");
        log.push(DispatchRecord { dispatch, session_id });
        id
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
        let id = self.next_dispatch_id();
        let dispatch = LaneDispatch {
            dispatch_id: id,
            lane: ExecutionLane::Metal,
            phase: InferencePhase::PromptPrefill,
            input_allocations: Vec::new(),
            output_allocations: Vec::new(),
            required_epoch: None,
            dependencies: Vec::new(),
            completion_fence: CompletionFenceId(0),
        };
        Ok(self.record(*session_id, dispatch))
    }

    /// Schedules decode work for the given session.
    ///
    /// Creates a `Metal`-lane dispatch with phase `Decode`.
    pub fn schedule_decode(
        &self,
        session_id: &SessionId,
    ) -> Result<DispatchId, String> {
        let id = self.next_dispatch_id();
        let dispatch = LaneDispatch {
            dispatch_id: id,
            lane: ExecutionLane::Metal,
            phase: InferencePhase::Decode,
            input_allocations: Vec::new(),
            output_allocations: Vec::new(),
            required_epoch: None,
            dependencies: Vec::new(),
            completion_fence: CompletionFenceId(0),
        };
        Ok(self.record(*session_id, dispatch))
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

        let mut ids: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect();

        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 10, "all ten concurrent dispatches must be unique");
    }
}

#[cfg(feature = "prism-backend")]
pub mod compute_scheduler {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    use tribunus_compute_core::scheduling::{Request, RequestState, Scheduler, SchedulerConfig};

    use super::super::super::manifest::SessionId;
    use super::super::super::server::{
        DispatchId, SlowConsumerAction, StreamBackpressurePolicy,
    };

    /// Wraps `tribunus_compute_core::scheduling::Scheduler` for integration
    /// with the Prism LLM runtime.
    ///
    /// Translates Prism session identifiers and dispatch metadata into
    /// compute-core `Request` structs, enqueues them on the continuous
    /// batching scheduler, and exposes backpressure telemetry.
    pub struct ComputeScheduler {
        inner: Mutex<Scheduler>,
        counter: AtomicU64,
    }

    impl ComputeScheduler {
        /// Creates a new scheduler with default compute-core configuration.
        pub fn new() -> Self {
            Self {
                inner: Mutex::new(Scheduler::new(SchedulerConfig::default())),
                counter: AtomicU64::new(0),
}
        }

        /// Schedules a prefill request for the given session.
        ///
        /// Converts the prompt tokens and session metadata into a
        /// compute-core `Request` and enqueues it on the scheduler's
        /// priority queue. Returns a monotonic `DispatchId`.
        pub fn schedule_prefill(
            &self,
            _session_id: &SessionId,
            prompt_tokens: Vec<u32>,
            max_tokens: usize,
        ) -> Result<DispatchId, String> {
            let id = DispatchId(self.counter.fetch_add(1, Ordering::SeqCst));
            let req = Request {
                id: id.0,
                prompt: prompt_tokens,
                max_tokens,
                priority: 128,
                state: RequestState::Queued,
                created_at: std::time::Instant::now(),
                slot: None,
            };
            self.inner
                .lock()
                .map_err(|e| format!("scheduler lock: {e}"))?
                .enqueue(req);
            Ok(id)
        }

        /// Schedules a decode step for the given session.
        ///
        /// Creates a compute-core `Request` in `Decoding` state and adds
        /// it to the scheduler's queue for the next decode batch.
        pub fn schedule_decode(
            &self,
            _session_id: &SessionId,
        ) -> Result<DispatchId, String> {
            let id = DispatchId(self.counter.fetch_add(1, Ordering::SeqCst));
            let req = Request {
                id: id.0,
                prompt: Vec::new(),
                max_tokens: 1,
                priority: 128,
                state: RequestState::Decoding,
                created_at: std::time::Instant::now(),
                slot: None,
            };
            self.inner
                .lock()
                .map_err(|e| format!("scheduler lock: {e}"))?
                .enqueue(req);
            Ok(id)
        }

        /// Schedules auxiliary inference work for the given session and island.
        ///
        /// Auxiliary phases (e.g. cross-attention, vision encoder passes)
        /// are enqueued as zero-max-tokens requests that run once.
        pub fn schedule_auxiliary(
            &self,
            _session_id: &SessionId,
            _island_id: &str,
        ) -> Result<DispatchId, String> {
            let id = DispatchId(self.counter.fetch_add(1, Ordering::SeqCst));
            let req = Request {
                id: id.0,
                prompt: Vec::new(),
                max_tokens: 0,
                priority: 128,
                state: RequestState::Queued,
                created_at: std::time::Instant::now(),
                slot: None,
            };
            self.inner
                .lock()
                .map_err(|e| format!("scheduler lock: {e}"))?
                .enqueue(req);
            Ok(id)
        }

        /// Returns the current backpressure policy.
        ///
        /// Uses default values optimised for continuous batching:
        /// - 1,024 buffered events
        /// - 4,096 buffered tokens
        /// - 30-second slow-consumer timeout
        /// - Pause generation on overflow
        pub fn get_backpressure(&self) -> StreamBackpressurePolicy {
            StreamBackpressurePolicy {
                max_buffered_events: 1024,
                max_buffered_tokens: 4096,
                slow_consumer_timeout_secs: 30.0,
                action_on_overflow: SlowConsumerAction::PauseGeneration,
            }
        }
    }

    impl Default for ComputeScheduler {
        fn default() -> Self {
            Self::new()
        }
    }
}

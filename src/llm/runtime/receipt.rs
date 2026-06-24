// ── Prism LLM Inference — Receipt Store ──────────────────────────────────
//
// Builds, persists, and retrieves MultiIslandInferenceReceipt instances.
// The ReceiptStore maintains an in-memory registry of active receipts and
// serializes completed receipts to JSON on finalize().

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use super::super::manifest::{MaterializationEvent, SessionId};
use crate::image::types::ArtifactDigest;
use super::super::server::{
    CImageId, ContextProfileId, CoreMlVisibilityState,
    InferenceAdmissionReceipt, InferenceCancelledReceipt, InferenceExecutionPolicy,
    InferenceFailureReceipt, InferenceOutputReceipt, InferenceTerminalState, KvEpochReceipt,
    LaneExecutionReceipt, MemoryPressureReceipt, MultiIslandInferenceReceipt, ReceptionId,
    RequestId, WeightEvictionStatus, WeightResidencyReceipt,
};

/// Thread-safe store that builds, persists, and retrieves inference receipts.
///
/// Receipts are accumulated in-memory via `record_*` calls and finalised
/// with [`finalize`], which writes the completed receipt to disk as JSON.
pub struct ReceiptStore {
    receipts: Mutex<HashMap<SessionId, MultiIslandInferenceReceipt>>,
    output_dir: PathBuf,
}

impl ReceiptStore {
    /// Creates a new ReceiptStore that writes finalised receipts to
    /// `output_dir`.
    pub fn new(output_dir: String) -> Self {
        Self {
            receipts: Mutex::new(HashMap::new()),
            output_dir: PathBuf::from(output_dir),
        }
    }

    /// Creates a base receipt for the given session and inserts it into the
    /// store. The receipt starts with empty/placeholder values for mandatory
    /// fields; callers populate those fields via [`record_admission`] and
    /// [`record_residency`] before recording lane or epoch events.
    pub fn create_base_receipt(
        &self,
        session_id: &SessionId,
        request_id: &RequestId,
    ) -> MultiIslandInferenceReceipt {
        let now = iso_timestamp();
        let receipt = MultiIslandInferenceReceipt {
            receipt_id: ReceptionId(uuid::Uuid::new_v4()),
            session_id: *session_id,
            request_id: request_id.clone(),
            terminal_state: InferenceTerminalState::RefusedBeforeExecution,
            cimage_digest: ArtifactDigest(String::new()),
            context_profile: ContextProfileId(String::new()),
            admission: InferenceAdmissionReceipt {
                cimage_id: CImageId(String::new()),
                context_profile: ContextProfileId(String::new()),
                execution_policy: InferenceExecutionPolicy::RequireMetalDecode,
                admitted: false,
                refusal_reason: None,
            },
            weight_residency: WeightResidencyReceipt {
                cimage_digest: ArtifactDigest(String::new()),
                cache_hit: false,
                initial_load_bytes: 0,
                decode_step_reload_count: 0,
                active_weight_leases: 0,
                metal_visible: false,
                accelerate_visible: false,
                coreml_auxiliary_visibility: CoreMlVisibilityState::NotVisible,
                materialization_events: Vec::new(),
                eviction_status: WeightEvictionStatus::Retained,
            },
            lane_receipts: Vec::new(),
            kv_history: Vec::new(),
            materialization_events: Vec::new(),
            output: None,
            failure: None,
            cancellation: None,
            memory_pressure_history: Vec::new(),
            started_at: now,
            completed_at: String::new(),
        };
        self.receipts
            .lock()
            .expect("receipts lock poisoned")
            .insert(*session_id, receipt.clone());
        receipt
    }

    /// Records the admission receipt for a session, replacing any earlier
    /// admission data on the stored receipt.
    pub fn record_admission(&self, session_id: &SessionId, admission: InferenceAdmissionReceipt) {
        if let Some(receipt) = self
            .receipts
            .lock()
            .expect("receipts lock poisoned")
            .get_mut(session_id)
        {
            receipt.admission = admission;
        }
    }

    /// Records the weight-residency receipt for a session.
    pub fn record_residency(&self, session_id: &SessionId, residency: WeightResidencyReceipt) {
        if let Some(receipt) = self
            .receipts
            .lock()
            .expect("receipts lock poisoned")
            .get_mut(session_id)
        {
            receipt.weight_residency = residency;
        }
    }

    /// Appends a lane execution receipt to the session's receipt.
    pub fn record_lane(&self, session_id: &SessionId, lane: LaneExecutionReceipt) {
        if let Some(receipt) = self
            .receipts
            .lock()
            .expect("receipts lock poisoned")
            .get_mut(session_id)
        {
            receipt.lane_receipts.push(lane);
        }
    }

    /// Appends a KV epoch receipt to the session's receipt.
    pub fn record_kv_epoch(&self, session_id: &SessionId, epoch: KvEpochReceipt) {
        if let Some(receipt) = self
            .receipts
            .lock()
            .expect("receipts lock poisoned")
            .get_mut(session_id)
        {
            receipt.kv_history.push(epoch);
        }
    }

    /// Appends a materialization event to the session's receipt.
    pub fn record_materialization(&self, session_id: &SessionId, event: MaterializationEvent) {
        if let Some(receipt) = self
            .receipts
            .lock()
            .expect("receipts lock poisoned")
            .get_mut(session_id)
        {
            receipt.materialization_events.push(event);
        }
    }

    /// Appends a memory-pressure receipt to the session's history.
    pub fn record_memory_pressure(&self, session_id: &SessionId, pressure: MemoryPressureReceipt) {
        if let Some(receipt) = self
            .receipts
            .lock()
            .expect("receipts lock poisoned")
            .get_mut(session_id)
        {
            receipt.memory_pressure_history.push(pressure);
        }
    }

    /// Records the output receipt for a session.
    pub fn record_output(&self, session_id: &SessionId, output: InferenceOutputReceipt) {
        if let Some(receipt) = self
            .receipts
            .lock()
            .expect("receipts lock poisoned")
            .get_mut(session_id)
        {
            receipt.output = Some(output);
        }
    }

    /// Records the failure receipt for a session.
    pub fn record_failure(&self, session_id: &SessionId, failure: InferenceFailureReceipt) {
        if let Some(receipt) = self
            .receipts
            .lock()
            .expect("receipts lock poisoned")
            .get_mut(session_id)
        {
            receipt.failure = Some(failure);
        }
    }

    /// Records the cancellation receipt for a session.
    pub fn record_cancellation(
        &self,
        session_id: &SessionId,
        cancellation: InferenceCancelledReceipt,
    ) {
        if let Some(receipt) = self
            .receipts
            .lock()
            .expect("receipts lock poisoned")
            .get_mut(session_id)
        {
            receipt.cancellation = Some(cancellation);
        }
    }

    /// Finalises the receipt for a session by setting the terminal state and
    /// completion timestamp. The completed receipt is serialised to JSON in
    /// the output directory and returned.
    ///
    /// Returns an error if no receipt exists for the session or if the JSON
    /// serialisation or file write fails.
    pub fn finalize(
        &self,
        session_id: &SessionId,
        terminal_state: InferenceTerminalState,
    ) -> Result<MultiIslandInferenceReceipt, String> {
        let mut guard = self.receipts.lock().expect("receipts lock poisoned");
        let receipt = guard
            .get_mut(session_id)
            .ok_or_else(|| format!("no receipt found for session {session_id:?}"))?;

        receipt.terminal_state = terminal_state;
        receipt.completed_at = iso_timestamp();

        let final_receipt = receipt.clone();
        let filename = format!(
            "receipt_{}_{}.json",
            session_id.0, final_receipt.receipt_id.0
        );
        let path = self.output_dir.join(&filename);

        // Ensure the output directory exists.
        fs::create_dir_all(&self.output_dir)
            .map_err(|e| format!("failed to create output directory: {e}"))?;

        let json = serde_json::to_string_pretty(&final_receipt)
            .map_err(|e| format!("failed to serialise receipt: {e}"))?;

        fs::write(&path, &json).map_err(|e| format!("failed to write receipt file: {e}"))?;

        Ok(final_receipt)
    }

    /// Returns a clone of the receipt for the given session, or `None` if
    /// no receipt has been created yet.
    pub fn get_receipt(&self, session_id: &SessionId) -> Option<MultiIslandInferenceReceipt> {
        self.receipts
            .lock()
            .expect("receipts lock poisoned")
            .get(session_id)
            .cloned()
    }
}

/// Returns the current wall-clock time as an ISO 8601 string
/// (UTC, second precision).
fn iso_timestamp() -> String {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();

    let days = secs / 86400;
    let day_secs = secs % 86400;

    let hours = day_secs / 3600;
    let minutes = (day_secs % 3600) / 60;
    let seconds = day_secs % 60;

    // Year/month/day from days since 1970-01-01.
    let mut y = 1970i64;
    let mut remaining = days as i64;

    loop {
        let days_in_year = if is_leap(y) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        y += 1;
    }

    let leap = is_leap(y);
    let month_days: [i64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];

    let mut m = 1u32;
    for &md in &month_days {
        if remaining < md {
            break;
        }
        remaining -= md;
        m += 1;
    }
    let d = remaining as u32 + 1;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, m, d, hours, minutes, seconds
    )
}

fn is_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

#[cfg(test)]
mod tests {
    use super::super::super::server::{
        InferenceFailureClass, InferenceSessionState, MemoryPressureLevel,
    };
    use super::super::super::manifest::{ExecutionLane, InferencePhase};
    use super::super::super::server::KvEpochState;
    use super::super::super::server::KvEpochId;
    use super::*;

    #[test]
    fn test_create_base_receipt() {
        let store = ReceiptStore::new("/tmp/prism-test-receipts".to_string());
        let sid = SessionId(uuid::Uuid::new_v4());
        let rid = RequestId(uuid::Uuid::new_v4());

        let receipt = store.create_base_receipt(&sid, &rid);

        assert_eq!(receipt.session_id, sid);
        assert_eq!(receipt.request_id, rid);
        assert!(!receipt.started_at.is_empty());
        assert!(receipt.completed_at.is_empty());
        assert_eq!(receipt.lane_receipts.len(), 0);
        assert_eq!(receipt.kv_history.len(), 0);
        assert_eq!(receipt.materialization_events.len(), 0);
        assert_eq!(receipt.memory_pressure_history.len(), 0);
        assert!(receipt.output.is_none());
        assert!(receipt.failure.is_none());
        assert!(receipt.cancellation.is_none());
        assert_eq!(
            receipt.terminal_state,
            InferenceTerminalState::RefusedBeforeExecution
        );

        // Verify it's retrievable.
        let retrieved = store.get_receipt(&sid).expect("receipt should exist");
        assert_eq!(retrieved.receipt_id, receipt.receipt_id);
    }

    #[test]
    fn test_record_and_finalize() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let store = ReceiptStore::new(dir.path().to_string_lossy().to_string());
        let sid = SessionId(uuid::Uuid::new_v4());
        let rid = RequestId(uuid::Uuid::new_v4());

        store.create_base_receipt(&sid, &rid);

        // Record admission.
        store.record_admission(
            &sid,
            InferenceAdmissionReceipt {
                cimage_id: CImageId("test-cimage".to_string()),
                context_profile: ContextProfileId("default".to_string()),
                execution_policy: InferenceExecutionPolicy::RequireMetalDecode,
                admitted: true,
                refusal_reason: None,
            },
        );

        // Record weight residency.
        store.record_residency(
            &sid,
            WeightResidencyReceipt {
                cimage_digest: ArtifactDigest("abc123".to_string()),
                cache_hit: true,
                initial_load_bytes: 4096,
                decode_step_reload_count: 0,
                active_weight_leases: 1,
                metal_visible: true,
                accelerate_visible: false,
                coreml_auxiliary_visibility: CoreMlVisibilityState::NotVisible,
                materialization_events: Vec::new(),
                eviction_status: WeightEvictionStatus::Retained,
            },
        );

        // Record KV epoch.
        store.record_kv_epoch(
            &sid,
            KvEpochReceipt {
                epoch_id: KvEpochId(1),
                parent_epoch: None,
                logical_context_length: 128,
                state: KvEpochState::Active,
            },
        );

        // Record lane.
        store.record_lane(
            &sid,
            LaneExecutionReceipt {
                lane: ExecutionLane::Metal,
                metal: None,
                accelerate: None,
                coreml: None,
            },
        );

        // Record memory pressure.
        store.record_memory_pressure(
            &sid,
            MemoryPressureReceipt {
                level: MemoryPressureLevel::Normal,
                timestamp: iso_timestamp(),
                action_taken: "none".to_string(),
            },
        );

        // Finalize.
        let finalized = store
            .finalize(&sid, InferenceTerminalState::Succeeded)
            .expect("finalize should succeed");

        assert_eq!(finalized.terminal_state, InferenceTerminalState::Succeeded);
        assert!(!finalized.completed_at.is_empty());
        assert_eq!(finalized.kv_history.len(), 1);
        assert_eq!(finalized.lane_receipts.len(), 1);
        assert_eq!(finalized.memory_pressure_history.len(), 1);

        // Receipt file should exist on disk.
        let filename = format!("receipt_{}_{}.json", sid.0, finalized.receipt_id.0);
        assert!(
            dir.path().join(&filename).exists(),
            "receipt file should exist on disk"
        );
    }

    #[test]
    fn test_finalize_missing_session() {
        let store = ReceiptStore::new("/tmp/prism-test-receipts".to_string());
        let sid = SessionId(uuid::Uuid::new_v4());

        let result = store.finalize(&sid, InferenceTerminalState::Cancelled);
        assert!(result.is_err(), "should error on missing session");
    }

    #[test]
    fn test_record_cancellation() {
        let store = ReceiptStore::new("/tmp/prism-test-receipts".to_string());
        let sid = SessionId(uuid::Uuid::new_v4());
        let rid = RequestId(uuid::Uuid::new_v4());
        store.create_base_receipt(&sid, &rid);

        store.record_cancellation(
            &sid,
            InferenceCancelledReceipt {
                session_id: sid,
                request_id: rid,
                state_at_cancellation: InferenceSessionState::Decoding,
                active_epoch: None,
                completed_decode_tokens: 42,
                cleanup_completed: true,
            },
        );

        let receipt = store.get_receipt(&sid).expect("receipt exists");
        assert!(receipt.cancellation.is_some());
        assert_eq!(
            receipt.cancellation.as_ref().unwrap().completed_decode_tokens,
            42
        );
    }

    #[test]
    fn test_record_output_and_failure() {
        let store = ReceiptStore::new("/tmp/prism-test-receipts".to_string());
        let sid = SessionId(uuid::Uuid::new_v4());
        let rid = RequestId(uuid::Uuid::new_v4());
        store.create_base_receipt(&sid, &rid);

        store.record_output(
            &sid,
            InferenceOutputReceipt {
                total_tokens: 256,
                tokens_per_second: 30.5,
                total_latency_ms: 8400.0,
                metal_decode_latency_ms: 7200.0,
            },
        );

        store.record_failure(
            &sid,
            InferenceFailureReceipt {
                class: InferenceFailureClass::MetalPrefillFailed,
                phase: InferencePhase::PromptPrefill,
                lane: Some(ExecutionLane::Metal),
                retryable: true,
                recovery_action: None,
            },
        );

        let receipt = store.get_receipt(&sid).expect("receipt exists");
        assert_eq!(receipt.output.as_ref().unwrap().total_tokens, 256);
        assert!(receipt.failure.is_some());
    }

    #[test]
    fn test_iso_timestamp_format() {
        let ts = iso_timestamp();
        // Basic ISO 8601 check: YYYY-MM-DDTHH:MM:SSZ
        assert_eq!(ts.len(), 20, "timestamp should be 20 chars: {ts}");
        assert!(ts.ends_with('Z'), "timestamp should end with Z: {ts}");
        assert_eq!(&ts[4..5], "-", "expected - at position 4: {ts}");
        assert_eq!(&ts[7..8], "-", "expected - at position 7: {ts}");
        assert_eq!(&ts[10..11], "T", "expected T at position 10: {ts}");
    }
}

//! Canonical receipt envelope and ReceiptBus — the unified receipt infrastructure
//! for the Prism ECS monolith.
//!
//! Every subsystem produces [`CanonicalReceipt`]s with content-addressed
//! [`ReceiptId`]s. [`ReceiptBus`] broadcasts them to [`ReceiptSubscriber`]s
//! and buffers them for test/audit draining.
//!
//! ## Thread safety
//!
//! [`ReceiptBus::emit`] is `&self` — all internal state uses `parking_lot::Mutex`.
//! Subscribers are dispatched via an async [`mpsc::UnboundedSender`] channel.
//! The synchronous [`ReceiptSubscriber::on_receipt`] trait method is retained
//! as a compatibility path for manual use outside the bus; new subscribers
//! should use the async channel instead.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

// ── ReceiptKind ─────────────────────────────────────────────────────────────

/// Categorisation of a canonical receipt by subsystem and event type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReceiptKind {
    // Compilation
    SourceValidated,
    Admitted,
    Bound,
    Sealed,
    Failed,
    // Runtime
    ModelLoaded,
    RequestAdmitted,
    PrefillCompleted,
    TokenDecoded,
    GenerationCompleted,
    // Validation
    PhaseValidated,
    ParityCheckPassed,
    ParityCheckFailed,
    CalibrationCompleted,
    // Distillation
    JobCreated,
    CalibrationStarted,
    TensorCandidatesIdentified,
    PromotionEligible,
    PromotionCommitted,
    // Server
    RequestReceived,
    StreamStarted,
    StreamCancelled,
    StreamCompleted,
    // Inference
    PhaseReceipt,
    StepReceipt,
    MetricsReported,
    // Evidence / Arena
    ArenaCreated,
    ArenaReleased,
    ArenaLeased,
    StateMutation,
    // Artifact ingestion
    ArtifactDiscovered,
    ArtifactIngested,
    // Extension
    Other(String),
}

impl std::fmt::Display for ReceiptKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl ReceiptKind {
    /// Return a stable string representation of this receipt kind.
    ///
    /// For unit variants this returns `snake_case` identical to the serde
    /// representation. For `Other(s)` it returns the inner string directly.
    pub fn as_str(&self) -> &str {
        match self {
            Self::SourceValidated => "source_validated",
            Self::Admitted => "admitted",
            Self::Bound => "bound",
            Self::Sealed => "sealed",
            Self::Failed => "failed",
            Self::ModelLoaded => "model_loaded",
            Self::RequestAdmitted => "request_admitted",
            Self::PrefillCompleted => "prefill_completed",
            Self::TokenDecoded => "token_decoded",
            Self::GenerationCompleted => "generation_completed",
            Self::PhaseValidated => "phase_validated",
            Self::ParityCheckPassed => "parity_check_passed",
            Self::ParityCheckFailed => "parity_check_failed",
            Self::CalibrationCompleted => "calibration_completed",
            Self::JobCreated => "job_created",
            Self::CalibrationStarted => "calibration_started",
            Self::TensorCandidatesIdentified => "tensor_candidates_identified",
            Self::PromotionEligible => "promotion_eligible",
            Self::PromotionCommitted => "promotion_committed",
            Self::RequestReceived => "request_received",
            Self::StreamStarted => "stream_started",
            Self::StreamCancelled => "stream_cancelled",
            Self::StreamCompleted => "stream_completed",
            Self::PhaseReceipt => "phase_receipt",
            Self::StepReceipt => "step_receipt",
            Self::MetricsReported => "metrics_reported",
            Self::ArenaCreated => "arena_created",
            Self::ArenaReleased => "arena_released",
            Self::ArenaLeased => "arena_leased",
            Self::StateMutation => "state_mutation",
            Self::ArtifactDiscovered => "artifact_discovered",
            Self::ArtifactIngested => "artifact_ingested",
            Self::Other(s) => s.as_str(),
        }
    }
}

// ── ReceiptId ───────────────────────────────────────────────────────────────

/// Content-addressed receipt identifier.
///
/// `ReceiptId = blake3(payload_hash || entity_id_bytes || epoch_le || seq_le)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReceiptId(#[serde(with = "hex_array")] pub [u8; 32]);

impl ReceiptId {
    /// Compute a content-addressed ID from its components.
    pub fn compute(payload_hash: [u8; 32], entity_id: Option<&str>, epoch: u64, seq: u64) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&payload_hash);
        match entity_id {
            Some(id) => {
                hasher.update(id.as_bytes());
            }
            None => {}
        }
        hasher.update(&epoch.to_le_bytes());
        hasher.update(&seq.to_le_bytes());
        let mut out = [0u8; 32];
        out.copy_from_slice(hasher.finalize().as_bytes());
        Self(out)
    }
}

impl std::fmt::Display for ReceiptId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl From<[u8; 32]> for ReceiptId {
    fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

// ── CanonicalReceipt ────────────────────────────────────────────────────────

/// A type-erased, content-addressed receipt emitted by any subsystem.
///
/// Follows an append-only semantic — receipts are never modified after emission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalReceipt {
    pub id: ReceiptId,
    pub entity_id: Option<String>,
    pub world_epoch: u64,
    pub event_seq: u64,
    pub kind: ReceiptKind,
    pub payload: serde_json::Value,
    pub parent_ids: Vec<ReceiptId>,
    pub payload_hash: [u8; 32],
    pub source: String,
    #[serde(with = "serde_systemtime")]
    pub timestamp: std::time::SystemTime,
}

impl CanonicalReceipt {
    /// Minimal constructor — sets kind, payload, source, and timestamp.
    ///
    /// ## Content addressing vs causal identity
    ///
    /// `ReceiptId` is computed from `blake3(payload_hash || epoch || seq)`.
    /// Because [`new`](Self::new) always passes `entity_id: None, epoch: 0, seq: 0`,
    /// two receipts with identical payloads get the **same ID** regardless of
    /// source, timestamp, or causal context. This is correct for evidence
    /// deduplication (identical evidence blobs are the same content) but does
    /// **not** provide unique event identity.
    ///
    /// Use [`CanonicalReceiptBuilder`] with explicit `epoch`/`seq`/`entity_id`
    /// when causal envelope identity is required. The constitutional ECS
    /// migration will introduce `MessageId` + `Envelope<T>` for that purpose.
    pub fn new(kind: ReceiptKind, payload: serde_json::Value, source: impl Into<String>) -> Self {
        let source = source.into();
        let payload_str = serde_json::to_string(&payload).unwrap_or_default();
        let payload_hash = blake3::hash(payload_str.as_bytes());
        let mut ph = [0u8; 32];
        ph.copy_from_slice(payload_hash.as_bytes());
        let id = ReceiptId::compute(ph, None, 0, 0);
        Self {
            id,
            entity_id: None,
            world_epoch: 0,
            event_seq: 0,
            kind,
            payload,
            parent_ids: Vec::new(),
            payload_hash: ph,
            source,
            timestamp: std::time::SystemTime::now(),
        }
    }
}

// ── CanonicalReceiptBuilder ─────────────────────────────────────────────────

/// Builder for [`CanonicalReceipt`].
///
/// # Panics
///
/// `build()` panics if `source` is empty (it is required).
#[derive(Debug, Default)]
pub struct CanonicalReceiptBuilder {
    entity_id: Option<String>,
    world_epoch: u64,
    event_seq: u64,
    kind: Option<ReceiptKind>,
    payload: Option<serde_json::Value>,
    parent_ids: Vec<ReceiptId>,
    source: Option<String>,
    timestamp: Option<std::time::SystemTime>,
}

impl CanonicalReceiptBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_entity_id(mut self, entity_id: impl Into<String>) -> Self {
        self.entity_id = Some(entity_id.into());
        self
    }

    pub fn with_epoch(mut self, epoch: u64) -> Self {
        self.world_epoch = epoch;
        self
    }

    pub fn with_seq(mut self, seq: u64) -> Self {
        self.event_seq = seq;
        self
    }

    pub fn with_kind(mut self, kind: ReceiptKind) -> Self {
        self.kind = Some(kind);
        self
    }

    pub fn with_payload(mut self, payload: serde_json::Value) -> Self {
        self.payload = Some(payload);
        self
    }

    pub fn with_parent(mut self, parent: ReceiptId) -> Self {
        self.parent_ids.push(parent);
        self
    }

    pub fn with_parents(mut self, parents: Vec<ReceiptId>) -> Self {
        self.parent_ids = parents;
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn with_timestamp(mut self, ts: std::time::SystemTime) -> Self {
        self.timestamp = Some(ts);
        self
    }

    pub fn build(self) -> CanonicalReceipt {
        let source = self
            .source
            .expect("CanonicalReceiptBuilder::build: source is required");
        let kind = self
            .kind
            .expect("CanonicalReceiptBuilder::build: kind is required");
        let payload = self.payload.unwrap_or(serde_json::Value::Null);
        let payload_str = serde_json::to_string(&payload).unwrap_or_default();
        let payload_hash_arr = blake3::hash(payload_str.as_bytes());
        let mut payload_hash = [0u8; 32];
        payload_hash.copy_from_slice(payload_hash_arr.as_bytes());
        let id = ReceiptId::compute(
            payload_hash,
            self.entity_id.as_deref(),
            self.world_epoch,
            self.event_seq,
        );
        CanonicalReceipt {
            id,
            entity_id: self.entity_id,
            world_epoch: self.world_epoch,
            event_seq: self.event_seq,
            kind,
            payload,
            parent_ids: self.parent_ids,
            payload_hash,
            source,
            timestamp: self.timestamp.unwrap_or_else(std::time::SystemTime::now),
        }
    }
}

// ── ReceiptSubscriber trait ─────────────────────────────────────────────────

/// A subscriber that receives receipts dispatched by [`ReceiptBus`].
///
/// Implementations are `Send + Sync` so they can be registered from any thread.
///
/// # Filtering
///
/// Return `Some(vec![…])` from [`kind_filter`](Self::kind_filter) to only
/// receive specific receipt kinds. Return `None` to receive all receipts.
pub trait ReceiptSubscriber: Send + Sync {
    /// Receive a receipt synchronously.
    ///
    /// **Deprecated** — new subscribers should use the async channel from
    /// [`ReceiptBus::subscribe`] instead and leave this default no-op.
    fn on_receipt(&mut self, _receipt: &CanonicalReceipt) {}
    fn kind_filter(&self) -> Option<Vec<ReceiptKind>> {
        None
    }
}

// ── AsyncSubscriber ───────────────────────────────────────────────────────────

/// A subscriber registered with [`ReceiptBus`], backed by an async channel.
///
/// The [`subscriber`] field retains the [`Box<dyn ReceiptSubscriber>`] for
/// backward compat. New subscribers should use the async channel exclusively:
/// call [`subscribe`](ReceiptBus::subscribe), spawn a worker task that drains
/// the returned [`mpsc::UnboundedReceiver`], and never touch [`ReceiptSubscriber::on_receipt`].
pub struct AsyncSubscriber {
    /// The original synchronous subscriber (compat path).
    pub subscriber: Box<dyn ReceiptSubscriber>,
    /// Non-blocking sender — [`emit`](ReceiptBus::emit) only calls `try_send`.
    pub sender: mpsc::UnboundedSender<CanonicalReceipt>,
    /// Optional kind filter (cached from the subscriber at registration time).
    pub filter: Option<Vec<ReceiptKind>>,
}

// ── ReceiptBus ──────────────────────────────────────────────────────────────

/// Default maximum number of receipts retained in the replay buffer.
const DEFAULT_MAX_BUFFER: usize = 10_000;

/// Central hub for emitting and subscribing to [`CanonicalReceipt`]s.
///
/// Every receipt is buffered for test/audit draining and forwarded to
/// matching subscribers.
///
/// # Thread safety
///
/// All methods are thread-safe. The subscriber lock is held only for the
/// duration of the (non-blocking) `try_send` call, not for subscriber processing.
pub struct ReceiptBus {
    subscribers: Mutex<Vec<AsyncSubscriber>>,
    buffer: Mutex<Vec<CanonicalReceipt>>,
    max_buffer: usize,
}

impl std::fmt::Debug for ReceiptBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReceiptBus")
            .field("subscriber_count", &self.subscribers.lock().len())
            .field("buffered_count", &self.buffer.lock().len())
            .field("max_buffer", &self.max_buffer)
            .finish()
    }
}

impl ReceiptBus {
    pub fn new() -> Self {
        Self::with_max_buffer(DEFAULT_MAX_BUFFER)
    }

    /// Create a bus with a bounded replay buffer.
    ///
    /// When the buffer exceeds `max` the oldest receipts are dropped.
    /// Set to `usize::MAX` for unbounded growth (not recommended in production).
    pub fn with_max_buffer(max: usize) -> Self {
        Self {
            subscribers: Mutex::new(Vec::new()),
            buffer: Mutex::new(Vec::new()),
            max_buffer: max,
        }
    }

    /// Register a subscriber and return its async channel receiver.
    ///
    /// The subscriber is stored alongside an [`mpsc::UnboundedSender`].
    /// Every future [`emit`](Self::emit) call that matches the subscriber's
    /// [`kind_filter`](ReceiptSubscriber::kind_filter) sends the receipt
    /// through the channel without blocking.
    ///
    /// Spawn a worker task that drains the returned receiver to process receipts.
    ///
    /// ## Compat note
    ///
    /// The synchronous [`ReceiptSubscriber::on_receipt`] method is **not** called
    /// by the bus after this migration. Callers that depended on synchronous
    /// dispatch must switch to reading from the returned receiver.
    pub fn subscribe(
        &self,
        subscriber: Box<dyn ReceiptSubscriber>,
    ) -> mpsc::UnboundedReceiver<CanonicalReceipt> {
        let (tx, rx) = mpsc::unbounded_channel();
        let filter = subscriber.kind_filter();
        self.subscribers.lock().push(AsyncSubscriber {
            subscriber,
            sender: tx,
            filter,
        });
        rx
    }

    pub fn emit(&self, receipt: CanonicalReceipt) {
        {
            let mut buf = self.buffer.lock();
            if buf.len() >= self.max_buffer {
                let excess = buf.len().saturating_sub(self.max_buffer - 1);
                buf.drain(..excess);
            }
            buf.push(receipt.clone());
        }
        let subs = self.subscribers.lock();
        for sub in subs.iter() {
            let matches = sub
                .filter
                .as_ref()
                .map_or(true, |kinds| kinds.contains(&receipt.kind));
            if matches {
                // Non-blocking send — subscriber lock is never held across processing
                let _ = sub.sender.send(receipt.clone());
            }
        }
    }

    pub fn drain(&self) -> Vec<CanonicalReceipt> {
        std::mem::take(&mut *self.buffer.lock())
    }

    pub fn subscriber_count(&self) -> usize {
        self.subscribers.lock().len()
    }

    pub fn buffered_count(&self) -> usize {
        self.buffer.lock().len()
    }
}

impl Default for ReceiptBus {
    fn default() -> Self {
        Self::with_max_buffer(DEFAULT_MAX_BUFFER)
    }
}

// ── PgReceiptSubscriber (stub) ──────────────────────────────────────────────

/// A stub [`ReceiptSubscriber`] that logs receipts via `tracing::info!`.
///
/// Phase 2 replaced this with a real Postgres-backed subscriber
/// at `compute-core/src/ecs/pg_receipt_subscriber.rs`.
#[cfg(not(feature = "server-dashboard"))]
#[derive(Debug)]
pub struct PgReceiptSubscriber {
    name: String,
}

#[cfg(not(feature = "server-dashboard"))]
impl PgReceiptSubscriber {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[cfg(not(feature = "server-dashboard"))]
impl ReceiptSubscriber for PgReceiptSubscriber {
    fn on_receipt(&mut self, receipt: &CanonicalReceipt) {
        tracing::info!(
            target: "pg_receipt_subscriber",
            subscriber = %self.name,
            kind = ?receipt.kind,
            id = %receipt.id,
            "receipt received (stub — no Postgres connection)"
        );
    }

    fn kind_filter(&self) -> Option<Vec<ReceiptKind>> {
        None
    }
}

// ── serde helpers ───────────────────────────────────────────────────────────

/// Serialize/deserialize `SystemTime` as `u64` nanoseconds since Unix epoch.
mod serde_systemtime {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn serialize<S: Serializer>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error> {
        let duration = time.duration_since(UNIX_EPOCH).unwrap_or_default();
        serializer.serialize_u64(duration.as_nanos() as u64)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<SystemTime, D::Error> {
        let nanos = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + std::time::Duration::from_nanos(nanos))
    }
}

/// Serialize/deserialize `[u8; 32]` as a lowercase hex string.
mod hex_array {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error> {
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        serializer.serialize_str(&hex)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[u8; 32], D::Error> {
        let hex = String::deserialize(deserializer)?;
        let mut out = [0u8; 32];
        if hex.len() != 64 {
            return Err(serde::de::Error::custom(format!(
                "expected 64 hex chars, got {}",
                hex.len()
            )));
        }
        for i in 0..32 {
            out[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
                .map_err(|e| serde::de::Error::custom(format!("hex decode error: {e}")))?;
        }
        Ok(out)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receipt_id_determinism() {
        let hash = [0xabu8; 32];
        let id1 = ReceiptId::compute(hash, Some("entity"), 1, 2);
        let id2 = ReceiptId::compute(hash, Some("entity"), 1, 2);
        assert_eq!(id1, id2, "same inputs must produce same id");
        let id3 = ReceiptId::compute(hash, Some("entity"), 1, 3);
        assert_ne!(id1, id3, "different seq must produce different id");
    }

    #[test]
    fn test_receipt_id_display() {
        let id = ReceiptId([
            0x00, 0x01, 0xff, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x00, 0x01, 0xff, 0xab, 0xcd, 0xef,
            0x12, 0x34, 0x00, 0x01, 0xff, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x00, 0x01, 0xff, 0xab,
            0xcd, 0xef, 0x12, 0x34,
        ]);
        let s = format!("{id}");
        assert_eq!(s.len(), 64);
        assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_builder_minimal() {
        let receipt = CanonicalReceiptBuilder::new()
            .with_kind(ReceiptKind::ModelLoaded)
            .with_payload(serde_json::json!({"model": "test"}))
            .with_source("test")
            .build();
        assert_eq!(receipt.kind, ReceiptKind::ModelLoaded);
        assert_eq!(receipt.source, "test");
        assert!(!format!("{}", receipt.id).is_empty());
    }

    #[test]
    fn test_builder_full() {
        let now = std::time::SystemTime::now();
        let receipt = CanonicalReceiptBuilder::new()
            .with_kind(ReceiptKind::Bound)
            .with_payload(serde_json::json!({"phase": "sealing"}))
            .with_source("compiler")
            .with_entity_id("artifact:abc123")
            .with_epoch(42)
            .with_seq(7)
            .with_timestamp(now)
            .build();
        assert_eq!(receipt.entity_id.as_deref(), Some("artifact:abc123"));
        assert_eq!(receipt.world_epoch, 42);
        assert_eq!(receipt.event_seq, 7);
    }

    #[test]
    fn test_builder_panics_without_source() {
        let result = std::panic::catch_unwind(|| {
            CanonicalReceiptBuilder::new()
                .with_kind(ReceiptKind::Failed)
                .build();
        });
        assert!(result.is_err(), "build without source should panic");
    }

    #[test]
    fn test_bus_emit_and_buffer() {
        let bus = ReceiptBus::new();
        let receipt = CanonicalReceiptBuilder::new()
            .with_kind(ReceiptKind::Admitted)
            .with_payload(serde_json::json!({}))
            .with_source("test")
            .build();
        bus.emit(receipt);
        assert_eq!(bus.buffered_count(), 1);
        let drained = bus.drain();
        assert_eq!(drained.len(), 1);
        assert_eq!(bus.buffered_count(), 0);
    }

    #[test]
    fn test_bus_subscriber_dispatched() {
        let bus = ReceiptBus::new();
        // Dummy subscriber — actual processing happens on the channel receiver
        struct TestSub;
        impl ReceiptSubscriber for TestSub {}
        let mut rx = bus.subscribe(Box::new(TestSub));

        bus.emit(
            CanonicalReceiptBuilder::new()
                .with_kind(ReceiptKind::Bound)
                .with_payload(serde_json::json!({}))
                .with_source("test")
                .build(),
        );

        let received = rx.try_recv().expect("emit should send via channel");
        assert_eq!(received.kind, ReceiptKind::Bound);
    }

    #[test]
    fn test_kind_filter() {
        let bus = ReceiptBus::new();
        struct FilterSub;
        impl ReceiptSubscriber for FilterSub {
            fn kind_filter(&self) -> Option<Vec<ReceiptKind>> {
                Some(vec![ReceiptKind::Sealed])
            }
        }
        let mut rx = bus.subscribe(Box::new(FilterSub));

        bus.emit(
            CanonicalReceiptBuilder::new()
                .with_kind(ReceiptKind::Bound)
                .with_payload(serde_json::json!({}))
                .with_source("test")
                .build(),
        );
        // Bound should be filtered out — nothing in channel
        assert!(
            rx.try_recv().is_err(),
            "Bound receipt should be filtered out"
        );

        bus.emit(
            CanonicalReceiptBuilder::new()
                .with_kind(ReceiptKind::Sealed)
                .with_payload(serde_json::json!({}))
                .with_source("test")
                .build(),
        );
        let received = rx.try_recv().expect("Sealed receipt should pass filter");
        assert_eq!(received.kind, ReceiptKind::Sealed);
    }
    #[test]
    fn test_serde_roundtrip() {
        let receipt = CanonicalReceiptBuilder::new()
            .with_kind(ReceiptKind::StreamStarted)
            .with_payload(serde_json::json!({"stream_id": "s-123"}))
            .with_source("server")
            .with_entity_id("stream:s-123")
            .with_epoch(1)
            .with_seq(5)
            .build();
        let json = serde_json::to_string(&receipt).unwrap();
        let deserialized: CanonicalReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(receipt.id, deserialized.id);
        assert_eq!(receipt.kind, deserialized.kind);
        assert_eq!(receipt.source, deserialized.source);
        assert_eq!(receipt.event_seq, deserialized.event_seq);
        assert_eq!(receipt.world_epoch, deserialized.world_epoch);
    }

    #[test]
    fn test_receipt_id_serde_roundtrip() {
        let id = ReceiptId::compute([0xab; 32], Some("test"), 1, 2);
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: ReceiptId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
        assert_eq!(json, format!("\"{id}\""));
    }

    #[test]
    fn test_kind_serde_snake_case() {
        let json = serde_json::to_string(&ReceiptKind::SourceValidated).unwrap();
        assert_eq!(json, "\"source_validated\"");
        let json = serde_json::to_string(&ReceiptKind::ArtifactDiscovered).unwrap();
        assert_eq!(json, "\"artifact_discovered\"");
        let json = serde_json::to_string(&ReceiptKind::Other("custom_event".into())).unwrap();
        assert_eq!(json, r#"{"other":"custom_event"}"#);
    }

    #[test]
    fn test_kind_deser_snake_case() {
        let kind: ReceiptKind = serde_json::from_str("\"token_decoded\"").unwrap();
        assert_eq!(kind, ReceiptKind::TokenDecoded);
        let kind: ReceiptKind = serde_json::from_str("\"promotion_committed\"").unwrap();
        assert_eq!(kind, ReceiptKind::PromotionCommitted);
    }

    #[test]
    fn test_receipt_id_from_and_into() {
        let bytes = [
            0xde, 0xad, 0xbe, 0xef, 0x00, 0x01, 0x02, 0x03, 0xde, 0xad, 0xbe, 0xef, 0x00, 0x01,
            0x02, 0x03, 0xde, 0xad, 0xbe, 0xef, 0x00, 0x01, 0x02, 0x03, 0xde, 0xad, 0xbe, 0xef,
            0x00, 0x01, 0x02, 0x03,
        ];
        let id: ReceiptId = bytes.into();
        assert_eq!(id.0, bytes);
    }

    #[test]
    fn test_receipt_new() {
        let receipt = CanonicalReceipt::new(
            ReceiptKind::MetricsReported,
            serde_json::json!({"metric": "test"}),
            "test_system",
        );
        assert_eq!(receipt.kind, ReceiptKind::MetricsReported);
        assert_eq!(receipt.source, "test_system");
        assert!(receipt.parent_ids.is_empty());
        assert!(receipt.entity_id.is_none());
    }
}

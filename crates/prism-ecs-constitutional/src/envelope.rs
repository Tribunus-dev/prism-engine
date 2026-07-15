use serde::{Deserialize, Serialize};

use crate::types::{
    AggregateSequence, CausationId, CorrelationId, DomainId, IdempotencyKey, MessageId, Timestamp,
    WorldEpoch,
};

/// Causal envelope for every command, effect, event, and receipt.
///
/// Wraps a payload `T` with the full causal chain (correlation, causation),
/// content-addressed identity, temporal ordering, and domain routing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Envelope<T> {
    /// Content-addressed identifier (blake3 of the causal fields + payload).
    pub id: MessageId,

    /// Groups related messages into a single logical operation.
    pub correlation_id: CorrelationId,

    /// The message that caused this one (optional for root messages).
    pub causation_id: Option<CausationId>,

    /// Stable domain identifier for routing.
    pub target: DomainId,

    /// Epoch in which this envelope was created.
    pub originating_epoch: WorldEpoch,

    /// Idempotency key for safe retry.
    pub idempotency_key: IdempotencyKey,

    /// Wall-clock timestamp.
    pub timestamp: Timestamp,

    /// Per-entity sequence number for ordered replay.
    pub aggregate_sequence: AggregateSequence,

    /// The inner payload (command, effect, event, or receipt).
    pub payload: T,
}

impl<T: Serialize> Envelope<T> {
    /// Compute the content-addressed `MessageId` from the envelope contents.
    ///
    /// The hash covers: `correlation_id || causation_id || epoch_le || seq_le || target || payload_canonical_json`
    /// This guarantees deterministic identity regardless of serialization format.
    pub fn compute_id(&self) -> MessageId {
        let mut hasher = blake3::Hasher::new();

        // correlation_id (UUID bytes)
        hasher.update(self.correlation_id.0.as_bytes());

        // causation_id (opaque string, bytes)
        if let Some(cid) = &self.causation_id {
            hasher.update(cid.0.as_bytes());
        }

        // originating_epoch (little-endian u64)
        hasher.update(&self.originating_epoch.0.to_le_bytes());

        // aggregate_sequence (little-endian u64)
        hasher.update(&self.aggregate_sequence.0.to_le_bytes());

        // target domain (UUID bytes)
        hasher.update(self.target.0.as_bytes());

        // payload as canonical JSON (deterministic key order)
        let canonical = serde_json_canonicalizer::to_string(&self.payload)
            .expect("Envelope::compute_id: payload serialization should never fail");
        hasher.update(canonical.as_bytes());

        MessageId::new(hasher.finalize().into())
    }

    /// Map the payload while preserving all envelope metadata.
    /// The content-addressed ID is recomputed from the new payload,
    /// maintaining the invariant that `id == compute_id()`.
    pub fn map<U: Serialize>(self, f: impl FnOnce(T) -> U) -> Envelope<U> {
        let mut envelope = Envelope {
            id: self.id,
            correlation_id: self.correlation_id,
            causation_id: self.causation_id,
            target: self.target,
            originating_epoch: self.originating_epoch,
            idempotency_key: self.idempotency_key,
            timestamp: self.timestamp,
            aggregate_sequence: self.aggregate_sequence,
            payload: f(self.payload),
        };
        envelope.id = envelope.compute_id();
        envelope
    }
}

impl<T> Envelope<T> {
    /// Borrow the payload.
    pub fn payload(&self) -> &T {
        &self.payload
    }

    /// Consume the envelope and return the payload.
    pub fn into_payload(self) -> T {
        self.payload
    }
}

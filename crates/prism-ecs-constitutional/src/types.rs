use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// Re-export WorldEpoch from prism-ecs-core (shared type across crates).
pub use prism_ecs_core::WorldEpoch;

/// Per-entity aggregate sequence — ordered events within one entity's history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AggregateSequence(pub u64);

/// Component schema identity — stable across process restarts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ComponentSchemaId(pub u64);

/// Schema version for migration detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SchemaVersion(pub u32);

/// Content-addressed message identity (blake3 hash).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(#[serde(with = "hex_array")] pub [u8; 32]);

impl MessageId {
    /// Construct from a raw 32-byte hash.
    pub fn new(hash: [u8; 32]) -> Self {
        Self(hash)
    }

    /// Compute a content-addressed MessageId from arbitrary bytes (blake3).
    pub fn compute(data: &[u8]) -> Self {
        let hash = blake3::hash(data);
        Self(*hash.as_bytes())
    }

    /// Access the underlying bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Consume into the underlying bytes.
    pub fn into_bytes(self) -> [u8; 32] {
        self.0
    }
}

impl From<[u8; 32]> for MessageId {
    fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl fmt::Display for MessageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Correlation ID — groups related messages (commands, events, receipts).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CorrelationId(pub uuid::Uuid);

impl CorrelationId {
    /// Create a new random correlation ID.
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl Default for CorrelationId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<uuid::Uuid> for CorrelationId {
    fn from(uuid: uuid::Uuid) -> Self {
        Self(uuid)
    }
}

/// Causation ID — the message that caused this one.
///
/// Serialized as an opaque string (e.g. a MessageId hex, a UUID, or a URI).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CausationId(pub String);

impl CausationId {
    /// Create a causation ID from a message ID.
    pub fn from_message_id(id: &MessageId) -> Self {
        Self(id.to_string())
    }
}

impl From<String> for CausationId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for CausationId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

/// Idempotency key — safe-to-retry marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IdempotencyKey(pub uuid::Uuid);

impl IdempotencyKey {
    /// Create a new random idempotency key.
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl Default for IdempotencyKey {
    fn default() -> Self {
        Self::new()
    }
}

impl From<uuid::Uuid> for IdempotencyKey {
    fn from(uuid: uuid::Uuid) -> Self {
        Self(uuid)
    }
}

/// Stable domain ID — cross-process, survives restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DomainId(pub uuid::Uuid);

impl DomainId {
    /// Create a new random domain ID.
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl Default for DomainId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<uuid::Uuid> for DomainId {
    fn from(uuid: uuid::Uuid) -> Self {
        Self(uuid)
    }
}

/// Wall-clock timestamp with nanosecond precision.
///
/// Stored as nanoseconds since Unix epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Timestamp(pub u64);

impl Timestamp {
    /// Create a timestamp from nanoseconds since Unix epoch.
    pub fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Returns the current wall-clock time as a `Timestamp`.
    pub fn now() -> Self {
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before Unix epoch");
        Self(duration.as_nanos() as u64)
    }

    /// Return nanoseconds since Unix epoch.
    pub fn as_nanos(&self) -> u64 {
        self.0
    }
}

/// Entity kind for entity identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityKindId(pub u64);

/// Schema key — stable protocol identifier independent of crate names.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct SchemaKey {
    pub namespace: &'static str,
    pub id: u32,
    pub version: u32,
}

/// 256-bit digest for component value identification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Digest256(pub [u8; 32]);

// ── Authority-bearing primitives (B-2 newtypes for the `cmd!` macro) ────────
//
// Every value below is a transparent newtype around a primitive. The type
// says what the value is. `#[serde(transparent)]` ensures the wire format
// is unchanged — existing serialized commands continue to deserialize
// correctly. The `cmd!` macro in `lifecycle_command.rs` is the primary
// consumer; call sites are updated in the B-2 follow-up change.

/// Fencing generation: monotonic per resource; replaced on lease acquire.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Generation(pub u32);

/// World epoch: increments on every `WorldTxn` commit. Read by stale-fencing.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Epoch(pub u64);

/// Event sequence: monotonic per `EventStore`; never reused.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Sequence(pub u64);

/// Command identity: assigned at ingress; never reused.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct CommandId(pub u64);

/// Filesystem path: not a free `String`.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct FilePath(pub String);

/// Format tag: e.g. `"gguf"`, `"cimage"`, `"safetensors"`. Validated against
/// the registered format set at construction.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Format(pub String);

/// Rejection reason: human-readable, validated, not a free `String`.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct RejectionReason(pub String);

/// Adapter handle: backend-specific opaque token. The adapter is the only
/// authority that can interpret the value.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct AdapterHandle(pub String);

/// Backend config: free-form `key=value` text. Validated by the backend.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Config(pub String);

/// Receipt identity: monotonic per work entity; never reused.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct ReceiptId(pub String);

/// Lease token: opaque to the constitutional layer; verified by the
/// dispatcher at effect time. Replaces the `String` token used in
/// `LifecycleCommandResult::LeaseAcquired::token`.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct LeaseToken(pub String);

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Serialize/deserialize `[u8; 32]` as a lowercase hex string.
pub(crate) mod hex_array {
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

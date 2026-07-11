use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// Global world epoch — total commit order counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorldEpoch(pub u64);

impl PartialOrd for WorldEpoch {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.0.cmp(&other.0))
    }
}

impl Ord for WorldEpoch {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

/// Per-entity aggregate sequence — ordered events within one entity's history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AggregateSequence(pub u64);

/// Component schema identity — stable across process restarts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

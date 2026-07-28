//! Shared types for the `compute_image_runtime` surface — content hash
//! newtype and execution shape class.
//!
//! These types are shared by the various sub-modules of the
//! `compute_image_runtime` namespace. The engine's
//! `compute-core/src/ecs/integration::ContentHash` is the same
//! authority (a 64-bit content-addressed hash), and the engine's
//! `compute-core/src/ecs/compute_image::execution_shape::ExecutionShapeClass`
//! is the same authority (a 6-variant enum of execution shapes). The
//! constitutional surface re-implements them so the data types here
//! can stand on their own.

use serde::{Deserialize, Serialize};

/// A content-addressed hash value used to identify objects in the
/// content store. Wraps a raw `u64` for compact serialization.
///
/// This is the constitutional newtype for what the engine previously
/// exposed as `crate::integration::ContentHash`. The two are
/// semantically equivalent; engine callers of the legacy path continue
/// to use the engine's wrapper, while constitutional callers use this
/// newtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash(pub u64);

impl ContentHash {
    /// The zero hash — used as a sentinel / unset value.
    pub const ZERO: ContentHash = ContentHash(0);

    /// Create from a raw `u64`.
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    /// Create from a hex-encoded SHA-256 digest string. The conversion
    /// is intentionally simple (a 64-bit hasher) — the canonical full
    /// 256-bit digests are produced by the content-store pipeline and
    /// are not re-derived here.
    pub fn from_hex(hex: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        hex.hash(&mut hasher);
        Self(hasher.finish())
    }

    /// Return the inner `u64`.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl Default for ContentHash {
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ContentHash({})", self.0)
    }
}

impl From<u64> for ContentHash {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<ContentHash> for u64 {
    fn from(value: ContentHash) -> Self {
        value.0
    }
}

/// Shape-specialized execution variants.
///
/// Each variant identifies a distinct execution shape class — the
/// runtime selects a compiled phase program whose
/// [`ExecutionShapeClass`] best matches the incoming request shape.
///
/// This is the constitutional enum for what the engine previously
/// exposed as
/// `crate::ecs::compute_image::execution_shape::ExecutionShapeClass`.
/// The two are semantically equivalent; engine callers of the legacy
/// path continue to use the engine's wrapper, while constitutional
/// callers use this enum.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub enum ExecutionShapeClass {
    /// Single-token decode (autoregressive generation, one step).
    #[default]
    Decode1,
    /// Batched decode with up to `max_batch` concurrent sequences.
    DecodeBatch {
        /// Maximum concurrent sequences.
        max_batch: u32,
    },
    /// Prefix prefill with up to `tokens` KV entries.
    PrefillBucket {
        /// Maximum KV entries.
        tokens: u32,
    },
    /// Chunked prefill — processes `chunk_tokens` per micro-batch.
    ChunkedPrefill {
        /// Tokens per micro-batch.
        chunk_tokens: u32,
    },
    /// Mixed batch — interleaved decode/prefill within the same invocation.
    MixedBatch,
    /// Diffusion forward — processes image/video canvas tokens.
    DiffusionForward {
        /// Maximum canvas tokens.
        max_canvas_tokens: u32,
    },
}

impl ExecutionShapeClass {
    /// Human-readable label for this shape class.
    pub fn variant_name(&self) -> &'static str {
        match self {
            Self::Decode1 => "Decode1",
            Self::DecodeBatch { .. } => "DecodeBatch",
            Self::PrefillBucket { .. } => "PrefillBucket",
            Self::ChunkedPrefill { .. } => "ChunkedPrefill",
            Self::MixedBatch => "MixedBatch",
            Self::DiffusionForward { .. } => "DiffusionForward",
        }
    }
}

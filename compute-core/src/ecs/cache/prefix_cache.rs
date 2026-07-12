//! Stub re-implementation of the deleted prefix_cache.
//!
//! The original file was removed but callers in `distributed_kv.rs` and
//! `session.rs` still depend on these types and functions.  This stub
//! provides the minimum API surface to keep them compiling.
//!
//! TODO: restore or rewrite a real prefix cache.

use std::hash::{Hash, Hasher};

/// A content-addressable hash for a KV block — matches a block of token IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockHash(pub [u8; 32]);

/// Number of tokens per prefix block.
pub const PREFIX_BLOCK_SIZE: usize = 64;

/// Block-aware prefix cache (stub).
///
/// The real implementation used content-based hashing to share KV blocks
/// across sequences.  This stub retains the public API without any actual
/// caching behaviour.
pub struct BlockAwarePrefixCache {
    _capacity: usize,
}

impl BlockAwarePrefixCache {
    /// Create a new prefix cache with the given local capacity (in blocks).
    pub fn new(capacity: usize) -> Self {
        Self {
            _capacity: capacity,
        }
    }

    /// Compute a content hash for a block of token IDs.
    ///
    /// Uses a simple deterministic hash (the real implementation used a
    /// stronger content-addressable hash).
    pub fn compute_block_hash(tokens: &[u32]) -> BlockHash {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &t in tokens {
            t.hash(&mut hasher);
        }
        let h = hasher.finish();
        let mut bytes = [0u8; 32];
        bytes[..8].copy_from_slice(&h.to_ne_bytes());
        BlockHash(bytes)
    }
}

/// Check whether a shared prefix of the given token sequence is cached.
///
/// Returns `Some((matched_hashes, prefix_token_count))` when a prefix exists,
/// or `None` when even the first block is not in cache.
///
/// Stub: always returns `None` (no prefix caching).
pub fn check_shared_prefix(tokens: &[u32]) -> Option<(Vec<BlockHash>, usize)> {
    let _ = tokens;
    None
}

/// Insert a token sequence (starting at `start_offset`) into the shared cache.
///
/// Stub: no-op.
pub fn insert_shared_prefix(tokens: &[u32], start_offset: usize) {
    let _ = (tokens, start_offset);
}

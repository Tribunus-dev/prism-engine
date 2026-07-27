//! Prefill orchestration system (constitutional home).
//!
//! Orchestrates chunked ANE prefill with IOSurface-backed KV output.
//! This is the constitutional-side counterpart to the engine's
//! `PrefillOrchestrator`; the engine file is the legacy duplicate
//! and is deleted in step 58.
//!
//! # Placeholder
//!
//! The engine's orchestrator holds raw Core ML pointers, IOSurface
//! handles, and BTreeMap-keyed island pointers. These are FFI-side
//! and move to `prism-ecs-kernel::backend::ane` in the ANE migration.
//! The constitutional side has a placeholder with the public API
//! and an invariant test; the full implementation is added when
//! the engine's prefill orchestrator callers migrate.

use std::collections::BTreeMap;

/// A precompiled Core ML ANE prefill island for a specific chunk size.
#[derive(Debug, Clone)]
pub struct AnePrefillIsland {
    pub chunk_size: usize,
    /// Opaque pointer to the compiled island. The constitutional
    /// side stores it as a raw `usize` to keep the runtime crate
    /// FFI-independent; the kernel side (step 50) re-introduces
    /// the proper pointer type.
    pub ptr: usize,
}

/// Constitutional-side prefill orchestrator.
///
/// The full implementation is added when the engine's prefill
/// orchestrator callers migrate.
pub struct PrefillOrchestrator {
    /// BTreeMap of chunk size → island pointer (placeholder).
    islands: BTreeMap<usize, usize>,
    /// Maximum sequence length.
    pub max_seq_len: usize,
}

impl PrefillOrchestrator {
    pub fn new(islands: Vec<AnePrefillIsland>, max_seq_len: usize) -> Self {
        let mut map = BTreeMap::new();
        for i in islands {
            map.insert(i.chunk_size, i.ptr);
        }
        Self {
            islands: map,
            max_seq_len,
        }
    }

    /// Select the largest compiled chunk size that fits the
    /// remaining token count.
    pub fn select_optimal_chunk_size(&self, remaining: usize) -> Option<usize> {
        // The largest chunk ≤ remaining, or the smallest available.
        for (&size, _) in self.islands.iter().rev() {
            if remaining >= size {
                return Some(size);
            }
        }
        self.islands.keys().next().copied()
    }

    /// Pad a token slice to the required static chunk size.
    pub fn pad_token_chunk(tokens: &[u32], required: usize, pad_id: u32) -> Vec<u32> {
        let mut chunk = tokens.to_vec();
        chunk.truncate(required);
        chunk.resize(required, pad_id);
        chunk
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `prefill_orchestration` system.

    use super::*;

    fn island(size: usize) -> AnePrefillIsland {
        AnePrefillIsland {
            chunk_size: size,
            ptr: size as usize, // placeholder
        }
    }

    #[test]
    fn select_optimal_chunk_picks_largest_fitting() {
        // Architectural invariant: the optimal chunk size is the
        // largest compiled size that does not exceed the remaining
        // token count.
        let islands = vec![island(64), island(256), island(1024)];
        let orch = PrefillOrchestrator::new(islands, 4096);
        assert_eq!(orch.select_optimal_chunk_size(200), Some(64));
        assert_eq!(orch.select_optimal_chunk_size(1024), Some(1024));
        assert_eq!(orch.select_optimal_chunk_size(5000), Some(1024));
    }

    #[test]
    fn select_optimal_chunk_falls_back_to_smallest() {
        // Architectural invariant: when no chunk fits the remaining
        // count, fall back to the smallest available chunk (the
        // engine may pad the prompt).
        let islands = vec![island(1024), island(2048)];
        let orch = PrefillOrchestrator::new(islands, 4096);
        // remaining=10 < 1024; no chunk fits, fall back to 1024.
        assert_eq!(orch.select_optimal_chunk_size(10), Some(1024));
    }

    #[test]
    fn pad_token_chunk_resizes_correctly() {
        // Architectural invariant: pad_token_chunk truncates the
        // input to `required` and pads with `pad_id` if shorter.
        let tokens = vec![1, 2, 3];
        let padded = PrefillOrchestrator::pad_token_chunk(&tokens, 5, 99);
        assert_eq!(padded, vec![1, 2, 3, 99, 99]);
    }

    #[test]
    fn pad_token_chunk_truncates_if_too_long() {
        // Architectural invariant: pad_token_chunk truncates the
        // input when it is longer than `required`.
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let padded = PrefillOrchestrator::pad_token_chunk(&tokens, 4, 99);
        assert_eq!(padded.len(), 4);
        assert_eq!(padded, vec![1, 2, 3, 4]);
    }
}

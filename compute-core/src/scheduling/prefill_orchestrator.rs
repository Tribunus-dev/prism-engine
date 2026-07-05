//! Phase 2d: ANE chunked prefill runtime orchestrator.

use std::collections::BTreeMap;

use crate::arena::Arena;
use crate::coreml_state::StatefulPrefillContext;
use crate::models::embedding::TokenEmbedding;

/// A precompiled Core ML ANE prefill island for a specific chunk size.
pub struct AnePrefillIsland {
    pub chunk_size: usize,
    pub ptr: std::ptr::NonNull<std::ffi::c_void>,
}

/// Orchestrator for chunked ANE prefill with IOSurface-backed KV output.
pub struct PrefillOrchestrator {
    /// Model pointers pre-extracted to avoid borrow conflicts with state_ctx.
    island_ptrs: BTreeMap<usize, std::ptr::NonNull<std::ffi::c_void>>,
    state_ctx: Option<StatefulPrefillContext>,
    embedding: TokenEmbedding,
    max_seq_len: usize,
}

impl PrefillOrchestrator {
    pub fn new(
        islands: Vec<AnePrefillIsland>,
        embedding: TokenEmbedding,
        max_seq_len: usize,
    ) -> Self {
        let mut island_ptrs = BTreeMap::new();
        for i in islands {
            island_ptrs.insert(i.chunk_size, i.ptr);
        }
        PrefillOrchestrator {
            island_ptrs,
            state_ctx: None,
            embedding,
            max_seq_len,
        }
    }

    /// Select the largest compiled chunk size ≤ remaining tokens.
    pub fn select_optimal_chunk_size(&self, remaining: usize) -> Option<usize> {
        for (&size, _) in self.island_ptrs.iter().rev() {
            if remaining >= size {
                return Some(size);
            }
        }
        self.island_ptrs.keys().next().copied()
    }

    /// Pad a token slice to the required static chunk size.
    pub fn pad_token_chunk(tokens: &[u32], required: usize, pad_id: u32) -> Vec<u32> {
        let mut chunk = tokens.to_vec();
        chunk.truncate(required);
        chunk.resize(required, pad_id);
        chunk
    }

    /// Run chunked ANE prefill over a prompt.
    pub fn execute_chunked_prefill(&mut self, prompt_tokens: &[u32]) -> Result<usize, String> {
        let total = prompt_tokens.len();
        if total > self.max_seq_len {
            return Err(format!(
                "Prompt length {} exceeds max KV capacity {}",
                total, self.max_seq_len
            ));
        }

        // Pre-extract model pointers — avoids borrow conflicts with state_ctx.
        let mut model_ptrs: Vec<std::ptr::NonNull<std::ffi::c_void>> = Vec::new();
        for (_, p) in &self.island_ptrs {
            model_ptrs.push(*p);
        }
        if model_ptrs.is_empty() {
            return Err("No prefill islands".to_string());
        }

        // Lazy-init MLState.
        if self.state_ctx.is_none() {
            self.state_ctx =
                Some(StatefulPrefillContext::new(model_ptrs[0].as_ptr())?);
        }

        let hidden = self.embedding.hidden_dim() as u32;
        let mut seq_offset = 0;
        while seq_offset < total {
            let remaining = total - seq_offset;
            let chunk_size = self
                .select_optimal_chunk_size(remaining)
                .ok_or("No island fits remaining tokens")?;

            let padded = Self::pad_token_chunk(
                &prompt_tokens[seq_offset..],
                chunk_size,
                self.embedding.pad_token_id(),
            );

            // CPU-side FP16 embedding lookup
            let activations = self.embedding.lookup(&padded);

            // IOSurface-backed arenas and Core ML stateful prefill
            let ctx = self.state_ctx.as_mut().unwrap();
            let mut input_arena =
                Arena::new(chunk_size as u32, hidden, mlx_rs::Dtype::Float16)
                    .map_err(|e| format!("input arena: {e}"))?;
            input_arena.lock().map_err(|e| format!("input lock: {e}"))?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    activations.as_ptr(),
                    input_arena.info.base_address as *mut u16,
                    activations.len(),
                );
            }
            input_arena.unlock().map_err(|e| format!("input unlock: {e}"))?;

            let mut output_arena =
                Arena::new(chunk_size as u32, hidden, mlx_rs::Dtype::Float16)
                    .map_err(|e| format!("output arena: {e}"))?;

            ctx.prefill_chunk(
                model_ptrs[0].as_ptr(),
                &input_arena.info,
                &mut output_arena.info,
                chunk_size as u32,
                hidden,
                hidden,
            )?;

            seq_offset += std::cmp::min(remaining, chunk_size);
            // ctx drops here, releasing the mutable borrow on self.state_ctx
        }

        Ok(seq_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_emb() -> TokenEmbedding {
        let weights = (0..32).map(|i| i as u16).collect();
        TokenEmbedding::new(weights, 4, 8, 0)
    }

    #[test]
    fn test_select_optimal_chunk_size() {
        let islands = vec![
            AnePrefillIsland {
                chunk_size: 128,
                ptr: std::ptr::NonNull::dangling(),
            },
            AnePrefillIsland {
                chunk_size: 256,
                ptr: std::ptr::NonNull::dangling(),
            },
            AnePrefillIsland {
                chunk_size: 512,
                ptr: std::ptr::NonNull::dangling(),
            },
        ];
        let orch = PrefillOrchestrator::new(islands, dummy_emb(), 4096);
        assert_eq!(orch.select_optimal_chunk_size(1024), Some(512));
        assert_eq!(orch.select_optimal_chunk_size(300), Some(256));
        assert_eq!(orch.select_optimal_chunk_size(200), Some(128));
        assert_eq!(orch.select_optimal_chunk_size(50), Some(128));
    }

    #[test]
    fn test_pad_token_chunk() {
        let padded = PrefillOrchestrator::pad_token_chunk(&[1, 2, 3], 5, 0);
        assert_eq!(padded.len(), 5);
        assert_eq!(padded[0], 1);
        assert_eq!(padded[4], 0);

        let exact = PrefillOrchestrator::pad_token_chunk(&[1, 2, 3, 4], 4, 0);
        assert_eq!(exact, vec![1, 2, 3, 4]);
    }
}

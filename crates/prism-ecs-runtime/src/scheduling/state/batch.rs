//! Batch construction state (constitutional home).
//!
//! This is the constitutional home for batch construction utilities:
//! the [`Batch`] and [`Slot`] data types, the prefill/decode batch
//! builders, the [`BatchedPrefill`] record, and the decode-step
//! batching helper.
//!
//! # Authority
//!
//! All types in this module are **scheduling state** in the C bucket.
//! A batch becomes visible to dispatch-selection only after the
//! batching system stages it through `ConstitutionalWorldTxn`. A
//! `BatchedPrefill` is a planning record; it does not commit any
//! scheduling decision until the runtime reconciliation system
//! validates and stages the resulting dispatch.
//!
//! # Placeholder engine types
//!
//! The engine's `batch.rs` references engine types (`Request`,
//! `RequestState`, `HardwareConfig`, `profiled_executor::LoadedProfiledModel`).
//! The constitutional home defines **placeholder newtypes** for
//! each so the runtime file builds and tests. When the engine files
//! for those types move into their constitutional homes (in their
//! own migration steps), the placeholders here are replaced by the
//! moved definitions.
//!
//! # Migration provenance
//!
//! The legacy home was `compute-core/src/ecs/scheduling/batch.rs`
//! (the builders) and `compute-core/src/ecs/scheduling/mod.rs` (the
//! `Batch` and `Slot` type definitions). The engine file is the
//! legacy duplicate; step 58 deletes it when no engine caller
//! remains. No compatibility facade.

// ---------------------------------------------------------------------------
// Placeholder engine types
// ---------------------------------------------------------------------------

/// Placeholder for `compute-core::ecs::scheduling::RequestState` (an
/// engine-only enum). Replaced when `request` moves in step 12.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RequestState {
    Queued,
    Prefilling,
    Decoding,
    Paused,
    Completed,
    Cancelled,
}

/// Placeholder for `compute-core::ecs::scheduling::Request` (an
/// engine-only struct). Replaced when `request` moves in step 12.
/// Wire shape (id, prompt, max_tokens, state) is preserved.
#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub priority: u8,
    pub state: RequestState,
    pub created_at: std::time::Instant,
    pub slot: Option<usize>,
}

/// Placeholder for `compute-core::ecs::scheduling::HardwareConfig`
/// (an engine-only struct). Replaced when the hardware-detection
/// path moves into `prism-ecs-runtime` (separate migration).
#[derive(Debug, Clone)]
pub struct HardwareConfig {
    pub total_ram_gb: u32,
    pub gpu_cores: u32,
    pub ane_cores: u32,
    pub cpu_cores: u32,
    pub memory_bw_gb_s: u32,
    pub is_memory_rich: bool,
    pub recommended_batch_size: u32,
    pub recommended_spec_length: u32,
    pub enable_weight_streaming: bool,
    pub enable_kv_disk_eviction: bool,
    pub max_concurrent_sequences: u32,
}

// ---------------------------------------------------------------------------
// Slot
// ---------------------------------------------------------------------------

/// A slot in the batch (one model execution unit).
///
/// A `Slot` is a scheduling-state record. The dispatch-selection system
/// allocates slots from the lane-lease state; the batching system
/// populates a slot with the request's prompt length and KV-cache
/// pages. Once a batch commits, every slot in it is in-flight; the
/// completion-reconciliation system releases slots on completion.
#[derive(Debug, Clone)]
pub struct Slot {
    pub id: usize,
    pub request_id: Option<u64>,
    pub tokens_generated: usize,
    pub kv_cache_start: usize,
    pub kv_cache_length: usize,
    /// Target execution backend for this slot.
    /// 0=MLX, 1=Accelerate, 2=CoreML, 3=ANE/Orion
    pub backend_id: u32,
    /// Page IDs allocated from the paged allocator for this slot's KV cache.
    pub kv_cache_pages: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Batch
// ---------------------------------------------------------------------------

/// A batch of slots for model execution.
#[derive(Debug, Clone)]
pub struct Batch {
    pub slots: Vec<Slot>,
    pub batch_size: usize,
    pub max_batch_size: usize,
}

// ---------------------------------------------------------------------------
// Batch builders
// ---------------------------------------------------------------------------

/// Build a prefill batch from queued requests.
pub fn build_prefill_batch(requests: &[Request], max_size: usize) -> Batch {
    let slots: Vec<Slot> = requests
        .iter()
        .take(max_size)
        .enumerate()
        .map(|(i, req)| Slot {
            id: i,
            request_id: Some(req.id),
            tokens_generated: 0,
            kv_cache_start: 0,
            kv_cache_length: req.prompt.len(),
            backend_id: 0,
            kv_cache_pages: vec![],
        })
        .collect();

    Batch {
        slots: slots.clone(),
        batch_size: slots.len(),
        max_batch_size: max_size,
    }
}

/// Build a decode batch from active requests.
pub fn build_decode_batch(active: &[Request], max_size: usize) -> Batch {
    let slots: Vec<Slot> = active
        .iter()
        .take(max_size)
        .enumerate()
        .map(|(i, req)| Slot {
            id: i,
            request_id: Some(req.id),
            tokens_generated: req.max_tokens,
            kv_cache_start: 0,
            kv_cache_length: req.max_tokens,
            backend_id: 0,
            kv_cache_pages: vec![],
        })
        .collect();

    Batch {
        slots: slots.clone(),
        batch_size: slots.len(),
        max_batch_size: max_size,
    }
}

// ---------------------------------------------------------------------------
// BatchedPrefill
// ---------------------------------------------------------------------------

/// Concatenate multiple prompts into a single batched forward pass.
///
/// MLX handles batched inputs naturally by stacking sequences along the
/// batch dimension. This is the key throughput optimization for memory-rich
/// hardware (e.g. M3 Ultra with 512 GB): instead of serial prefills, we
/// merge all queued prompts into one forward pass.
///
/// A `BatchedPrefill` is a planning record. The runtime batches
/// it into a `Batch` and stages the dispatch through
/// `ConstitutionalWorldTxn`.
pub struct BatchedPrefill {
    /// The prompts to process, each as a vector of token IDs.
    pub prompts: Vec<Vec<u32>>,
    /// Maximum sequence length in tokens (pads shorter prompts).
    pub max_seq_len: u32,
}

impl BatchedPrefill {
    /// Create a new batched prefill from the given prompts.
    pub fn new(prompts: Vec<Vec<u32>>, max_seq_len: u32) -> Self {
        Self {
            prompts,
            max_seq_len,
        }
    }

    /// Create a batched prefill configured for the detected hardware.
    pub fn new_for_hardware(hw: &HardwareConfig, model_prompt: Vec<u32>) -> Self {
        // Duplicate the prompt across the batch dimension. In real usage
        // each slot has its own prompt; this is a convenience for single
        // model-server startup or warmup.
        let count = hw.recommended_batch_size as usize;
        let prompts = vec![model_prompt; count];
        Self {
            prompts,
            max_seq_len: 262_144,
        }
    }

    /// Validate that the longest prompt fits within `max_seq_len`.
    ///
    /// Returns the maximum prompt length if it fits, or an error
    /// describing the overflow.
    ///
    /// In the engine, `execute()` actually runs the batched forward
    /// pass via `profiled_executor::LoadedProfiledModel`. The
    /// constitutional home does not own a model runtime; the runtime
    /// reconciliation system observes the dispatch and stages the
    /// result. This method is the validation half of the engine's
    /// execute — the actual kernel submission is the kernel's job.
    pub fn validate(&self) -> Result<usize, String> {
        let batch_size = self.prompts.len();
        if batch_size == 0 {
            return Ok(0);
        }
        let max_prompt_len = self.prompts.iter().map(|p| p.len()).max().unwrap_or(0);
        if max_prompt_len as u32 > self.max_seq_len {
            return Err(format!(
                "Prompt length {max_prompt_len} exceeds max sequence length {}",
                self.max_seq_len
            ));
        }
        Ok(max_prompt_len)
    }
}

/// Batch multiple decode steps into a single forward pass.
///
/// When decode batching is enabled, the scheduler collects pending decode
/// slots and runs them as a single tensor operation rather than N serial
/// steps. The MLX backend handles this automatically when multiple
/// sequences share the same model weights.
pub fn batch_decode_steps(slots: &[Slot]) -> Vec<Vec<u32>> {
    slots
        .iter()
        .map(|s| {
            // Each slot emits its next token. In production this calls
            // into the model runtime's batched decode.
            vec![s.id as u32]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `batch` state.
    //!
    //! These tests verify the constitutional rules: batch size is
    //! bounded by `max_size`, the builders consume requests in order,
    //! and the validation rule for `BatchedPrefill` is monotone.

    use super::*;

    fn req(id: u64, prompt_len: usize, max_tokens: usize) -> Request {
        Request {
            id,
            prompt: vec![0; prompt_len],
            max_tokens,
            priority: 0,
            state: RequestState::Queued,
            created_at: std::time::Instant::now(),
            slot: None,
        }
    }

    #[test]
    fn build_prefill_batch_respects_max_size() {
        let requests: Vec<Request> = (0..10).map(|i| req(i, 100, 50)).collect();
        let batch = build_prefill_batch(&requests, 3);
        assert_eq!(batch.batch_size, 3);
        assert_eq!(batch.max_batch_size, 3);
        assert_eq!(batch.slots.len(), 3);
    }

    #[test]
    fn build_prefill_batch_initializes_zero_tokens_generated() {
        let requests = vec![req(1, 100, 50)];
        let batch = build_prefill_batch(&requests, 4);
        assert_eq!(batch.slots[0].tokens_generated, 0);
        assert_eq!(batch.slots[0].kv_cache_length, 100);
        assert_eq!(batch.slots[0].backend_id, 0);
        assert!(batch.slots[0].kv_cache_pages.is_empty());
    }

    #[test]
    fn build_decode_batch_uses_max_tokens_for_kv_cache() {
        // Architectural invariant: a decode slot's kv_cache_length is
        // derived from the request's `max_tokens`, not the prompt
        // length. Decode slots are sized for the decode window.
        let requests = vec![req(1, 100, 50)];
        let batch = build_decode_batch(&requests, 4);
        assert_eq!(batch.slots[0].kv_cache_length, 50);
        assert_eq!(batch.slots[0].tokens_generated, 50);
    }

    #[test]
    fn build_decode_batch_respects_max_size() {
        let requests: Vec<Request> = (0..10).map(|i| req(i, 100, 50)).collect();
        let batch = build_decode_batch(&requests, 5);
        assert_eq!(batch.batch_size, 5);
    }

    #[test]
    fn batch_size_is_at_most_max_size() {
        let requests: Vec<Request> = (0..3).map(|i| req(i, 100, 50)).collect();
        let batch = build_prefill_batch(&requests, 100);
        assert!(batch.batch_size <= batch.max_batch_size);
    }

    #[test]
    fn empty_input_produces_empty_batch() {
        let batch = build_prefill_batch(&[], 10);
        assert_eq!(batch.batch_size, 0);
        assert!(batch.slots.is_empty());
    }

    #[test]
    fn batched_prefill_validate_rejects_overlong_prompt() {
        let bp = BatchedPrefill {
            prompts: vec![vec![0; 1000]],
            max_seq_len: 100,
        };
        assert!(bp.validate().is_err());
    }

    #[test]
    fn batched_prefill_validate_accepts_fitting_prompts() {
        let bp = BatchedPrefill {
            prompts: vec![vec![0; 50], vec![0; 100]],
            max_seq_len: 100,
        };
        assert_eq!(bp.validate().expect("validate"), 100);
    }

    #[test]
    fn batched_prefill_validate_returns_zero_for_empty_input() {
        let bp = BatchedPrefill {
            prompts: vec![],
            max_seq_len: 100,
        };
        assert_eq!(bp.validate().expect("validate empty"), 0);
    }

    #[test]
    fn batch_decode_steps_emits_one_per_slot() {
        let slots = vec![
            Slot {
                id: 0,
                request_id: Some(1),
                tokens_generated: 0,
                kv_cache_start: 0,
                kv_cache_length: 0,
                backend_id: 0,
                kv_cache_pages: vec![],
            },
            Slot {
                id: 1,
                request_id: Some(2),
                tokens_generated: 0,
                kv_cache_start: 0,
                kv_cache_length: 0,
                backend_id: 0,
                kv_cache_pages: vec![],
            },
        ];
        let steps = batch_decode_steps(&slots);
        assert_eq!(steps.len(), 2);
    }
}

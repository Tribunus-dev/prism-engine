//! Public execution surface — `run_batch` / `run_prefill` / `run_decode` /
//! `reset_kv_cache` on [`UnifiedRuntime`].
//!
//! This module owns the canonical authority for the orchestrator's
//! end-to-end inference entry points. The public methods set the
//! execution mode, build a `WorkloadScenario`, and delegate the
//! per-token logits to [`super::dispatch::dispatch_tokens`] (which
//! tries the AOT plan, then a UOp program, then the CPU reference
//! path). The KV cache is owned exclusively by these methods.
//!
//! No tensor arithmetic, no kernel contract, no ANE device calls
//! live here — those live in [`super::dispatch`] and
//! [`super::super::certification`].

use super::super::RuntimeError;
use super::dispatch::{argmax_token, dispatch_tokens};
use super::UnifiedRuntime;

impl UnifiedRuntime {
    /// Run batch inference on the loaded model.
    ///
    /// Processes all input tokens in parallel, producing one logit vector
    /// per token. This is the GEMM-heavy code path used for scoring or
    /// classification.
    ///
    pub fn run_batch(&mut self, input_tokens: &[u32]) -> Result<Vec<f32>, RuntimeError> {
        self.mode = super::ExecutionMode::Batch;
        self.dispatch_tokens_via_helper(input_tokens)
    }

    /// Run batch inference with an explicit workload batch shape. The batch
    /// value is policy metadata for strategy selection; callers remain
    /// responsible for packing the corresponding token buffer.
    pub fn run_batch_for_workload(
        &mut self,
        input_tokens: &[u32],
        batch_size: u32,
    ) -> Result<Vec<f32>, RuntimeError> {
        if batch_size == 0 {
            return Err(RuntimeError::ExecutionFailed(
                "batch workload size must be nonzero".into(),
            ));
        }
        if input_tokens.is_empty() || input_tokens.len() % batch_size as usize != 0 {
            return Err(RuntimeError::ExecutionFailed(
                "batch workload tokens must contain a nonempty, whole number of sequences".into(),
            ));
        }
        let previous = self.requested_batch_size.replace(batch_size);
        self.mode = super::ExecutionMode::Batch;
        let result = self.dispatch_tokens_via_helper(input_tokens);
        self.requested_batch_size = previous;
        result
    }

    /// Run autoregressive prefill.
    ///
    /// Processes all prompt tokens in a single forward pass, populating the
    /// KV cache and returning the first generated token(s). After prefill
    /// the caller switches to [`run_decode`](Self::run_decode) for each
    /// subsequent token.
    ///
    pub fn run_prefill(&mut self, input_tokens: &[u32]) -> Result<Vec<u32>, RuntimeError> {
        if input_tokens.is_empty() {
            return Err(RuntimeError::ExecutionFailed(
                "prefill requires tokens".into(),
            ));
        }
        let logits = self.dispatch_tokens_via_helper(input_tokens)?;
        self.kv_cache = Some(vec![input_tokens
            .iter()
            .flat_map(|t| t.to_ne_bytes())
            .collect()]);
        self.mode = super::ExecutionMode::RealtimePrefill;
        Ok(vec![argmax_token(&logits)])
    }

    /// Run canonical realtime prefill and return its logits. This is the
    /// adapter-facing form used by runtimes that own token sampling and KV
    /// lifecycle themselves.
    pub fn run_prefill_logits(&mut self, input_tokens: &[u32]) -> Result<Vec<f32>, RuntimeError> {
        if input_tokens.is_empty() {
            return Err(RuntimeError::ExecutionFailed(
                "prefill requires tokens".into(),
            ));
        }
        let logits = self.dispatch_tokens_via_helper(input_tokens)?;
        self.kv_cache = Some(vec![input_tokens
            .iter()
            .flat_map(|token| token.to_ne_bytes())
            .collect()]);
        self.mode = super::ExecutionMode::RealtimePrefill;
        Ok(logits)
    }

    /// Run a single autoregressive decode step.
    ///
    /// Consumes the last generated token (stored in KV cache state),
    /// runs a single-token forward pass, and returns the next token ID.
    ///
    /// Must be preceded by a call to [`run_prefill`](Self::run_prefill).
    ///
    pub fn run_decode(&mut self) -> Result<u32, RuntimeError> {
        let cache = self
            .kv_cache
            .as_ref()
            .ok_or_else(|| RuntimeError::UnsupportedMode("decode requires prefill".into()))?;
        // WAIVER: KV cache slots are written by `run_prefill` /
        // `run_decode` as `Vec<NeBytes<u32>>` (4-byte aligned). The
        // `rchunks_exact(4)` iterator yields infallible 4-byte slices.
        // The `try_into().unwrap()` is structurally guarded by the chunk
        // size. Pre-existing — survived the orchestrator decomposition.
        let last = cache
            .first()
            .and_then(|bytes| bytes.rchunks_exact(4).next())
            .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
            .ok_or_else(|| RuntimeError::ExecutionFailed("decode cache is empty".into()))?;
        let logits = self.dispatch_tokens_via_helper(&[last])?;
        self.mode = super::ExecutionMode::RealtimeDecode;
        if let Some(cache) = self.kv_cache.as_mut() {
            cache[0].extend_from_slice(&argmax_token(&logits).to_ne_bytes());
        }
        Ok(argmax_token(&logits))
    }

    /// Run one canonical realtime decode step and return logits to the caller
    /// that owns sampling.
    pub fn run_decode_logits(&mut self) -> Result<Vec<f32>, RuntimeError> {
        let cache = self
            .kv_cache
            .as_ref()
            .ok_or_else(|| RuntimeError::UnsupportedMode("decode requires prefill".into()))?;
        // WAIVER: same KV-cache alignment invariant as `run_decode` —
        // chunks are 4-byte aligned, the `try_into` is infallible.
        let last = cache
            .first()
            .and_then(|bytes| bytes.rchunks_exact(4).next())
            .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
            .ok_or_else(|| RuntimeError::ExecutionFailed("decode cache is empty".into()))?;
        let logits = self.dispatch_tokens_via_helper(&[last])?;
        self.mode = super::ExecutionMode::RealtimeDecode;
        if let Some(cache) = self.kv_cache.as_mut() {
            cache[0].extend_from_slice(&argmax_token(&logits).to_ne_bytes());
        }
        Ok(logits)
    }

    /// Decode a caller-supplied token while retaining the canonical KV state.
    pub fn run_decode_logits_for_token(&mut self, token: u32) -> Result<Vec<f32>, RuntimeError> {
        {
            let cache = self
                .kv_cache
                .as_mut()
                .ok_or_else(|| RuntimeError::UnsupportedMode("decode requires prefill".into()))?;
            let slot = cache
                .first_mut()
                .ok_or_else(|| RuntimeError::ExecutionFailed("decode cache is empty".into()))?;
            slot.extend_from_slice(&token.to_ne_bytes());
        }
        let logits = self.dispatch_tokens_via_helper(&[token])?;
        self.mode = super::ExecutionMode::RealtimeDecode;
        if let Some(slot) = self.kv_cache.as_mut().and_then(|cache| cache.first_mut()) {
            slot.extend_from_slice(&argmax_token(&logits).to_ne_bytes());
        }
        Ok(logits)
    }

    /// Reset the KV cache without reloading the model.
    ///
    /// After calling this, the runtime is back to a fresh prefill-ready
    /// state. The loaded tensors and kernels remain intact.
    pub fn reset_kv_cache(&mut self) {
        self.kv_cache = None;
        self.mode = super::ExecutionMode::Batch;
        self.requested_batch_size = None;
        self.last_workload_selection = None;
    }

    /// Internal helper that forwards the public run_* methods to the
    /// dispatch layer. This is a one-line wrapper so the run_* methods
    /// don't need to know which submodule owns the dispatch.
    fn dispatch_tokens_via_helper(&mut self, input_tokens: &[u32]) -> Result<Vec<f32>, RuntimeError> {
        dispatch_tokens(self, input_tokens)
    }
}

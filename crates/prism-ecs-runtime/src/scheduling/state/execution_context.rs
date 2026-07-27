//! Execution context (constitutional home).
//!
//! Runtime state bundled for phase execution. The execution context is
//! the record the phase runner consumes; the runtime constructs it
//! and stages the dispatch through `ConstitutionalWorldTxn`.
//!
//! # Authority
//!
//! The execution context is **scheduling state** in the C bucket.
//! Once a dispatch commits, the context is frozen for the duration of
//! the dispatch. The completion-reconciliation system reads the
//! committed context (via the receipt) when reconciling the result.
//!
//! # Placeholder engine types
//!
//! The engine's `ExecutionContext` carries MLX-specific data
//! (`mlx_rs::Array` hidden states, `LayerWeights` for the model, etc.).
//! The constitutional home defines minimal placeholder types matching
//! the engine's wire shape. The MLX-specific types stay in the engine
//! until they migrate with `prism-ecs-compile`.
//!
//! # Migration provenance
//!
//! The legacy home was `compute-core/src/ecs/scheduling/execution_context.rs`.
//! The engine file is the legacy duplicate; step 58 deletes it.

use std::any::Any;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Placeholder engine types
// ---------------------------------------------------------------------------

/// Placeholder for `compute-core::ecs::ane::sink_detector::AneSinkDetector`.
/// Replaced when ANE types migrate.
#[derive(Debug, Clone, Default)]
pub struct AneSinkDetector {
    _placeholder: (),
}

impl AneSinkDetector {
    /// Placeholder for the engine's `check` method.
    pub fn check(&mut self, _attention_weights: &[f32]) -> Result<bool, String> {
        Ok(false)
    }
}

/// Placeholder for `compute-core::ecs::kv_cache::LiveKvCache`.
#[derive(Debug, Clone, Default)]
pub struct LiveKvCache {
    _placeholder: (),
}

/// Placeholder for `compute-core::profiled_model::LayerWeights`.
#[derive(Debug, Clone, Default)]
pub struct LayerWeights {
    _placeholder: (),
}

/// Placeholder for `mlx_rs::Array` (the engine uses an MLX array for
/// the hidden state; the constitutional home treats it as opaque
/// bytes to keep the runtime crate MLX-independent).
#[derive(Debug, Clone, Default)]
pub struct HiddenStateArray {
    _bytes: Vec<u8>,
}

// ---------------------------------------------------------------------------
// ExecutionContext
// ---------------------------------------------------------------------------

/// Runtime state passed to every phase runner.
pub struct ExecutionContext {
    /// Unique request ID being processed.
    pub request_id: u64,
    /// Current token position in decode.
    pub token_position: usize,
    /// ANE-based sink detector for adaptive window sizing.
    pub sink_detector: Option<AneSinkDetector>,
    /// Whether this is a prefill or decode pass.
    pub is_prefill: bool,
    /// Input token IDs for this step (set by caller before dispatch).
    pub token_ids: Vec<i32>,
    /// Current hidden state activation flowing through the DAG.
    /// Populated by the caller before phase dispatch; updated by runners.
    pub hidden_state: Option<HiddenStateArray>,
    /// Per-layer KV caches for the active sequence.
    pub kv_caches: Vec<LiveKvCache>,
    /// Model weights indexed by layer index.
    pub layer_weights: Arc<Vec<LayerWeights>>,
    /// Opaque backend context. Concrete runners downcast this to access
    /// the MLX executor, Metal device, or Core ML state belonging to
    /// the current inference session.
    pub backend: Option<Box<dyn Any + Send>>,
}

impl ExecutionContext {
    /// Create an empty/default context for testing.
    pub fn new_empty() -> Self {
        Self {
            request_id: 0,
            token_position: 0,
            sink_detector: None,
            is_prefill: true,
            token_ids: Vec::new(),
            hidden_state: None,
            kv_caches: Vec::new(),
            layer_weights: Arc::new(Vec::new()),
            backend: None,
        }
    }

    /// Run the ANE sink detector on attention weights from the last layer.
    /// Returns whether the adaptive window should grow, or None if the
    /// detector isn't loaded.
    pub fn detect_sink_grow(&mut self, attention_weights: &[f32]) -> Option<bool> {
        self.sink_detector.as_mut()?.check(attention_weights).ok()
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `execution_context` state.

    use super::*;

    #[test]
    fn new_empty_is_prefill_and_no_state() {
        // Architectural invariant: a freshly constructed context is
        // in prefill mode with no token position, no hidden state,
        // no sink detector, no KV caches, no layer weights, and no
        // backend. The fields are independent — the caller populates
        // them in any order.
        let c = ExecutionContext::new_empty();
        assert_eq!(c.request_id, 0);
        assert_eq!(c.token_position, 0);
        assert!(c.sink_detector.is_none());
        assert!(c.is_prefill);
        assert!(c.token_ids.is_empty());
        assert!(c.hidden_state.is_none());
        assert!(c.kv_caches.is_empty());
        assert!(c.layer_weights.is_empty());
        assert!(c.backend.is_none());
    }

    #[test]
    fn detect_sink_grow_returns_none_without_detector() {
        // Architectural invariant: a context without a sink detector
        // returns None from detect_sink_grow, not a default value.
        let mut c = ExecutionContext::new_empty();
        assert!(c.detect_sink_grow(&[]).is_none());
    }

    #[test]
    fn detect_sink_grow_with_detector_returns_some() {
        // Architectural invariant: a context with a sink detector
        // returns Some(decision), even on empty attention weights.
        let mut c = ExecutionContext::new_empty();
        c.sink_detector = Some(AneSinkDetector::default());
        let result = c.detect_sink_grow(&[]);
        assert!(result.is_some());
    }
}

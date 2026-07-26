//! Attention-sink window — the canonical authority for sink+window attention
//! in the runtime.
//!
//! This module owns the design idea behind the engine's "attention sink
//! reuse" pattern (formerly `compute-core/src/ecs/core/executor.rs::SinkState`)
//! re-implemented in Prism's domain:
//!
//! 1. During **prefill**, the first `num_permanent_sinks` tokens' K/V
//!    projections are captured as permanent "sinks" — they are treated as
//!    always-attendable anchors regardless of the cache size.
//! 2. During **decode**, attention is computed over the union of the
//!    captured sinks and a **sliding window** of the most recent tokens.
//! 3. The window size is **adaptive**: it grows when attention entropy
//!    indicates uncertainty and shrinks when the model is confident,
//!    bounded between `window_size` and `4 * window_size`.
//!
//! The actual K/V storage is supplied by a backend through the
//! [`SinkStore`] trait; this module is backend-neutral. The MLX-specific
//! sink K/V tensors that previously lived in `SinkState { sink_k, sink_v }`
//! are now opaque `SinkHandle` values the backend allocates and retrieves.
//!
//! # Position semantics
//!
//! Positions are 0-indexed and refer to the position of a token in the
//! running sequence. The first prefill token has position 0; the first
//! decode step writes to position `prefill_len`. Sinks occupy the
//! contiguous range `[0, num_permanent_sinks)`. The sliding window occupies
//! the contiguous range `[window_start, cached_seq)`, with
//! `window_start >= num_permanent_sinks`.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Identifier for a captured sink batch. Backend-allocated; opaque to the
/// runtime. The runtime treats the id as a value to be passed back to the
/// backend when computing attention; it does not interpret its contents.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SinkHandle(pub String);

/// Opaque per-layer storage the backend uses to materialise captured
/// sink K/V on demand. The runtime holds [`SinkHandle`] ids; the backend
/// owns the storage.
pub trait SinkStore: Send + Sync {
    /// Drop all sink state for the given handle. Called when the
    /// corresponding request terminates or the layer is preempted.
    fn release(&self, handle: &SinkHandle) -> Result<(), SinkError>;
}

/// Errors raised by the sink layer. Categorised per the constitutional
/// pattern: `Rejected` for preflight failures, `Failed` for effect failures.
#[derive(Debug, Error)]
pub enum SinkError {
    /// The caller asked to use a window that has not been initialised.
    /// (Preflight failure — caught before the attention effect runs.)
    #[error("sink window not initialised: {0}")]
    Rejected(&'static str),

    /// The backend failed to materialise or release sink storage.
    /// (Effect failure — backend reported an error.)
    #[error("sink store error: {0}")]
    Failed(String),
}

/// Configuration for a sink window. Constructed by the schedule when a
/// layer is set up; the runtime uses it to compute attention ranges
/// without holding any per-layer K/V directly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SinkWindowConfig {
    /// Number of initial tokens treated as permanent sinks.
    pub num_permanent_sinks: u32,
    /// Base size of the sliding window alongside the sinks.
    pub window_size: u32,
    /// Maximum multiplier on `window_size` for the adaptive window.
    /// Bounded by 4x in the runtime; backends may cap lower.
    pub max_window_multiplier: u32,
}

impl SinkWindowConfig {
    /// Construct a sink window configuration with a 4x adaptive cap.
    pub fn new(num_permanent_sinks: u32, window_size: u32) -> Self {
        Self {
            num_permanent_sinks,
            window_size,
            max_window_multiplier: 4,
        }
    }

    /// Hard upper bound on the adaptive window — the attention layer must
    /// never request a window larger than this.
    pub fn max_window(&self) -> u32 {
        self.window_size.saturating_mul(self.max_window_multiplier)
    }
}

impl Default for SinkWindowConfig {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

/// The contiguous range of cache positions the attention layer should
/// attend to. `sinks` is always `[0, num_permanent_sinks)`; the window is
/// `[window.0, window.1)`. A degenerate window has `window.0 == window.1`
/// (zero length) — the attention layer must handle that case without
/// reading from the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionRange {
    /// Number of sink positions (positions `[0, sinks)` are always
    /// included in attention).
    pub sinks: u32,
    /// Half-open range `[start, end)` of cache positions the attention
    /// layer attends to alongside the sinks.
    pub window: (u32, u32),
}

impl AttentionRange {
    /// Number of positions the attention layer reads from the cache.
    /// The total attended length is `sinks + window.1 - window.0`.
    pub fn window_len(&self) -> u32 {
        self.window.1.saturating_sub(self.window.0)
    }

    pub fn total_len(&self) -> u32 {
        self.sinks + self.window_len()
    }
}

/// The state of a sink window across a request's lifetime.
///
/// The runtime constructs one [`SinkWindow`] per (layer, request) pair. The
/// window is initialised empty; prefill populates the captured sinks;
/// decode advances the sliding window. The adaptive window is updated
/// after every decode step based on attention entropy.
#[derive(Debug, Clone)]
pub struct SinkWindow {
    config: SinkWindowConfig,
    /// Whether [`capture`](Self::capture) has been called successfully at
    /// least once. The window is unusable until then.
    populated: bool,
    /// Most recently observed attention entropy, in nats. Used to drive
    /// adaptive window growth/shrink.
    last_entropy: f32,
    /// Current adaptive window size, in positions. Bounded by
    /// `[config.window_size, config.max_window()]`.
    adaptive_window: u32,
}

impl SinkWindow {
    /// Construct a new, empty sink window. The window is not yet usable;
    /// call [`capture`](Self::capture) after prefill completes.
    pub fn new(config: SinkWindowConfig) -> Self {
        let adaptive_window = config.window_size;
        Self {
            config,
            populated: false,
            last_entropy: 0.0,
            adaptive_window,
        }
    }

    pub fn config(&self) -> &SinkWindowConfig {
        &self.config
    }

    pub fn is_populated(&self) -> bool {
        self.populated
    }

    pub fn last_entropy(&self) -> f32 {
        self.last_entropy
    }

    pub fn adaptive_window(&self) -> u32 {
        self.adaptive_window
    }

    /// Mark the window as populated. The runtime calls this after the
    /// prefill pass has finished and the captured sinks are durably
    /// stored by the backend.
    pub fn mark_populated(&mut self) {
        self.populated = true;
    }

    /// Reset the window to its initial state. The next decode step sees
    /// a freshly-initialised adaptive window.
    pub fn reset(&mut self) {
        self.populated = false;
        self.last_entropy = 0.0;
        self.adaptive_window = self.config.window_size;
    }

    /// Compute the [`AttentionRange`] for the current state, given the
    /// total number of cached positions after the latest append.
    ///
    /// Returns `SinkError::Rejected` if the window has not been
    /// initialised by prefill. The returned range is a pure function of
    /// `(num_permanent_sinks, cached_seq, adaptive_window)`; the runtime
    /// applies it to whatever KV store the backend exposes.
    pub fn attention_range(&self, cached_seq: u32) -> Result<AttentionRange, SinkError> {
        if !self.populated {
            return Err(SinkError::Rejected(
                "sink window used before prefill capture",
            ));
        }
        if cached_seq == 0 {
            // No cache positions exist yet; only the (empty) sink range
            // is attendable. The attention layer must treat this as a
            // no-op.
            return Ok(AttentionRange {
                sinks: self.config.num_permanent_sinks,
                window: (0, 0),
            });
        }

        let sink_end = self.config.num_permanent_sinks;
        let cached = cached_seq;
        let window_start = sink_end.max(cached.saturating_sub(self.adaptive_window));
        // Defensive: window_start must never exceed cached_seq, even
        // when the adaptive window is 0.
        let window_start = window_start.min(cached);
        let window_end = cached;

        Ok(AttentionRange {
            sinks: self.config.num_permanent_sinks,
            window: (window_start, window_end),
        })
    }

    /// Update the adaptive window from observed attention entropy.
    ///
    /// The engine's heuristic:
    /// - `entropy > ln(adaptive_window) * 0.8`: uncertainty is high, grow
    ///   the window by 1.5x (capped at `max_window`).
    /// - `entropy < ln(adaptive_window) * 0.3 * 0.8`: confidence is high,
    ///   shrink the window by 2/3 (floored at `window_size`).
    /// - Otherwise leave the window unchanged.
    ///
    /// `entropy` is in nats (natural log). The implementation uses the
    /// natural log to match the engine's prior behaviour; backends that
    /// report entropy in bits must convert before calling.
    pub fn update_adaptive_window(&mut self, entropy: f32) {
        self.last_entropy = entropy;
        let base = (self.adaptive_window as f32).ln().max(1.0);
        let grow_threshold = base * 0.8;
        let shrink_threshold = base * 0.8 * 0.3;

        if entropy > grow_threshold && self.adaptive_window < self.config.max_window() {
            // Grow by 1.5x, bounded by the hard cap.
            let grown = self.adaptive_window.saturating_mul(3) / 2;
            self.adaptive_window = grown.min(self.config.max_window());
        } else if entropy < shrink_threshold && self.adaptive_window > self.config.window_size
        {
            // Shrink by 2/3, floored at the base window size.
            let shrunk = (self.adaptive_window.saturating_mul(2) / 3)
                .max(self.config.window_size);
            self.adaptive_window = shrunk;
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(sinks: u32, window: u32) -> SinkWindowConfig {
        SinkWindowConfig::new(sinks, window)
    }

    #[test]
    fn empty_window_rejects_attention_range() {
        let w = SinkWindow::new(cfg(4, 16));
        // Not populated → rejected.
        let r = w.attention_range(64);
        assert!(matches!(r, Err(SinkError::Rejected(_))));
    }

    #[test]
    fn attention_range_window_does_not_overlap_sinks() {
        let mut w = SinkWindow::new(cfg(4, 16));
        w.mark_populated();
        let r = w.attention_range(64).expect("populated");
        // Sinks occupy [0, 4); the window is [64 - 16, 64) = [48, 64).
        assert_eq!(r.sinks, 4);
        assert_eq!(r.window, (48, 64));
        // Sanity: window does not overlap the sink range.
        assert!(r.window.0 >= r.sinks);
    }

    #[test]
    fn attention_range_when_cache_smaller_than_window() {
        // The cache is shorter than the configured window — window clamps
        // to the available cache, and never starts before the sinks end.
        let mut w = SinkWindow::new(cfg(2, 64));
        w.mark_populated();
        let r = w.attention_range(8).expect("populated");
        // sink_end = 2; cached = 8; window_start = max(2, 8 - 64) = 2; end = 8.
        assert_eq!(r.sinks, 2);
        assert_eq!(r.window, (2, 8));
    }

    #[test]
    fn attention_range_when_cache_empty() {
        // Defensive: cached_seq == 0 — only sinks are attendable, window
        // is degenerate.
        let mut w = SinkWindow::new(cfg(4, 16));
        w.mark_populated();
        let r = w.attention_range(0).expect("populated");
        assert_eq!(r.sinks, 4);
        assert_eq!(r.window, (0, 0));
        assert_eq!(r.window_len(), 0);
    }

    #[test]
    fn adaptive_window_grows_under_high_entropy() {
        let mut w = SinkWindow::new(cfg(2, 8));
        w.mark_populated();
        // ln(8) ≈ 2.08; threshold = 2.08 * 0.8 ≈ 1.66. Pick entropy = 5.
        w.update_adaptive_window(5.0);
        // Grew by 1.5x: 8 * 3 / 2 = 12.
        assert_eq!(w.adaptive_window(), 12);
    }

    #[test]
    fn adaptive_window_capped_at_max_window() {
        let mut w = SinkWindow::new(cfg(2, 8));
        w.mark_populated();
        // Repeated high-entropy updates should saturate at max_window = 32.
        for _ in 0..20 {
            w.update_adaptive_window(10.0);
        }
        assert_eq!(w.adaptive_window(), 32);
    }

    #[test]
    fn adaptive_window_shrinks_under_low_entropy() {
        let mut w = SinkWindow::new(cfg(2, 8));
        w.mark_populated();
        // Grow first.
        w.update_adaptive_window(5.0);
        assert_eq!(w.adaptive_window(), 12);
        // ln(12) ≈ 2.48; threshold = 2.48 * 0.24 ≈ 0.60. entropy = 0.0 < 0.60.
        w.update_adaptive_window(0.0);
        // Shrink by 2/3: 12 * 2 / 3 = 8 (back to base).
        assert_eq!(w.adaptive_window(), 8);
    }

    #[test]
    fn adaptive_window_floored_at_base_window() {
        let mut w = SinkWindow::new(cfg(2, 8));
        w.mark_populated();
        // Already at base. Low entropy should not drop below base.
        w.update_adaptive_window(0.0);
        assert_eq!(w.adaptive_window(), 8);
    }

    #[test]
    fn reset_clears_populated_flag_and_adaptive_window() {
        let mut w = SinkWindow::new(cfg(2, 8));
        w.mark_populated();
        w.update_adaptive_window(5.0);
        assert!(w.is_populated());
        assert!(w.adaptive_window() > 8);

        w.reset();
        assert!(!w.is_populated());
        assert_eq!(w.adaptive_window(), 8);
        assert_eq!(w.last_entropy(), 0.0);
    }

    #[test]
    fn attention_range_after_window_growth() {
        let mut w = SinkWindow::new(cfg(2, 8));
        w.mark_populated();
        // After grow, adaptive_window = 12; window = [64 - 12, 64).
        w.update_adaptive_window(5.0);
        let r = w.attention_range(64).expect("populated");
        assert_eq!(r.window, (52, 64));
    }

    #[test]
    fn max_window_is_four_times_base() {
        let c = SinkWindowConfig::new(2, 16);
        assert_eq!(c.max_window(), 64);
    }

    #[test]
    fn sink_handle_equality_is_string_based() {
        let a = SinkHandle("sink-layer-3".to_string());
        let b = SinkHandle("sink-layer-3".to_string());
        let c = SinkHandle("sink-layer-4".to_string());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn attention_range_total_len_sums_sinks_and_window() {
        let mut w = SinkWindow::new(cfg(4, 16));
        w.mark_populated();
        let r = w.attention_range(64).expect("populated");
        // 4 sinks + (64 - 48) = 4 + 16 = 20.
        assert_eq!(r.total_len(), 20);
    }
}

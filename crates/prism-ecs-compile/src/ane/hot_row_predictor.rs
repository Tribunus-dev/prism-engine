//! Hot-row predictor — config and statistics for the ANE LM-head
//! weight-prefetch path.
//!
//! Authority: the canonical hot-row predictor config + outcome
//! statistics surface.
//!
//! The actual ANE-backed Core ML model loading and IOSurface
//! zero-copy inference are engine-coupled. The engine's
//! `legacy_ane/hot_row_predictor.rs` wraps a `HotRowPredictor`
//! from this surface with a Core ML backend; the constitutional
//! surface provides the backend-neutral config + statistics, and
//! records outcomes without touching the ANE.
//!
//! # Workflow
//!
//! 1. Caller loads an `AnePredictorBackend` (engine-coupled, lives in
//!    `legacy_ane/`) that owns the Core ML model.
//! 2. Caller calls `HotRowPredictor::new(config, backend)` to construct
//!    a predictor with the configuration.
//! 3. On each decode step, caller calls `predict(hidden_state)` — the
//!    predictor delegates to the backend and caches the result in
//!    `last_prediction`.
//! 4. After the actual next token is sampled, caller calls
//!    `record_outcome(token)` to update the hit-rate statistics.

use crate::ane::AneError;

/// Backend-agnostic config for [`HotRowPredictor`].
///
/// `hidden_size` is the model's hidden-state dimension (e.g. 3840 for
/// Llama-3 8B). `num_candidates` is the number of candidate token
/// IDs the predictor returns per call (e.g. 64 — enough to cover the
/// typical 5-10 token speculative batch without false drops).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HotRowPredictorConfig {
    /// Hidden state dimension (e.g. 3840).
    pub hidden_size: u32,
    /// Number of candidate token IDs to predict (e.g. 64).
    pub num_candidates: u32,
}

impl HotRowPredictorConfig {
    /// Create a new config.
    pub fn new(hidden_size: u32, num_candidates: u32) -> Self {
        Self {
            hidden_size,
            num_candidates,
        }
    }
}

/// Hit-rate statistics for the hot-row predictor.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HotRowPredictorStats {
    /// Number of predictions made (length of `total_predictions`).
    pub total_predictions: u64,
    /// Number of predictions where the actual next token was in the
    /// predicted candidate set.
    pub hits: u64,
    /// Hit rate in `[0.0, 1.0]`. Returns 0.0 when no predictions have
    /// been recorded.
    pub prediction_hit_rate: f64,
}

impl HotRowPredictorStats {
    /// Update the cached hit rate from `hits` and `total_predictions`.
    fn update(&mut self) {
        let total = self.total_predictions.max(1);
        self.prediction_hit_rate = self.hits as f64 / total as f64;
    }
}

/// Backend trait that performs the actual ANE inference for the
/// hot-row predictor.
///
/// The engine's `legacy_ane/` provides a `CoreMLHotRowPredictorBackend`
/// that owns a `CoreAiModel` loaded with `CpuAndNeuralEngine` compute
/// units. Other backends (e.g. a CPU-only simulator for tests) can
/// implement this trait and produce identical candidate token lists.
pub trait HotRowPredictorBackend {
    /// Run the predictor on `hidden_state` and return up to
    /// `num_candidates` candidate token IDs, sorted by confidence
    /// descending.
    fn predict(
        &self,
        hidden_state: &[f32],
        num_candidates: u32,
    ) -> Result<Vec<u32>, AneError>;
}

/// Backend-neutral ANE hot-row predictor.
///
/// Owns the candidate-set statistics and the last prediction, but
/// delegates the actual Core ML inference to a [`HotRowPredictorBackend`]
/// supplied at construction time.
pub struct HotRowPredictor {
    /// Public config (read-only after construction).
    pub config: HotRowPredictorConfig,
    /// Backend that performs the actual ANE inference.
    backend: Box<dyn HotRowPredictorBackend>,
    /// Previous predictions, recorded for debug/statistics.
    pub last_prediction: Vec<u32>,
    /// Running statistics.
    pub stats: HotRowPredictorStats,
}

impl HotRowPredictor {
    /// Construct a new predictor with a backend.
    ///
    /// The backend must be a `Box<dyn HotRowPredictorBackend>` — engine
    /// callers use the engine's `CoreMLHotRowPredictorBackend`; tests
    /// can use a CPU-only simulator.
    pub fn new(
        config: HotRowPredictorConfig,
        backend: Box<dyn HotRowPredictorBackend>,
    ) -> Self {
        Self {
            config,
            backend,
            last_prediction: Vec::new(),
            stats: HotRowPredictorStats::default(),
        }
    }

    /// Run the predictor on `hidden_state` and return candidate token IDs.
    ///
    /// Updates `last_prediction` and increments `total_predictions` on
    /// success. The caller is responsible for calling [`Self::record_outcome`]
    /// after the actual next token is sampled so the hit-rate statistic
    /// reflects reality.
    pub fn predict(&mut self, hidden_state: &[f32]) -> Result<Vec<u32>, AneError> {
        if hidden_state.len() > self.config.hidden_size as usize {
            return Err(AneError::PreflightRejected {
                reason: "hidden state length exceeds configured hidden_size",
            });
        }
        let candidates = self.backend.predict(hidden_state, self.config.num_candidates)?;
        self.last_prediction = candidates.clone();
        self.stats.total_predictions = self.stats.total_predictions.saturating_add(1);
        Ok(candidates)
    }

    /// Record whether the sampled next token was in the predicted set.
    pub fn record_outcome(&mut self, actual_token: u32) {
        if self.last_prediction.contains(&actual_token) {
            self.stats.hits = self.stats.hits.saturating_add(1);
        }
        self.stats.update();
    }

    /// Report hit-rate statistics as a human-readable string.
    pub fn report_hit_rate(&self) -> String {
        let pct = if self.stats.total_predictions > 0 {
            self.stats.prediction_hit_rate * 100.0
        } else {
            0.0
        };
        format!(
            "HotRowPredictor: {}/{} hits ({:.1}%)",
            self.stats.hits, self.stats.total_predictions, pct,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Test backend that returns a fixed candidate set.
    struct FixedBackend {
        candidates: Vec<u32>,
        call_count: Mutex<u32>,
    }

    impl HotRowPredictorBackend for FixedBackend {
        fn predict(
            &self,
            _hidden_state: &[f32],
            num_candidates: u32,
        ) -> Result<Vec<u32>, AneError> {
            *self.call_count.lock().unwrap() += 1;
            Ok(self.candidates[..num_candidates as usize].to_vec())
        }
    }

    #[test]
    fn predict_increments_total() {
        let config = HotRowPredictorConfig::new(4, 2);
        let backend = Box::new(FixedBackend {
            candidates: vec![10, 20, 30],
            call_count: Mutex::new(0),
        });
        let mut predictor = HotRowPredictor::new(config, backend);
        let result = predictor.predict(&[0.1, 0.2, 0.3, 0.4]).unwrap();
        assert_eq!(result, vec![10, 20]);
        assert_eq!(predictor.stats.total_predictions, 1);
        assert_eq!(predictor.last_prediction, vec![10, 20]);
    }

    #[test]
    fn record_outcome_updates_hit_rate() {
        let config = HotRowPredictorConfig::new(4, 2);
        let backend = Box::new(FixedBackend {
            candidates: vec![10, 20, 30],
            call_count: Mutex::new(0),
        });
        let mut predictor = HotRowPredictor::new(config, backend);
        predictor.predict(&[0.1, 0.2, 0.3, 0.4]).unwrap();
        predictor.record_outcome(20); // hit
        assert_eq!(predictor.stats.hits, 1);
        assert!((predictor.stats.prediction_hit_rate - 1.0).abs() < 1e-9);

        predictor.predict(&[0.1, 0.2, 0.3, 0.4]).unwrap();
        predictor.record_outcome(99); // miss
        assert_eq!(predictor.stats.hits, 1);
        assert_eq!(predictor.stats.total_predictions, 2);
        assert!((predictor.stats.prediction_hit_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn predict_rejects_oversized_input() {
        let config = HotRowPredictorConfig::new(2, 2);
        let backend = Box::new(FixedBackend {
            candidates: vec![10, 20, 30],
            call_count: Mutex::new(0),
        });
        let mut predictor = HotRowPredictor::new(config, backend);
        let result = predictor.predict(&[0.1, 0.2, 0.3, 0.4]);
        assert!(matches!(
            result,
            Err(AneError::PreflightRejected { .. })
        ));
    }
}

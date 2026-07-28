//! ANE attention-sink detector — config + CPU entropy heuristic.
//!
//! Authority: the canonical attention-sink detector config, the
//! backend trait, and the CPU entropy heuristic.
//!
//! The actual ANE-backed Core ML model loading and IOSurface
//! zero-copy inference are engine-coupled. The engine's
//! `legacy_ane/sink_detector.rs` wraps an `AneSinkDetector` from
//! this surface with a `CoreMLSinkDetectorBackend`; the
//! constitutional surface provides the config + the backend trait
//! + the CPU entropy heuristic used when no backend is available.
//!
//! # Behaviour
//!
//! The detector monitors attention-weight entropy and predicts
//! whether the adaptive window needs to grow. The ANE path runs
//! a tiny MLP that maps the last layer's attention weight
//! distribution `[1, seq_len]` to a scalar in `[0, 1]` where
//! `> 0.5` means "window should grow". The CPU fallback uses a
/// normalised entropy heuristic with a `0.8` threshold.

use crate::ane::AneError;

/// Backend-agnostic config for [`AneSinkDetector`].
///
/// `max_seq_len` is the maximum sequence length the detector
/// supports (e.g. 4096 for a long-context model). The backend may
/// pad shorter inputs up to this length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AneSinkDetectorConfig {
    /// Maximum sequence length the detector supports.
    pub max_seq_len: u32,
}

impl AneSinkDetectorConfig {
    /// Create a new config.
    pub fn new(max_seq_len: u32) -> Self {
        Self { max_seq_len }
    }
}

/// Backend trait that performs the actual ANE inference for the
/// attention-sink detector.
///
/// The engine's `legacy_ane/sink_detector.rs` provides a
/// `CoreMLSinkDetectorBackend`; tests can use a CPU simulator or
/// `None` (which forces the CPU entropy heuristic).
pub trait SinkDetectorBackend {
    /// Run the detector on `attention_weights` and return the
    /// probability that the window should grow, in `[0.0, 1.0]`.
    fn predict(&self, attention_weights: &[f32]) -> Result<f32, AneError>;
}

/// ANE attention-sink detector.
///
/// Owns the detector config, the optional backend, and the
/// running statistics. The actual Core ML inference delegates to
/// the backend; when no backend is set, the CPU entropy heuristic
/// runs instead.
pub struct AneSinkDetector {
    /// Public config (read-only after construction).
    pub config: AneSinkDetectorConfig,
    /// Backend that performs the actual ANE inference.
    backend: Option<Box<dyn SinkDetectorBackend>>,
    /// Number of predictions made.
    pub total_predictions: u64,
    /// Number of times the detector recommended growing the window.
    pub grow_recommendations: u64,
}

impl AneSinkDetector {
    /// Construct a new sink detector with an optional backend.
    ///
    /// When `backend` is `Some`, the ANE path is used. When it is
    /// `None`, the CPU entropy heuristic is used as a fallback.
    pub fn new(
        config: AneSinkDetectorConfig,
        backend: Option<Box<dyn SinkDetectorBackend>>,
    ) -> Self {
        Self {
            config,
            backend,
            total_predictions: 0,
            grow_recommendations: 0,
        }
    }

    /// Check if the adaptive window should grow based on `attention_weights`.
    ///
    /// `attention_weights` is a flat slice of `f32` softmax
    /// probabilities (one head's distribution over the KV
    /// sequence). Returns `true` if the window should grow
    /// (high uncertainty / high entropy).
    pub fn check(&mut self, attention_weights: &[f32]) -> Result<bool, AneError> {
        self.total_predictions = self.total_predictions.saturating_add(1);

        let should_grow = if let Some(backend) = &self.backend {
            let grow_prob = backend.predict(attention_weights)?;
            grow_prob > 0.5
        } else {
            cpu_entropy_should_grow(attention_weights)
        };

        if should_grow {
            self.grow_recommendations =
                self.grow_recommendations.saturating_add(1);
        }
        Ok(should_grow)
    }

    /// Report detection statistics as a human-readable string.
    pub fn report_stats(&self) -> String {
        let pct = if self.total_predictions > 0 {
            (self.grow_recommendations as f64 / self.total_predictions as f64) * 100.0
        } else {
            0.0
        };
        format!(
            "AneSinkDetector: {}/{} grow recommendations ({:.1}%)",
            self.grow_recommendations, self.total_predictions, pct,
        )
    }
}

/// CPU-side entropy heuristic: compare distribution entropy to a
/// threshold. Returns `true` when the distribution is close to
/// uniform (normalised entropy > 0.8).
pub fn cpu_entropy_should_grow(attention_weights: &[f32]) -> bool {
    let n = attention_weights.len();
    if n < 2 {
        return false;
    }

    let mut entropy = 0.0f32;
    for &p in attention_weights {
        if p > 0.0 {
            entropy -= p * p.log(std::f32::consts::E);
        }
    }

    // Normalize by max possible entropy (uniform distribution).
    let max_entropy = (n as f32).ln();
    if max_entropy <= 0.0 {
        return false;
    }
    let normalized = entropy / max_entropy;

    // Threshold: >0.8 means highly uncertain (near-uniform) → grow window.
    normalized > 0.8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_heuristic_short_input_returns_false() {
        assert!(!cpu_entropy_should_grow(&[]));
        assert!(!cpu_entropy_should_grow(&[0.5]));
    }

    #[test]
    fn cpu_heuristic_uniform_grows() {
        let n = 100;
        let uniform: Vec<f32> = vec![1.0 / n as f32; n];
        assert!(cpu_entropy_should_grow(&uniform));
    }

    #[test]
    fn cpu_heuristic_peaked_does_not_grow() {
        let mut peaked = vec![0.0f32; 100];
        peaked[0] = 0.99;
        peaked[1] = 0.01;
        assert!(!cpu_entropy_should_grow(&peaked));
    }

    #[test]
    fn detector_uses_cpu_when_no_backend() {
        let config = AneSinkDetectorConfig::new(64);
        let mut detector = AneSinkDetector::new(config, None);
        let uniform: Vec<f32> = vec![0.01; 100];
        let result = detector.check(&uniform).unwrap();
        assert!(result);
        assert_eq!(detector.total_predictions, 1);
        assert_eq!(detector.grow_recommendations, 1);
    }

    #[test]
    fn detector_uses_backend_when_provided() {
        use std::sync::Mutex;
        struct FixedBackend(Mutex<f32>);
        impl SinkDetectorBackend for FixedBackend {
            fn predict(&self, _weights: &[f32]) -> Result<f32, AneError> {
                Ok(*self.0.lock().unwrap())
            }
        }
        let config = AneSinkDetectorConfig::new(64);
        let backend: Box<dyn SinkDetectorBackend> =
            Box::new(FixedBackend(Mutex::new(0.7)));
        let mut detector = AneSinkDetector::new(config, Some(backend));
        let result = detector.check(&[0.1, 0.2, 0.3]).unwrap();
        assert!(result, "0.7 > 0.5 should trigger grow");
        assert_eq!(detector.grow_recommendations, 1);
    }
}

//! ANE-backed draft model — config and backend contract for speculative
//! decoding.
//!
//! Authority: the canonical draft-model config + backend contract for
//! the ANE-backed Core ML speculative-decoding path.
//!
//! The actual Core ML model loading, IOSurface zero-copy inference,
//! and autoregressive loop are engine-coupled. The engine's
//! `legacy_ane/draft_model.rs` provides a `CoreMLDraftBackend` that
//! implements [`DraftBackend`] and wraps an IOSurface-backed `Arena`;
//! the constitutional surface provides the config + the backend trait.
//!
//! # Architecture
//!
//! ```text
//!  CPU (tokenize + sample)          ANE (transformer)
//!        │                               │
//!        ├── prefix tokens ──► IOSurface ──► Core ML model ──► IOSurface ──► logits
//!        │                                   (CpuAndNeuralEngine)
//!        └────────── softmax + argmax ◄──────┘
//! ```

use crate::ane::sampling::greedy_argmax;
use crate::ane::AneError;

/// Backend-agnostic config for [`AneDraftModel`].
///
/// `vocab_size` is the vocabulary size (e.g. 32000 for Llama-3).
/// `seq_len` is the maximum prefix + draft tokens per forward pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AneDraftModelConfig {
    /// Vocabulary size (number of logits per output position).
    pub vocab_size: u32,
    /// Maximum sequence length (prefix + output tokens) per forward pass.
    pub seq_len: u32,
}

impl AneDraftModelConfig {
    /// Construct a new config.
    pub fn new(vocab_size: u32, seq_len: u32) -> Self {
        Self {
            vocab_size,
            seq_len,
        }
    }
}

/// Result of one ANE forward pass: the raw logits for the last
/// position of the input sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftForwardOutput {
    /// Logits for the predicted next token, length `vocab_size`.
    pub logits: Vec<f32>,
}

/// Backend trait that performs the actual ANE inference for the
/// draft model.
///
/// The engine's `legacy_ane/draft_model.rs` provides a
/// `CoreMLDraftBackend` that owns a `CoreAiModel` and an
/// `Arena` pair. Tests can use a CPU-only simulator.
pub trait DraftBackend {
    /// Run a single forward pass on `tokens` and return the logits
    /// for the last input position.
    fn forward(&self, tokens: &[u32]) -> Result<DraftForwardOutput, AneError>;
}

/// Errors raised by ANE backends.
///
/// Categorised per the constitutional pattern: `PreflightRejected` for
/// input validation, `EffectFailed` for backend-level failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DraftBackendError {
    /// The backend rejected the input during preflight.
    PreflightRejected {
        /// Static reason for the rejection.
        reason: &'static str,
    },
    /// The backend effect (forward pass, autoregressive loop) failed.
    EffectFailed(String),
}

impl std::fmt::Display for DraftBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PreflightRejected { reason } => write!(f, "preflight rejected: {reason}"),
            Self::EffectFailed(s) => write!(f, "effect failed: {s}"),
        }
    }
}

impl std::error::Error for DraftBackendError {}

/// Backend-neutral ANE draft model.
///
/// Owns the prefix accumulated across `speculate` calls and the
/// sampled token statistics. The actual autoregressive loop lives
/// in this surface (it operates on the backend's logits) but the
/// per-step Core ML forward pass delegates to the backend.
pub struct AneDraftModel {
    /// Public config (read-only after construction).
    pub config: AneDraftModelConfig,
    /// Backend that performs the actual Core ML forward pass.
    backend: Box<dyn DraftBackend>,
    /// Accumulated prefix tokens for KV-cache continuity across calls.
    prefix: Vec<u32>,
}

impl AneDraftModel {
    /// Construct a new draft model with the given config and backend.
    pub fn new(
        config: AneDraftModelConfig,
        backend: Box<dyn DraftBackend>,
    ) -> Self {
        Self {
            config,
            backend,
            prefix: Vec::new(),
        }
    }

    /// Read-only access to the accumulated prefix.
    pub fn prefix(&self) -> &[u32] {
        &self.prefix
    }

    /// Run a single forward pass on `tokens` and return the logits
    /// for the last position.
    pub fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>, AneError> {
        if tokens.is_empty() {
            return Err(AneError::PreflightRejected {
                reason: "AneDraftModel::forward: empty token sequence",
            });
        }
        if tokens.len() > self.config.seq_len as usize {
            return Err(AneError::PreflightRejected {
                reason: "AneDraftModel::forward: token count exceeds seq_len",
            });
        }
        let out = self.backend.forward(tokens)?;
        Ok(out.logits)
    }

    /// Generate `n_tokens` speculative tokens given a `prefix`.
    ///
    /// Returns `(token_ids, log_probabilities)` where each
    /// log-probability is the natural log of the softmax probability
    /// assigned to the sampled token by the draft model at its
    /// position.
    pub fn speculate(
        &mut self,
        prefix: &[u32],
        n_tokens: usize,
    ) -> Result<(Vec<u32>, Vec<f32>), AneError> {
        if n_tokens == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if n_tokens > self.config.seq_len as usize {
            return Err(AneError::PreflightRejected {
                reason: "AneDraftModel: n_tokens exceeds seq_len",
            });
        }
        let total_prefix_len = self.prefix.len() + prefix.len();
        if total_prefix_len == 0 {
            return Err(AneError::PreflightRejected {
                reason: "AneDraftModel: empty prefix",
            });
        }
        if total_prefix_len + n_tokens - 1 > self.config.seq_len as usize {
            return Err(AneError::PreflightRejected {
                reason: "AneDraftModel: total length (prefix + n_tokens) exceeds seq_len",
            });
        }

        let mut input: Vec<u32> =
            Vec::with_capacity(total_prefix_len + n_tokens);
        input.extend_from_slice(&self.prefix);
        input.extend_from_slice(prefix);

        let mut tokens = Vec::with_capacity(n_tokens);
        let mut log_probs = Vec::with_capacity(n_tokens);

        for _ in 0..n_tokens {
            let logits = self.backend.forward(&input)?.logits;
            let token = greedy_argmax(&logits);

            // Compute log-probability with numerically-stable softmax.
            let max_logit =
                logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
            let token_idx = token as usize;
            let prob = if token_idx < logits.len() && sum > 0.0 {
                (logits[token_idx] - max_logit).exp() / sum
            } else {
                0.0
            };
            let log_prob = if prob > 0.0 {
                prob.ln()
            } else {
                f32::NEG_INFINITY
            };

            tokens.push(token);
            log_probs.push(log_prob);

            // Append the sampled token for the next autoregressive step.
            input.push(token);
        }

        // Save the consumed prefix so the caller's next `speculate()`
        // call can continue where we left off.
        self.prefix.extend_from_slice(prefix);

        Ok((tokens, log_probs))
    }

    /// Reset the internal prefix buffer.
    pub fn reset(&mut self) {
        self.prefix.clear();
    }
}

/// Multi-core ANE draft orchestrator.
///
/// Runs N copies of an [`AneDraftModel`] (one per ANE core) in
/// parallel, each producing a different speculative continuation.
/// The M1–M4 Apple Neural Engine has 16 cores; M3 Ultra has 32.
pub struct AneMultiCoreDraft {
    /// One draft model per ANE core.
    drafts: Vec<AneDraftModel>,
}

impl AneMultiCoreDraft {
    /// Create a new multi-core draft with 16 copies of the model.
    ///
    /// The same [`AneDraftModelConfig`] is used for all 16; the
    /// caller supplies a factory closure that produces one backend
    /// per core (engine callers use a `CoreMLDraftBackend`; tests
    /// can use CPU simulators).
    pub fn new<F>(
        config: AneDraftModelConfig,
        n: usize,
        mut backend_factory: F,
    ) -> Result<Self, AneError>
    where
        F: FnMut() -> Result<Box<dyn DraftBackend>, AneError>,
    {
        let mut drafts = Vec::with_capacity(n);
        for _ in 0..n {
            let backend = backend_factory()?;
            drafts.push(AneDraftModel::new(config.clone(), backend));
        }
        Ok(Self { drafts })
    }

    /// Read-only access to the underlying drafts.
    pub fn drafts(&self) -> &[AneDraftModel] {
        &self.drafts
    }

    /// Number of drafts (one per ANE core).
    pub fn num_drafts(&self) -> usize {
        self.drafts.len()
    }

    /// Reset all drafts for a new generation sequence.
    pub fn reset_all(&mut self) {
        for draft in &mut self.drafts {
            draft.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Test backend that returns a fixed logits vector.
    struct FixedLogitsBackend {
        logits: Vec<f32>,
    }

    impl DraftBackend for FixedLogitsBackend {
        fn forward(&self, _tokens: &[u32]) -> Result<DraftForwardOutput, AneError> {
            Ok(DraftForwardOutput {
                logits: self.logits.clone(),
            })
        }
    }

    #[test]
    fn forward_returns_logits() {
        let config = AneDraftModelConfig::new(4, 4);
        let backend = Box::new(FixedLogitsBackend {
            logits: vec![1.0, 2.0, 5.0, 0.5],
        });
        let model = AneDraftModel::new(config, backend);
        let logits = model.forward(&[1, 2, 3]).unwrap();
        assert_eq!(logits, vec![1.0, 2.0, 5.0, 0.5]);
    }

    #[test]
    fn forward_rejects_empty_input() {
        let config = AneDraftModelConfig::new(4, 4);
        let backend = Box::new(FixedLogitsBackend {
            logits: vec![1.0, 2.0, 5.0, 0.5],
        });
        let model = AneDraftModel::new(config, backend);
        let result = model.forward(&[]);
        assert!(matches!(
            result,
            Err(AneError::PreflightRejected { .. })
        ));
    }

    #[test]
    fn speculate_returns_greedy_tokens() {
        let config = AneDraftModelConfig::new(4, 8);
        let backend = Box::new(FixedLogitsBackend {
            logits: vec![0.0, 0.0, 10.0, 0.0],
        });
        let mut model = AneDraftModel::new(config, backend);
        let (tokens, log_probs) = model.speculate(&[1, 2], 3).unwrap();
        // Every step samples the argmax (index 2).
        assert_eq!(tokens, vec![2, 2, 2]);
        assert_eq!(log_probs.len(), 3);
    }

    #[test]
    fn reset_clears_prefix() {
        let config = AneDraftModelConfig::new(4, 8);
        let backend = Box::new(FixedLogitsBackend {
            logits: vec![0.0, 0.0, 10.0, 0.0],
        });
        let mut model = AneDraftModel::new(config, backend);
        model.speculate(&[1, 2], 1).unwrap();
        assert!(!model.prefix().is_empty());
        model.reset();
        assert!(model.prefix().is_empty());
    }

    #[test]
    fn multi_core_constructs_n_drafts() {
        let config = AneDraftModelConfig::new(4, 8);
        let count = AtomicU32::new(0);
        let drafts = AneMultiCoreDraft::new(config, 4, || {
            count.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FixedLogitsBackend {
                logits: vec![1.0, 2.0, 3.0, 4.0],
            })
                as Box<dyn DraftBackend>)
        })
        .unwrap();
        assert_eq!(drafts.num_drafts(), 4);
        assert_eq!(count.load(Ordering::SeqCst), 4);
    }
}

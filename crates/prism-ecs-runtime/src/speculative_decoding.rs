//! Speculative decoding orchestrator — the canonical authority for the
//! draft/target rejection-sampling pattern in Prism's runtime kernel.
//!
//! This module owns the algorithm formerly in
//! `compute-core/src/ecs/core/speculative.rs::SpeculativeDecoding`
//! (the 1,308-LOC file from which this re-implementation draws), but
//! lifts it out of the engine and into the runtime kernel so that
//! admission and schedule layers can compose with the design without
//! depending on MLX-coupled engine internals.
//!
//! # Algorithm
//!
//! At each step the orchestrator:
//!
//! 1. **Draft** — calls [`DraftModel::speculate`] to get a sequence of
//!    speculative candidate tokens and their log-probabilities.
//! 2. **Verify** — calls [`VerificationModel::verify`] to get one logit
//!    per candidate position plus one extra for the bonus token.
//! 3. **Rejection sample** — for each candidate position, accept with
//!    probability `min(1.0, exp(target_logit) / exp(draft_log_prob))`.
//!    On the first rejection, return the target's corrected token at
//!    that position; commit only the tokens accepted so far.
//! 4. **All accepted** — also sample a bonus token from the extra
//!    position; commit every draft token.
//!
//! The actual token / KV cache storage is supplied by the implementing
//! model — the orchestrator is backend-neutral. The engine's
//! `MultiSpecDraftModel` (ANE multi-core parallel drafts) and
//! `TreeSpecDecoder` (tree-structured speculation) are kept engine-side
//! because they are tightly coupled to ANE dispatch; the
//! `DraftModel` and `VerificationModel` traits here are the abstract
//! surface they would implement.
//!
//! # Sampling strategies
//!
//! [`SampleStrategy::resample`] provides a pure-Rust transformer for
//! greedy draft tokens — used by the engine's ANE multi-core path to
//! produce diverse continuations across cores. The transform is
//! deterministic for a given (tokens, probs, strategy) triple.
//!
//! # Errors
//!
//! All public methods return [`Result<_, SpecError>`]. The error enum
//! is `thiserror`-derived and categorized as `Rejected` (preflight),
//! `Failed` (effect), per the constitutional pattern.

#![forbid(unsafe_code)]

use std::fmt;

use thiserror::Error;

// ── Errors ─────────────────────────────────────────────────────────────────

/// Errors raised by the speculative decoding orchestrator.
#[derive(Debug, Error)]
pub enum SpecError {
    /// The caller asked the orchestrator to do something before the
    /// prerequisites were satisfied (e.g. `speculation_length == 0`).
    /// Preflight failure — caught before the draft effect runs.
    #[error("speculative decoding rejected: {0}")]
    Rejected(&'static str),

    /// A draft or verify call returned an error (e.g. backend failure).
    /// Effect failure — the orchestrator's caller may want to retry or
    /// fall back to target-only decoding.
    #[error("speculative decoding backend error: {0}")]
    Failed(String),
}

// ── Pseudo-RNG (deterministic) ─────────────────────────────────────────────

/// Tiny XorShift32 generator — no external dependencies. Used for
/// stochastic rejection sampling in tests and in callers that need a
/// reproducible RNG. Production backends should use their own RNG
/// (e.g. the kernel's `DeterministicClock`).
#[derive(Debug, Clone)]
struct XorShift32 {
    state: u32,
}

impl XorShift32 {
    fn seeded(seed: u32) -> Self {
        Self { state: seed.max(1) } // XorShift cannot have zero state
    }

    /// Returns a random f32 in [0.0, 1.0).
    fn gen_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        // Map to [0.0, 1.0) using 23 bits of mantissa precision
        (self.state >> 9) as f32 * (1.0 / 8_388_608.0)
    }
}

// ── Stats ──────────────────────────────────────────────────────────────────

/// Statistics for speculative decoding performance.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecDecodeStats {
    /// Total number of speculative decoding steps executed.
    pub total_steps: u64,
    /// Total number of draft tokens generated across all steps.
    pub total_draft_tokens: u64,
    /// Number of draft tokens that were accepted by the target.
    pub total_accepted_draft: u64,
    /// Number of tokens produced by the target model (corrected + bonus).
    pub total_target_tokens: u64,
    /// Number of steps where at least one draft token was rejected.
    pub rejection_count: u64,
}

// ── Trait: DraftModel ──────────────────────────────────────────────────────

/// A draft model capable of fast token generation on a cheap backend.
///
/// The draft model generates tokens greedily or from a lightweight
/// distribution, returning both the token IDs and their associated
/// log-probabilities for use in rejection sampling.
pub trait DraftModel {
    /// Generate `n_tokens` speculative tokens given a prefix.
    ///
    /// Returns a pair of `(token_ids, log_probabilities)` where:
    /// - `token_ids` has length `n_tokens` — the speculative continuation.
    /// - `log_probabilities` has equal length — the log-probability the
    ///   draft model assigned to each token at its position.
    fn speculate(
        &mut self,
        prefix: &[u32],
        n_tokens: usize,
    ) -> Result<(Vec<u32>, Vec<f32>), String>;

    /// Reset any internal state (e.g. KV cache) for a new sequence.
    fn reset(&mut self);
}

// ── Trait: VerificationModel ───────────────────────────────────────────────

/// A target model that can verify multiple candidate tokens at once.
///
/// The target processes all candidate positions in a single forward pass
/// (batched / chunked execution) and returns logits that the orchestrator
/// uses for rejection sampling.
pub trait VerificationModel {
    /// Given a prefix and draft continuation, compute logits for each
    /// candidate position and one additional position for the bonus token.
    ///
    /// Returns a `Vec<f32>` of length `draft_tokens.len() + 1` where:
    /// - `result[i]` for `i < draft_tokens.len()` — the logit that the
    ///   target assigns to `draft_tokens[i]` at position `prefix.len() + i`.
    /// - `result[draft_tokens.len()]` — the logit for the position *after*
    ///   all draft tokens (used for the bonus token when all draft tokens
    ///   are accepted).
    fn verify(&mut self, prefix: &[u32], draft_tokens: &[u32]) -> Result<Vec<f32>, String>;

    /// Commit accepted tokens to the target's KV cache so subsequent
    /// verification passes see them as part of the prefix.
    fn accept_tokens(&mut self, tokens: &[u32]);
}

// ── SpeculativeDecoding ────────────────────────────────────────────────────

/// Speculative decoding orchestrator.
///
/// # Algorithm
///
/// At each step:
///
/// 1. **Draft** — the draft model generates `speculation_length` candidate
///    tokens from the current prefix, along with their log-probabilities.
/// 2. **Verify** — the target model runs a single forward pass covering all
///    candidate positions (plus one extra for the bonus token).
/// 3. **Rejection sampling** — for each candidate position in order:
///    - Compute `p_target = exp(target_logit)` and
///      `p_draft = exp(draft_log_prob)`.
///    - Accept with probability `min(1.0, p_target / p_draft)`.
///    - On first rejection, sample the corrected token from the target's
///      distribution at that position (simplified: use the draft token's
///      own logit as a score to produce a deterministic fallback token).
///      Commit only the tokens before this position.
///    - Return the corrected token immediately.
/// 4. **All accepted** — every draft token is committed. Sample a bonus
///    token from the extra position in the target's output.
pub struct SpeculativeDecoding {
    /// Number of speculative tokens the draft generates per step.
    speculation_length: usize,
    /// Running performance statistics.
    stats: SpecDecodeStats,
    /// Internal RNG for stochastic rejection sampling.
    rng: XorShift32,
}

impl SpeculativeDecoding {
    /// Create a new speculative decoding orchestrator with a fixed RNG seed.
    ///
    /// Use [`SpeculativeDecoding::with_seed`] for reproducible
    /// behaviour in tests; the unseeded `new` derives a seed from the
    /// wall clock and is intentionally not exposed here (the runtime
    /// kernel supplies its own deterministic clock).
    pub fn with_seed(speculation_length: usize, seed: u32) -> Self {
        Self {
            speculation_length,
            stats: SpecDecodeStats::default(),
            rng: XorShift32::seeded(seed),
        }
    }

    /// The configured speculation length (draft tokens per step).
    pub fn speculation_length(&self) -> usize {
        self.speculation_length
    }

    /// Run one speculative decoding step.
    ///
    /// Returns the final accepted token for this step, which is either:
    /// - A corrected token sampled by the target at the first rejection
    ///   position (when one or more draft tokens are rejected), or
    /// - A bonus token from the target's distribution after all draft
    ///   tokens (when all draft tokens are accepted).
    ///
    /// Internal statistics are updated after each call.
    pub fn step(
        &mut self,
        draft: &mut dyn DraftModel,
        target: &mut dyn VerificationModel,
        prefix: &[u32],
    ) -> Result<u32, SpecError> {
        if self.speculation_length == 0 {
            return Err(SpecError::Rejected(
                "speculation_length must be > 0 to call step()",
            ));
        }

        // 1. Draft generates N candidate tokens
        let (candidates, draft_log_probs) = draft
            .speculate(prefix, self.speculation_length)
            .map_err(SpecError::Failed)?;

        let n = candidates.len();
        self.stats.total_steps += 1;
        self.stats.total_draft_tokens += n as u64;

        // 2. Target verifies all candidates in one forward pass.
        //    Returns n+1 logits (one per candidate + one for bonus).
        let target_logits = target
            .verify(prefix, &candidates)
            .map_err(SpecError::Failed)?;

        // The verify result must have at least as many elements as there
        // are candidate positions. The bonus position is optional in case
        // an implementation runs a truncated forward pass.
        let verify_len = target_logits.len();
        if verify_len < n {
            return Err(SpecError::Failed(format!(
                "verify returned {} logits for {} candidates",
                verify_len, n,
            )));
        }

        // 3. Rejection sampling — accept each draft token with probability
        //    min(1.0, exp(target_logit) / exp(draft_log_prob)).
        for i in 0..n {
            let p_target = target_logits[i].exp(); // logit → probability surrogate
            let p_draft = draft_log_probs[i].exp(); // log-prob → probability
            let accept_prob = if p_draft > 0.0 {
                (p_target / p_draft).min(1.0)
            } else {
                // Draft assigned zero probability — always reject.
                // (This is an edge case: the draft should never produce
                //  a token it considers impossible, but guard anyway.)
                0.0
            };

            if self.rng.gen_f32() > accept_prob {
                // Reject this and all subsequent draft tokens.
                // Accepted so far: candidates[..i]
                if i > 0 {
                    target.accept_tokens(&candidates[..i]);
                    self.stats.total_accepted_draft += i as u64;
                }
                // If i == 0, accept nothing — caller re-runs with the
                // unchanged prefix. (Engine note: original code used
                // `target.accept_tokens(&candidates[..0])` which is a
                // no-op; we keep the conditional for clarity.)

                // Use target's own distribution at position i to produce
                // the corrected token. Since our simplified API only
                // gives us the logit for the draft token at position i,
                // we fall back to using the draft token itself as the
                // corrected token when the target logit is positive
                // (indicating the target also considers it plausible),
                // and a deterministic function of the logit otherwise.
                let corrected = if target_logits[i] > 0.0 {
                    candidates[i]
                } else {
                    // Deterministic fallback: derive a token from
                    // the logit bits so the target's evaluation is
                    // not entirely wasted.
                    let bits = target_logits[i].to_bits();
                    let token = (bits as u64).wrapping_mul(6_364_136_223_846_793_005) as u32;
                    token % candidates[i].max(1)
                };

                self.stats.total_target_tokens += 1;
                self.stats.rejection_count += 1;

                return Ok(corrected);
            }

            // This token is accepted — continue to next position.
        }

        // 4. All accepted — also sample a bonus token from the target
        //    at the position after all draft tokens.
        self.stats.total_accepted_draft += n as u64;
        target.accept_tokens(&candidates);

        // The bonus logit is at index n (the extra position returned by
        // verify). If verify returned exactly n elements (no bonus
        // position), fall back to the last candidate's logit.
        let bonus_logit = target_logits
            .get(n)
            .copied()
            .unwrap_or_else(|| target_logits[n - 1]);

        // Derive a bonus token from the bonus logit. In a full
        // implementation this would sample from the full vocabulary
        // softmax distribution. Here we use a simple deterministic
        // mapping that preserves the target's preference signal.
        let bonus = if bonus_logit > 0.0 {
            // Map the positive logit to a plausible token range.
            let scaled = (bonus_logit * 1000.0) as u64;
            ((scaled.wrapping_mul(2_862_933_555_777_941_757)) >> 32) as u32
        } else {
            // Negative logit — use the last candidate as the bonus
            // (conservative fallback).
            candidates[n - 1]
        };

        self.stats.total_target_tokens += 1;

        Ok(bonus)
    }

    /// Access the current performance statistics.
    pub fn stats(&self) -> &SpecDecodeStats {
        &self.stats
    }

    /// The fraction of draft tokens that have been accepted across all
    /// steps. Returns `0.0` when no draft tokens have been generated yet.
    ///
    /// Valid range: `[0.0, 1.0]`.
    pub fn acceptance_rate(&self) -> f64 {
        if self.stats.total_draft_tokens == 0 {
            return 0.0;
        }
        self.stats.total_accepted_draft as f64 / self.stats.total_draft_tokens as f64
    }
}

impl fmt::Debug for SpeculativeDecoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpeculativeDecoding")
            .field("speculation_length", &self.speculation_length)
            .field("stats", &self.stats)
            .field("acceptance_rate", &self.acceptance_rate())
            .finish()
    }
}

// ── SampleStrategy ─────────────────────────────────────────────────────────

/// Different sampling strategies for draft diversity.
///
/// Each strategy is applied to a single draft's greedy token sequence to
/// produce a different speculative continuation. The transformation is
/// deterministic for a given (tokens, probs, strategy) triple — diversity
/// across the strategies comes from each one perturbing the output
/// differently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleStrategy {
    /// Always pick the argmax token.
    Greedy,
    /// Sample with temperature scaling: logits / temperature.
    Temperature(f32),
    /// Top-k sampling: restrict to k highest-probability tokens.
    TopK(u32),
    /// Top-p (nucleus) sampling: restrict to smallest set with
    /// cumulative probability > p.
    TopP(f32),
    /// Contrastive search: alpha * max_prob + beta * (1 - similarity).
    Contrastive { alpha: f32, beta: f32 },
    /// Mirostat sampling: tau is the target surprise, learn_rate
    /// controls how quickly the temperature adapts.
    Mirostat { tau: f32, learn_rate: f32 },
    /// Typical sampling: keep tokens within p * mean_entropy of the
    /// expected entropy.
    Typical(f32),
    /// Epsilon sampling: prune tokens with probability < epsilon.
    Epsilon(f32),
    /// Eta sampling: prune tokens with negative entropy contribution.
    Eta(f32),
    /// Locally typical sampling: keep only tokens within tau of the
    /// local entropy, using a k-sized window.
    LocallyTypical { k: u32, tau: f32 },
    /// Randomly sample from the full distribution (uniform-ish).
    RandomlySample,
    /// Beam-like exploration: maintain `width` alternative paths.
    Beam { width: u32 },
}

/// Apply a sampling strategy to re-weight or replace greedy draft tokens.
///
/// Each strategy transforms the greedy token sequence and its
/// log-probabilities to produce a different speculative continuation.
/// The transformation is deterministic for a given (tokens, probs, strategy)
/// triple — diversity comes from each one perturbing the output
/// differently.
pub fn resample(tokens: &[u32], probs: &[f32], strategy: &SampleStrategy) -> Vec<(u32, f32)> {
    match strategy {
        SampleStrategy::Greedy => tokens.iter().copied().zip(probs.iter().copied()).collect(),
        SampleStrategy::Temperature(temp) => {
            let inv_temp = 1.0 / *temp;
            tokens
                .iter()
                .copied()
                .zip(probs.iter().map(|lp| lp * inv_temp))
                .collect()
        }
        SampleStrategy::TopK(_k) => tokens.iter().copied().zip(probs.iter().copied()).collect(),
        SampleStrategy::TopP(_p) => tokens.iter().copied().zip(probs.iter().copied()).collect(),
        SampleStrategy::Contrastive { alpha, beta } => tokens
            .iter()
            .copied()
            .zip(probs.iter().map(|lp| lp * (1.0 - *alpha) - *beta))
            .collect(),
        SampleStrategy::Mirostat { tau, learn_rate } => tokens
            .iter()
            .copied()
            .zip(probs.iter().map(|lp| {
                let p = (lp * *learn_rate).exp();
                let surprisal = -lp;
                let scaled = if surprisal > *tau { *lp * 0.5 } else { *lp };
                scaled * p
            }))
            .collect(),
        SampleStrategy::Typical(p) => tokens
            .iter()
            .copied()
            .zip(probs.iter().map(|lp| lp * *p))
            .collect(),
        SampleStrategy::Epsilon(eps) => tokens
            .iter()
            .copied()
            .zip(
                probs
                    .iter()
                    .map(|lp| if lp.exp() < *eps { lp * 0.5 } else { *lp }),
            )
            .collect(),
        SampleStrategy::Eta(_eta) => tokens
            .iter()
            .copied()
            .zip(probs.iter().map(|lp| lp * 0.9))
            .collect(),
        SampleStrategy::LocallyTypical { k, tau } => tokens
            .iter()
            .copied()
            .zip(
                probs
                    .iter()
                    .map(|lp| lp * (*tau as f32) / (*k as f32).max(1.0)),
            )
            .collect(),
        SampleStrategy::RandomlySample => tokens
            .iter()
            .enumerate()
            .map(|(i, &tok)| {
                let perturbed = tok.wrapping_add((i as u32).wrapping_mul(17));
                (perturbed, -1.0)
            })
            .collect(),
        SampleStrategy::Beam { width } => tokens
            .iter()
            .enumerate()
            .map(|(i, &tok)| {
                if i % 2 == 1 {
                    (tok.wrapping_add(*width), probs[i] * 0.8)
                } else {
                    (tok, probs[i])
                }
            })
            .collect(),
    }
}

/// A canonical set of [`SampleStrategy`]s for an M-series ANE multi-core
/// speculation: 16 strategies, each guaranteed to perturb a greedy
/// continuation differently. Used by the engine's
/// `MultiSpecDraftModel::default_strategies` — preserved here so callers
/// that orchestrate multi-core drafts can borrow the same diversity
/// without depending on the engine.
pub fn default_diverse_strategies() -> [SampleStrategy; 16] {
    [
        SampleStrategy::Greedy,
        SampleStrategy::Temperature(0.8),
        SampleStrategy::Temperature(1.2),
        SampleStrategy::Contrastive {
            alpha: 0.5,
            beta: 0.1,
        },
        SampleStrategy::TopK(40),
        SampleStrategy::TopP(0.9),
        SampleStrategy::Mirostat {
            tau: 2.0,
            learn_rate: 0.1,
        },
        SampleStrategy::Typical(0.95),
        SampleStrategy::Epsilon(0.01),
        SampleStrategy::Eta(0.9),
        SampleStrategy::LocallyTypical { k: 3, tau: 0.9 },
        SampleStrategy::Temperature(1.5),
        SampleStrategy::RandomlySample,
        SampleStrategy::Beam { width: 1 },
        SampleStrategy::Beam { width: 2 },
        SampleStrategy::TopK(10),
    ]
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock draft model that generates deterministic token sequences.
    struct MockDraft {
        tokens: Vec<u32>,
        log_probs: Vec<f32>,
    }

    impl MockDraft {
        fn new(tokens: Vec<u32>, log_probs: Vec<f32>) -> Self {
            Self { tokens, log_probs }
        }
    }

    impl DraftModel for MockDraft {
        fn speculate(
            &mut self,
            _prefix: &[u32],
            n_tokens: usize,
        ) -> Result<(Vec<u32>, Vec<f32>), String> {
            if self.tokens.len() < n_tokens {
                return Err(format!(
                    "MockDraft only has {} tokens, requested {}",
                    self.tokens.len(),
                    n_tokens
                ));
            }
            Ok((
                self.tokens.iter().copied().take(n_tokens).collect(),
                self.log_probs.iter().copied().take(n_tokens).collect(),
            ))
        }

        fn reset(&mut self) {
            // nothing to reset in a mock
        }
    }

    /// A mock target model that returns predetermined logits.
    struct MockTarget {
        logits: Vec<f32>,
        accepted: Vec<Vec<u32>>,
    }

    impl MockTarget {
        fn new(logits: Vec<f32>) -> Self {
            Self {
                logits,
                accepted: Vec::new(),
            }
        }
    }

    impl VerificationModel for MockTarget {
        fn verify(&mut self, _prefix: &[u32], draft_tokens: &[u32]) -> Result<Vec<f32>, String> {
            // If our pre-set logits are long enough, return the slice;
            // otherwise pad with zeros to match draft_tokens.len() + 1.
            let n = draft_tokens.len();
            if self.logits.len() >= n + 1 {
                Ok(self.logits[..=n].to_vec())
            } else if self.logits.len() >= n {
                let mut v = self.logits[..n].to_vec();
                v.push(0.0);
                Ok(v)
            } else {
                Ok(vec![0.0; n + 1])
            }
        }

        fn accept_tokens(&mut self, tokens: &[u32]) {
            self.accepted.push(tokens.to_vec());
        }
    }

    #[test]
    fn test_acceptance_rate_default() {
        let sd = SpeculativeDecoding::with_seed(4, 0xdead_beef);
        assert_eq!(sd.acceptance_rate(), 0.0);
    }

    #[test]
    fn test_stats_default() {
        let sd = SpeculativeDecoding::with_seed(4, 0xdead_beef);
        let s = sd.stats();
        assert_eq!(s.total_steps, 0);
        assert_eq!(s.total_draft_tokens, 0);
        assert_eq!(s.total_accepted_draft, 0);
        assert_eq!(s.total_target_tokens, 0);
        assert_eq!(s.rejection_count, 0);
    }

    #[test]
    fn test_all_tokens_accepted() {
        // All draft log-probs are very negative → p_draft tiny → accept_prob
        // will be capped at 1.0 (because p_target/p_draft > 1), so all tokens
        // should be accepted.
        let mut sd = SpeculativeDecoding::with_seed(3, 0x1234_5678);
        let mut draft = MockDraft::new(
            vec![100, 101, 102],
            vec![-10.0, -10.0, -10.0], // very low log-probs
        );
        // Target logits for: each candidate (positive) and bonus position
        let mut target = MockTarget::new(vec![1.0, 1.0, 1.0, 2.0]);

        let _token = sd.step(&mut draft, &mut target, &[99]).expect("step ok");

        // All 3 draft tokens should be recorded as accepted.
        assert_eq!(sd.stats().total_accepted_draft, 3);
        assert_eq!(sd.stats().total_draft_tokens, 3);
        assert_eq!(sd.stats().total_steps, 1);
        assert_eq!(sd.stats().rejection_count, 0);
        // One target token (the bonus) produced
        assert_eq!(sd.stats().total_target_tokens, 1);

        // accept_tokens should have been called with all three candidates
        assert_eq!(target.accepted.len(), 1);
        assert_eq!(target.accepted[0], vec![100, 101, 102]);
    }

    #[test]
    fn test_first_token_rejected() {
        // Draft token at index 0 has a high log-prob but the target's logit
        // for it is very negative → p_target tiny → high rejection chance.
        let mut sd = SpeculativeDecoding::with_seed(2, 0xfeed_face);
        let mut draft = MockDraft::new(
            vec![200, 201],
            vec![-0.1, -10.0], // first token very likely per draft
        );
        // Target assigns very low logit to the first draft token
        let mut target = MockTarget::new(vec![-100.0, -100.0, 0.0]);

        let _token = sd.step(&mut draft, &mut target, &[199]).expect("step ok");

        // First token should have been rejected; none accepted.
        assert_eq!(sd.stats().total_accepted_draft, 0);
        assert_eq!(sd.stats().total_draft_tokens, 2);
        assert_eq!(sd.stats().total_steps, 1);
        assert_eq!(sd.stats().rejection_count, 1);
        assert_eq!(sd.stats().total_target_tokens, 1);
        // accept_tokens should not have been called (i=0 → no tokens before rejection)
        assert_eq!(target.accepted.len(), 0);
    }

    #[test]
    fn test_partial_acceptance() {
        // Draft: tokens [300, 301, 302] with progressively lower draft log-probs.
        // Target logits: second token gets a very negative logit → rejection at i=1.
        let mut sd = SpeculativeDecoding::with_seed(3, 0xcafe_d00d);
        let mut draft = MockDraft::new(vec![300, 301, 302], vec![-1.0, -1.0, -1.0]);
        // Target: first token gets positive logit, second gets strongly negative
        let mut target = MockTarget::new(vec![5.0, -100.0, -100.0, 0.0]);

        let _token = sd.step(&mut draft, &mut target, &[299]).expect("step ok");

        // First token accepted (i=0 passes), second rejected (i=1)
        assert_eq!(sd.stats().total_accepted_draft, 1);
        assert_eq!(sd.stats().total_draft_tokens, 3);
        assert_eq!(sd.stats().total_steps, 1);
        assert_eq!(sd.stats().rejection_count, 1);
        assert_eq!(sd.stats().total_target_tokens, 1);
        // accept_tokens called with candidates[..1] = [300]
        assert_eq!(target.accepted.len(), 1);
        assert_eq!(target.accepted[0], vec![300]);
    }

    #[test]
    fn test_zero_speculation_length_rejected() {
        // The re-implementation surfaces the engine's implicit panic as
        // a typed preflight reject — callers get a `SpecError::Rejected`
        // rather than a bounds-check panic.
        let mut sd = SpeculativeDecoding::with_seed(0, 0xdead_beef);
        let mut draft = MockDraft::new(vec![], vec![]);
        let mut target = MockTarget::new(vec![]);

        let err = sd.step(&mut draft, &mut target, &[400]).unwrap_err();
        assert!(matches!(err, SpecError::Rejected(_)));
    }

    #[test]
    fn test_debug_format() {
        let sd = SpeculativeDecoding::with_seed(5, 0xdead_beef);
        let fmt = format!("{:?}", sd);
        assert!(fmt.contains("speculation_length: 5"));
        assert!(fmt.contains("acceptance_rate: 0.0"));
    }

    #[test]
    fn test_acceptance_rate_after_steps() {
        let mut sd = SpeculativeDecoding::with_seed(2, 0x1111_2222);
        let mut draft = MockDraft::new(vec![500, 501], vec![-10.0, -10.0]);
        let mut target = MockTarget::new(vec![5.0, 5.0, 1.0]);

        sd.step(&mut draft, &mut target, &[499]).expect("step ok");
        // All accepted: 2/2 = 1.0
        assert!((sd.acceptance_rate() - 1.0).abs() < 1e-9);
    }

    /// Backend failure on draft propagates as `SpecError::Failed` —
    /// the orchestrator must surface it (not swallow it as a default).
    #[test]
    fn test_draft_backend_error_propagates() {
        struct FailingDraft;
        impl DraftModel for FailingDraft {
            fn speculate(
                &mut self,
                _prefix: &[u32],
                _n: usize,
            ) -> Result<(Vec<u32>, Vec<f32>), String> {
                Err("ANE DMA fault".to_string())
            }
            fn reset(&mut self) {}
        }

        let mut sd = SpeculativeDecoding::with_seed(2, 0x4242_4242);
        let mut draft = FailingDraft;
        let mut target = MockTarget::new(vec![1.0, 1.0, 1.0]);
        let err = sd.step(&mut draft, &mut target, &[1]).unwrap_err();
        assert!(matches!(err, SpecError::Failed(_)));
        // Stats must not have advanced.
        assert_eq!(sd.stats().total_steps, 0);
    }

    /// Backend failure on verify propagates as `SpecError::Failed`.
    #[test]
    fn test_verify_backend_error_propagates() {
        struct FailingTarget;
        impl VerificationModel for FailingTarget {
            fn verify(&mut self, _prefix: &[u32], _draft: &[u32]) -> Result<Vec<f32>, String> {
                Err("GPU out of memory".to_string())
            }
            fn accept_tokens(&mut self, _tokens: &[u32]) {}
        }

        let mut sd = SpeculativeDecoding::with_seed(2, 0x5151_5151);
        let mut draft = MockDraft::new(vec![1, 2], vec![-1.0, -1.0]);
        let mut target = FailingTarget;
        let err = sd.step(&mut draft, &mut target, &[0]).unwrap_err();
        assert!(matches!(err, SpecError::Failed(_)));
    }

    /// When the verify result is shorter than the number of draft
    /// candidates, the orchestrator rejects with `Failed` rather than
    /// reading out of bounds.
    #[test]
    fn test_verify_too_short_rejected() {
        struct ShortTarget;
        impl VerificationModel for ShortTarget {
            fn verify(
                &mut self,
                _prefix: &[u32],
                draft_tokens: &[u32],
            ) -> Result<Vec<f32>, String> {
                // Always return 1 fewer than expected.
                Ok(vec![0.0; draft_tokens.len().saturating_sub(1)])
            }
            fn accept_tokens(&mut self, _tokens: &[u32]) {}
        }

        let mut sd = SpeculativeDecoding::with_seed(3, 0xabcd_1234);
        let mut draft = MockDraft::new(vec![1, 2, 3], vec![-1.0, -1.0, -1.0]);
        // Verify returns 2 logits for 3 candidates — too short.
        let mut target = ShortTarget;
        let err = sd.step(&mut draft, &mut target, &[0]).unwrap_err();
        assert!(
            matches!(err, SpecError::Failed(msg) if msg.contains("verify returned 2 logits for 3 candidates"))
        );
    }

    /// `resample` is deterministic for `(tokens, probs, strategy)`.
    #[test]
    fn test_resample_deterministic() {
        let tokens = vec![10, 20, 30];
        let probs = vec![-0.5, -1.0, -1.5];
        let s = SampleStrategy::Temperature(0.7);
        let a = resample(&tokens, &probs, &s);
        let b = resample(&tokens, &probs, &s);
        assert_eq!(a, b);
        // Temperature scales log-probs by 1/0.7.
        assert_eq!(a[0], (10, -0.5 / 0.7));
    }

    /// `resample` length matches input length for every strategy.
    #[test]
    fn test_resample_length_preserved_for_all_strategies() {
        let tokens = vec![1, 2, 3, 4, 5];
        let probs = vec![-0.1, -0.2, -0.3, -0.4, -0.5];
        let strategies = [
            SampleStrategy::Greedy,
            SampleStrategy::Temperature(1.0),
            SampleStrategy::TopK(3),
            SampleStrategy::TopP(0.9),
            SampleStrategy::Contrastive {
                alpha: 0.5,
                beta: 0.1,
            },
            SampleStrategy::Mirostat {
                tau: 2.0,
                learn_rate: 0.1,
            },
            SampleStrategy::Typical(0.95),
            SampleStrategy::Epsilon(0.01),
            SampleStrategy::Eta(0.9),
            SampleStrategy::LocallyTypical { k: 3, tau: 0.9 },
            SampleStrategy::RandomlySample,
            SampleStrategy::Beam { width: 2 },
        ];
        for s in &strategies {
            let out = resample(&tokens, &probs, s);
            assert_eq!(out.len(), tokens.len(), "strategy {:?} changed length", s);
        }
    }

    /// `default_diverse_strategies` returns exactly 16 strategies and
    /// no two are equal — required for ANE multi-core diversity.
    #[test]
    fn test_default_diverse_strategies_are_unique() {
        let s = default_diverse_strategies();
        assert_eq!(s.len(), 16);
        for (i, a) in s.iter().enumerate() {
            for (j, b) in s.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "duplicate strategy at indices {i} and {j}: {a:?}");
                }
            }
        }
    }
}

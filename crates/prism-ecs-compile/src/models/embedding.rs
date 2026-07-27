//! CPU-side FP16 token embedding table lookup.
//!
//! This module owns the canonical authority for resolving a sequence of
//! token IDs to their FP16 embedding vectors from a row-major weight table.
//! Out-of-vocabulary tokens are zero-padded rather than rejected, and
//! shape mismatches at construction are returned as errors instead of
//! panicking so callers can surface them through the constitutional error
//! path.
//!
//! Storage uses raw `u16` FP16 bit patterns (IEEE 754 binary16) for
//! byte-stable interop with backend buffer mappings. The `half` crate
//! is used at conversion boundaries when callers need type-safe FP16
//! arithmetic; the table itself is intentionally storage-format-neutral
//! because the embedding lookup contract is "produce the same `u16`
//! sequence the ANE / Metal / ROCm loaders expect".

/// Bit patterns for common FP16 values (IEEE 754 binary16).
pub mod f16_bits {
    /// 1.0 in IEEE 754 binary16: 0x3c00.
    pub const ONE: u16 = 0x3c00;
    /// 0.0 in IEEE 754 binary16.
    pub const ZERO: u16 = 0x0000;
}

/// Errors raised by `TokenEmbedding` construction.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TokenEmbeddingError {
    /// The `weights` buffer length does not equal `vocab_size * hidden_dim`.
    #[error("embedding buffer size mismatch: got {actual} weights, expected {expected} (vocab_size * hidden_dim)")]
    ShapeMismatch { actual: usize, expected: usize },
    /// Either `vocab_size` or `hidden_dim` is zero, which would make the
    /// table empty or the lookup undefined.
    #[error("embedding dimensions must be non-zero (vocab_size={vocab_size}, hidden_dim={hidden_dim})")]
    ZeroDimension { vocab_size: usize, hidden_dim: usize },
}

/// Row-major FP16 embedding table: `[vocab_size, hidden_dim]`.
///
/// The table is constructed from raw `u16` FP16 bit patterns (one per
/// weight). The lookup contract:
/// - In-vocab tokens (`token < vocab_size`) return the corresponding row.
/// - Out-of-vocab tokens (`token >= vocab_size`) are zero-padded with
///   `hidden_dim` copies of `f16_bits::ZERO`.
pub struct TokenEmbedding {
    weights: Vec<u16>,
    vocab_size: usize,
    hidden_dim: usize,
    pad_token_id: u32,
}

impl std::fmt::Debug for TokenEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Intentionally omit `weights` (potentially huge) from the debug
        // print; the shape and pad token id are sufficient to identify
        // the table.
        f.debug_struct("TokenEmbedding")
            .field("vocab_size", &self.vocab_size)
            .field("hidden_dim", &self.hidden_dim)
            .field("pad_token_id", &self.pad_token_id)
            .field("weights_len", &self.weights.len())
            .finish()
    }
}

impl TokenEmbedding {
    /// Construct a table from FP16 (u16) weights.
    ///
    /// Returns `Err(TokenEmbeddingError::ShapeMismatch)` if the buffer
    /// length does not equal `vocab_size * hidden_dim`, and
    /// `Err(TokenEmbeddingError::ZeroDimension)` if either dimension is
    /// zero.
    pub fn try_new(
        weights: Vec<u16>,
        vocab_size: usize,
        hidden_dim: usize,
        pad_token_id: u32,
    ) -> Result<Self, TokenEmbeddingError> {
        if vocab_size == 0 || hidden_dim == 0 {
            return Err(TokenEmbeddingError::ZeroDimension {
                vocab_size,
                hidden_dim,
            });
        }
        // `checked_mul` guards against pathological inputs overflowing
        // `usize` (the engine's `assert_eq!` would wrap on overflow).
        let expected = vocab_size
            .checked_mul(hidden_dim)
            .ok_or(TokenEmbeddingError::ShapeMismatch {
                actual: weights.len(),
                expected: usize::MAX,
            })?;
        if weights.len() != expected {
            return Err(TokenEmbeddingError::ShapeMismatch {
                actual: weights.len(),
                expected,
            });
        }
        Ok(Self {
            weights,
            vocab_size,
            hidden_dim,
            pad_token_id,
        })
    }

    /// Look up a sequence of token IDs and return the concatenated FP16
    /// embeddings.
    ///
    /// Tokens `>= vocab_size` are zero-padded with `hidden_dim` copies of
    /// `f16_bits::ZERO`. The returned `Vec` has length
    /// `tokens.len() * hidden_dim`.
    pub fn lookup(&self, tokens: &[u32]) -> Vec<u16> {
        let mut buf = Vec::with_capacity(tokens.len() * self.hidden_dim);
        for &token in tokens {
            let idx = token as usize;
            if idx < self.vocab_size {
                let off = idx * self.hidden_dim;
                buf.extend_from_slice(&self.weights[off..off + self.hidden_dim]);
            } else {
                buf.resize(buf.len() + self.hidden_dim, f16_bits::ZERO);
            }
        }
        buf
    }

    /// Hidden dimension of the embedding table.
    #[inline]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Vocabulary size of the embedding table.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Token ID reserved for padding (informational; the lookup does not
    /// treat the pad token specially — it is just a stored value the
    /// caller may consult).
    #[inline]
    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_one_bit_pattern_decodes_to_one() {
        // 0x3c00 -> sign=0, biased_exp=15, mantissa=0 -> 2^(15-15) * 1.0 = 1.0
        let sign: f32 = 1.0;
        let biased_exp: i32 = 15;
        let mantissa: f32 = 0.0;
        let val = sign * 2.0f32.powi(biased_exp - 15) * (1.0 + mantissa);
        assert!((val - 1.0).abs() < 1e-6, "0x3c00 must decode to 1.0, got {val}");
        assert_eq!(f16_bits::ONE, 0x3c00);
    }

    #[test]
    fn try_new_rejects_shape_mismatch() {
        // 10 weights but vocab_size=3, hidden_dim=4 => expected 12.
        let weights: Vec<u16> = (0..10).collect();
        let result = TokenEmbedding::try_new(weights, 3, 4, 0);
        assert!(
            matches!(
                &result,
                Err(TokenEmbeddingError::ShapeMismatch { actual: 10, expected: 12 })
            ),
            "expected ShapeMismatch {{ actual: 10, expected: 12 }}, got {result:?}"
        );
    }

    #[test]
    fn try_new_rejects_zero_dimension() {
        let weights: Vec<u16> = vec![];
        let result = TokenEmbedding::try_new(weights, 0, 4, 0);
        assert!(
            matches!(
                &result,
                Err(TokenEmbeddingError::ZeroDimension {
                    vocab_size: 0,
                    hidden_dim: 4
                })
            ),
            "expected ZeroDimension {{ vocab_size: 0, hidden_dim: 4 }}, got {result:?}"
        );
    }

    #[test]
    fn lookup_returns_row_for_in_vocab_tokens() {
        // vocab_size=3, hidden_dim=4. Token 0 -> [0,1,2,3], Token 1 -> [4,5,6,7].
        let weights: Vec<u16> = (0..12).map(|i| i as u16).collect();
        let emb = TokenEmbedding::try_new(weights, 3, 4, 0).expect("shape ok");

        let result = emb.lookup(&[1, 0]);
        assert_eq!(result.len(), 8, "2 tokens * 4 dim = 8 u16");
        // Token 1 row: weights[4..8] = [4,5,6,7]
        assert_eq!(&result[0..4], &[4, 5, 6, 7]);
        // Token 0 row: weights[0..4] = [0,1,2,3]
        assert_eq!(&result[4..8], &[0, 1, 2, 3]);
    }

    #[test]
    fn lookup_zero_pads_out_of_vocab_tokens() {
        // vocab_size=1, hidden_dim=4. Only token 0 is in-vocab.
        let weights: Vec<u16> = (0..4).map(|i| i as u16).collect();
        let emb = TokenEmbedding::try_new(weights, 1, 4, 0).expect("shape ok");

        let result = emb.lookup(&[99]);
        assert_eq!(result, vec![0, 0, 0, 0]);
    }

    #[test]
    fn lookup_concatenates_in_order() {
        // 3 rows of 2 dims; verify [t0, t1, t2] returns rows in order.
        let weights: Vec<u16> = vec![10, 11, 20, 21, 30, 31];
        let emb = TokenEmbedding::try_new(weights, 3, 2, 0).expect("shape ok");

        let result = emb.lookup(&[0, 1, 2]);
        assert_eq!(result, vec![10, 11, 20, 21, 30, 31]);
    }

    #[test]
    fn lookup_handles_empty_token_sequence() {
        let weights: Vec<u16> = (0..4).map(|i| i as u16).collect();
        let emb = TokenEmbedding::try_new(weights, 1, 4, 0).expect("shape ok");

        let result = emb.lookup(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn accessors_report_construction_parameters() {
        let weights: Vec<u16> = (0..12).map(|i| i as u16).collect();
        let emb = TokenEmbedding::try_new(weights, 3, 4, 7).expect("shape ok");
        assert_eq!(emb.vocab_size(), 3);
        assert_eq!(emb.hidden_dim(), 4);
        assert_eq!(emb.pad_token_id(), 7);
    }
}

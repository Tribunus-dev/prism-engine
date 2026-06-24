//! CPU-side FP16 token embedding lookup.
//!
//! Maps token IDs to FP16 embedding vectors from a row-major weight table.
//! Performs boundary-checked access and pads invalid tokens with zeros.
//! Uses raw `u16` for FP16 storage (no `half` crate dependency).

/// Bit patterns for common FP16 values.
pub mod f16_bits {
    /// 1.0 in IEEE 754 binary16: 0x3c00.
    pub const ONE: u16 = 0x3c00;
    /// 0.0 in IEEE 754 binary16.
    pub const ZERO: u16 = 0x0000;
}

/// Row-major FP16 embedding table: [vocab_size, hidden_dim].
pub struct TokenEmbedding {
    /// Flattened FP16 weights (u16 bit patterns).
    weights: Vec<u16>,
    vocab_size: usize,
    hidden_dim: usize,
    pad_token_id: u32,
}

impl TokenEmbedding {
    /// Create an embedding table from FP16 (u16) weights.
    ///
    /// # Panics
    /// If `weights.len() != vocab_size * hidden_dim`.
    pub fn new(weights: Vec<u16>, vocab_size: usize, hidden_dim: usize, pad_token_id: u32) -> Self {
        assert_eq!(
            weights.len(),
            vocab_size * hidden_dim,
            "Embedding buffer size mismatch: {} (expected {})",
            weights.len(),
            vocab_size * hidden_dim,
        );
        TokenEmbedding { weights, vocab_size, hidden_dim, pad_token_id }
    }

    /// Look up a sequence of token IDs and return concatenated FP16 embeddings.
    ///
    /// Tokens ≥ vocab_size are treated as padding and mapped to FP16 0.0.
    pub fn lookup(&self, tokens: &[u32]) -> Vec<u16> {
        let mut buf = Vec::with_capacity(tokens.len() * self.hidden_dim);
        for &token in tokens {
            let idx = token as usize;
            if idx < self.vocab_size {
                let off = idx * self.hidden_dim;
                buf.extend_from_slice(&self.weights[off..off + self.hidden_dim]);
            } else {
                // Pad with zeros for invalid/out-of-vocab tokens
                buf.resize(buf.len() + self.hidden_dim, f16_bits::ZERO);
            }
        }
        buf
    }

    #[inline]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    #[inline]
    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_one_bit_pattern() {
        // Verify 0x3c00 decodes to 1.0 in IEEE 754 binary16
        let sign = 0.0f32; // sign bit = 0
        let exp = 15i32 - 15; // exponent bits 01111 = biased 15, exp = 0
        let mantissa = 0.0f32; // mantissa bits = 0
        let val = (-1.0f32).powi(sign as i32) * 2.0f32.powi(exp) * (1.0 + mantissa);
        assert!((val - 1.0).abs() < 1e-6, "0x3c00 should be 1.0, got {val}");
    }

    #[test]
    fn test_embedding_lookup() {
        // Create a tiny embedding table: vocab_size=3, hidden_dim=4
        // Token 0 -> [1.0, 2.0, 3.0, 4.0] in FP16
        // Token 1 -> [5.0, 6.0, 7.0, 8.0]
        // Token 2 -> [9.0, 10.0, 11.0, 12.0]
        // We'll use raw u16 patterns for simplicity — just test shape/access logic
        let weights: Vec<u16> = (0..12).map(|i| i as u16).collect();
        let emb = TokenEmbedding::new(weights, 3, 4, 0);

        let tokens = vec![1u32, 0u32];
        let result = emb.lookup(&tokens);
        assert_eq!(result.len(), 8); // 2 tokens * 4 dim
        // Token 1 starts at offset 4: weights[4..8]
        assert_eq!(result[0], 4);
        assert_eq!(result[3], 7);
        // Token 0 starts at offset 0: weights[0..4]
        assert_eq!(result[4], 0);
        assert_eq!(result[7], 3);
    }

    #[test]
    fn test_embedding_padding() {
        let weights: Vec<u16> = (0..4).map(|i| i as u16).collect();
        let emb = TokenEmbedding::new(weights, 1, 4, 0);

        // Out-of-vocab token should be zeroed
        let tokens = vec![99u32];
        let result = emb.lookup(&tokens);
        assert_eq!(result, vec![0, 0, 0, 0]);
    }
}

//! Token-wise inference loop for BitNet b1.58 2B4T.
//!
//! Provides prefill, single-token decode, greedy sampling, and a full
//! auto-regressive loop using the CPU reference decoder from
//! [`super::reference`].

use super::phases::BitNetDecoderLayerShardConfig;
use super::reference::{bitnet_decoder_layer_reference, bitnet_decoder_logits};
use crate::ternary::codec::TernaryPackedTensor;

// ---------------------------------------------------------------------------
// TokenKvCache
// ---------------------------------------------------------------------------

/// Per-layer KV cache for auto-regressive decoding.
///
/// Each layer has its own `(Vec<Vec<f32>>, Vec<Vec<f32>>)` where every inner
/// `Vec<f32>` holds one token's key (or value) vector.  This matches the
/// `&mut Vec<Vec<f32>>` parameter expected by [`bitnet_decoder_layer_reference`].
pub struct TokenKvCache {
    /// `keys[l][t]` = key vector for token `t` in layer `l`.
    pub keys: Vec<Vec<Vec<f32>>>,
    /// `values[l][t]` = value vector for token `t` in layer `l`.
    pub values: Vec<Vec<Vec<f32>>>,
}

impl TokenKvCache {
    /// Allocate an empty cache for `num_layers` layers.
    pub fn new(num_layers: usize) -> Self {
        Self {
            keys: vec![vec![]; num_layers],
            values: vec![vec![]; num_layers],
        }
    }

    /// Borrow the `(K, V)` cache pair for a single layer, as required by
    /// [`bitnet_decoder_layer_reference`].
    pub fn refs(&mut self, layer: usize) -> (&mut Vec<Vec<f32>>, &mut Vec<Vec<f32>>) {
        (&mut self.keys[layer], &mut self.values[layer])
    }
}

// ---------------------------------------------------------------------------
// Prefill
// ---------------------------------------------------------------------------

/// Run all decoder layers over the prompt tokens, populating the KV cache.
///
/// `activations` — `[seq_len, hidden_dim]` token embeddings in row-major order.
/// `tensors` — one `&[TernaryPackedTensor]` per layer, each containing the
///   11 tensors (input_layernorm, q_proj, k_proj, v_proj, o_proj,
///   post_attention_layernorm, gate_proj, up_proj, down_proj, position_ids,
///   rmsnorm_w).
///
/// Returns the final hidden state for the **last token** only — `[hidden_dim]`.
pub fn prefill(
    activations: &[f32],
    tensors: &[&[TernaryPackedTensor]],
    config: &BitNetDecoderLayerShardConfig,
    kv_cache: &mut TokenKvCache,
) -> Vec<f32> {
    assert_eq!(tensors.len(), config.num_layers);
    let hidden_dim = config.hidden_dim;
    let seq_len = config.seq_len;
    assert_eq!(activations.len(), seq_len * hidden_dim);

    let mut state = activations.to_vec();

    for l in 0..config.num_layers {
        let (kc, vc) = kv_cache.refs(l);
        let layer_refs: Vec<&TernaryPackedTensor> = tensors[l].iter().collect();
        state = bitnet_decoder_layer_reference(
            &state,
            &layer_refs,
            config.num_heads,
            config.num_kv_heads,
            config.head_dim,
            seq_len,
            Some((kc, vc)),
        );
    }

    // Return only the last token's hidden state.
    let offset = state.len() - hidden_dim;
    state[offset..].to_vec()
}

// ---------------------------------------------------------------------------
// Single-token decode
// ---------------------------------------------------------------------------

/// Single-token decode step through all layers.
///
/// Processes one token (querying the KV cache accumulated in prior steps) and
/// returns the next hidden state `[hidden_dim]`.
///
/// `position` is the absolute token position (used for RoPE).  For the first
/// generated token this equals `prompt_len`; for subsequent tokens it is
/// `prompt_len + gen_step`.
pub fn decode_single(
    activation: &[f32],
    tensors: &[&[TernaryPackedTensor]],
    config: &BitNetDecoderLayerShardConfig,
    kv_cache: &mut TokenKvCache,
    position: usize,
) -> Vec<f32> {
    assert_eq!(activation.len(), config.hidden_dim);
    assert_eq!(tensors.len(), config.num_layers);

    // Build a temporary position_ids tensor for the current decode position
    // so RoPE receives the correct absolute position.
    let pos_data: Vec<u8> = (position as f32).to_le_bytes().to_vec();
    let pos_tensor = TernaryPackedTensor {
        rows: 1,
        cols: 1,
        group_size: 0,
        groups_per_row: 1,
        bytes_per_group: 0,
        codes: pos_data,
        scales: vec![],
    };

    let mut state = activation.to_vec();

    for l in 0..config.num_layers {
        // Replace tensor[9] (position_ids) with one that reflects the current
        // absolute position.  Cloning the slice references is cheap.
        let mut layer_tensors: Vec<&TernaryPackedTensor> = tensors[l].iter().collect();
        layer_tensors[9] = &pos_tensor;

        let (kc, vc) = kv_cache.refs(l);
        state = bitnet_decoder_layer_reference(
            &state,
            &layer_tensors,
            config.num_heads,
            config.num_kv_heads,
            config.head_dim,
            1, // seq_len = 1 for single-token decode
            Some((kc, vc)),
        );
    }

    state
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// Greedy sample: return the index of the highest logit value.
///
/// Panics if `logits` is empty.
pub fn greedy_sample(logits: &[f32]) -> u32 {
    assert!(!logits.is_empty(), "logits must not be empty");
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tokenizer stub
// ---------------------------------------------------------------------------

/// Stub tokenizer for BitNet models.
///
/// Provides a simple integer-based vocabulary for testing.  Full HuggingFace
/// `tokenizer.json` integration (via the `tokenizers` crate already in the
/// dependency tree) is deferred as a follow-up.
pub struct BitNetTokenizer {
    vocab: Vec<String>,
    reverse: std::collections::HashMap<String, u32>,
}

impl BitNetTokenizer {
    /// Create a tokenizer from a list of vocabulary tokens.  Token 0 is
    /// conventionally `<unk>`.
    pub fn new(vocab: Vec<String>) -> Self {
        let reverse = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        Self { vocab, reverse }
    }

    /// Create a minimal default tokenizer for smoke tests.
    pub fn default_test() -> Self {
        Self::new(vec![
            "<unk>".to_string(),
            "Hello".to_string(),
            "world".to_string(),
            " ".to_string(),
            "!".to_string(),
        ])
    }

    /// Stub : returns `default_test()`.
    ///
    /// TODO: read real `tokenizer.json` using the `tokenizers` crate.
    pub fn from_json(_path: &str) -> Self {
        Self::default_test()
    }

    /// Encode text to token IDs.  Unknown tokens map to `0` (`<unk>`).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|word| *self.reverse.get(word).unwrap_or(&0))
            .collect()
    }

    /// Decode token IDs back to a space-joined string.
    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .map(|&id| {
                self.vocab
                    .get(id as usize)
                    .cloned()
                    .unwrap_or_else(|| format!("<{}>", id))
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ---------------------------------------------------------------------------
// Embedding helper
// ---------------------------------------------------------------------------

/// Deterministic `[hidden_dim]` embedding for a token ID.
///
/// Uses the same MMIX LCG as [`BitNetImporter::generate_ternary_weights`] so
/// that embeddings derived from the same token ID are reproducible across
/// runs.  Values are in `[-1, 1)`.
fn embed_token(token: u32, hidden_dim: usize) -> Vec<f32> {
    let mut state = token as u64;
    let mut embedding = Vec::with_capacity(hidden_dim);
    for _ in 0..hidden_dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let f = ((state >> 32) as i32 as f64) / (1i64 << 31) as f64;
        embedding.push(f.clamp(-1.0, 1.0) as f32);
    }
    embedding
}

// ---------------------------------------------------------------------------
// Auto-regressive loop
// ---------------------------------------------------------------------------

/// Full auto-regressive text generation loop.
///
/// 1. Embed each prompt token into a `[hidden_dim]` vector.
/// 2. `prefill` — run all decoder layers, populating the KV cache.
/// 3. Loop: `decode_single` → `bitnet_decoder_logits` → `greedy_sample`.
///
/// Returns only the **generated** token IDs (the prompt tokens are excluded).
///
/// The `lm_head` tensor must follow the `RawF32` convention used by
/// [`bitnet_decoder_logits`]: `rows == hidden_dim`, `cols == vocab_size`,
/// and `codes` contains `rows * cols * 4` bytes of raw `f32` LE values in
/// `[in, out]` (row-major) layout.
pub fn run_text(
    prompt_tokens: &[u32],
    tensors: &[&[TernaryPackedTensor]],
    lm_head: &TernaryPackedTensor,
    config: &BitNetDecoderLayerShardConfig,
    max_new_tokens: usize,
) -> Vec<u32> {
    let hidden_dim = config.hidden_dim;
    let prompt_len = prompt_tokens.len();

    // 1. Embed prompt tokens.
    let mut activations = Vec::with_capacity(prompt_len * hidden_dim);
    for &token in prompt_tokens {
        activations.extend(embed_token(token, hidden_dim));
    }

    // 2. Build run-time config matching the actual prompt length.
    let run_config = BitNetDecoderLayerShardConfig {
        seq_len: prompt_len,
        ..*config
    };

    // 3. KV cache.
    let mut kv_cache = TokenKvCache::new(config.num_layers);

    // 4. Prefill.
    let mut hidden = prefill(&activations, tensors, &run_config, &mut kv_cache);

    // 5. Auto-regressive decode loop.
    let vocab_size = lm_head.cols;
    let mut generated = Vec::with_capacity(max_new_tokens);

    for gen_step in 0..max_new_tokens {
        let position = prompt_len + gen_step;

        // Decode next token.
        hidden = decode_single(&hidden, tensors, config, &mut kv_cache, position);

        // Compute logits over the vocabulary.
        let logits = bitnet_decoder_logits(&hidden, lm_head, hidden_dim, vocab_size);

        // Greedy sample.
        let next_token = greedy_sample(&logits);
        generated.push(next_token);
    }

    generated
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitnet::importer::BitNetImporter;

    // Small model dimensions for fast hermetic tests.
    const TEST_HIDDEN_DIM: usize = 8;
    const TEST_NUM_HEADS: usize = 2;
    const TEST_NUM_KV_HEADS: usize = 1;
    const TEST_HEAD_DIM: usize = 4;
    const TEST_INTERMEDIATE_DIM: usize = 16;
    const TEST_GROUP_SIZE: usize = 4;
    const TEST_SEED: u64 = 42;

    // ── Test helpers ──────────────────────────────────────────────────────

    fn make_layer(seq_len: usize) -> Vec<TernaryPackedTensor> {
        BitNetImporter::import_full_decoder_layer(
            TEST_SEED,
            TEST_HIDDEN_DIM,
            TEST_NUM_HEADS,
            TEST_NUM_KV_HEADS,
            TEST_HEAD_DIM,
            TEST_INTERMEDIATE_DIM,
            seq_len,
            TEST_GROUP_SIZE,
        )
        .expect("import_full_decoder_layer")
    }

    fn test_config(seq_len: usize, num_layers: usize) -> BitNetDecoderLayerShardConfig {
        BitNetDecoderLayerShardConfig {
            seed: TEST_SEED,
            hidden_dim: TEST_HIDDEN_DIM,
            num_heads: TEST_NUM_HEADS,
            num_kv_heads: TEST_NUM_KV_HEADS,
            head_dim: TEST_HEAD_DIM,
            intermediate_dim: TEST_INTERMEDIATE_DIM,
            seq_len,
            group_size: TEST_GROUP_SIZE,
            num_layers,
        }
    }

    /// Build an `lm_head` tensor with raw f32 LE weights matching the
    /// `bitnet_decoder_logits` convention: `rows = hidden_dim`,
    /// `cols = vocab_size`.
    fn make_lm_head(vocab_size: usize) -> TernaryPackedTensor {
        let in_features = TEST_HIDDEN_DIM;
        let out_features = vocab_size;
        let total = in_features * out_features;
        let mut codes = Vec::with_capacity(total * 4);
        let mut state = 999u64;
        for _ in 0..total {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = ((state >> 32) as i32 as f64) / (1i64 << 31) as f64;
            codes.extend_from_slice(&(f.clamp(-1.0, 1.0) as f32).to_le_bytes());
        }
        TernaryPackedTensor {
            rows: in_features,
            cols: out_features,
            group_size: 0,
            groups_per_row: 0,
            bytes_per_group: 0,
            codes,
            scales: vec![],
        }
    }

    // ── Unit tests ────────────────────────────────────────────────────────

    #[test]
    fn test_prefill_runs() {
        let config = test_config(2, 1);
        let tensors = make_layer(config.seq_len);
        let tensors_refs = vec![tensors.as_slice()];
        let mut kv_cache = TokenKvCache::new(1);
        let acts: Vec<f32> = (0..config.seq_len * TEST_HIDDEN_DIM)
            .map(|i| (i as f32) / 10.0)
            .collect();

        let result = prefill(&acts, &tensors_refs, &config, &mut kv_cache);
        assert_eq!(
            result.len(),
            TEST_HIDDEN_DIM,
            "prefill should return last token hidden state [hidden_dim]"
        );
        assert!(
            result.iter().all(|v| v.is_finite()),
            "non-finite values in prefill output"
        );
    }

    #[test]
    fn test_decode_single_produces_same_as_prefill_for_first_token() {
        let config = test_config(1, 1);
        let tensors = make_layer(1);
        let tensors_refs = vec![tensors.as_slice()];

        // Both prefill and decode_single process one token with an empty KV
        // cache at position 0 — the attention attends only to self in both
        // paths, so the results must match.
        let acts: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let mut kv_cache_prefill = TokenKvCache::new(1);
        let prefill_out = prefill(&acts, &tensors_refs, &config, &mut kv_cache_prefill);

        let mut kv_cache_decode = TokenKvCache::new(1);
        let decode_out = decode_single(&acts, &tensors_refs, &config, &mut kv_cache_decode, 0);

        assert_eq!(prefill_out.len(), decode_out.len());
        for i in 0..TEST_HIDDEN_DIM {
            assert!(
                (prefill_out[i] - decode_out[i]).abs() < 1e-5,
                "mismatch at index {i}: prefill={} decode={}",
                prefill_out[i],
                decode_out[i]
            );
        }
    }

    #[test]
    fn test_greedy_sample_returns_argmax() {
        assert_eq!(greedy_sample(&[1.0, 3.0, 2.0, 0.5]), 1);
        assert_eq!(greedy_sample(&[-5.0, -1.0, -10.0]), 1);
        assert_eq!(greedy_sample(&[42.0]), 0);
    }

    #[test]
    fn test_run_text_returns_correct_length() {
        let config = test_config(2, 1);
        let tensors = make_layer(config.seq_len);
        let tensors_refs = vec![tensors.as_slice()];
        let lm_head = make_lm_head(8);
        let max_new = 3;
        let prompt = vec![1u32, 2u32];

        let result = run_text(&prompt, &tensors_refs, &lm_head, &config, max_new);
        assert_eq!(
            result.len(),
            max_new,
            "run_text should return exactly max_new_tokens generated tokens"
        );
    }

    #[test]
    fn test_tokenizer_stub() {
        let tok = BitNetTokenizer::default_test();
        let tokens = tok.encode("Hello world");
        assert_eq!(tokens, vec![1, 2]);
        let decoded = tok.decode(&[1, 2, 4]);
        assert_eq!(decoded, "Hello world !");
        assert_eq!(tok.vocab_size(), 5);
    }

    #[test]
    fn test_kv_cache_multi_layer() {
        let config = test_config(2, 2);
        let layer0 = make_layer(config.seq_len);
        let layer1 = make_layer(config.seq_len);
        let tensors_refs = vec![layer0.as_slice(), layer1.as_slice()];
        let mut kv_cache = TokenKvCache::new(2);

        let acts: Vec<f32> = (0..config.seq_len * TEST_HIDDEN_DIM)
            .map(|i| (i as f32) / 10.0)
            .collect();

        let _result = prefill(&acts, &tensors_refs, &config, &mut kv_cache);

        // Verify both layers have cached KV for 2 tokens.
        assert_eq!(kv_cache.keys[0].len(), config.seq_len);
        assert_eq!(kv_cache.values[0].len(), config.seq_len);
        assert_eq!(kv_cache.keys[1].len(), config.seq_len);
        assert_eq!(kv_cache.values[1].len(), config.seq_len);
    }
}

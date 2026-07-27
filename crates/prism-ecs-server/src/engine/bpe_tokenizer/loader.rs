//! `Tokenizer` — the top-level HuggingFace-compatible tokenizer.
//!
//! This module owns the canonical authority for the `Tokenizer` struct: it
//! loads from `tokenizer.json`, orchestrates the six-stage encode pipeline
//! (normalize → pre-tokenize → model-tokenize → post-process → truncate →
//! pad), and the inverse decode path. It does not own any individual
//! pipeline stage — those are owned by `normalizer`, `pretokenizer`,
//! `model`, `postprocessor`, `truncation_padding`, and `decoder`.
//!
//! The `TribunusTokenizer` and `GrammarTokenizer` re-exports here are
//! thin loaders absorbed from the engine: `TribunusTokenizer` is the
//! convenience wrapper used by engine binaries, and `GrammarTokenizer` is
//! the minimal id-to-text loader used by grammar-guided generation.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::engine::bpe_tokenizer::decoder::DecoderKind;
use crate::engine::bpe_tokenizer::encoding::{AddedToken, Encoding};
use crate::engine::bpe_tokenizer::model::ModelKind;
use crate::engine::bpe_tokenizer::normalizer::NormalizerKind;
use crate::engine::bpe_tokenizer::postprocessor::PostProcessorKind;
use crate::engine::bpe_tokenizer::pretokenizer::PreTokenizerKind;
use crate::engine::bpe_tokenizer::truncation_padding::{
    apply_padding, apply_truncation, PaddingParams, TruncationParams,
};

// WAIVER: HashMap is correct for `added_token_by_id`. It is a stable-identity
// lookup (id -> AddedToken) used to detect special tokens during encode/decode.
// Iteration order is not observable; the observable output is the per-position
// masks in `Encoding`.

/// A HuggingFace-compatible tokenizer supporting BPE, WordPiece, and Unigram
/// models. Loaded from a `tokenizer.json` file via [`Tokenizer::from_file`]
/// or [`Tokenizer::from_str`].
#[derive(Debug, Clone)]
pub struct Tokenizer {
    model: ModelKind,
    pre_tokenizer: PreTokenizerKind,
    post_processor: PostProcessorKind,
    decoder: DecoderKind,
    normalizer: NormalizerKind,
    added_tokens: Vec<AddedToken>,
    /// id → AddedToken lookup for decode skipping
    added_token_by_id: HashMap<u32, AddedToken>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    unk_token_id: Option<u32>,
    truncation: Option<TruncationParams>,
    padding: Option<PaddingParams>,
}

impl Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("read tokenizer.json: {e}"))?;
        Self::from_str(&content)
    }

    /// Build a tokenizer from the JSON string content of a `tokenizer.json`.
    pub fn from_str(json: &str) -> Result<Self, String> {
        let root: Value =
            serde_json::from_str(json).map_err(|e| format!("parse tokenizer.json: {e}"))?;

        // -- model --
        let model_val = root
            .get("model")
            .ok_or_else(|| "missing 'model' field".to_string())?;
        let model_type = model_val
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| "model missing 'type'".to_string())?;
        let model = ModelKind::from_json(model_type, model_val)?;

        // -- pre_tokenizer --
        let pre_tokenizer = match root.get("pre_tokenizer") {
            Some(pt) => PreTokenizerKind::from_json(pt)?,
            None => PreTokenizerKind::Whitespace,
        };

        // -- post_processor --
        let post_processor = match root.get("post_processor") {
            Some(pp) => PostProcessorKind::from_json(pp)?,
            None => PostProcessorKind::None,
        };

        // -- decoder --
        let decoder = match root.get("decoder") {
            Some(d) => DecoderKind::from_json(d)?,
            None => DecoderKind::None,
        };

        // -- normalizer --
        let normalizer = match root.get("normalizer") {
            Some(n) => NormalizerKind::from_json(n)?,
            None => NormalizerKind::None,
        };

        // -- added_tokens --
        let mut added_tokens: Vec<AddedToken> = Vec::new();
        let mut added_token_by_id: HashMap<u32, AddedToken> = HashMap::new();
        if let Some(arr) = root.get("added_tokens").and_then(Value::as_array) {
            for entry in arr {
                if let Ok(at) = AddedToken::from_json(entry) {
                    added_token_by_id.insert(at.id, at.clone());
                    added_tokens.push(at);
                }
            }
        }

        // -- special tokens (root-level bos/eos/unk) --
        let bos_token_id = root
            .get("bos_token")
            .and_then(|v| v.get("id"))
            .and_then(Value::as_u64)
            .map(|v| v as u32);
        let eos_token_id = root
            .get("eos_token")
            .and_then(|v| v.get("id"))
            .and_then(Value::as_u64)
            .map(|v| v as u32);
        let unk_token_id = root
            .get("unk_token")
            .and_then(|v| v.get("id"))
            .and_then(Value::as_u64)
            .map(|v| v as u32);

        // Fallback: scan added_tokens for special tokens if root-level is missing
        let bos_token_id = bos_token_id.or_else(|| {
            added_tokens
                .iter()
                .find(|t| {
                    t.special && matches!(t.content.as_str(), "<s>" | "<bos>" | "<|begin_of_text|>")
                })
                .map(|t| t.id)
        });
        let eos_token_id = eos_token_id.or_else(|| {
            added_tokens
                .iter()
                .find(|t| {
                    t.special
                        && matches!(
                            t.content.as_str(),
                            "</s>" | "<eos>" | "<|end_of_text|>" | "<|eot_id|>"
                        )
                })
                .map(|t| t.id)
        });
        let unk_token_id = unk_token_id.or_else(|| {
            added_tokens
                .iter()
                .find(|t| t.special && t.content == "<unk>")
                .map(|t| t.id)
        });

        Ok(Self {
            model,
            pre_tokenizer,
            post_processor,
            decoder,
            normalizer,
            added_tokens,
            added_token_by_id,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            truncation: None,
            padding: None,
        })
    }

    /// Configure truncation parameters.
    pub fn with_truncation(mut self, params: Option<TruncationParams>) -> Self {
        self.truncation = params;
        self
    }

    /// Configure padding parameters.
    pub fn with_padding(mut self, params: Option<PaddingParams>) -> Self {
        self.padding = params;
        self
    }

    /// Encode text into token IDs.
    ///
    /// `add_special_tokens` controls whether BOS/EOS (from the post-processor
    /// template) and configured special tokens are prepended/appended.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding, String> {
        // 1. Normalize
        let normalized = self.normalizer.normalize(text);

        // 2. Pre-tokenize
        let words = self.pre_tokenizer.pre_tokenize(&normalized);

        // 3. Model tokenize each word
        let mut encoding = Encoding::empty();
        for (word_idx, word) in words.iter().enumerate() {
            let tokens = self.model.tokenize(word);
            for token in &tokens {
                if let Some(id) = self.model.token_to_id(token) {
                    encoding.push(id, Some(word_idx as u32), false, 0);
                } else {
                    // Unknown token — use UNK if available
                    if let Some(unk_id) = self.unk_token_id {
                        encoding.push(unk_id, Some(word_idx as u32), true, 0);
                    }
                }
            }
        }

        // 4. Post-process (add special tokens like BOS/EOS from template)
        if add_special_tokens {
            let mut processed = Encoding::empty();
            let encoded_ids: Vec<u32> = encoding.ids.clone();
            let result = self.post_processor.post_process(
                &encoded_ids,
                &encoding,
                self.bos_token_id,
                self.eos_token_id,
            );
            // If the post-processor returns tokens, merge them
            if let Some((new_ids, new_type_ids, new_special_mask)) = result {
                processed.ids = new_ids;
                processed.attention_mask = vec![1; processed.ids.len()];
                processed.type_ids = new_type_ids;
                // Word ids shift — leave as None for special tokens
                let n_special = processed.ids.len() - encoding.ids.len();
                processed.word_ids = vec![None; n_special];
                processed.word_ids.extend(encoding.word_ids.iter().copied());
                processed.special_tokens_mask = new_special_mask;
                encoding = processed;
            }
        }

        // 5. Truncation
        if let Some(ref trunc) = self.truncation {
            let overflow = apply_truncation(&mut encoding, trunc);
            encoding.overflowing = overflow;
        }

        // 6. Padding
        if let Some(ref pad) = self.padding {
            apply_padding(&mut encoding, pad);
        }

        Ok(encoding)
    }

    /// Decode token IDs back into text.
    ///
    /// If `skip_special_tokens` is `true`, special tokens (BOS, EOS, UNK, and
    /// any added token marked as special) are excluded from the output.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, String> {
        let mut tokens: Vec<String> = Vec::new();

        for &id in ids {
            if skip_special_tokens {
                if self.bos_token_id == Some(id)
                    || self.eos_token_id == Some(id)
                    || self.unk_token_id == Some(id)
                {
                    continue;
                }
                if self.added_token_by_id.contains_key(&id)
                    && self.added_token_by_id[&id].special
                {
                    continue;
                }
            }

            match self.model.id_to_token(id) {
                Some(t) => tokens.push(t.to_string()),
                None => {
                    // Unknown id — skip in decode
                }
            }
        }

        let decoded = self.decoder.decode(&tokens);
        Ok(decoded)
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    /// Look up a token's ID.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        // Check added tokens first (special tokens can shadow model tokens)
        for at in &self.added_tokens {
            if at.content == token {
                return Some(at.id);
            }
        }
        self.model.token_to_id(token)
    }

    /// Look up an ID's token.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        // Check added tokens first
        if let Some(at) = self.added_token_by_id.get(&id) {
            return Some(&at.content);
        }
        self.model.id_to_token(id)
    }

    /// Get the BOS token ID.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Get the EOS token ID.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Get the UNK token ID.
    pub fn unk_token_id(&self) -> Option<u32> {
        self.unk_token_id
    }
}

// ════════════════════════════════════════════════════════════════════════════
// TribunusTokenizer — engine-side convenience wrapper (absorbed from
// compute-core/src/ecs/core/tokenizer.rs).
// ════════════════════════════════════════════════════════════════════════════

/// A HuggingFace-compatible tokenizer loaded from a directory containing
/// `tokenizer.json`. Thin wrapper around [`Tokenizer`] absorbed from the
/// engine's `compute-core/src/ecs/core/tokenizer.rs` to keep engine binaries'
/// public path stable (`tribunus_compute_core::tokenizer::TribunusTokenizer`).
#[derive(Debug, Clone)]
pub struct TribunusTokenizer {
    inner: Tokenizer,
}

impl TribunusTokenizer {
    /// Load a tokenizer from a directory containing `tokenizer.json`.
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let path = dir.join("tokenizer.json");
        if !path.exists() {
            return Err(format!("tokenizer file not found: {}", path.display()));
        }
        let inner = Tokenizer::from_file(&path)
            .map_err(|e| format!("failed to load tokenizer from {}: {}", path.display(), e))?;
        Ok(Self { inner })
    }

    /// Encode a prompt string into token IDs (u32).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| format!("tokenizer encode failed: {}", e))?;
        Ok(encoding.ids)
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, String> {
        let ids: Vec<u32> = tokens.to_vec();
        self.inner
            .decode(&ids, true)
            .map_err(|e| format!("tokenizer decode failed: {}", e))
    }
}

// ════════════════════════════════════════════════════════════════════════════
// GrammarTokenizer — minimal id→text loader absorbed from
// compute-core/src/ecs/parsing/tokenizer/mod.rs. Only the token_id → text
// mapping is needed for grammar masking, not the full pipeline.
// ════════════════════════════════════════════════════════════════════════════

/// Minimal tokenizer for grammar masking. Just the `token_id → text` mapping,
/// not the full tokenizer pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct GrammarTokenizer {
    /// token_id → decoded text
    pub id_to_text: Vec<String>,
}

impl GrammarTokenizer {
    /// Load tokenizer from a `tokenizer.json` file. Expects the standard
    /// HuggingFace tokenizer.json format with a `model.vocab` dictionary
    /// mapping strings to integers, or `added_tokens` for special tokens.
    pub fn load(tokenizer_path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(tokenizer_path)
            .map_err(|e| format!("failed to read tokenizer.json: {}", e))?;
        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("invalid tokenizer.json: {}", e))?;

        let mut id_to_text: Vec<String> = Vec::new();

        // Try model.vocab first (standard HF format)
        if let Some(vocab) = json.get("model").and_then(|m| m.get("vocab")) {
            if let Some(obj) = vocab.as_object() {
                for (_token, id_val) in obj {
                    if let Some(id) = id_val.as_u64() {
                        let id = id as usize;
                        if id >= id_to_text.len() {
                            id_to_text.resize(id + 1, String::new());
                        }
                        id_to_text[id] = _token.to_string();
                    }
                }
            }
        }

        // Also check for added_tokens
        if let Some(added) = json.get("added_tokens").and_then(|a| a.as_array()) {
            for entry in added {
                if let (Some(id), Some(content)) = (
                    entry.get("id").and_then(|v| v.as_u64()),
                    entry.get("content").and_then(|v| v.as_str()),
                ) {
                    let id = id as usize;
                    if id >= id_to_text.len() {
                        id_to_text.resize(id + 1, String::new());
                    }
                    id_to_text[id] = content.to_string();
                }
            }
        }

        if id_to_text.is_empty() {
            return Err("tokenizer.json has no vocabulary entries".to_string());
        }

        Ok(GrammarTokenizer { id_to_text })
    }

    /// Create a new tokenizer from an existing id→text mapping.
    pub fn new(id_to_text: Vec<String>) -> Self {
        GrammarTokenizer { id_to_text }
    }

    /// Decode a token ID to its text representation.
    pub fn decode(&self, token_id: u32) -> &str {
        let id = token_id as usize;
        if id < self.id_to_text.len() {
            &self.id_to_text[id]
        } else {
            ""
        }
    }

    /// The vocabulary size (number of known tokens).
    pub fn vocab_size(&self) -> usize {
        self.id_to_text.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_full_json() -> String {
        r#"{
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": {
                    "h": 0, "e": 1, "l": 2, "o": 3, " ": 4,
                    "w": 5, "r": 6, "d": 7, "he": 8, "llo": 9,
                    "hello": 10, "world": 11
                },
                "merges": [
                    "h e", "he l", "hel l", "hell o",
                    "w o", "wo r", "wor l", "worl d"
                ]
            },
            "added_tokens": [
                {"id": 12, "content": "<s>", "single_word": false, "special": true},
                {"id": 13, "content": "</s>", "single_word": false, "special": true},
                {"id": 14, "content": "<unk>", "single_word": false, "special": true}
            ],
            "bos_token": {"id": 12, "content": "<s>"},
            "eos_token": {"id": 13, "content": "</s>"},
            "unk_token": {"id": 14, "content": "<unk>"}
        }"#
        .to_string()
    }

    #[test]
    fn tokenizer_from_str_resolves_bos_eos_unk_from_added_tokens() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        assert_eq!(t.vocab_size(), 12);
        assert_eq!(t.bos_token_id(), Some(12));
        assert_eq!(t.eos_token_id(), Some(13));
        assert_eq!(t.unk_token_id(), Some(14));
    }

    #[test]
    fn encode_no_special_returns_only_model_ids() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let enc = t.encode("hello world", false).unwrap();
        assert_eq!(enc.ids, vec![10, 4, 11]);
        assert_eq!(enc.attention_mask, vec![1, 1, 1]);
        assert_eq!(enc.special_tokens_mask, vec![0, 0, 0]);
    }

    #[test]
    fn encode_with_special_wraps_with_bos_and_eos() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let enc = t.encode("hello world", true).unwrap();
        // No explicit template, so BOS/EOS are prepended/appended
        assert_eq!(enc.ids, vec![12, 10, 4, 11, 13]);
        assert_eq!(enc.attention_mask, vec![1, 1, 1, 1, 1]);
        assert_eq!(enc.special_tokens_mask, vec![1, 0, 0, 0, 1]);
    }

    #[test]
    fn decode_round_trip_without_specials() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let text = t.decode(&[10, 4, 11], false).unwrap();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn decode_skips_special_when_requested() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let text = t.decode(&[12, 10, 4, 11, 13], true).unwrap();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn encode_unknown_chars_use_unk() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let enc = t.encode("xyz", false).unwrap();
        assert_eq!(enc.ids, vec![14, 14, 14]);
    }

    #[test]
    fn encode_empty_string_is_empty() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let enc = t.encode("", false).unwrap();
        assert!(enc.ids.is_empty());
    }

    #[test]
    fn encode_cjk_chars_use_unk() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let enc = t.encode("你好", false).unwrap();
        assert_eq!(enc.ids, vec![14, 14]); // unknown chars
    }

    #[test]
    fn truncation_reduces_length_to_max() {
        let t = Tokenizer::from_str(&make_full_json())
            .unwrap()
            .with_truncation(Some(TruncationParams {
                max_length: 2,
                strategy: crate::engine::bpe_tokenizer::truncation_padding::TruncationStrategy::LongestFirst,
                stride: 0,
            }));
        let enc = t.encode("hello world", false).unwrap();
        assert_eq!(enc.ids.len(), 2);
    }

    #[test]
    fn padding_extends_to_multiple_with_right_pad() {
        let t = Tokenizer::from_str(&make_full_json())
            .unwrap()
            .with_padding(Some(PaddingParams {
                pad_token_id: 0,
                pad_token: "<pad>".to_string(),
                pad_to_multiple_of: Some(8),
                pad_left: false,
            }));
        let enc = t.encode("hello world", false).unwrap();
        assert_eq!(enc.ids.len(), 8); // 3 tokens → padded to 8
        assert_eq!(enc.attention_mask, vec![1, 1, 1, 0, 0, 0, 0, 0]);
        assert_eq!(enc.ids[3..], vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn padding_extends_to_multiple_with_left_pad() {
        let t = Tokenizer::from_str(&make_full_json())
            .unwrap()
            .with_padding(Some(PaddingParams {
                pad_token_id: 0,
                pad_token: "<pad>".to_string(),
                pad_to_multiple_of: Some(8),
                pad_left: true,
            }));
        let enc = t.encode("hello world", false).unwrap();
        assert_eq!(enc.ids.len(), 8);
        assert_eq!(enc.attention_mask, vec![0, 0, 0, 0, 0, 1, 1, 1]);
        assert_eq!(enc.ids[..5], vec![0, 0, 0, 0, 0]);
        assert_eq!(&enc.ids[5..], &[10, 4, 11]);
    }

    #[test]
    fn from_file_loads_and_round_trips() {
        let json = make_full_json();
        let dir = std::env::temp_dir().join(format!("tok_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, &json).unwrap();

        let t = Tokenizer::from_file(&path).unwrap();
        let enc = t.encode("hello world", true).unwrap();
        assert_eq!(enc.ids, vec![12, 10, 4, 11, 13]);
        let decoded = t.decode(&enc.ids, true).unwrap();
        assert_eq!(decoded, "hello world");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_str_supports_wordpiece_model_type() {
        let json = r#"{"model": {"type": "WordPiece", "vocab": {"[CLS]": 0}}}"#;
        let t = Tokenizer::from_str(json);
        assert!(t.is_ok());
    }

    #[test]
    fn from_str_rejects_unknown_model_type() {
        let json = r#"{"model": {"type": "FancyNewModel", "vocab": {}}}"#;
        let t = Tokenizer::from_str(json);
        assert!(t.is_err());
    }

    #[test]
    fn wordpiece_full_encode() {
        let json = r###"{
            "version": "1.0",
            "model": {
                "type": "WordPiece",
                "vocab": {
                    "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
                    "Hello": 3, ",": 4, "world": 5, "!": 6
                },
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##"
            },
            "pre_tokenizer": {"type": "BertPreTokenizer"}
        }"###;
        let t = Tokenizer::from_str(json).unwrap();
        let enc = t.encode("Hello, world!", false).unwrap();
        assert_eq!(enc.ids, vec![3, 4, 5, 6]);
    }

    #[test]
    fn no_overflow_when_input_fits_max_length() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let enc = t.encode("hello world", false).unwrap();
        assert!(enc.overflowing.is_empty());
    }

    #[test]
    fn word_ids_track_pre_tokenized_word_index() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let enc = t.encode("hello world", false).unwrap();
        // "hello" → word 0, " " → word 1, "world" → word 2
        assert!(enc.word_ids.iter().all(|w| w.is_some()));
        assert_eq!(enc.word_ids[0], Some(0));
        assert_eq!(enc.word_ids[1], Some(1));
        assert_eq!(enc.word_ids[2], Some(2));
    }

    // ── TribunusTokenizer tests ──

    #[test]
    fn tribunus_tokenizer_wraps_inner_tokenizer() {
        let json = make_full_json();
        let dir = std::env::temp_dir().join(format!("trib_tok_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, &json).unwrap();

        let t = TribunusTokenizer::from_dir(&dir).unwrap();
        let ids = t.encode("hello world").unwrap();
        // TribunusTokenizer::encode does not add special tokens
        assert_eq!(ids, vec![10, 4, 11]);
        let text = t.decode(&ids).unwrap();
        assert_eq!(text, "hello world");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tribunus_tokenizer_from_missing_dir_errors() {
        let dir = std::env::temp_dir().join(format!("trib_missing_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let result = TribunusTokenizer::from_dir(&dir);
        assert!(result.is_err());
    }

    // ── GrammarTokenizer tests ──

    #[test]
    fn grammar_tokenizer_new_round_trips_id_to_text() {
        let tokenizer = GrammarTokenizer::new(vec![
            "hello".to_string(),
            "world".to_string(),
            " ".to_string(),
            "a".to_string(),
        ]);
        assert_eq!(tokenizer.decode(0), "hello");
        assert_eq!(tokenizer.decode(1), "world");
        assert_eq!(tokenizer.decode(3), "a");
        assert_eq!(tokenizer.decode(99), ""); // out of range
        assert_eq!(tokenizer.vocab_size(), 4);
    }

    #[test]
    fn grammar_tokenizer_loads_vocab_from_json() {
        let json = serde_json::json!({
            "model": {
                "vocab": {
                    "hello": 0,
                    "world": 1
                }
            }
        });
        let dir = std::env::temp_dir().join(format!("grammar_tok_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, serde_json::to_string(&json).unwrap()).unwrap();

        let t = GrammarTokenizer::load(&path).unwrap();
        assert_eq!(t.decode(0), "hello");
        assert_eq!(t.decode(1), "world");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn grammar_tokenizer_load_includes_added_tokens() {
        let json = serde_json::json!({
            "model": {
                "vocab": {
                    "hello": 0,
                    "world": 1
                }
            },
            "added_tokens": [
                {"id": 2, "content": "<pad>"}
            ]
        });
        let dir = std::env::temp_dir().join(format!("grammar_tok_added_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, serde_json::to_string(&json).unwrap()).unwrap();

        let t = GrammarTokenizer::load(&path).unwrap();
        assert_eq!(t.decode(0), "hello");
        assert_eq!(t.decode(1), "world");
        assert_eq!(t.decode(2), "<pad>");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn grammar_tokenizer_rejects_empty_vocab() {
        let json = serde_json::json!({
            "model": {
                "vocab": {}
            }
        });
        let dir = std::env::temp_dir().join(format!("grammar_tok_empty_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, serde_json::to_string(&json).unwrap()).unwrap();

        let result = GrammarTokenizer::load(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}

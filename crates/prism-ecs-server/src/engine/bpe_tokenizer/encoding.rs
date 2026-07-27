//! `Encoding` — the canonical result of encoding a single input string.
//!
//! This module owns the canonical shape of an encoded token sequence and its
//! per-position masks. It does not own model types, pre-tokenization,
//! post-processing, or any pipeline orchestration.
//!
//! Ordering: `ids[i]`, `attention_mask[i]`, `type_ids[i]`, `word_ids[i]`,
//! `special_tokens_mask[i]` all index the same position. `overflowing` carries
//! additional encodings when truncation splits the input into overlapping
//! windows.

use serde_json::Value;

/// Full encoding result: token IDs, masks, and overflow information.
#[derive(Debug, Clone)]
pub struct Encoding {
    /// Token ID sequence, in order.
    pub ids: Vec<u32>,
    /// One bit per position (1 = real token, 0 = padding).
    pub attention_mask: Vec<u32>,
    /// Segment / type ids (e.g., 0 for prompt A, 1 for prompt B).
    pub type_ids: Vec<u32>,
    /// Maps each output token back to its original pre-tokenized word.
    /// `None` for special tokens appended by the post-processor.
    pub word_ids: Vec<Option<u32>>,
    /// 1 at positions that are special tokens (BOS, EOS, etc.).
    pub special_tokens_mask: Vec<u32>,
    /// Overflow encodings when truncation splits the input.
    pub overflowing: Vec<Encoding>,
}

impl Encoding {
    /// Build an empty encoding. Internal constructor for the pipeline.
    pub(crate) fn empty() -> Self {
        Self {
            ids: Vec::new(),
            attention_mask: Vec::new(),
            type_ids: Vec::new(),
            word_ids: Vec::new(),
            special_tokens_mask: Vec::new(),
            overflowing: Vec::new(),
        }
    }

    /// Append a single token position to the encoding. Internal helper used by
    /// the model-tokenize and post-process pipeline stages.
    pub(crate) fn push(&mut self, id: u32, word_id: Option<u32>, is_special: bool, type_id: u32) {
        self.ids.push(id);
        self.attention_mask.push(1);
        self.type_ids.push(type_id);
        self.word_ids.push(word_id);
        self.special_tokens_mask
            .push(if is_special { 1 } else { 0 });
    }
}

/// Parse an `AddedToken` entry from a `tokenizer.json` `added_tokens` array.
///
/// The shape lives here (alongside `Encoding`) because added tokens are the
/// inverse direction of encoding: they are the canonical mapping between
/// token text and the position the model will see after post-processing.
#[derive(Debug, Clone)]
pub struct AddedToken {
    /// Authority-bearing token id assigned by the model.
    pub id: u32,
    /// The literal text the token represents.
    pub content: String,
    /// True for BOS/EOS/SEP/CLS — they get `skip_special_tokens` treatment.
    pub special: bool,
    /// If true, treat the token as atomic (never split by the pre-tokenizer).
    pub single_word: bool,
    /// If true, strip a leading space when matching.
    pub lstrip: bool,
    /// If true, strip a trailing space when matching.
    pub rstrip: bool,
    /// If true, normalize the token through the configured normalizer.
    pub normalized: bool,
}

impl AddedToken {
    /// Parse from a HuggingFace `added_tokens` JSON entry.
    pub(crate) fn from_json(v: &Value) -> Result<Self, String> {
        let id = v
            .get("id")
            .and_then(Value::as_u64)
            .ok_or_else(|| "added_token missing id".to_string())? as u32;
        let content = v
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| "added_token missing content".to_string())?
            .to_string();
        let special = v.get("special").and_then(Value::as_bool).unwrap_or(false);
        let single_word = v
            .get("single_word")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let lstrip = v.get("lstrip").and_then(Value::as_bool).unwrap_or(false);
        let rstrip = v.get("rstrip").and_then(Value::as_bool).unwrap_or(false);
        let normalized = v.get("normalized").and_then(Value::as_bool).unwrap_or(true);
        Ok(Self {
            id,
            content,
            special,
            single_word,
            lstrip,
            rstrip,
            normalized,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_encoding_is_all_empty() {
        let e = Encoding::empty();
        assert!(e.ids.is_empty());
        assert!(e.attention_mask.is_empty());
        assert!(e.type_ids.is_empty());
        assert!(e.word_ids.is_empty());
        assert!(e.special_tokens_mask.is_empty());
        assert!(e.overflowing.is_empty());
    }

    #[test]
    fn push_advances_all_masks_in_lockstep() {
        let mut e = Encoding::empty();
        e.push(7, Some(0), false, 0);
        e.push(8, Some(1), true, 0);
        assert_eq!(e.ids, vec![7, 8]);
        assert_eq!(e.attention_mask, vec![1, 1]);
        assert_eq!(e.word_ids, vec![Some(0), Some(1)]);
        assert_eq!(e.special_tokens_mask, vec![0, 1]);
    }

    #[test]
    fn added_token_defaults_when_optional_fields_absent() {
        let v = serde_json::json!({"id": 5, "content": "<pad>"});
        let at = AddedToken::from_json(&v).unwrap();
        assert_eq!(at.id, 5);
        assert_eq!(at.content, "<pad>");
        assert!(!at.special);
        assert!(!at.single_word);
        assert!(!at.lstrip);
        assert!(!at.rstrip);
        // `normalized` defaults to `true` in HuggingFace convention
        assert!(at.normalized);
    }

    #[test]
    fn added_token_rejects_missing_id() {
        let v = serde_json::json!({"content": "<pad>"});
        assert!(AddedToken::from_json(&v).is_err());
    }

    #[test]
    fn added_token_rejects_missing_content() {
        let v = serde_json::json!({"id": 5});
        assert!(AddedToken::from_json(&v).is_err());
    }
}

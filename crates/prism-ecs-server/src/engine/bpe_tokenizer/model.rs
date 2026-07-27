//! Model types — BPE, WordPiece, and Unigram subword models.
//!
//! This module owns the canonical authority for subword model construction
//! from `tokenizer.json`. It does not own pre-tokenization, normalization,
//! post-processing, or the top-level `Tokenizer` orchestration.
//!
//! Each model exposes the same three operations: `tokenize(word) -> Vec<String>`,
//! `token_to_id(token)`, `id_to_token(id) -> Option<&str>`. The `ModelKind`
//! enum dispatches to the right backend.

use serde_json::Value;
use std::collections::HashMap;

// WAIVER: HashMap is correct for `vocab` and `merges` lookups. These are
// stable-identity lookup tables (token string -> id, merge pair -> rank)
// whose iteration order is never observable. `Encoding.ids` is the observable
// output and is order-stable. Re-sorted iteration over the vocab/merges is
// never required.

// ════════════════════════════════════════════════════════════════════════════
// ModelKind — dispatcher
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub(crate) enum ModelKind {
    Bpe(BpeModel),
    WordPiece(WordPieceModel),
    Unigram(UnigramModel),
}

impl ModelKind {
    /// Parse a `ModelKind` from the `model` field of `tokenizer.json`.
    pub(crate) fn from_json(model_type: &str, v: &Value) -> Result<Self, String> {
        match model_type {
            "BPE" => Ok(ModelKind::Bpe(BpeModel::from_json(v)?)),
            "WordPiece" | "WordPieceModel" => {
                Ok(ModelKind::WordPiece(WordPieceModel::from_json(v)?))
            }
            "Unigram" => Ok(ModelKind::Unigram(UnigramModel::from_json(v)?)),
            other => Err(format!("unsupported model type: {other}")),
        }
    }

    /// Tokenize a single pre-tokenized word into model-level tokens.
    pub(crate) fn tokenize(&self, word: &str) -> Vec<String> {
        match self {
            ModelKind::Bpe(m) => m.tokenize(word),
            ModelKind::WordPiece(m) => m.tokenize(word),
            ModelKind::Unigram(m) => m.tokenize(word),
        }
    }

    /// Resolve a token string to its vocabulary id.
    pub(crate) fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            ModelKind::Bpe(m) => m.token_to_id(token),
            ModelKind::WordPiece(m) => m.token_to_id(token),
            ModelKind::Unigram(m) => m.token_to_id(token),
        }
    }

    /// Resolve a vocabulary id to its token string.
    pub(crate) fn id_to_token(&self, id: u32) -> Option<&str> {
        match self {
            ModelKind::Bpe(m) => m.id_to_token(id),
            ModelKind::WordPiece(m) => m.id_to_token(id),
            ModelKind::Unigram(m) => m.id_to_token(id),
        }
    }

    /// Number of entries in the vocabulary.
    pub(crate) fn vocab_size(&self) -> usize {
        match self {
            ModelKind::Bpe(m) => m.vocab_size(),
            ModelKind::WordPiece(m) => m.vocab_size(),
            ModelKind::Unigram(m) => m.vocab_size(),
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// BPE — Byte Pair Encoding
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub(crate) struct BpeModel {
    pub(crate) vocab: HashMap<String, u32>,
    id_to_token: Vec<String>,
    merges: HashMap<(String, String), u32>,
    /// Whether to use byte-level internal encoding (GPT-2 style)
    byte_fallback: bool,
    /// Unicode-to-byte reverse mapping for byte_fallback decoding
    byte_decoder: Option<HashMap<char, u8>>,
}

impl BpeModel {
    pub(crate) fn from_json(v: &Value) -> Result<Self, String> {
        let vocab_obj = v
            .get("vocab")
            .ok_or_else(|| "BPE model missing 'vocab'".to_string())?
            .as_object()
            .ok_or_else(|| "BPE vocab is not an object".to_string())?;

        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut max_id = 0u32;
        for (token, id_val) in vocab_obj {
            let id = id_val
                .as_u64()
                .ok_or_else(|| format!("invalid vocab id for token '{}'", token))?
                as u32;
            vocab.insert(token.clone(), id);
            if id > max_id {
                max_id = id;
            }
        }

        let mut id_to_token: Vec<String> = Vec::with_capacity(max_id as usize + 1);
        id_to_token.resize(max_id as usize + 1, String::new());
        for (token, &id) in &vocab {
            if (id as usize) < id_to_token.len() {
                id_to_token[id as usize] = token.clone();
            }
        }

        let _drop_out = v.get("dropout").and_then(Value::as_f64).unwrap_or(0.0);

        let byte_fallback = v
            .get("byte_fallback")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let mut merges: HashMap<(String, String), u32> = HashMap::new();
        if let Some(merges_arr) = v.get("merges").and_then(Value::as_array) {
            for (i, entry) in merges_arr.iter().enumerate() {
                let line = entry
                    .as_str()
                    .ok_or_else(|| format!("merges[{}] is not a string", i))?;
                if let Some(space) = line.rfind(' ') {
                    let left = line[..space].to_string();
                    let right = line[space + 1..].to_string();
                    merges.insert((left, right), i as u32);
                }
            }
        }

        // Build byte_decoder for byte_fallback mode
        let byte_decoder = if byte_fallback {
            Some(
                bytes_to_unicode()
                    .into_iter()
                    .map(|(b, c)| (c, b))
                    .collect(),
            )
        } else {
            None
        };

        Ok(Self {
            vocab,
            id_to_token,
            merges,
            byte_fallback,
            byte_decoder,
        })
    }

    pub(crate) fn tokenize(&self, word: &str) -> Vec<String> {
        // If the whole word is already in vocab, return it immediately
        if self.vocab.contains_key(word) {
            return vec![word.to_string()];
        }

        if self.byte_fallback {
            // Byte-level BPE (GPT-2 style): convert each byte to its unicode char
            let byte_encoding: String = word
                .as_bytes()
                .into_iter()
                .map(|&b| bytes_to_unicode().get(&b).copied().unwrap_or(b as char))
                .collect();
            self.bpe_merge(&byte_encoding)
        } else {
            // Character-level BPE: start with individual chars
            self.bpe_merge(word)
        }
    }

    fn bpe_merge(&self, word: &str) -> Vec<String> {
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if tokens.len() <= 1 {
            return tokens;
        }

        loop {
            let mut best_rank = u32::MAX;
            let mut best_i: Option<usize> = None;

            for i in 0..tokens.len().saturating_sub(1) {
                let key = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&rank) = self.merges.get(&key) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = Some(i);
                    }
                }
            }

            let i = match best_i {
                Some(i) => i,
                None => break,
            };

            tokens[i] = format!("{}{}", tokens[i], tokens[i + 1]);
            tokens.remove(i + 1);
            if tokens.len() <= 1 {
                break;
            }
        }

        tokens
    }

    pub(crate) fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    pub(crate) fn id_to_token(&self, id: u32) -> Option<&str> {
        let idx = id as usize;
        if idx < self.id_to_token.len() && !self.id_to_token[idx].is_empty() {
            Some(self.id_to_token[idx].as_str())
        } else {
            None
        }
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Decode a token's unicode mapping back to raw bytes for byte-level BPE.
    /// Returns `None` if the model is not in `byte_fallback` mode or if no
    /// characters in the token map back to bytes.
    pub(crate) fn decode_bytes(&self, token: &str) -> Option<Vec<u8>> {
        if let Some(ref decoder) = self.byte_decoder {
            let bytes: Vec<u8> = token
                .chars()
                .filter_map(|c| decoder.get(&c).copied())
                .collect();
            if !bytes.is_empty() {
                return Some(bytes);
            }
        }
        None
    }
}

// ════════════════════════════════════════════════════════════════════════════
// WordPiece
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub(crate) struct WordPieceModel {
    pub(crate) vocab: HashMap<String, u32>,
    id_to_token: Vec<String>,
    unk_token: String,
    unk_id: u32,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
}

impl WordPieceModel {
    pub(crate) fn from_json(v: &Value) -> Result<Self, String> {
        let vocab_obj = v
            .get("vocab")
            .ok_or_else(|| "WordPiece model missing 'vocab'".to_string())?
            .as_object()
            .ok_or_else(|| "WordPiece vocab is not an object".to_string())?;

        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut max_id = 0u32;
        for (token, id_val) in vocab_obj {
            let id = id_val
                .as_u64()
                .ok_or_else(|| format!("invalid vocab id for token '{}'", token))?
                as u32;
            vocab.insert(token.clone(), id);
            if id > max_id {
                max_id = id;
            }
        }

        let mut id_to_token: Vec<String> = Vec::with_capacity(max_id as usize + 1);
        id_to_token.resize(max_id as usize + 1, String::new());
        for (token, &id) in &vocab {
            if (id as usize) < id_to_token.len() {
                id_to_token[id as usize] = token.clone();
            }
        }

        let unk_token = v
            .get("unk_token")
            .and_then(Value::as_str)
            .unwrap_or("[UNK]")
            .to_string();
        let unk_id = vocab.get(&unk_token).copied().unwrap_or(0);
        let continuing_subword_prefix = v
            .get("continuing_subword_prefix")
            .and_then(Value::as_str)
            .unwrap_or("##")
            .to_string();
        let max_input_chars_per_word = v
            .get("max_input_chars_per_word")
            .and_then(Value::as_u64)
            .unwrap_or(100) as usize;

        Ok(Self {
            vocab,
            id_to_token,
            unk_token,
            unk_id,
            continuing_subword_prefix,
            max_input_chars_per_word,
        })
    }

    pub(crate) fn tokenize(&self, word: &str) -> Vec<String> {
        if word.len() > self.max_input_chars_per_word || word.is_empty() {
            return vec![self.unk_token.clone()];
        }

        if self.vocab.contains_key(word) {
            return vec![word.to_string()];
        }

        // Greedy longest-match segmentation (forward maximum matching)
        let chars: Vec<char> = word.chars().collect();
        let n = chars.len();
        let mut tokens: Vec<String> = Vec::new();
        let mut i = 0;

        while i < n {
            let mut found = false;
            let max_j = n.saturating_sub(i).min(self.max_input_chars_per_word);

            // Try longest substring first
            for j in (1..=max_j).rev() {
                let sub: String = chars[i..i + j].iter().collect();
                let candidate = if i > 0 {
                    // Subword starting from second character onwards uses prefix
                    format!("{}{}", self.continuing_subword_prefix, sub)
                } else {
                    sub
                };

                if self.vocab.contains_key(&candidate) {
                    tokens.push(candidate);
                    i += j;
                    found = true;
                    break;
                }
            }

            if !found {
                // If no subword found at the start of the word, the whole word
                // maps to [UNK] (standard WordPiece behavior).
                if i == 0 {
                    return vec![self.unk_token.clone()];
                }
                tokens.push(self.unk_token.clone());
                i += 1; // Skip one char
            }
        }

        tokens
    }

    pub(crate) fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    pub(crate) fn id_to_token(&self, id: u32) -> Option<&str> {
        let idx = id as usize;
        if idx < self.id_to_token.len() && !self.id_to_token[idx].is_empty() {
            Some(self.id_to_token[idx].as_str())
        } else {
            None
        }
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Unigram
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub(crate) struct UnigramModel {
    /// (token, score) sorted by score descending
    pieces: Vec<(String, f64)>,
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, u32>,
    unk_id: u32,
    unk_token: String,
}

impl UnigramModel {
    pub(crate) fn from_json(v: &Value) -> Result<Self, String> {
        let vocab_arr = v
            .get("vocab")
            .ok_or_else(|| "Unigram model missing 'vocab'".to_string())?
            .as_array()
            .ok_or_else(|| "Unigram vocab is not an array".to_string())?;

        let unk_token = v
            .get("unk_token")
            .and_then(Value::as_str)
            .unwrap_or("<unk>")
            .to_string();
        let unk_id = v
            .get("unk_id")
            .and_then(Value::as_u64)
            .map(|v| v as u32)
            .unwrap_or(0);

        let mut pieces: Vec<(String, f64)> = Vec::with_capacity(vocab_arr.len());
        let mut token_to_id: HashMap<String, u32> = HashMap::new();

        for (i, entry) in vocab_arr.iter().enumerate() {
            if let Some(obj) = entry.as_object() {
                let token = obj
                    .get("0")
                    .or_else(|| obj.get("piece"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let score = obj
                    .get("1")
                    .or_else(|| obj.get("score"))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                pieces.push((token.to_string(), score));
                token_to_id.insert(token.to_string(), i as u32);
            } else if let Some(arr) = entry.as_array() {
                if arr.len() >= 2 {
                    let token = arr[0].as_str().unwrap_or("");
                    let score = arr[1].as_f64().unwrap_or(0.0);
                    pieces.push((token.to_string(), score));
                    token_to_id.insert(token.to_string(), i as u32);
                }
            }
        }

        // Sort by score descending (higher score = higher probability)
        pieces.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut id_to_token: Vec<String> = Vec::with_capacity(pieces.len());
        for (token, _) in &pieces {
            id_to_token.push(token.clone());
        }

        let unk_id = token_to_id.get(&unk_token).copied().unwrap_or(unk_id);

        Ok(Self {
            pieces,
            id_to_token,
            token_to_id,
            unk_id,
            unk_token,
        })
    }

    /// Encode a word using Viterbi DP over the unigram lattice.
    pub(crate) fn tokenize(&self, word: &str) -> Vec<String> {
        if word.is_empty() || self.pieces.is_empty() {
            return vec![self.unk_token.clone()];
        }

        let chars: Vec<char> = word.chars().collect();
        let n = chars.len();

        // Build a char-position-based lattice.
        // dp[i] = (best_score, best_prev_i, best_token_output)
        // where output tokens are collected on the forward pass.
        //
        // We use log-probabilities: score = log(P(piece))
        // dp[i] = max over j < i of dp[j].0 + score of chars[j..i]

        let mut dp: Vec<(f64, Option<usize>, Option<&str>)> =
            vec![(f64::NEG_INFINITY, None, None); n + 1];
        dp[0] = (0.0, None, None);

        for end in 1..=n {
            // Limit search window to avoid O(n²) on long words
            let start = if end > 50 { end - 50 } else { 0 };
            for start_pos in start..end {
                let sub: String = chars[start_pos..end].iter().collect();
                if let Some(&id) = self.token_to_id.get(&sub) {
                    let score = self.pieces[id as usize].1; // log-prob (negative)
                    let prev_score = dp[start_pos].0;
                    let candidate = prev_score + score;
                    if candidate > dp[end].0 {
                        dp[end] = (
                            candidate,
                            Some(start_pos),
                            Some(&self.id_to_token[id as usize]),
                        );
                    }
                }
            }
        }

        // Backtrack
        let mut tokens: Vec<String> = Vec::new();
        let mut pos = n;
        while pos > 0 {
            if let Some((_, Some(prev), Some(tok))) = dp.get(pos) {
                tokens.push(tok.to_string());
                pos = *prev;
            } else {
                // No match — use UNK for this character
                // Try to find the first character as UNK
                let single = chars[pos - 1].to_string();
                if let Some(&id) = self.token_to_id.get(&single) {
                    tokens.push(self.id_to_token[id as usize].clone());
                } else {
                    tokens.push(self.unk_token.clone());
                }
                pos -= 1;
            }
        }

        tokens.reverse();
        tokens
    }

    pub(crate) fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub(crate) fn id_to_token(&self, id: u32) -> Option<&str> {
        let idx = id as usize;
        if idx < self.id_to_token.len() {
            Some(self.id_to_token[idx].as_str())
        } else {
            None
        }
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// bytes_to_unicode — GPT-2 byte-to-unicode mapping
// ════════════════════════════════════════════════════════════════════════════

/// Build byte-to-unicode mapping (GPT-2 style). Printable ASCII bytes and
/// Latin-1 supplement bytes map to themselves; everything else maps to a
/// sequential code point starting at U+0100.
pub(crate) fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut map = HashMap::new();
    let mut n = 0u32;
    for b in 0..=255u8 {
        if matches!(b, 33..=126 | 161..=172 | 174..=255) {
            map.insert(b, char::from_u32(b as u32).unwrap_or('\u{FFFD}'));
        } else {
            map.insert(b, char::from_u32(256 + n).unwrap_or('\u{FFFD}'));
            n += 1;
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── BPE tests ──

    fn build_simple_bpe() -> BpeModel {
        let mut vocab: HashMap<String, u32> = HashMap::new();
        vocab.insert("h".to_string(), 0);
        vocab.insert("e".to_string(), 1);
        vocab.insert("l".to_string(), 2);
        vocab.insert("o".to_string(), 3);
        vocab.insert(" ".to_string(), 4);
        vocab.insert("w".to_string(), 5);
        vocab.insert("r".to_string(), 6);
        vocab.insert("d".to_string(), 7);
        vocab.insert("he".to_string(), 8);
        vocab.insert("llo".to_string(), 9);
        vocab.insert("hello".to_string(), 10);
        vocab.insert("world".to_string(), 11);

        let max_id = 11u32;
        let mut id_to_token: Vec<String> = Vec::with_capacity(max_id as usize + 1);
        id_to_token.resize(max_id as usize + 1, String::new());
        for (token, &id) in &vocab {
            id_to_token[id as usize] = token.clone();
        }

        let mut merges: HashMap<(String, String), u32> = HashMap::new();
        merges.insert(("h".to_string(), "e".to_string()), 0);
        merges.insert(("he".to_string(), "l".to_string()), 1);
        merges.insert(("hel".to_string(), "l".to_string()), 2);
        merges.insert(("hell".to_string(), "o".to_string()), 3);
        merges.insert(("w".to_string(), "o".to_string()), 4);
        merges.insert(("wo".to_string(), "r".to_string()), 5);
        merges.insert(("wor".to_string(), "l".to_string()), 6);
        merges.insert(("worl".to_string(), "d".to_string()), 7);

        BpeModel {
            vocab,
            id_to_token,
            merges,
            byte_fallback: false,
            byte_decoder: None,
        }
    }

    #[test]
    fn bpe_whole_word_in_vocab_returns_singleton() {
        let bpe = build_simple_bpe();
        assert_eq!(bpe.tokenize("hello"), vec!["hello".to_string()]);
        assert_eq!(bpe.tokenize("world"), vec!["world".to_string()]);
    }

    #[test]
    fn bpe_unknown_word_emits_chars_when_no_merges_match() {
        let bpe = build_simple_bpe();
        assert_eq!(bpe.tokenize("xyz"), vec!["x".to_string(), "y".to_string(), "z".to_string()]);
    }

    #[test]
    fn bpe_vocab_and_id_to_token_roundtrip() {
        let bpe = build_simple_bpe();
        assert_eq!(bpe.token_to_id("hello"), Some(10));
        assert_eq!(bpe.id_to_token(10), Some("hello"));
        assert_eq!(bpe.id_to_token(11), Some("world"));
        assert_eq!(bpe.vocab_size(), 12);
        // Unmapped slot in id_to_token returns None
        assert!(bpe.id_to_token(999).is_none());
    }

    // ── WordPiece tests ──

    fn build_wordpiece() -> WordPieceModel {
        let mut vocab: HashMap<String, u32> = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("[CLS]".to_string(), 1);
        vocab.insert("[SEP]".to_string(), 2);
        vocab.insert("Hello".to_string(), 3);
        vocab.insert("##ing".to_string(), 4);
        vocab.insert("play".to_string(), 5);

        let max_id = 5u32;
        let mut id_to_token: Vec<String> = Vec::with_capacity(max_id as usize + 1);
        id_to_token.resize(max_id as usize + 1, String::new());
        for (token, &id) in &vocab {
            id_to_token[id as usize] = token.clone();
        }

        WordPieceModel {
            vocab,
            id_to_token,
            unk_token: "[UNK]".to_string(),
            unk_id: 0,
            continuing_subword_prefix: "##".to_string(),
            max_input_chars_per_word: 100,
        }
    }

    #[test]
    fn wordpiece_whole_word_in_vocab() {
        let wp = build_wordpiece();
        assert_eq!(wp.tokenize("Hello"), vec!["Hello".to_string()]);
    }

    #[test]
    fn wordpiece_subword_segmentation() {
        let wp = build_wordpiece();
        assert_eq!(
            wp.tokenize("playing"),
            vec!["play".to_string(), "##ing".to_string()]
        );
    }

    #[test]
    fn wordpiece_unknown_whole_word_yields_unk() {
        let wp = build_wordpiece();
        assert_eq!(wp.tokenize("xyz"), vec!["[UNK]".to_string()]);
    }

    // ── Unigram tests ──

    fn build_unigram_json() -> Value {
        serde_json::json!({
            "type": "Unigram",
            "vocab": [
                {"0": "<unk>", "1": 0.0},
                {"0": "hello", "1": -1.0},
                {"0": "world", "1": -2.0},
                {"0": "h", "1": -3.0},
                {"0": "e", "1": -3.0},
                {"0": "l", "1": -3.0},
                {"0": "o", "1": -3.0},
                {"0": "w", "1": -4.0},
                {"0": "r", "1": -4.0},
                {"0": "d", "1": -4.0}
            ],
            "unk_token": "<unk>",
            "unk_id": 0
        })
    }

    #[test]
    fn unigram_chooses_highest_scoring_piece() {
        let model = UnigramModel::from_json(&build_unigram_json()).unwrap();
        assert_eq!(model.tokenize("hello"), vec!["hello".to_string()]);
    }

    #[test]
    fn unigram_falls_back_to_unk_for_unseen_chars() {
        let model = UnigramModel::from_json(&build_unigram_json()).unwrap();
        let tokens = model.tokenize("xyz");
        assert!(tokens.iter().any(|t| t == "<unk>"));
    }

    // ── ModelKind dispatch tests ──

    #[test]
    fn model_kind_rejects_unknown_type() {
        let v = serde_json::json!({"vocab": {}});
        let err = ModelKind::from_json("FancyNewModel", &v).unwrap_err();
        assert!(err.contains("unsupported model type"));
    }

    #[test]
    fn model_kind_dispatches_bpe() {
        let v = serde_json::json!({
            "type": "BPE",
            "vocab": {"a": 0, "b": 1},
            "merges": []
        });
        let m = ModelKind::from_json("BPE", &v).unwrap();
        assert_eq!(m.vocab_size(), 2);
    }

    #[test]
    fn model_kind_dispatches_wordpiece_alias() {
        let v = serde_json::json!({
            "vocab": {"a": 0},
            "unk_token": "[UNK]"
        });
        let m = ModelKind::from_json("WordPieceModel", &v).unwrap();
        assert_eq!(m.vocab_size(), 1);
    }

    // ── bytes_to_unicode tests ──

    #[test]
    fn bytes_to_unicode_covers_all_256_bytes() {
        let m = bytes_to_unicode();
        assert_eq!(m.len(), 256);
        for b in 0..=255u8 {
            assert!(m.contains_key(&b), "missing mapping for byte {}", b);
        }
    }

    #[test]
    fn bytes_to_unicode_preserves_printable_ascii() {
        let m = bytes_to_unicode();
        // ASCII 'A' is 65 → maps to itself
        assert_eq!(m.get(&b'A').copied(), Some('A'));
        assert_eq!(m.get(&b'z').copied(), Some('z'));
    }
}

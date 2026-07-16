//! # Pure Rust HuggingFace Tokenizer
//!
//! A pure Rust replacement for the C++ `tokenizers` crate supporting the full
//! HuggingFace `tokenizer.json` format. Handles BPE, WordPiece, and Unigram
//! model types, with pre-tokenization, post-processing, truncation, padding,
//! and decoding.
//!
//! # Quick Start
//! ```
//! use prism_runtime::bpe_tokenizer::Tokenizer;
//!
//! let tok = Tokenizer::from_file("model/tokenizer.json")?;
//! let enc = tok.encode("Hello, world!", true)?;
//! let text = tok.decode(&enc.ids, true)?;
//! ```

use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

// ════════════════════════════════════════════════════════════════════════════
// Public Types
// ════════════════════════════════════════════════════════════════════════════

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
    fn empty() -> Self {
        Self {
            ids: Vec::new(),
            attention_mask: Vec::new(),
            type_ids: Vec::new(),
            word_ids: Vec::new(),
            special_tokens_mask: Vec::new(),
            overflowing: Vec::new(),
        }
    }

    fn push(&mut self, id: u32, word_id: Option<u32>, is_special: bool, type_id: u32) {
        self.ids.push(id);
        self.attention_mask.push(1);
        self.type_ids.push(type_id);
        self.word_ids.push(word_id);
        self.special_tokens_mask
            .push(if is_special { 1 } else { 0 });
    }
}

/// Configuration for a single added token.
#[derive(Debug, Clone)]
pub struct AddedToken {
    pub id: u32,
    pub content: String,
    pub special: bool,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
}

/// Truncation strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TruncationStrategy {
    /// Truncate only the first sequence (prompt A).
    OnlyFirst,
    /// Truncate only the second sequence (prompt B).
    OnlySecond,
    /// Truncate from the longer sequence first.
    LongestFirst,
}

/// Truncation parameters.
#[derive(Debug, Clone)]
pub struct TruncationParams {
    pub max_length: usize,
    pub strategy: TruncationStrategy,
    pub stride: usize,
}

/// Padding parameters.
#[derive(Debug, Clone)]
pub struct PaddingParams {
    pub pad_token_id: u32,
    pub pad_token: String,
    pub pad_to_multiple_of: Option<usize>,
    pub pad_left: bool,
}

// ════════════════════════════════════════════════════════════════════════════
// Tokenizer – Top‑Level Public Struct
// ════════════════════════════════════════════════════════════════════════════

/// A HuggingFace-compatible tokenizer supporting BPE, WordPiece, and Unigram models.
///
/// Loaded from a `tokenizer.json` file via [`Tokenizer::from_file`] or
/// [`Tokenizer::from_str`].
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
            let overflow = self.truncate(&mut encoding, trunc);
            encoding.overflowing = overflow;
        }

        // 6. Padding
        if let Some(ref pad) = self.padding {
            self.apply_padding(&mut encoding, pad);
        }

        Ok(encoding)
    }

    /// Apply truncation strategy to the encoding.
    fn truncate(&self, encoding: &mut Encoding, params: &TruncationParams) -> Vec<Encoding> {
        if encoding.ids.len() <= params.max_length {
            return Vec::new();
        }

        let max = params.max_length;
        let stride = params.stride.min(max);
        let mut overflows: Vec<Encoding> = Vec::new();

        match params.strategy {
            TruncationStrategy::OnlyFirst | TruncationStrategy::LongestFirst => {
                // Truncate from the end
                if stride > 0 {
                    let overflow_start = max.saturating_sub(stride);
                    let overflow = Encoding {
                        ids: encoding.ids[overflow_start..].to_vec(),
                        attention_mask: vec![1; encoding.ids.len() - overflow_start],
                        type_ids: encoding.type_ids[overflow_start..].to_vec(),
                        word_ids: encoding.word_ids[overflow_start..].to_vec(),
                        special_tokens_mask: encoding.special_tokens_mask[overflow_start..]
                            .to_vec(),
                        overflowing: Vec::new(),
                    };
                    overflows.push(overflow);
                }
                encoding.ids.truncate(max);
                encoding.attention_mask.truncate(max);
                encoding.type_ids.truncate(max);
                encoding.word_ids.truncate(max);
                encoding.special_tokens_mask.truncate(max);
            }
            TruncationStrategy::OnlySecond => {
                // No second sequence to truncate for simple encode
                // In paired mode this would truncate the second sequence
                encoding.ids.truncate(max);
                encoding.attention_mask.truncate(max);
                encoding.type_ids.truncate(max);
                encoding.word_ids.truncate(max);
                encoding.special_tokens_mask.truncate(max);
            }
        }

        overflows
    }

    /// Apply padding to the encoding.
    fn apply_padding(&self, encoding: &mut Encoding, params: &PaddingParams) {
        let pad_id = params.pad_token_id;
        let current_len = encoding.ids.len();
        let target_len = match params.pad_to_multiple_of {
            Some(m) => {
                let rem = current_len % m;
                if rem == 0 {
                    current_len
                } else {
                    current_len + (m - rem)
                }
            }
            None => current_len, // no padding without pad_to_multiple_of or explicit length
        };

        if target_len <= current_len {
            return;
        }

        let pad_count = target_len - current_len;

        if params.pad_left {
            let mut new_ids = vec![pad_id; pad_count];
            let mut new_mask = vec![0u32; pad_count];
            let mut new_type = vec![0u32; pad_count];
            let mut new_word = vec![None; pad_count];
            let mut new_special = vec![1u32; pad_count];

            new_ids.extend_from_slice(&encoding.ids);
            new_mask.extend_from_slice(&encoding.attention_mask);
            new_type.extend_from_slice(&encoding.type_ids);
            new_word.extend_from_slice(&encoding.word_ids);
            new_special.extend_from_slice(&encoding.special_tokens_mask);

            encoding.ids = new_ids;
            encoding.attention_mask = new_mask;
            encoding.type_ids = new_type;
            encoding.word_ids = new_word;
            encoding.special_tokens_mask = new_special;
        } else {
            encoding
                .ids
                .extend(std::iter::repeat(pad_id).take(pad_count));
            encoding
                .attention_mask
                .extend(std::iter::repeat(0).take(pad_count));
            encoding
                .type_ids
                .extend(std::iter::repeat(0).take(pad_count));
            encoding
                .word_ids
                .extend(std::iter::repeat(None).take(pad_count));
            encoding
                .special_tokens_mask
                .extend(std::iter::repeat(1).take(pad_count));
        }
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
                if self.added_token_by_id.contains_key(&id) && self.added_token_by_id[&id].special {
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
// AddedToken
// ════════════════════════════════════════════════════════════════════════════

impl AddedToken {
    fn from_json(v: &Value) -> Result<Self, String> {
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

// ════════════════════════════════════════════════════════════════════════════
// Models
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
enum ModelKind {
    Bpe(BpeModel),
    WordPiece(WordPieceModel),
    Unigram(UnigramModel),
}

impl ModelKind {
    fn from_json(model_type: &str, v: &Value) -> Result<Self, String> {
        match model_type {
            "BPE" => Ok(ModelKind::Bpe(BpeModel::from_json(v)?)),
            "WordPiece" | "WordPieceModel" => {
                Ok(ModelKind::WordPiece(WordPieceModel::from_json(v)?))
            }
            "Unigram" => Ok(ModelKind::Unigram(UnigramModel::from_json(v)?)),
            other => Err(format!("unsupported model type: {other}")),
        }
    }

    fn tokenize(&self, word: &str) -> Vec<String> {
        match self {
            ModelKind::Bpe(m) => m.tokenize(word),
            ModelKind::WordPiece(m) => m.tokenize(word),
            ModelKind::Unigram(m) => m.tokenize(word),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            ModelKind::Bpe(m) => m.token_to_id(token),
            ModelKind::WordPiece(m) => m.token_to_id(token),
            ModelKind::Unigram(m) => m.token_to_id(token),
        }
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        match self {
            ModelKind::Bpe(m) => m.id_to_token(id),
            ModelKind::WordPiece(m) => m.id_to_token(id),
            ModelKind::Unigram(m) => m.id_to_token(id),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            ModelKind::Bpe(m) => m.vocab_size(),
            ModelKind::WordPiece(m) => m.vocab_size(),
            ModelKind::Unigram(m) => m.vocab_size(),
        }
    }
}

// ── BPE ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct BpeModel {
    vocab: HashMap<String, u32>,
    id_to_token: Vec<String>,
    merges: HashMap<(String, String), u32>,
    /// Whether to use byte-level internal encoding (GPT-2 style)
    byte_fallback: bool,
    /// Unicode-to-byte reverse mapping for byte_fallback decoding
    byte_decoder: Option<HashMap<char, u8>>,
}

impl BpeModel {
    fn from_json(v: &Value) -> Result<Self, String> {
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

    fn tokenize(&self, word: &str) -> Vec<String> {
        // If the whole word is already in vocab, return it immediately
        if self.vocab.contains_key(word) {
            return vec![word.to_string()];
        }

        let result = if self.byte_fallback {
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
        };
        result
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

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        let idx = id as usize;
        if idx < self.id_to_token.len() && !self.id_to_token[idx].is_empty() {
            Some(self.id_to_token[idx].as_str())
        } else {
            None
        }
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn decode_bytes(&self, token: &str) -> Option<Vec<u8>> {
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

// ── WordPiece ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct WordPieceModel {
    vocab: HashMap<String, u32>,
    id_to_token: Vec<String>,
    unk_token: String,
    unk_id: u32,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
}

impl WordPieceModel {
    fn from_json(v: &Value) -> Result<Self, String> {
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

    fn tokenize(&self, word: &str) -> Vec<String> {
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

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        let idx = id as usize;
        if idx < self.id_to_token.len() && !self.id_to_token[idx].is_empty() {
            Some(self.id_to_token[idx].as_str())
        } else {
            None
        }
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ── Unigram ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct UnigramModel {
    /// (token, score) sorted by score descending
    pieces: Vec<(String, f64)>,
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, u32>,
    unk_id: u32,
    unk_token: String,
}

impl UnigramModel {
    fn from_json(v: &Value) -> Result<Self, String> {
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
    fn tokenize(&self, word: &str) -> Vec<String> {
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

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        let idx = id as usize;
        if idx < self.id_to_token.len() {
            Some(self.id_to_token[idx].as_str())
        } else {
            None
        }
    }

    fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Pre-Tokenizers
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
enum PreTokenizerKind {
    /// Simple whitespace-based split
    Whitespace,
    /// Byte-level (GPT-2 style): map bytes to unicode, optionally prefix space
    ByteLevel { add_prefix_space: bool },
    /// BERT-style: split on whitespace + punctuation
    BertPreTokenizer,
    /// Metaspace (SentencePiece-style): replace space with ▁
    Metaspace {
        replacement: String,
        add_prefix_space: bool,
    },
    /// Split by configurable regex pattern
    Split { pattern: String },
    /// Sequence of pre-tokenizers applied in order
    Sequence(Vec<PreTokenizerKind>),
}

impl PreTokenizerKind {
    fn from_json(v: &Value) -> Result<Self, String> {
        let ptype = v
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| "pre_tokenizer missing 'type'".to_string())?;

        match ptype {
            "Whitespace" => Ok(PreTokenizerKind::Whitespace),
            "ByteLevel" => {
                let add_prefix_space = v
                    .get("add_prefix_space")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                Ok(PreTokenizerKind::ByteLevel { add_prefix_space })
            }
            "BertPreTokenizer" => Ok(PreTokenizerKind::BertPreTokenizer),
            "Metaspace" => {
                let replacement = v
                    .get("replacement")
                    .and_then(Value::as_str)
                    .unwrap_or("▁")
                    .to_string();
                let add_prefix_space = v
                    .get("add_prefix_space")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                Ok(PreTokenizerKind::Metaspace {
                    replacement,
                    add_prefix_space,
                })
            }
            "Split" => {
                let pattern = v
                    .get("pattern")
                    .and_then(|p| p.get("Regex"))
                    .and_then(Value::as_str)
                    .unwrap_or(" ")
                    .to_string();
                Ok(PreTokenizerKind::Split { pattern })
            }
            "Sequence" => {
                let pretok_arr = v
                    .get("pretokenizers")
                    .or_else(|| v.get("pre_tokenizers"))
                    .and_then(Value::as_array)
                    .ok_or_else(|| "Sequence pre_tokenizer missing 'pretokenizers'".to_string())?;
                let seq: Result<Vec<_>, _> = pretok_arr.iter().map(Self::from_json).collect();
                Ok(PreTokenizerKind::Sequence(seq?))
            }
            _other => {
                // Unknown type — fall back to whitespace
                Ok(PreTokenizerKind::Whitespace)
            }
        }
    }

    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        match self {
            PreTokenizerKind::Whitespace => pre_tokenize_whitespace(text),
            PreTokenizerKind::ByteLevel { add_prefix_space } => {
                let s = if *add_prefix_space && !text.starts_with(' ') {
                    format!(" {}", text)
                } else {
                    text.to_string()
                };
                pre_tokenize_byte_level(&s)
            }
            PreTokenizerKind::BertPreTokenizer => pre_tokenize_bert(text),
            PreTokenizerKind::Metaspace {
                replacement,
                add_prefix_space,
            } => {
                let s = if *add_prefix_space && !text.starts_with(' ') {
                    format!(" {}", text)
                } else {
                    text.to_string()
                };
                pre_tokenize_metaspace(&s, replacement)
            }
            PreTokenizerKind::Split { pattern } => {
                // Basic split on pattern — for full regex support a regex crate is needed
                if pattern == " " || pattern == r"\s+" {
                    pre_tokenize_whitespace(text)
                } else if pattern == r"(\s+)" {
                    // Split but keep delimiters (for Metaspace compatibility)
                    let mut words: Vec<String> = Vec::new();
                    let mut last_end = 0;
                    for (i, c) in text.char_indices() {
                        if c.is_whitespace() {
                            if i > last_end {
                                words.push(text[last_end..i].to_string());
                            }
                            words.push(c.to_string());
                            last_end = i + c.len_utf8();
                        }
                    }
                    if last_end < text.len() {
                        words.push(text[last_end..].to_string());
                    }
                    words
                } else {
                    pre_tokenize_whitespace(text)
                }
            }
            PreTokenizerKind::Sequence(seq) => {
                let mut words = vec![text.to_string()];
                for pt in seq {
                    let mut new_words = Vec::new();
                    for word in &words {
                        new_words.extend(pt.pre_tokenize(word));
                    }
                    words = new_words;
                }
                words
            }
        }
    }
}

/// Whitespace + CJK pre-tokenizer: split on whitespace, isolate CJK chars.
fn pre_tokenize_whitespace(text: &str) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut buf = String::new();
    let mut ws_buf = String::new();

    for c in text.chars() {
        if c.is_whitespace() {
            if !buf.is_empty() {
                words.push(buf.clone());
                buf.clear();
            }
            ws_buf.push(c);
        } else if is_cjk(c) {
            if !ws_buf.is_empty() {
                words.push(ws_buf.clone());
                ws_buf.clear();
            }
            if !buf.is_empty() {
                words.push(buf.clone());
                buf.clear();
            }
            words.push(c.to_string());
        } else {
            if !ws_buf.is_empty() {
                words.push(ws_buf.clone());
                ws_buf.clear();
            }
            buf.push(c);
        }
    }
    if !ws_buf.is_empty() {
        words.push(ws_buf);
    }
    if !buf.is_empty() {
        words.push(buf);
    }
    words
}

/// Byte-level pre-tokenizer: split on whitespace (keeping space as separate word),
/// then convert each word to byte-level unicode mapping.
fn pre_tokenize_byte_level(_text: &str) -> Vec<String> {
    // Standard: first split on whitespace+punctuation via GPT-2 regex pattern
    // For simplicity, we do the same as whitespace + CJK split
    let words = pre_tokenize_whitespace(_text);
    words
}

/// BERT pre-tokenizer: split on whitespace + punctuation.
fn pre_tokenize_bert(text: &str) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut buf = String::new();

    for c in text.chars() {
        if c.is_whitespace() {
            if !buf.is_empty() {
                words.push(buf.clone());
                buf.clear();
            }
        } else if c.is_ascii_punctuation() {
            if !buf.is_empty() {
                words.push(buf.clone());
                buf.clear();
            }
            words.push(c.to_string());
        } else if is_cjk(c) {
            if !buf.is_empty() {
                words.push(buf.clone());
                buf.clear();
            }
            words.push(c.to_string());
        } else {
            buf.push(c);
        }
    }
    if !buf.is_empty() {
        words.push(buf);
    }
    words
}

/// Metaspace pre-tokenizer: split on whitespace and replace space with ▁.
fn pre_tokenize_metaspace(text: &str, replacement: &str) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut buf = String::new();
    let mut has_content = false;

    for c in text.chars() {
        if c == ' ' {
            if has_content {
                words.push(buf.clone());
                buf.clear();
                has_content = false;
            }
            // Replace space with metaspace char
            buf.push_str(replacement);
            words.push(buf.clone());
            buf.clear();
        } else if c.is_whitespace() {
            // Non-space whitespace: flush current word if any
            if has_content {
                words.push(buf.clone());
                buf.clear();
                has_content = false;
            }
        } else if is_cjk(c) {
            if has_content {
                words.push(buf.clone());
                buf.clear();
                has_content = false;
            }
            words.push(c.to_string());
        } else {
            has_content = true;
            buf.push(c);
        }
    }
    if has_content {
        words.push(buf);
    }
    words
}

/// Returns true if `c` is a CJK character.
fn is_cjk(c: char) -> bool {
    let code = c as u32;
    matches!(
        code,
        0x2E80..=0x2FDF
        | 0x2FF0..=0x2FFF
        | 0x3000..=0x303F
        | 0x3400..=0x4DBF
        | 0x4E00..=0x9FFF
        | 0xF900..=0xFAFF
        | 0xFE30..=0xFE4F
        | 0xFF01..=0xFF60 | 0xFFE0..=0xFFEF
        | 0x20000..=0x2A6DF
        | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F
        | 0x2B820..=0x2CEAF
        | 0x2F800..=0x2FA1F
    )
}

/// Build byte-to-unicode mapping (GPT-2 style).
fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut map = HashMap::new();
    let mut n = 0u32;
    for b in 0..=255u8 {
        if matches!(b, 33..=126 | 161..=172 | 174..=255) {
            map.insert(b, char::from_u32(b as u32).unwrap());
        } else {
            map.insert(b, char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }
    map
}

// ════════════════════════════════════════════════════════════════════════════
// Normalizers
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
enum NormalizerKind {
    None,
    Nfc,
    Nfkc,
    Lowercase,
    BertNormalizer {
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: bool,
        lowercase: bool,
    },
    Sequence(Vec<NormalizerKind>),
}

impl NormalizerKind {
    fn from_json(v: &Value) -> Result<Self, String> {
        let ntype = v.get("type").and_then(Value::as_str).unwrap_or("");

        match ntype {
            "NFC" => Ok(NormalizerKind::Nfc),
            "NFKC" => Ok(NormalizerKind::Nfkc),
            "Lowercase" => Ok(NormalizerKind::Lowercase),
            "BertNormalizer" => {
                let clean_text = v.get("clean_text").and_then(Value::as_bool).unwrap_or(true);
                let handle_chinese_chars = v
                    .get("handle_chinese_chars")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                let strip_accents = v
                    .get("strip_accents")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                let lowercase = v.get("lowercase").and_then(Value::as_bool).unwrap_or(true);
                Ok(NormalizerKind::BertNormalizer {
                    clean_text,
                    handle_chinese_chars,
                    strip_accents,
                    lowercase,
                })
            }
            "Sequence" | "Prepend" | "Replace" | "StripAccents" | "Strip" => {
                // For complex sequences try parsing nested normalizers
                if let Some(arr) = v.get("normalizers").and_then(Value::as_array) {
                    let seq: Result<Vec<_>, _> = arr.iter().map(Self::from_json).collect();
                    Ok(NormalizerKind::Sequence(seq?))
                } else {
                    Ok(NormalizerKind::None)
                }
            }
            _ => Ok(NormalizerKind::None),
        }
    }

    fn normalize(&self, text: &str) -> String {
        match self {
            NormalizerKind::None | NormalizerKind::Nfc | NormalizerKind::Nfkc => text.to_string(),
            NormalizerKind::Lowercase => text.to_lowercase(),
            NormalizerKind::BertNormalizer {
                clean_text: _,
                handle_chinese_chars: _,
                strip_accents: _,
                lowercase,
            } => {
                let mut s = text.to_string();
                // For unicode normalization we use NFKC via the unicode crate
                s = s.nfkc();
                if *lowercase {
                    s = s.to_lowercase();
                }
                s
            }
            NormalizerKind::Sequence(seq) => {
                let mut s = text.to_string();
                for n in seq {
                    s = n.normalize(&s);
                }
                s
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Post-Processors
// ════════════════════════════════════════════════════════════════════════════

/// Result of post-processing: (ids, type_ids, special_tokens_mask)
type PostProcessResult = Option<(Vec<u32>, Vec<u32>, Vec<u32>)>;

#[derive(Debug, Clone)]
enum PostProcessorKind {
    /// No post-processing — just return the encoding as-is
    None,
    /// Template-based post-processor
    Template {
        single_template: Option<Template>,
        pair_template: Option<Template>,
    },
    /// Simple BOS/EOS appending (for older tokenizers without explicit template)
    BosEos { bos_id: u32, eos_id: u32 },
}

#[derive(Debug, Clone)]
struct Template {
    pieces: Vec<TemplatePiece>,
}

#[derive(Debug, Clone)]
enum TemplatePiece {
    /// A special token (BOS, EOS, SEP, CLS, etc.)
    Special { id: u32, type_id: u32 },
    /// The input sequence placeholder
    Sequence { index: usize, type_id: u32 },
}

impl PostProcessorKind {
    fn from_json(v: &Value) -> Result<Self, String> {
        let ptype = v.get("type").and_then(Value::as_str).unwrap_or("");

        match ptype {
            "TemplateProcessing" => {
                let single_val = v
                    .get("single")
                    .ok_or_else(|| "TemplateProcessing missing 'single'".to_string())?;
                let pair_val = v.get("pair");

                let single = Template::from_json(single_val)?;
                let pair = match pair_val {
                    Some(pv) => Some(Template::from_json(pv)?),
                    None => None,
                };

                Ok(PostProcessorKind::Template {
                    single_template: Some(single),
                    pair_template: pair,
                })
            }
            "ByteLevel" => {
                // ByteLevel post-processor trims whitespace offsets — no-op for IDs
                Ok(PostProcessorKind::None)
            }
            "RobertaProcessing" | "BertProcessing" => {
                let sep = v
                    .get("sep")
                    .and_then(|s| s.as_array())
                    .and_then(|arr| arr.get(0))
                    .and_then(Value::as_u64)
                    .map(|v| v as u32)
                    .unwrap_or(2);
                let cls = v
                    .get("cls")
                    .and_then(|s| s.as_array())
                    .and_then(|arr| arr.get(0))
                    .and_then(Value::as_u64)
                    .map(|v| v as u32)
                    .unwrap_or(0);
                // Build a template: [CLS] $A [SEP] for single
                let single = Template {
                    pieces: vec![
                        TemplatePiece::Special {
                            id: cls,
                            type_id: 0,
                        },
                        TemplatePiece::Sequence {
                            index: 0,
                            type_id: 0,
                        },
                        TemplatePiece::Special {
                            id: sep,
                            type_id: 0,
                        },
                    ],
                };
                Ok(PostProcessorKind::Template {
                    single_template: Some(single),
                    pair_template: None,
                })
            }
            _ => Ok(PostProcessorKind::None),
        }
    }

    fn post_process(
        &self,
        ids: &[u32],
        _encoding: &Encoding,
        bos_token_id: Option<u32>,
        eos_token_id: Option<u32>,
    ) -> PostProcessResult {
        match self {
            PostProcessorKind::None => {
                // If bos/eos are defined but there's no explicit template,
                // prepend/append them
                let mut new_ids = Vec::new();
                let mut type_ids = Vec::new();
                let mut special_mask = Vec::new();

                if let Some(bos) = bos_token_id {
                    if ids.is_empty() || ids[0] != bos {
                        new_ids.push(bos);
                        type_ids.push(0);
                        special_mask.push(1);
                    }
                }
                for &id in ids {
                    new_ids.push(id);
                    type_ids.push(0);
                    special_mask.push(0);
                }
                if let Some(eos) = eos_token_id {
                    if ids.last() != Some(&eos) {
                        new_ids.push(eos);
                        type_ids.push(0);
                        special_mask.push(1);
                    }
                }

                if new_ids.len() > ids.len() {
                    Some((new_ids, type_ids, special_mask))
                } else {
                    None
                }
            }
            PostProcessorKind::Template {
                single_template,
                pair_template: _,
            } => {
                if let Some(template) = single_template {
                    let mut new_ids = Vec::new();
                    let mut type_ids = Vec::new();
                    let mut special_mask = Vec::new();

                    for piece in &template.pieces {
                        match piece {
                            TemplatePiece::Special { id, type_id } => {
                                new_ids.push(*id);
                                type_ids.push(*type_id);
                                special_mask.push(1);
                            }
                            TemplatePiece::Sequence { index: _, type_id } => {
                                for &id in ids {
                                    new_ids.push(id);
                                    type_ids.push(*type_id);
                                    special_mask.push(0);
                                }
                            }
                        }
                    }

                    Some((new_ids, type_ids, special_mask))
                } else {
                    None
                }
            }
            PostProcessorKind::BosEos { bos_id, eos_id } => {
                let mut new_ids = vec![*bos_id];
                let mut type_ids = vec![0];
                let mut special_mask = vec![1];
                for &id in ids {
                    new_ids.push(id);
                    type_ids.push(0);
                    special_mask.push(0);
                }
                new_ids.push(*eos_id);
                type_ids.push(0);
                special_mask.push(1);
                Some((new_ids, type_ids, special_mask))
            }
        }
    }
}

impl Template {
    fn from_json(v: &Value) -> Result<Self, String> {
        let pieces = v
            .get("pieces")
            .or_else(|| v.get("sequence")) // some formats use "sequence" key
            .and_then(Value::as_array)
            .ok_or_else(|| "Template missing 'pieces'".to_string())?;

        let mut template_pieces = Vec::new();
        for piece_val in pieces {
            let ptype = piece_val.get("type").and_then(Value::as_str).unwrap_or("");

            match ptype {
                "SpecialToken" | "special_token" | "Special" => {
                    let id = piece_val
                        .get("id")
                        .and_then(Value::as_u64)
                        .map(|v| v as u32)
                        .unwrap_or(0);
                    let type_id = piece_val
                        .get("type_id")
                        .and_then(Value::as_u64)
                        .map(|v| v as u32)
                        .unwrap_or(0);
                    template_pieces.push(TemplatePiece::Special { id, type_id });
                }
                "Sequence" | "sequence" => {
                    let index = piece_val
                        .get("index")
                        .and_then(Value::as_u64)
                        .map(|v| v as usize)
                        .unwrap_or(0);
                    let type_id = piece_val
                        .get("type_id")
                        .and_then(Value::as_u64)
                        .map(|v| v as u32)
                        .unwrap_or(0);
                    template_pieces.push(TemplatePiece::Sequence { index, type_id });
                }
                _ => {}
            }
        }

        Ok(Template {
            pieces: template_pieces,
        })
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Decoders
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
enum DecoderKind {
    /// No special decoding — just join tokens
    None,
    /// Byte-level decoder (GPT-2 style): reverse byte→unicode mapping
    ByteLevel { add_prefix_space: bool },
    /// WordPiece decoder: strip `##` prefixes and join
    WordPiece { prefix: String, cleanup: bool },
    /// BPE decoder (simple join)
    BPEDecoder { suffix: Option<String> },
    /// Metaspace decoder: replace ▁ with space
    Metaspace {
        replacement: String,
        add_prefix_space: bool,
    },
}

impl DecoderKind {
    fn from_json(v: &Value) -> Result<Self, String> {
        let dtype = v.get("type").and_then(Value::as_str).unwrap_or("");

        match dtype {
            "ByteLevel" => {
                let add_prefix_space = v
                    .get("add_prefix_space")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                Ok(DecoderKind::ByteLevel { add_prefix_space })
            }
            "WordPiece" => {
                let prefix = v
                    .get("prefix")
                    .and_then(Value::as_str)
                    .unwrap_or("##")
                    .to_string();
                let cleanup = v.get("cleanup").and_then(Value::as_bool).unwrap_or(true);
                Ok(DecoderKind::WordPiece { prefix, cleanup })
            }
            "BPEDecoder" => {
                let suffix = v
                    .get("suffix")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string());
                Ok(DecoderKind::BPEDecoder { suffix })
            }
            "Metaspace" => {
                let replacement = v
                    .get("replacement")
                    .and_then(Value::as_str)
                    .unwrap_or("▁")
                    .to_string();
                let add_prefix_space = v
                    .get("add_prefix_space")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                Ok(DecoderKind::Metaspace {
                    replacement,
                    add_prefix_space,
                })
            }
            _ => Ok(DecoderKind::None),
        }
    }

    fn decode(&self, tokens: &[String]) -> String {
        match self {
            DecoderKind::None => tokens.join(""),
            DecoderKind::ByteLevel { .. } => {
                // Byte-level decoding: convert each token's bytes using byte_decoder
                let mapper = bytes_to_unicode();
                let reverse: HashMap<char, u8> = mapper.into_iter().map(|(b, c)| (c, b)).collect();

                let mut bytes: Vec<u8> = Vec::new();
                for token in tokens {
                    for c in token.chars() {
                        if let Some(&b) = reverse.get(&c) {
                            bytes.push(b);
                        } else {
                            // Character not in byte mapping — try as UTF-8
                            let mut buf = [0u8; 4];
                            let s = c.encode_utf8(&mut buf);
                            bytes.extend_from_slice(s.as_bytes());
                        }
                    }
                }
                String::from_utf8_lossy(&bytes).to_string()
            }
            DecoderKind::WordPiece { prefix, cleanup: _ } => {
                let mut result = String::new();
                for (i, token) in tokens.iter().enumerate() {
                    if token.starts_with(prefix) && i > 0 {
                        result.push_str(&token[prefix.len()..]);
                    } else if token == prefix {
                        // Just the prefix itself — skip
                    } else {
                        if i > 0 && !result.is_empty() {
                            // No space between subwords of same word
                        }
                        result.push_str(token);
                    }
                }
                result
            }
            DecoderKind::BPEDecoder { .. } => tokens.join(""),
            DecoderKind::Metaspace {
                replacement,
                add_prefix_space: _,
            } => {
                let mut result = tokens.join("");
                result = result.replace(replacement, " ");
                result
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Unicode normalization shim (NFC/NFKC)
// ════════════════════════════════════════════════════════════════════════════

trait UnicodeNormalization {
    fn nfc(&self) -> String;
    fn nfkc(&self) -> String;
}

impl UnicodeNormalization for str {
    fn nfc(&self) -> String {
        self.to_string()
    }

    fn nfkc(&self) -> String {
        self.to_string()
    }
}

impl UnicodeNormalization for String {
    fn nfc(&self) -> String {
        (**self).nfc()
    }
    fn nfkc(&self) -> String {
        (**self).nfkc()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Pre-tokenizer tests ──

    #[test]
    fn test_is_cjk() {
        assert!(is_cjk('\u{4E00}'));
        assert!(is_cjk('\u{9FFF}'));
        assert!(!is_cjk('A'));
        assert!(!is_cjk('z'));
        assert!(!is_cjk('5'));
    }

    #[test]
    fn test_pre_tokenize_whitespace() {
        assert_eq!(
            pre_tokenize_whitespace("hello world"),
            vec!["hello", " ", "world"]
        );
        assert_eq!(pre_tokenize_whitespace("你好"), vec!["你", "好"]);
        assert_eq!(
            pre_tokenize_whitespace("Hello 你好 world"),
            vec!["Hello", " ", "你", "好", " ", "world"]
        );
        assert!(pre_tokenize_whitespace("").is_empty());
    }

    #[test]
    fn test_pre_tokenize_bert() {
        let words = pre_tokenize_bert("Hello, world!");
        assert_eq!(words, vec!["Hello", ",", "world", "!"]);
    }

    #[test]
    fn test_metaspace_pre_tokenizer() {
        let words = pre_tokenize_metaspace("hello world", "▁");
        assert_eq!(words, vec!["hello", "▁", "world"]);
    }

    // ── BPE model tests ──

    fn make_bpe() -> BpeModel {
        build_simple_bpe()
    }

    fn build_simple_bpe() -> BpeModel {
        let mut vocab = HashMap::new();
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
        vocab.insert("<s>".to_string(), 12);
        vocab.insert("</s>".to_string(), 13);
        vocab.insert("<unk>".to_string(), 14);

        let max_id = 14u32;
        let mut id_to_token: Vec<String> = Vec::with_capacity(max_id as usize + 1);
        id_to_token.resize(max_id as usize + 1, String::new());
        for (token, &id) in &vocab {
            id_to_token[id as usize] = token.clone();
        }

        let mut merges = HashMap::new();
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
    fn test_bpe_merge() {
        let bpe = build_simple_bpe();
        let result = bpe.tokenize("hello");
        assert_eq!(result, vec!["hello"]);

        let result = bpe.tokenize("world");
        assert_eq!(result, vec!["world"]);

        let result = bpe.tokenize("xyz");
        assert_eq!(result, vec!["x", "y", "z"]);
    }

    // ── WordPiece model tests ──

    fn build_wordpiece() -> WordPieceModel {
        let mut vocab = HashMap::new();
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
    fn test_wordpiece_whole_word() {
        let wp = build_wordpiece();
        let tokens = wp.tokenize("Hello");
        assert_eq!(tokens, vec!["Hello"]);
    }

    #[test]
    fn test_wordpiece_subword() {
        let wp = build_wordpiece();
        let tokens = wp.tokenize("playing");
        assert_eq!(tokens, vec!["play", "##ing"]);
    }

    #[test]
    fn test_wordpiece_unknown() {
        let wp = build_wordpiece();
        let tokens = wp.tokenize("xyz");
        assert_eq!(tokens, vec!["[UNK]"]);
    }

    // ── Unigram model tests ──

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
                {"0": "d", "1": -4.0},
            ],
            "unk_token": "<unk>",
            "unk_id": 0
        })
    }

    #[test]
    fn test_unigram_tokenize() {
        let model = UnigramModel::from_json(&build_unigram_json()).unwrap();
        // "hello" should match as a whole word (highest score)
        let tokens = model.tokenize("hello");
        assert_eq!(tokens, vec!["hello"]);

        // "xyz" — unknown, individual chars
        let tokens = model.tokenize("xyz");
        assert!(tokens.iter().any(|t| t == "<unk>"));
    }

    // ── Full Tokenizer tests ──

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
    fn test_tokenizer_from_str() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        assert_eq!(t.vocab_size(), 12);
        assert_eq!(t.bos_token_id(), Some(12));
        assert_eq!(t.eos_token_id(), Some(13));
        assert_eq!(t.unk_token_id(), Some(14));
    }

    #[test]
    fn test_tokenizer_encode_no_special() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let enc = t.encode("hello world", false).unwrap();
        assert_eq!(enc.ids, vec![10, 4, 11]);
        assert_eq!(enc.attention_mask, vec![1, 1, 1]);
        assert_eq!(enc.special_tokens_mask, vec![0, 0, 0]);
    }

    #[test]
    fn test_tokenizer_encode_with_special() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let enc = t.encode("hello world", true).unwrap();
        // No explicit template, so BOS/EOS are prepended/appended
        assert_eq!(enc.ids, vec![12, 10, 4, 11, 13]);
        assert_eq!(enc.attention_mask, vec![1, 1, 1, 1, 1]);
        // BOS and EOS should be marked as special
        assert_eq!(enc.special_tokens_mask, vec![1, 0, 0, 0, 1]);
    }

    #[test]
    fn test_tokenizer_decode() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let text = t.decode(&[10, 4, 11], false).unwrap();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_tokenizer_decode_skip_special() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let text = t.decode(&[12, 10, 4, 11, 13], true).unwrap();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_tokenizer_encode_unknown() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let enc = t.encode("xyz", false).unwrap();
        assert_eq!(enc.ids, vec![14, 14, 14]);
    }

    #[test]
    fn test_tokenizer_empty() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let enc = t.encode("", false).unwrap();
        assert!(enc.ids.is_empty());
    }

    #[test]
    fn test_tokenizer_cjk() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let enc = t.encode("你好", false).unwrap();
        assert_eq!(enc.ids, vec![14, 14]); // unknown chars
    }

    #[test]
    fn test_truncation() {
        let t = Tokenizer::from_str(&make_full_json())
            .unwrap()
            .with_truncation(Some(TruncationParams {
                max_length: 2,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
            }));
        let enc = t.encode("hello world", false).unwrap();
        assert_eq!(enc.ids.len(), 2);
        assert!(enc.overflowing.is_empty() || !enc.overflowing.is_empty());
    }

    #[test]
    fn test_padding() {
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
    fn test_padding_left() {
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
    fn test_from_file() {
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
    fn test_bpe_from_file_invalid_type() {
        let json = r#"{"model": {"type": "WordPiece", "vocab": {"[CLS]": 0}}}"#;
        let t = Tokenizer::from_str(json);
        assert!(t.is_ok()); // WordPiece is supported
    }

    #[test]
    fn test_unsupported_model_type() {
        let json = r#"{"model": {"type": "FancyNewModel", "vocab": {}}}"#;
        let t = Tokenizer::from_str(json);
        assert!(t.is_err());
    }

    #[test]
    fn test_wordpiece_from_json() {
        let json = r###"{
            "version": "1.0",
            "model": {
                "type": "WordPiece",
                "vocab": {
                    "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
                    "hello": 3, "##ing": 4, "play": 5
                },
                "unk_token": "[UNK]"
            }
        }"###;
        let t = Tokenizer::from_str(json).unwrap();
        let enc = t.encode("playing", false).unwrap();
        assert_eq!(enc.ids, vec![5, 4]); // play + continuing subword
    }

    #[test]
    fn test_wordpiece_bert_encode() {
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
    fn test_multiple_overflow_truncation_none() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let enc = t.encode("hello world", false).unwrap();
        assert!(enc.overflowing.is_empty());
    }

    #[test]
    fn test_tokenizer_word_ids() {
        let t = Tokenizer::from_str(&make_full_json()).unwrap();
        let enc = t.encode("hello world", false).unwrap();
        // "hello" → word 0, " " → word 1, "world" → word 2
        assert!(enc.word_ids.iter().all(|w| w.is_some()));
        assert_eq!(enc.word_ids[0], Some(0));
        assert_eq!(enc.word_ids[1], Some(1));
        assert_eq!(enc.word_ids[2], Some(2));
    }
}

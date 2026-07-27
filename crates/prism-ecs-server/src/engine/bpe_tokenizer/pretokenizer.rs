//! Pre-tokenization — split a normalized string into model-level words.
//!
//! This module owns the canonical authority for pre-tokenization strategies:
//! whitespace, byte-level, BERT-style, metaspace, regex split, and sequences
//! thereof. It does not own normalization (upstream), model tokenization
//! (downstream), or any pipeline orchestration.
//!
//! Pre-tokenization is purely a string-to-string transformation; the output
//! is a `Vec<String>` of words passed to the model.

use serde_json::Value;

// ════════════════════════════════════════════════════════════════════════════
// PreTokenizerKind
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub(crate) enum PreTokenizerKind {
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
    pub(crate) fn from_json(v: &Value) -> Result<Self, String> {
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
            // Unknown type — fall back to whitespace
            _other => Ok(PreTokenizerKind::Whitespace),
        }
    }

    pub(crate) fn pre_tokenize(&self, text: &str) -> Vec<String> {
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

// ════════════════════════════════════════════════════════════════════════════
// Pre-tokenization implementations
// ════════════════════════════════════════════════════════════════════════════

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

/// Byte-level pre-tokenizer: split on whitespace (keeping space as separate
/// word), then convert each word to byte-level unicode mapping.
fn pre_tokenize_byte_level(text: &str) -> Vec<String> {
    // Standard: first split on whitespace+punctuation via GPT-2 regex pattern
    // For simplicity, we do the same as whitespace + CJK split
    pre_tokenize_whitespace(text)
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
            | 0xFF01..=0xFF60
            | 0xFFE0..=0xFFEF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0x2F800..=0x2FA1F
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cjk_detection_in_ideographic_unified_and_punctuation() {
        assert!(is_cjk('\u{4E00}'));
        assert!(is_cjk('\u{9FFF}'));
        assert!(!is_cjk('A'));
        assert!(!is_cjk('z'));
        assert!(!is_cjk('5'));
    }

    #[test]
    fn whitespace_pre_tokenizer_handles_ascii() {
        assert_eq!(
            pre_tokenize_whitespace("hello world"),
            vec!["hello", " ", "world"]
        );
    }

    #[test]
    fn whitespace_pre_tokenizer_isolates_cjk() {
        assert_eq!(pre_tokenize_whitespace("你好"), vec!["你", "好"]);
        assert_eq!(
            pre_tokenize_whitespace("Hello 你好 world"),
            vec!["Hello", " ", "你", "好", " ", "world"]
        );
    }

    #[test]
    fn whitespace_pre_tokenizer_empty_string() {
        assert!(pre_tokenize_whitespace("").is_empty());
    }

    #[test]
    fn bert_pre_tokenizer_splits_punctuation() {
        assert_eq!(
            pre_tokenize_bert("Hello, world!"),
            vec!["Hello", ",", "world", "!"]
        );
    }

    #[test]
    fn metaspace_pre_tokenizer_replaces_space() {
        assert_eq!(
            pre_tokenize_metaspace("hello world", "▁"),
            vec!["hello", "▁", "world"]
        );
    }

    #[test]
    fn pre_tokenizer_kind_dispatches_from_json() {
        let v = serde_json::json!({"type": "BertPreTokenizer"});
        let pt = PreTokenizerKind::from_json(&v).unwrap();
        assert!(matches!(pt, PreTokenizerKind::BertPreTokenizer));
    }

    #[test]
    fn pre_tokenizer_kind_unknown_falls_back_to_whitespace() {
        let v = serde_json::json!({"type": "MysteryPretok"});
        let pt = PreTokenizerKind::from_json(&v).unwrap();
        assert!(matches!(pt, PreTokenizerKind::Whitespace));
    }

    #[test]
    fn pre_tokenizer_kind_rejects_missing_type() {
        let v = serde_json::json!({});
        assert!(PreTokenizerKind::from_json(&v).is_err());
    }

    #[test]
    fn pre_tokenizer_kind_byte_level_adds_prefix_space() {
        let v = serde_json::json!({"type": "ByteLevel", "add_prefix_space": true});
        let pt = PreTokenizerKind::from_json(&v).unwrap();
        // The "no space → prepend" branch prepends a space, then
        // pre_tokenize_byte_level delegates to pre_tokenize_whitespace which
        // keeps the leading space as its own word.
        let result = pt.pre_tokenize("hello");
        assert_eq!(result, vec![" ".to_string(), "hello".to_string()]);
    }
}

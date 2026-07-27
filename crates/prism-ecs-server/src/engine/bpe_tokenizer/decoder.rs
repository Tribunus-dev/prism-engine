//! Decoding — convert a sequence of model-level tokens back into text.
//!
//! This module owns the canonical authority for decoder construction from
//! `tokenizer.json` and the application of decoding to a list of model
//! tokens. It does not own encoding, post-processing, or pipeline
//! orchestration.
//!
//! Decoders operate on a `Vec<String>` of token strings (already
//! resolved from ids) and produce a single `String` output. Byte-level
//! decoding requires the bytes-to-unicode reverse mapping.

use std::collections::HashMap;

use serde_json::Value;

use crate::engine::bpe_tokenizer::model::bytes_to_unicode;

// WAIVER: HashMap is correct for the byte-decoder reverse mapping. It is a
// stable-identity lookup (char -> byte) whose iteration order is not
// observable; the decoded `String` output is byte-order stable.

#[derive(Debug, Clone)]
pub(crate) enum DecoderKind {
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
    pub(crate) fn from_json(v: &Value) -> Result<Self, String> {
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

    pub(crate) fn decode(&self, tokens: &[String]) -> String {
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
                let result = tokens.join("");
                result.replace(replacement, " ")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_decoder_joins_tokens() {
        let d = DecoderKind::None;
        assert_eq!(d.decode(&["hello".to_string(), " ".to_string(), "world".to_string()]), "hello world");
    }

    #[test]
    fn bpe_decoder_joins_tokens() {
        let d = DecoderKind::BPEDecoder { suffix: None };
        assert_eq!(d.decode(&["hello".to_string(), "world".to_string()]), "helloworld");
    }

    #[test]
    fn wordpiece_decoder_strips_continuing_prefix() {
        let d = DecoderKind::WordPiece {
            prefix: "##".to_string(),
            cleanup: true,
        };
        let tokens: Vec<String> = vec!["play".to_string(), "##ing".to_string()];
        assert_eq!(d.decode(&tokens), "playing");
    }

    #[test]
    fn wordpiece_decoder_skips_standalone_prefix() {
        let d = DecoderKind::WordPiece {
            prefix: "##".to_string(),
            cleanup: true,
        };
        // A standalone "##" token should be skipped
        assert_eq!(d.decode(&["##".to_string(), "hello".to_string()]), "hello");
    }

    #[test]
    fn metaspace_decoder_replaces_replacement_with_space() {
        let d = DecoderKind::Metaspace {
            replacement: "▁".to_string(),
            add_prefix_space: false,
        };
        let tokens: Vec<String> = vec!["hello".to_string(), "▁".to_string(), "world".to_string()];
        assert_eq!(d.decode(&tokens), "hello world");
    }

    #[test]
    fn byte_level_decoder_resolves_chars_to_bytes() {
        let d = DecoderKind::ByteLevel { add_prefix_space: false };
        // 'A' is a printable ASCII byte and maps to itself in bytes_to_unicode
        let tokens: Vec<String> = vec!["A".to_string()];
        assert_eq!(d.decode(&tokens), "A");
    }

    #[test]
    fn unknown_decoder_type_falls_back_to_none() {
        let v = serde_json::json!({"type": "FancyDecoder"});
        let d = DecoderKind::from_json(&v).unwrap();
        assert!(matches!(d, DecoderKind::None));
    }

    #[test]
    fn decoder_kind_dispatches_from_json() {
        let v = serde_json::json!({"type": "BPEDecoder"});
        let d = DecoderKind::from_json(&v).unwrap();
        assert!(matches!(d, DecoderKind::BPEDecoder { .. }));
    }
}

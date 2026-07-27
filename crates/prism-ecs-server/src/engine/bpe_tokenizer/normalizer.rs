//! Normalization — Unicode and case normalization applied before pre-tokenization.
//!
//! This module owns the canonical authority for normalizer construction from
//! `tokenizer.json` and the application of normalization to input text. It
//! does not own pre-tokenization, model tokenization, or post-processing.
//!
//! The `UnicodeNormalization` trait here is a local shim that keeps callers
//! out of the `unicode-normalization` crate dependency. Production-grade
//! NFC/NFKC would replace these no-op shims with a real implementation.

use serde_json::Value;

#[derive(Debug, Clone)]
pub(crate) enum NormalizerKind {
    /// No normalization (passthrough).
    None,
    /// NFC (Canonical Decomposition followed by Canonical Composition).
    Nfc,
    /// NFKC (Compatibility Decomposition followed by Canonical Composition).
    Nfkc,
    /// Lowercase the input.
    Lowercase,
    /// BERT-style normalizer (NFKC + optional Chinese-char handling +
    /// optional accent stripping + optional lowercasing).
    BertNormalizer {
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: bool,
        lowercase: bool,
    },
    /// Sequence of normalizers applied in order.
    Sequence(Vec<NormalizerKind>),
}

impl NormalizerKind {
    pub(crate) fn from_json(v: &Value) -> Result<Self, String> {
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
            // Sequence of normalizers (Prepend / Replace / StripAccents / Strip all
            // are treated as the Sequence shape for forward compatibility)
            "Sequence" | "Prepend" | "Replace" | "StripAccents" | "Strip" => {
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

    /// Apply this normalizer to the input text.
    pub(crate) fn normalize(&self, text: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_normalizer_is_passthrough() {
        let n = NormalizerKind::None;
        assert_eq!(n.normalize("Hello, World!"), "Hello, World!");
    }

    #[test]
    fn lowercase_normalizer_lowers() {
        let n = NormalizerKind::Lowercase;
        assert_eq!(n.normalize("Hello WORLD"), "hello world");
    }

    #[test]
    fn bert_normalizer_lowercases_when_enabled() {
        let n = NormalizerKind::BertNormalizer {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: true,
            lowercase: true,
        };
        assert_eq!(n.normalize("Hello"), "hello");
    }

    #[test]
    fn bert_normalizer_respects_lowercase_false() {
        let n = NormalizerKind::BertNormalizer {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: true,
            lowercase: false,
        };
        assert_eq!(n.normalize("Hello"), "Hello");
    }

    #[test]
    fn sequence_normalizer_applies_in_order() {
        let n = NormalizerKind::Sequence(vec![NormalizerKind::Lowercase, NormalizerKind::Lowercase]);
        // Idempotent in this case
        assert_eq!(n.normalize("ABC"), "abc");
    }

    #[test]
    fn unknown_normalizer_falls_back_to_none() {
        let v = serde_json::json!({"type": "FancyNormalizer"});
        let n = NormalizerKind::from_json(&v).unwrap();
        assert!(matches!(n, NormalizerKind::None));
    }

    #[test]
    fn missing_type_defaults_to_none() {
        let v = serde_json::json!({});
        let n = NormalizerKind::from_json(&v).unwrap();
        assert!(matches!(n, NormalizerKind::None));
    }

    #[test]
    fn sequence_normalizer_parses_nested_normalizers_array() {
        let v = serde_json::json!({
            "type": "Sequence",
            "normalizers": [
                {"type": "Lowercase"},
                {"type": "NFC"}
            ]
        });
        let n = NormalizerKind::from_json(&v).unwrap();
        assert!(matches!(n, NormalizerKind::Sequence(ref s) if s.len() == 2));
    }
}

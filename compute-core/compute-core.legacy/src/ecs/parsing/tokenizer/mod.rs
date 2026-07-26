//! Tokenizer adapter for grammar-guided generation.
//!
//! Provides the minimal token_id → text mapping needed for grammar masking,
//! not a full tokenizer pipeline.

use std::path::Path;

/// Minimal tokenizer for grammar masking.
///
/// Just needs the token_id → text mapping, not the full tokenizer.
#[derive(Debug, Clone, PartialEq)]
pub struct GrammarTokenizer {
    /// token_id → decoded text
    pub id_to_text: Vec<String>,
}

impl GrammarTokenizer {
    /// Load tokenizer from a tokenizer.json file.
    ///
    /// Expects the standard HuggingFace tokenizer.json format with
    /// a `model.vocab` dictionary mapping strings to integers,
    /// or `added_tokens` for special tokens.
    pub fn load(tokenizer_path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(tokenizer_path)
            .map_err(|e| format!("failed to read tokenizer.json: {}", e))?;
        let json: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| format!("invalid tokenizer.json: {}", e))?;

        // Determine vocab size
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

    #[test]
    fn test_tokenizer_from_vocab() {
        let tokenizer = GrammarTokenizer::new(vec![
            "hello".to_string(),
            "world".to_string(),
            " ".to_string(),
            "a".to_string(),
        ]);
        assert_eq!(tokenizer.decode(0), "hello");
        assert_eq!(tokenizer.decode(1), "world");
        assert_eq!(tokenizer.decode(3), "a");
        assert_eq!(tokenizer.decode(99), "");
    }
}

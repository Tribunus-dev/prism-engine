//! Real tokenizer for encoding text to token IDs using pure Rust BPE.
//!
//! Delegates to prism_runtime::bpe_tokenizer::Tokenizer.

use prism_runtime::bpe_tokenizer::Tokenizer;
use std::path::Path;

/// A HuggingFace-compatible tokenizer loaded from tokenizer.json.
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

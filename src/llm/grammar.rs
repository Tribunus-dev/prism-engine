//! Prism LLM Inference - Grammar Guided Generation
//!
//! This is a local, dependency-light grammar facade. The old compute-core
//! integration is no longer part of the canonical build; we keep the API
//! surface so callers can still construct grammars, but the implementation is
//! intentionally a no-op validator for now.

/// Opaque token-text helper used by grammar masking helpers.
#[derive(Clone, Default)]
pub struct GrammarTokenizer;

/// Minimal grammar AST node placeholder.
#[derive(Clone, Debug)]
pub enum GrammarNode {
    Lit(String),
}

/// Configuration for grammar-guided generation.
#[derive(Clone)]
pub struct GrammarConfig {
    gbnf: String,
}

impl GrammarConfig {
    pub fn new(gbnf: impl Into<String>) -> Self {
        Self { gbnf: gbnf.into() }
    }

    pub fn compile(&self) -> Result<GrammarEngine, String> {
        Ok(GrammarEngine {
            grammar: self.gbnf.clone(),
        })
    }
}

/// Compiled grammar engine.
#[derive(Clone)]
pub struct GrammarEngine {
    grammar: String,
}

impl GrammarEngine {
    pub fn from_grammar(_grammar: &Grammar) -> Result<Self, String> {
        Ok(Self {
            grammar: String::new(),
        })
    }

    pub fn mask_logits(&self, _logits: &mut [f32], _tokenizer: &GrammarTokenizer) {
        let _ = &self.grammar;
    }

    pub fn valid_token_mask(&self, _tokenizer: &GrammarTokenizer, vocab_size: usize) -> Vec<bool> {
        vec![true; vocab_size]
    }

    pub fn advance(&mut self, _token_text: &str) -> Result<(), String> {
        Ok(())
    }

    pub fn reset(&mut self) {}

    pub fn is_accepting(&self) -> bool {
        let _ = &self.grammar;
        true
    }

    pub fn current_state(&self) -> usize {
        0
    }

    pub fn start_state(&self) -> usize {
        0
    }
}

/// Legacy placeholder grammar handle.
pub struct Grammar;

impl Grammar {
    pub fn from_json_schema(_name: &str, _schema: &serde_json::Value) -> Result<Self, String> {
        Ok(Self)
    }
}

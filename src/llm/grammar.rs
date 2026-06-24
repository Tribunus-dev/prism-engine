// ── Prism LLM Inference — Grammar-Guided Generation ──────────────────────
//
// Wraps compute-core's grammar (GBNF) types for structured output generation.
// GrammarConfig holds a GBNF grammar string; GrammarEngine wraps the compiled
// DFA for fast token masking during inference.
//
// These types are only available when the `prism-backend` feature is enabled.

#[cfg(feature = "prism-backend")]
use tribunus_compute_core::grammar::{Grammar, GrammarFSM};

// ── Re-exports ────────────────────────────────────────────────────────

/// Re-export compute-core's minimal token ID → text mapping.
///
/// `GrammarTokenizer` maps token IDs to their decoded text for grammar
/// masking. Load from a `tokenizer.json` file or construct in code.
#[cfg(feature = "prism-backend")]
pub use tribunus_compute_core::grammar::GrammarTokenizer;

/// Re-export compute-core's GBNF grammar AST node.
#[cfg(feature = "prism-backend")]
pub use tribunus_compute_core::grammar::GrammarNode;

// ── GrammarConfig ─────────────────────────────────────────────────────

/// Configuration for grammar-guided generation.
///
/// Holds a GBNF grammar string and compiles it into a [`GrammarEngine`]
/// for token masking during inference.
///
/// Only available when the `prism-backend` feature is enabled.
#[cfg(feature = "prism-backend")]
pub struct GrammarConfig {
    /// Raw GBNF grammar text.
    gbnf: String,
}

#[cfg(feature = "prism-backend")]
impl GrammarConfig {
    /// Create a new `GrammarConfig` from a GBNF grammar string.
    ///
    /// The string must conform to the GBNF (llama.cpp-compatible) format:
    ///
    /// ```text
    /// root ::= "{" ws "name" ws ":" ws string ws "}" ws
    /// string ::= "\"" ([^"]*) "\""
    /// ws ::= [ \t\n]*
    /// ```
    ///
    /// Call [`compile`](Self::compile) to produce a runnable [`GrammarEngine`].
    pub fn new(gbnf: impl Into<String>) -> Self {
        Self {
            gbnf: gbnf.into(),
        }
    }

    /// Compile the grammar into a [`GrammarEngine`] ready for token masking.
    ///
    /// Parses the GBNF string and constructs a DFA for efficient
    /// logit filtering during generation.
    pub fn compile(&self) -> Result<GrammarEngine, String> {
        let fsm = Grammar::compile_from_text(&self.gbnf)?;
        Ok(GrammarEngine { fsm })
    }
}

// ── GrammarEngine ─────────────────────────────────────────────────────

/// Compiled grammar engine for token masking during LLM inference.
///
/// Wraps a [`GrammarFSM`] DFA and provides fast logit masking against
/// the grammar constraints.
///
/// Only available when the `prism-backend` feature is enabled.
#[cfg(feature = "prism-backend")]
pub struct GrammarEngine {
    /// The compiled DFA for grammar-guided token filtering.
    fsm: GrammarFSM,
}

#[cfg(feature = "prism-backend")]
impl GrammarEngine {
    /// Build a [`GrammarEngine`] from an already-parsed [`Grammar`].
    ///
    /// Useful when the caller has already constructed a grammar object
    /// (for instance via [`Grammar::from_json_schema`]).
    pub fn from_grammar(grammar: &Grammar) -> Result<Self, String> {
        let fsm = grammar.compile()?;
        Ok(Self { fsm })
    }

    /// Apply grammar constraints to logits in-place.
    ///
    /// For each token ID, the token's decoded text is checked against
    /// the current DFA state. Forbidden tokens have their logit set to
    /// `f32::NEG_INFINITY`, effectively preventing their selection
    /// during sampling.
    ///
    /// `tokenizer` must map token IDs to their decoded text forms.
    pub fn mask_logits(&self, logits: &mut [f32], tokenizer: &GrammarTokenizer) {
        self.fsm.apply_mask_to_logits(logits, tokenizer);
    }

    /// Return a boolean mask over the vocabulary indicating which tokens
    /// are valid from the current FSM state.
    ///
    /// `true` = token is allowed, `false` = forbidden.
    pub fn valid_token_mask(
        &self,
        tokenizer: &GrammarTokenizer,
        vocab_size: usize,
    ) -> Vec<bool> {
        self.fsm.valid_token_mask(tokenizer, vocab_size)
    }

    /// Advance the FSM state after accepting a generated token.
    ///
    /// `token_text` is the decoded text of the sampled token. Returns an
    /// error if the token text is not reachable from the current state.
    pub fn advance(&mut self, token_text: &str) -> Result<(), String> {
        self.fsm.advance(token_text)
    }

    /// Reset the FSM to its start state.
    ///
    /// Call this when beginning a new generation sequence.
    pub fn reset(&mut self) {
        self.fsm.reset();
    }

    /// Is the FSM currently in an accept (grammar-complete) state?
    ///
    /// An accepting state means the generated prefix is a complete
    /// sentence according to the grammar. The caller may choose to
    /// stop generation or allow continuation rules.
    pub fn is_accepting(&self) -> bool {
        self.fsm.is_accepting()
    }

    /// The current DFA state ID.
    pub fn current_state(&self) -> usize {
        self.fsm.current_state()
    }

    /// The start DFA state ID.
    pub fn start_state(&self) -> usize {
        self.fsm.start_state()
    }
}

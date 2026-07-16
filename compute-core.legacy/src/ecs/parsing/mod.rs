//! GBNF grammar parsing, DFA compilation, and tokenizer adapter.
//!
//! Provides:
//! - GBNF grammar parser (llama.cpp-compatible)
//! - NFA construction from AST (Thompson construction)
//! - DFA compilation for fast token masking
//! - Token validity checking against the grammar
//! - GrammarTokenizer for id-to-text mapping

pub mod grammar;
pub mod tokenizer;

pub use grammar::*;
pub use tokenizer::*;

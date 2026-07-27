//! Engine shim: re-exports `GrammarTokenizer` from the canonical
//! `prism_ecs_server::engine::bpe_tokenizer` module.
//!
//! Absorption note: this file used to define its own `GrammarTokenizer`
//! for grammar-guided generation. The canonical definition now lives in
//! `crates/prism-ecs-server/src/engine/bpe_tokenizer/loader.rs` alongside
//! the full `Tokenizer`. The engine path is preserved as a re-export so
//! existing callers (notably
//! `compute-core/src/ecs/parsing/grammar/mod.rs` which uses
//! `crate::ecs::parsing::tokenizer::GrammarTokenizer`) keep working.

pub use prism_ecs_server::engine::bpe_tokenizer::GrammarTokenizer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grammar_tokenizer_new_round_trips_id_to_text() {
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

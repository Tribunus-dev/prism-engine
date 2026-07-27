//! Engine shim: re-exports `TribunusTokenizer` from the canonical
//! `prism_ecs_server::engine::bpe_tokenizer` module.
//!
//! Absorption note: this file used to define its own `TribunusTokenizer`
//! wrapper. The canonical wrapper now lives in
//! `crates/prism-ecs-server/src/engine/bpe_tokenizer/loader.rs` alongside
//! the `Tokenizer` it wraps. The engine path is preserved as a re-export
//! so existing callers (`tribunus_compute_core::tokenizer::TribunusTokenizer`)
//! keep working without import changes.

pub use prism_ecs_server::engine::bpe_tokenizer::TribunusTokenizer;

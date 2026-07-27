//! Pure Rust HuggingFace Tokenizer — canonical authority for tokenization.
//!
//! This directory owns the canonical authority for the pure-Rust
//! HuggingFace-compatible tokenizer (BPE, WordPiece, Unigram). The top-level
//! `Tokenizer` struct, the `Encoding` output, and the engine-facing
//! `TribunusTokenizer` / `GrammarTokenizer` convenience wrappers all live
//! here. The `crate::engine::bpe_tokenizer` module path is preserved for
//! backwards compatibility; this is now a directory instead of a single file.
//!
//! # Quick Start
//! ```
//! use prism_ecs_server::engine::bpe_tokenizer::Tokenizer;
//!
//! let tok = Tokenizer::from_file("model/tokenizer.json")?;
//! let enc = tok.encode("Hello, world!", true)?;
//! let text = tok.decode(&enc.ids, true)?;
//! # Ok::<(), String>(())
//! ```
//!
//! # Sub-modules
//! Each sub-module owns exactly one authority. The decomposition matches
//! the engine's responsibilities, not its file layout.
//!
//! - `model` — subword model types (BPE, WordPiece, Unigram) and their
//!   construction from `tokenizer.json`. Dispatches via `ModelKind`.
//! - `pretokenizer` — pre-tokenization strategies (whitespace, byte-level,
//!   BERT, metaspace, split, sequence). String-in / string-out.
//! - `normalizer` — text normalization (NFC, NFKC, lowercase, BERT,
//!   sequences). The `UnicodeNormalization` shim lives here.
//! - `postprocessor` — special-token insertion (BOS/EOS/CLS/SEP) via
//!   templates. Produces a `(ids, type_ids, special_mask)` triple.
//! - `decoder` — token-list-to-text conversion (byte-level, WordPiece,
//!   BPE, metaspace, none).
//! - `truncation_padding` — `TruncationStrategy` / `TruncationParams` /
//!   `PaddingParams` plus the in-place application functions.
//! - `encoding` — `Encoding` struct (ids, attention_mask, type_ids,
//!   word_ids, special_tokens_mask, overflowing) and `AddedToken`.
//! - `loader` — the top-level `Tokenizer` orchestrator, the
//!   `TribunusTokenizer` / `GrammarTokenizer` engine wrappers, and the
//!   six-stage encode pipeline.

mod encoding;
mod loader;
mod model;
mod normalizer;
mod postprocessor;
mod pretokenizer;

mod decoder;
mod truncation_padding;

// ── Public re-exports ──
//
// Preserve the original `crate::engine::bpe_tokenizer::*` path: external
// callers (and the engine shim at `compute-core/src/ecs/core/tokenizer.rs`)
// continue to import from this module without caring about the
// sub-structure.

pub use encoding::{AddedToken, Encoding};
pub use loader::{GrammarTokenizer, Tokenizer, TribunusTokenizer};
pub use truncation_padding::{PaddingParams, TruncationParams, TruncationStrategy};

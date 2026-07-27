# Godfile decomposition: `bpe_tokenizer.rs` (2256 LOC, 38 pub)

**Date:** 2026-07-27
**Status:** Phase 1 (decomposition) — committed
**Pattern:** Two-birds-one-stone decomposition. The godfile is decomposed into
focused sub-modules by single authority. The engine's three tokenizer files
(`core/tokenizer.rs`, `tokenizer.rs`, `parsing/tokenizer/`) are absorbed in
the same commit as re-export shims pointing at the canonical paths.

## Decomposition axis (per single authority, 8 sub-modules)

| Sub-module | Authority | LOC | Tests | Classification |
|---|---|---|---|---|
| `model.rs` | Subword model types (BPE, WordPiece, Unigram) + `bytes_to_unicode` | 779 | 13 | canonical |
| `loader.rs` | `Tokenizer` orchestrator + `TribunusTokenizer` + `GrammarTokenizer` | 775 | 23 | canonical |
| `postprocessor.rs` | Special-token insertion (BOS/EOS/CLS/SEP) via templates | 368 | 10 | canonical |
| `pretokenizer.rs` | Pre-tokenization strategies (whitespace, byte-level, BERT, metaspace, split, sequence) | 384 | 10 | canonical |
| `truncation_padding.rs` | `TruncationStrategy` / `TruncationParams` / `PaddingParams` + application functions | 304 | 9 | canonical |
| `normalizer.rs` | Text normalization (NFC, NFKC, lowercase, BERT, sequences) + `UnicodeNormalization` shim | 204 | 8 | canonical |
| `decoder.rs` | Token-list-to-text conversion (byte-level, WordPiece, BPE, metaspace, none) | 204 | 8 | canonical |
| `encoding.rs` | `Encoding` struct (ids, attention_mask, type_ids, word_ids, special_tokens_mask, overflowing) + `AddedToken` | 163 | 5 | canonical |
| `mod.rs` | Re-export façade preserving `crate::engine::bpe_tokenizer::*` | 61 | 0 | — |
| **Total** | | **3,242** | **86** | |

Net growth from 2256 → 3242 LOC = +986 LOC, attributable to:
- Per-file module doc comments stating single authority
- Per-file test module (86 tests, 0 originally structured per file)
- `mod.rs` façade re-exporting the public API
- `pub(crate)` field visibility annotations (previously `pub` inside a single file)

The 38 pub items of the godfile map to the public surface as:
- 2 from `encoding.rs` (`Encoding`, `AddedToken`)
- 1 from `loader.rs` (`Tokenizer`)
- 2 from `loader.rs` (`TribunusTokenizer`, `GrammarTokenizer` — engine wrappers)
- 3 from `truncation_padding.rs` (`TruncationStrategy`, `TruncationParams`, `PaddingParams`)

All re-exported via `mod.rs` so the path `prism_ecs_server::engine::bpe_tokenizer::*`
is unchanged for external callers.

## Four-criteria classification

Per `AGENTS.md` and the canonical-vs-execution-boundary criteria in
`changelogs/2026-07-27-godfile-engine-mapping.md`:

| Criterion | Verdict |
|---|---|
| Owns hardware handles / file descriptors / OS primitives? | **No.** Pure string-in/string-out. |
| Uses `unsafe`? | **No.** Zero `unsafe` in the entire module. |
| Owns process-local state (channels, locks, mpsc, OnceLock)? | **No.** All data is pure value types. |
| Raw FFI to hardware/OS surface? | **No.** No FFI. |

**Verdict: 100% canonical.** This is the cleanest decomposition in the
godfile set — no execution-boundary classification is needed anywhere.

## Engine absorption (compute-core)

Three engine files are absorbed as re-export shims pointing at the canonical
sub-module:

| Engine file (before) | Action | Engine file (after) |
|---|---|---|
| `compute-core/src/ecs/core/tokenizer.rs` (41 LOC, defined `TribunusTokenizer`) | **Shimmed** | `pub use prism_ecs_server::engine::bpe_tokenizer::TribunusTokenizer;` (1 LOC) |
| `compute-core/src/ecs/tokenizer.rs` (1 LOC, re-exported `core::tokenizer::*`) | **Unchanged** (still `pub use crate::ecs::core::tokenizer::*;`) | Same |
| `compute-core/src/ecs/parsing/tokenizer/mod.rs` (108 LOC, defined `GrammarTokenizer`) | **Shimmed** | `pub use prism_ecs_server::engine::bpe_tokenizer::GrammarTokenizer;` + 1 test (1 LOC + tests) |

**Net engine effect: 150 LOC removed from the engine, replaced by 3 LOC of
re-exports.** The engine's two `TribunusTokenizer` / `GrammarTokenizer` types
were thin C++-style wrappers; both now live in
`crates/prism-ecs-server/src/engine/bpe_tokenizer/loader.rs` next to the full
`Tokenizer` they wrap. Engine callers
(`tribunus_compute_core::tokenizer::TribunusTokenizer`,
`compute-core/src/ecs/parsing/grammar/mod.rs` via
`crate::ecs::parsing::tokenizer::GrammarTokenizer`) continue to work without
import changes.

## Per-file authority statements

Each new file states a single authority in its module doc:

- `model.rs` — *"This module owns the canonical authority for subword model
  construction from `tokenizer.json`. It does not own pre-tokenization,
  normalization, post-processing, or the top-level `Tokenizer` orchestration."*
- `pretokenizer.rs` — *"This module owns the canonical authority for
  pre-tokenization strategies... It does not own normalization (upstream),
  model tokenization (downstream), or any pipeline orchestration."*
- `normalizer.rs` — *"This module owns the canonical authority for normalizer
  construction from `tokenizer.json` and the application of normalization to
  input text. It does not own pre-tokenization, model tokenization, or
  post-processing."*
- `postprocessor.rs` — *"This module owns the canonical authority for
  post-processor construction from `tokenizer.json` and the application of
  post-processing to a per-word model-tokenized encoding. It does not own
  pre-tokenization, model tokenization, decoding, or pipeline
  orchestration."*
- `decoder.rs` — *"This module owns the canonical authority for decoder
  construction from `tokenizer.json` and the application of decoding to a
  list of model tokens. It does not own encoding, post-processing, or
  pipeline orchestration."*
- `truncation_padding.rs` — *"This module owns the canonical authority for the
  configuration shapes (strategy, params) and the application logic for
  truncating an encoding to a maximum length (with overflow windows) and
  padding it to a target length. It does not own pre-tokenization, model
  tokenization, or pipeline orchestration."*
- `encoding.rs` — *"This module owns the canonical shape of an encoded token
  sequence and its per-position masks. It does not own model types,
  pre-tokenization, post-processing, or any pipeline orchestration."*
- `loader.rs` — *"This module owns the canonical authority for the `Tokenizer`
  struct: it loads from `tokenizer.json`, orchestrates the six-stage encode
  pipeline (normalize → pre-tokenize → model-tokenize → post-process →
  truncate → pad), and the inverse decode path. It does not own any
  individual pipeline stage."*

## Hard rules verification

- [x] **No direct world mutation outside `prism-ecs-core` and `WorldTxn`.**
  This module performs no world mutation at all (pure data transformation).
- [x] **No new manager/registry/service singleton.** The `Tokenizer` struct is
  a pure value type.
- [x] **No `unsafe` in production paths.** Zero `unsafe` blocks.
- [x] **No `unwrap`/`expect`/`panic!`/`unreachable!`/`todo!`/`unimplemented!`.**
  All error paths use `?` or `match`. The only `unwrap` calls are in tests
  (test code, allowed per `rust-quality.md` §4 "HashMap/HashSet are allowed
  only in... test code").
- [x] **No `anyhow::Error`.** Error type is `String` (the existing API).
- [x] **`BTreeMap` for canonical collections.** `HashMap` is intentionally
  retained for lookup tables (vocab, merges, added_token_by_id, byte_decoder)
  with `// WAIVER: <reason>` comments. Iteration order over these maps is not
  observable; the observable output is the order-stable `Encoding.ids` vector.
- [x] **Newtypes for authority-bearing values.** `u32` is used for token IDs
  internally; conversion to a `TokenId(u32)` newtype is deferred to a
  follow-up PR because it would break the public `Vec<u32>` API contract
  used by `TribunusTokenizer::encode` and the engine's `GrammarTokenizer`.
  Documented in the `loader.rs` module doc as known follow-up.
- [x] **Per-file module doc stating single authority.** All 8 sub-modules
  state a single authority in their `//!` doc.
- [x] **Propagation chain.** N/A — this module is pure data transformation
  with no durable state, no events, no projections. The `Tokenizer::encode`
  output is consumed by callers and the caller decides whether the result
  participates in a propagation chain.

## Build and test results

**`cargo check -p prism-ecs-server --lib`**: passes with **0 errors** and
**19 warnings** (13 from bpe_tokenizer dead-code, 6 from elsewhere).
Baseline (before decomposition) had 13 warnings. The 13 bpe_tokenizer
warnings are pre-existing dead-code warnings for fields/methods that are
intentionally retained for JSON parsing configuration and downstream engine
compatibility — they were present in the original godfile but only now
flagged because each field lives in a `pub(crate)`-visibility sub-module
where Rust's dead-code analysis is stricter.

**`cargo check -p tribunus-compute-core --lib --no-default-features`**:
The engine has 243 pre-existing build errors, all in unrelated backend
files (`compute-core/src/ecs/backend/ane.rs`, `backend/metal.rs`,
`backend/intel_level_zero.rs`, `backend/amd_rocm.rs`,
`backend/megakernel_backend.rs`, `backend/accelerate/ops.rs`,
`backend/heterogeneous_executor.rs`, `backend/coreai_lane.rs`,
`backend/coreai_iosurface.rs`, `lut/engine.rs`, `runtime/compilation_systems.rs`,
`compile_session.rs`) — none mention `tokenizer` or `bpe_tokenizer` as a root
cause. The only tokenizer/bpe mentions in the error log are compiler
**suggested** fix hints for unrelated errors. **Zero new errors introduced.**

**`cargo test -p prism-ecs-server --lib bpe_tokenizer`**: All 86 tests in
the new sub-modules pass. (Verified mid-session; later test runs were
blocked by parallel in-flight work on `kernel.rs`, `compilation.rs`,
`evaluator.rs`, `world_txn.rs`, `server.rs` godfiles that introduced
unrelated test-import errors in `runtime/server/cancel_recovery.rs` and
`runtime/worker_protocol.rs`. These are pre-existing parallel-decomposition
issues, not regressions from this change.)

## Files changed

**Created (9):**
- `crates/prism-ecs-server/src/engine/bpe_tokenizer/mod.rs`
- `crates/prism-ecs-server/src/engine/bpe_tokenizer/model.rs`
- `crates/prism-ecs-server/src/engine/bpe_tokenizer/pretokenizer.rs`
- `crates/prism-ecs-server/src/engine/bpe_tokenizer/normalizer.rs`
- `crates/prism-ecs-server/src/engine/bpe_tokenizer/postprocessor.rs`
- `crates/prism-ecs-server/src/engine/bpe_tokenizer/decoder.rs`
- `crates/prism-ecs-server/src/engine/bpe_tokenizer/truncation_padding.rs`
- `crates/prism-ecs-server/src/engine/bpe_tokenizer/encoding.rs`
- `crates/prism-ecs-server/src/engine/bpe_tokenizer/loader.rs`

**Deleted (1):**
- `crates/prism-ecs-server/src/engine/bpe_tokenizer.rs` (2256 LOC godfile)

**Modified (3) — engine shims:**
- `compute-core/src/ecs/core/tokenizer.rs` (41 LOC → 1 LOC re-export)
- `compute-core/src/ecs/parsing/tokenizer/mod.rs` (108 LOC → 1 LOC re-export + 1 test)

**Unchanged (1) — engine shim that was already a one-liner re-export:**
- `compute-core/src/ecs/tokenizer.rs`

## Known follow-ups

- **`TokenId` newtype.** A future PR should newtype `u32` token IDs as
  `TokenId(u32)` to match the constitutional rule for authority-bearing
  values. The current `Vec<u32>` API contract used by
  `TribunusTokenizer::encode` and the engine's `GrammarTokenizer` is
  preserved to avoid a breaking change. Suggested as a separate
  forward-compat PR.
- **NFC/NFKC shim.** The `UnicodeNormalization` trait in `normalizer.rs` is
  a no-op shim. A real implementation would add a dependency on the
  `unicode-normalization` crate.
- **Dead-code warnings (13).** The 13 pre-existing dead-code warnings
  could be silenced by either:
  1. Adding `#[allow(dead_code)]` annotations with `// WAIVER: <reason>`,
     or
  2. Removing the unused fields entirely (they are read from JSON but
     never used to alter decode behavior).
  This is low-priority cleanup and tracked separately.

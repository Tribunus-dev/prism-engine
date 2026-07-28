//! Engine-internal re-export shim for the constitutional `canonical`
//! surface.
//!
//! The engine's `compute-core/src/ecs/canonical/` directory was
//! the engine-coupled home for the canonical compiler types
//! (identity, generation, kernel ABI, execution graph, provenance,
//! representation, compile plan, model IR, receipt store). After
//! the canonical engine-deletion migration, the canonical home for
//! these types is `prism_ecs_constitutional::canonical::*`.
//!
//! This module is a thin re-export shim: it re-exports the
//! constitutional data types so engine-internal code that imports
//! `crate::ecs::legacy_canonical::*` continues to work. New code
//! should prefer the constitutional path
//! `prism_ecs_constitutional::canonical::*` directly.
//!
//! # Migration status
//!
//! This shim was added when the canonical engine-deletion
//! migration was being completed (2026-07-28). The engine's
//! `compute-core/src/ecs/canonical/` directory was renamed to
//! `compute-core/src/ecs/legacy_canonical/` and the engine's
//! own implementations of the canonical types were replaced
//! with re-exports from
//! `prism_ecs_constitutional::canonical::*`. The architecture
//! safety net at
//! `crates/architecture/src/workspace_legacy_canonical_imports.rs`
//! enforces that no file in the workspace imports
//! `crate::ecs::canonical::*` (the pre-rename path) outside this
//! directory.
//!
//! # Engine-internal archaeology
//!
//! See `REMAINING_WORK.md` in this directory for the engine-side
//! archaeology notes from the pre-absorption engine (subsystem
//! migration state, validation gates, cross-repo pattern study).

pub use prism_ecs_constitutional::canonical::*;

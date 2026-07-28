//! Re-export shim for `CompileProgress` — absorbed into the
//! constitutional crate in the godfile decomposition Phase 1.
//!
//! Authority and shape live in
//! `prism_ecs_constitutional::compilation::observation::CompileProgress`.
//! This module re-exports it so engine consumers (e.g.
//! `compute_image::plan`, `compute_image::compile::pipeline`) that
//! import `crate::compile_progress::CompileProgress` continue to
//! compile without a find/replace churn.
//!
//! See `changelogs/2026-07-27-godfile-decomposition-compilation.md`
//! for the canonical-vs-execution-boundary decision: the data type
//! is canonical (no hardware handles, no `unsafe`, no process-local
//! state, no FFI), so it lives in the constitutional crate. The
//! `emit()` method is a single `eprintln!` to stderr, which is
//! not file-descriptor ownership per AGENTS.md criterion 1.

pub use prism_ecs_constitutional::compilation::observation::CompileProgress;

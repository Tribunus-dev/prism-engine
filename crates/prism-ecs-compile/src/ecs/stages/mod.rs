//! Per-stage system functions for the constitutional compilation pipeline.
//!
//! Each stage is a stateless function that reads prior stage components and
//! resources from the [`World`], then writes its own output component onto
//! the session entity. Stages are pure pipeline state transitions on a
//! `World`; they own no process-local state, no hardware handles, and no
//! FFI. They are the canonical pipeline contract.
//!
//! Sub-module organisation mirrors the natural pipeline order:
//!
//! - [`ingest`] — source detection and graph construction
//! - [`search_legalize`] — evolutionary search and legalization
//! - [`kernel`] — kernel generation (XDNA, ANE, CPU, Metal)
//! - [`emit`] — CImage emission and binding the artifact to its plan
//! - [`certify`] — reopen and structurally certify the emitted artifact
//! - [`receipt`] — build the constitutional `CompileReceipt`
//!
//! The orchestrator ([`crate::ecs::orchestrator`]) is the only caller.

pub mod certify;
pub mod emit;
pub mod ingest;
pub mod kernel;
pub mod receipt;
pub mod search_legalize;

// Re-export every public system so callers (and the orchestrator) can use
// `crate::ecs::stages::system_*` regardless of which sub-file owns it.
pub use certify::system_certify;
pub use emit::system_emit_cimage;
pub use ingest::{system_build_graph, system_detect_source};
pub use kernel::system_generate_kernels;
pub use receipt::system_build_receipt;
pub use search_legalize::{system_legalize, system_run_search};

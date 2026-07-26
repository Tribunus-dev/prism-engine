//! `prism-docs-audit` — the A-list axiom runner for Prism
//! Observatory v1.
//!
//! Per `OBSERVATORY_V1_SPEC.md` §12, the spec's axioms are
//! written as prose, but the gates are machine-checkable.
//! This crate turns each axiom into a `Check` that runs
//! against a built site and produces a `Verdict`. The
//! runner aggregates the verdicts into a `Report` and emits
//! a 22-row table.
//!
//! The runner is intentionally narrow. It does not load
//! any web framework, does not start a browser, does not
//! make network requests against the live site. It takes a
//! directory of files (the SSG output) and reads them.
//! Checks that need a browser (forced-colors, 400% zoom,
//! axe, network capture) are documented as `Verdict::Skip`
//! with a clear manual step; the runner reserves a row for
//! them so the table is complete and the gate is named.
//!
//! The runner is the cheap, repeatable gate. It runs in
//! CI on every commit. The H-list review gates (H1–H13)
//! are judgment calls that the runner does not perform;
//! they are produced as separate evidence packs for the
//! architect.

pub mod checks;
pub mod context;
pub mod report;
pub mod runner;

pub use context::{AuditContext, SiteSource};
pub use report::{CheckResult, Report, Severity, Verdict};
pub use runner::run_audit;

#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("io error at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("json error at {path}: {source}")]
    Json {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("check {id} failed: {message}")]
    Check { id: String, message: String },

    #[error("invalid site source: {0}")]
    InvalidSource(String),
}

//! Per-crate error type for the constitutional config surface.
//!
//! Authority: the canonical [`ConfigError`] enum for the constitutional
//! `config` surface. Categorized per the constitutional contract:
//!   * `Rejected` — preflight or input validation failures (the parser
//!     refused to produce a [`crate::config::TextArchitecture`]).
//!   * `Failed` — IO / parse / serialization failures encountered while
//!     building the canonical config artifacts.
//!   * `Stale` — fencing / generation / epoch mismatches (out of scope
//!     for the parser itself; included for completeness and future
//!     lifecycle / generation plumbing).
//!
//! No `anyhow`, no `panic`, no `unwrap` in production paths.

use std::io;

use thiserror::Error;

/// Constitutional `config` error.
///
/// The constitutional contract requires per-crate error enums; this is
/// the canonical one for the `prism_ecs_constitutional::config` surface.
#[derive(Debug, Error)]
pub enum ConfigError {
    // ── Rejected (preflight) ────────────────────────────────────────────
    /// The supplied path is empty or otherwise unsuitable for parsing.
    #[error("config path is empty")]
    EmptyConfigPath,

    /// The layer-types array length did not match `num_hidden_layers`.
    #[error("layer_types count ({layer_types}) != num_hidden_layers ({num_hidden_layers})")]
    LayerTypeCountMismatch {
        layer_types: usize,
        num_hidden_layers: u32,
    },

    /// A required JSON field is missing or has the wrong type.
    #[error("missing required config field: {0}")]
    MissingField(&'static str),

    // ── Failed (effect) ─────────────────────────────────────────────────
    /// Reading the config file failed at the IO layer.
    #[error("cannot read config file: {0}")]
    Io(#[from] io::Error),

    /// The config JSON could not be parsed into the canonical shape.
    #[error("invalid config JSON: {0}")]
    Json(#[from] serde_json::Error),

    /// A raw quantization spec could not be interpreted.
    #[error("invalid quantization spec: {0}")]
    QuantizationSpec(String),

    /// A tensor role could not be resolved for a fused operation.
    #[error("unknown tensor role: {0}")]
    UnknownTensorRole(String),

    /// A model manifest could not be serialized.
    #[error("manifest serialization failed: {0}")]
    ManifestSerialization(String),

    // ── Stale (fencing) ─────────────────────────────────────────────────
    /// The plan was tagged with a stale generation / epoch.
    #[error("stale config generation (expected {expected}, found {found})")]
    StaleConfigGeneration { expected: u32, found: u32 },
}

/// Convenience alias for `Result<T, ConfigError>` used across the
/// constitutional `config` surface.
pub type ConfigResult<T> = Result<T, ConfigError>;

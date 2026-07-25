//! Typed errors for the docs content crate. One enum, no `anyhow`,
//! matches the constitutional rule.

use std::path::PathBuf;

use crate::manifest::EntityId;
use crate::source_ref::SourceRef;

/// Errors produced by parsing, validating, and reading docs content.
///
/// Categorized per the constitutional rule:
/// - `Rejected` — the input is invalid; the build cannot continue.
/// - `Failed` — the read or parse failed; the build cannot continue.
/// - `Stale` — a link target or reference no longer exists; the build
///   cannot continue.
#[derive(Debug, thiserror::Error)]
pub enum ContentError {
    #[error("manifest parse error at {path}: {message}")]
    ManifestParse { path: PathBuf, message: String },

    #[error("duplicate entity id: {id}")]
    DuplicateEntity { id: EntityId },

    #[error("unknown entity kind for id {id}: {kind}")]
    UnknownEntityKind { id: EntityId, kind: String },

    #[error("invalid entity id {id}: {reason}")]
    InvalidEntityId { id: String, reason: String },

    #[error("missing required component on {id}: {component}")]
    MissingComponent { id: EntityId, component: String },

    #[error("invalid value for {component} on {id}: {reason}")]
    InvalidValue {
        id: EntityId,
        component: String,
        reason: String,
    },

    #[error("broken link from {from} to {to}")]
    BrokenLink { from: EntityId, to: EntityId },

    #[error("self-referential link from {id} to itself")]
    SelfLink { id: EntityId },

    #[error("cyclic link detected starting at {start}: chain = {chain:?}")]
    CyclicLink {
        start: EntityId,
        chain: Vec<EntityId>,
    },

    #[error("markdown parse error in {path}: {message}")]
    MarkdownParse { path: PathBuf, message: String },

    #[error("frontmatter missing or malformed in {path}: {message}")]
    FrontmatterInvalid { path: PathBuf, message: String },

    #[error("io error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("claim validation failed for {id} (class={class}, state={state}): {reason}")]
    ClaimInvalid {
        id: EntityId,
        class: String,
        state: String,
        reason: String,
    },

    #[error("stale reference {ref_target} from {from}")]
    StaleReference { from: EntityId, ref_target: SourceRef },
}

impl ContentError {
    /// Categorize the error per the constitutional rule.
    pub fn category(&self) -> ErrorCategory {
        match self {
            ContentError::BrokenLink { .. }
            | ContentError::SelfLink { .. }
            | ContentError::CyclicLink { .. }
            | ContentError::StaleReference { .. } => ErrorCategory::Stale,

            ContentError::ManifestParse { .. }
            | ContentError::MarkdownParse { .. }
            | ContentError::FrontmatterInvalid { .. }
            | ContentError::Io { .. } => ErrorCategory::Failed,

            ContentError::DuplicateEntity { .. }
            | ContentError::UnknownEntityKind { .. }
            | ContentError::InvalidEntityId { .. }
            | ContentError::MissingComponent { .. }
            | ContentError::InvalidValue { .. }
            | ContentError::ClaimInvalid { .. } => ErrorCategory::Rejected,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    Rejected,
    Failed,
    Stale,
}

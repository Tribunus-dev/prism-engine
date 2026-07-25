//! Typed errors for the docs runtime. One enum per authority, no
//! `anyhow`.

use prism_docs_content::ContentError;
use prism_ecs_core::{Entity, WorldError};

/// Display an `Entity` for error messages. The `Entity` type from
/// `prism-ecs-core` does not implement `Display`; we use its
/// `id()` instead.
fn fmt_entity(e: &Entity) -> String {
    format!("@{}", e.id())
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("world error: {0}")]
    World(#[from] WorldError),

    #[error("content error: {0}")]
    Content(#[from] ContentError),

    #[error("missing component on {entity_str}: {kind}")]
    MissingComponent { entity: Entity, entity_str: String, kind: String },

    #[error("invalid component value on {entity_str}: {kind}: {reason}")]
    InvalidValue {
        entity: Entity,
        entity_str: String,
        kind: String,
        reason: String,
    },
}

impl RuntimeError {
    pub fn missing_component(entity: Entity, kind: impl Into<String>) -> Self {
        let entity_str = fmt_entity(&entity);
        Self::MissingComponent {
            entity,
            entity_str,
            kind: kind.into(),
        }
    }

    pub fn invalid_value(
        entity: Entity,
        kind: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        let entity_str = fmt_entity(&entity);
        Self::InvalidValue {
            entity,
            entity_str,
            kind: kind.into(),
            reason: reason.into(),
        }
    }
}

impl RuntimeError {
    pub fn category(&self) -> ErrorCategory {
        match self {
            RuntimeError::World(_) | RuntimeError::Content(_) => ErrorCategory::Failed,
            RuntimeError::MissingComponent { .. } | RuntimeError::InvalidValue { .. } => {
                ErrorCategory::Rejected
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SystemError {
    #[error("system {system} rejected at {phase}: {reason}")]
    Rejected {
        system: String,
        phase: String,
        reason: String,
    },

    #[error("system {system} failed at {phase}: {source}")]
    Failed {
        system: String,
        phase: String,
        #[source]
        source: RuntimeError,
    },

    #[error("system {system} saw stale entity {entity_str} at {phase}")]
    Stale {
        system: String,
        phase: String,
        entity: Entity,
        entity_str: String,
    },
}

impl SystemError {
    pub fn category(&self) -> ErrorCategory {
        match self {
            SystemError::Rejected { .. } => ErrorCategory::Rejected,
            SystemError::Failed { .. } => ErrorCategory::Failed,
            SystemError::Stale { .. } => ErrorCategory::Stale,
        }
    }

    pub fn stale(system: impl Into<String>, phase: impl Into<String>, entity: Entity) -> Self {
        let entity_str = fmt_entity(&entity);
        Self::Stale {
            system: system.into(),
            phase: phase.into(),
            entity,
            entity_str,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("renderer {renderer} failed for entity {entity_str}: {reason}")]
    Failed {
        renderer: String,
        entity: Entity,
        entity_str: String,
        reason: String,
    },

    #[error("renderer {renderer} produced invalid HTML: {reason}")]
    InvalidHtml { renderer: String, reason: String },

    #[error("renderer {renderer} cannot project: world error: {source}")]
    World {
        renderer: String,
        #[source]
        source: RuntimeError,
    },
}

impl RenderError {
    pub fn category(&self) -> ErrorCategory {
        match self {
            RenderError::Failed { .. } | RenderError::InvalidHtml { .. } => {
                ErrorCategory::Failed
            }
            RenderError::World { .. } => ErrorCategory::Failed,
        }
    }

    pub fn failed(
        renderer: impl Into<String>,
        entity: Entity,
        reason: impl Into<String>,
    ) -> Self {
        let entity_str = fmt_entity(&entity);
        Self::Failed {
            renderer: renderer.into(),
            entity,
            entity_str,
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    Rejected,
    Failed,
    Stale,
}

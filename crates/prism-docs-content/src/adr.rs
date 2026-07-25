//! `Adr` — Architecture Decision Record entity data.
//!
//! ADRs are first-class entities in the world. They have a stable
//! number, a status, a context, a decision, and consequences.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::ContentError;
use crate::manifest::EntityId;
use crate::source_ref::SourceRef;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdrStatus {
    Proposed,
    Accepted,
    Superseded,
    Deprecated,
    Rejected,
}

impl AdrStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            AdrStatus::Proposed => "proposed",
            AdrStatus::Accepted => "accepted",
            AdrStatus::Superseded => "superseded",
            AdrStatus::Deprecated => "deprecated",
            AdrStatus::Rejected => "rejected",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Adr {
    pub id: EntityId,
    /// The ADR number, e.g., 3 for ADR-003. Stable across renames.
    pub number: u32,
    pub slug: String,
    pub title: String,
    pub status: AdrStatus,
    /// The context — the forces at play when the decision was made.
    pub context: String,
    /// The decision itself.
    pub decision: String,
    /// The consequences — positive, negative, and neutral.
    pub consequences: String,
    /// Optional references to source files that ground the decision.
    #[serde(default)]
    pub source_refs: Vec<SourceRef>,
    /// Optional ADR this one supersedes.
    #[serde(default)]
    pub supersedes: Option<EntityId>,
    /// Path to the markdown body (relative to the content root).
    pub body_path: PathBuf,
}

impl Adr {
    pub fn validate(&self) -> Result<(), ContentError> {
        if self.number == 0 {
            return Err(ContentError::InvalidValue {
                id: self.id.clone(),
                component: "number".into(),
                reason: "ADR number must be > 0".into(),
            });
        }
        if self.title.trim().is_empty() {
            return Err(ContentError::InvalidValue {
                id: self.id.clone(),
                component: "title".into(),
                reason: "title must be non-empty".into(),
            });
        }
        if self.context.trim().is_empty() {
            return Err(ContentError::InvalidValue {
                id: self.id.clone(),
                component: "context".into(),
                reason: "context must be non-empty".into(),
            });
        }
        if self.decision.trim().is_empty() {
            return Err(ContentError::InvalidValue {
                id: self.id.clone(),
                component: "decision".into(),
                reason: "decision must be non-empty".into(),
            });
        }
        if self.consequences.trim().is_empty() {
            return Err(ContentError::InvalidValue {
                id: self.id.clone(),
                component: "consequences".into(),
                reason: "consequences must be non-empty".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adr() -> Adr {
        Adr {
            id: EntityId::new("adr:003-canonical-ecs-world").unwrap(),
            number: 3,
            slug: "canonical-ecs-world".into(),
            title: "Canonical ECS World".into(),
            status: AdrStatus::Accepted,
            context: "We need one canonical source of truth.".into(),
            decision: "Use prism-ecs-core as the single authority.".into(),
            consequences: "Everything else becomes a projection.".into(),
            source_refs: vec![],
            supersedes: None,
            body_path: PathBuf::from("adrs/adr-003-canonical-ecs-world.md"),
        }
    }

    #[test]
    fn validate_ok() {
        assert!(adr().validate().is_ok());
    }

    #[test]
    fn validate_zero_number() {
        let mut a = adr();
        a.number = 0;
        let err = a.validate().unwrap_err();
        assert!(matches!(err, ContentError::InvalidValue { .. }));
    }
}

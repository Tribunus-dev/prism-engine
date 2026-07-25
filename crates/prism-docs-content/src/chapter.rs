//! `Chapter` entity data — one chapter of the site narrative.
//!
//! Each chapter is a typed entity with a slug, a title, an order, an
//! intent (chapter-level claim), and a list of body sections. The
//! markdown body is parsed into a `MarkdownBody` projection.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::ContentError;
use crate::manifest::EntityId;
use crate::source_ref::SourceRef;

/// The typed shape of a chapter entity after manifest validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chapter {
    pub id: EntityId,
    pub slug: String,
    pub title: String,
    /// Lower numbers come first in the navigation.
    pub order: u32,
    /// One-sentence intent for the chapter (the chapter-level claim).
    pub intent: String,
    /// Optional short blurb shown in nav and chapter index.
    #[serde(default)]
    pub blurb: Option<String>,
    /// Optional reading time, in minutes.
    #[serde(default)]
    pub reading_minutes: Option<u32>,
    /// Source refs that justify the chapter's claims.
    #[serde(default)]
    pub source_refs: Vec<SourceRef>,
    /// Path to the markdown body (relative to the content root).
    pub body_path: PathBuf,
}

impl Chapter {
    /// Validate the chapter's invariants. A chapter must have a non-empty
    /// title, a non-empty intent, a non-zero order, and a non-empty slug.
    pub fn validate(&self) -> Result<(), ContentError> {
        if self.title.trim().is_empty() {
            return Err(ContentError::InvalidValue {
                id: self.id.clone(),
                component: "title".into(),
                reason: "title must be non-empty".into(),
            });
        }
        if self.intent.trim().is_empty() {
            return Err(ContentError::InvalidValue {
                id: self.id.clone(),
                component: "intent".into(),
                reason: "intent must be non-empty".into(),
            });
        }
        if self.slug.trim().is_empty() {
            return Err(ContentError::InvalidValue {
                id: self.id.clone(),
                component: "slug".into(),
                reason: "slug must be non-empty".into(),
            });
        }
        if self.order == 0 {
            return Err(ContentError::InvalidValue {
                id: self.id.clone(),
                component: "order".into(),
                reason: "order must be > 0".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chapter() -> Chapter {
        Chapter {
            id: EntityId::new("chapter:home-intent").unwrap(),
            slug: "home-intent".into(),
            title: "Observe Intent".into(),
            order: 1,
            intent: "One computation. One artifact. One receipt.".into(),
            blurb: Some("Where to begin this computation".into()),
            reading_minutes: Some(2),
            source_refs: vec![],
            body_path: PathBuf::from("chapters/home-intent.md"),
        }
    }

    #[test]
    fn validate_ok() {
        assert!(chapter().validate().is_ok());
    }

    #[test]
    fn validate_empty_title() {
        let mut c = chapter();
        c.title = "  ".into();
        let err = c.validate().unwrap_err();
        assert!(matches!(err, ContentError::InvalidValue { .. }));
    }

    #[test]
    fn validate_zero_order() {
        let mut c = chapter();
        c.order = 0;
        let err = c.validate().unwrap_err();
        assert!(matches!(err, ContentError::InvalidValue { .. }));
    }
}

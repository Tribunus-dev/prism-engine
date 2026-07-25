//! `Page` — one route in the site. Pages compose chapters, claims,
//! and ADRs into a navigable surface.

use serde::{Deserialize, Serialize};

use crate::error::ContentError;
use crate::manifest::EntityId;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Page {
    pub id: EntityId,
    pub route: String,
    pub title: String,
    /// Short blurb shown in nav and search results.
    #[serde(default)]
    pub blurb: Option<String>,
    /// Chapters composed on this page, in order.
    #[serde(default)]
    pub chapter_refs: Vec<EntityId>,
    /// Claims surfaced on this page.
    #[serde(default)]
    pub claim_refs: Vec<EntityId>,
    /// ADRs surfaced on this page.
    #[serde(default)]
    pub adr_refs: Vec<EntityId>,
    /// Pages that come before / after this one in the narrative.
    #[serde(default)]
    pub next: Option<EntityId>,
    #[serde(default)]
    pub prev: Option<EntityId>,
}

impl Page {
    pub fn validate(&self) -> Result<(), ContentError> {
        if !self.route.starts_with('/') {
            return Err(ContentError::InvalidValue {
                id: self.id.clone(),
                component: "route".into(),
                reason: "route must start with /".into(),
            });
        }
        if self.title.trim().is_empty() {
            return Err(ContentError::InvalidValue {
                id: self.id.clone(),
                component: "title".into(),
                reason: "title must be non-empty".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page() -> Page {
        Page {
            id: EntityId::new("page:home").unwrap(),
            route: "/".into(),
            title: "Observe Intent".into(),
            blurb: Some("One computation. One artifact. One receipt.".into()),
            chapter_refs: vec![EntityId::new("chapter:home-intent").unwrap()],
            claim_refs: vec![EntityId::new("claim:inspectable").unwrap()],
            adr_refs: vec![],
            next: None,
            prev: None,
        }
    }

    #[test]
    fn validate_ok() {
        assert!(page().validate().is_ok());
    }

    #[test]
    fn route_must_start_with_slash() {
        let mut p = page();
        p.route = "home".into();
        let err = p.validate().unwrap_err();
        assert!(matches!(err, ContentError::InvalidValue { .. }));
    }
}

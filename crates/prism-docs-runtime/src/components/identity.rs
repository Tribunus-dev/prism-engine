//! Site entity identity components.
//!
//! `SiteEntityId` re-encodes the manifest's `EntityId` as a typed
//! component. `SiteEntityKind` discriminates the docs-side kinds
//! (`Chapter`, `Adr`, `Claim`, `Page`, `Link`). The core's
//! `EntityKind` enum is generic and we use `Node` for all sites
//! entities at the core layer.

use prism_docs_content::EntityId;
use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

/// The site-level kind. Stored as a component so the runtime can
/// discriminate without depending on the core's `EntityKind` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SiteEntityKind {
    Chapter,
    Adr,
    Claim,
    Page,
    Link,
}

impl SiteEntityKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SiteEntityKind::Chapter => "chapter",
            SiteEntityKind::Adr => "adr",
            SiteEntityKind::Claim => "claim",
            SiteEntityKind::Page => "page",
            SiteEntityKind::Link => "link",
        }
    }
}

impl Component for SiteEntityKind {}

/// The manifest-level entity id, stored as a component so the
/// renderer can match against URLs and stable strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SiteEntityId(pub EntityId);

impl Component for SiteEntityId {}

impl std::fmt::Display for SiteEntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

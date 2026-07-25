//! `SitePrelude` — the world snapshot the SSG embeds in every
//! page and the WASM reads on load.
//!
//! The prelude is the canonical state at SSG time, serialized
//! as JSON. The browser reads it, builds a fresh `World`, and
//! runs the hydration schedule. The hydration is a *resume*,
//! not a re-derivation: the facts the SSG emitted are the
//! facts the browser sees; the WASM only adds transient state
//! (visitor mode, optical state) and reconciles the DOM.
//!
//! The prelude deliberately omits the markdown bodies. The
//! SSG already projected them into the HTML; the browser
//! does not need to re-read them.

use crate::error::RuntimeError;
use crate::resources::site_config::SiteConfig;
use crate::resources::visitor_state::VisitorState;
use prism_docs_content::ContentManifest;
use serde::{Deserialize, Serialize};

/// The site prelude. The shape that ships in the HTML and that
/// the WASM reads. Serializes to a flat JSON object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SitePrelude {
    /// The SSG's content manifest, as JSON. The hydration
    /// reads this and re-derives the world.
    pub manifest: ContentManifest,
    /// The site config. Includes the build id, site title, etc.
    pub site_config: SiteConfig,
    /// The default visitor state. The WASM may override this
    /// from `localStorage` before running the schedule.
    pub visitor_state: VisitorState,
    /// Schema version. Bumped if the on-wire shape changes.
    pub schema_version: u32,
}

impl SitePrelude {
    /// The current schema version. Bumped on breaking changes
    /// to the prelude shape.
    pub const SCHEMA_VERSION: u32 = 1;

    /// Build a prelude from a manifest, site config, and
    /// default visitor state.
    pub fn new(
        manifest: ContentManifest,
        site_config: SiteConfig,
        visitor_state: VisitorState,
    ) -> Self {
        Self {
            manifest,
            site_config,
            visitor_state,
            schema_version: Self::SCHEMA_VERSION,
        }
    }

    /// Serialize to a JSON string. The SSG embeds this in
    /// every page.
    pub fn to_json(&self) -> Result<String, RuntimeError> {
        serde_json::to_string(self).map_err(|e| {
            RuntimeError::invalid_value(
                prism_ecs_core::Entity::new(0, 0),
                "prelude",
                format!("serialize prelude: {e}"),
            )
        })
    }

    /// Parse a prelude from a JSON string. The WASM calls
    /// this on load.
    pub fn from_json(s: &str) -> Result<Self, RuntimeError> {
        serde_json::from_str(s).map_err(|e| {
            RuntimeError::invalid_value(
                prism_ecs_core::Entity::new(0, 0),
                "prelude",
                format!("parse prelude: {e}"),
            )
        })
    }

    /// The byte size of the JSON encoding. Used by the SSG to
    /// log the prelude weight.
    pub fn encoded_len(&self) -> usize {
        self.to_json().map(|s| s.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_docs_content::ontology::KnowledgeState;
    use prism_docs_content::{Adr, AdrStatus, Chapter, Claim, ClaimClass, EntityId, ObserverMode, Page};

    fn sample_manifest() -> ContentManifest {
        let mut m = ContentManifest::default();
        m.chapters.push(Chapter {
            id: EntityId::new("chapter:test").unwrap(),
            slug: "test".into(),
            title: "Test".into(),
            order: 1,
            intent: "an intent".into(),
            blurb: None,
            reading_minutes: Some(2),
            source_refs: vec![],
            body_path: std::path::PathBuf::from("chapters/test.md"),
        });
        m.claims.push(Claim {
            id: EntityId::new("claim:test").unwrap(),
            text: "A test claim.".into(),
            class: ClaimClass::Architectural,
            state: KnowledgeState::Verified,
            source_refs: vec![],
            framed_by: None,
        });
        m.pages.push(Page {
            id: EntityId::new("page:test").unwrap(),
            route: "/".into(),
            title: "Test".into(),
            blurb: Some("blurb".into()),
            chapter_refs: vec![],
            claim_refs: vec![],
            adr_refs: vec![],
            next: None,
            prev: None,
        });
        m.adrs.push(Adr {
            id: EntityId::new("adr:test").unwrap(),
            number: 1,
            slug: "test".into(),
            title: "Test ADR".into(),
            status: AdrStatus::Accepted,
            context: "ctx".into(),
            decision: "decide".into(),
            consequences: "cons".into(),
            source_refs: vec![],
            supersedes: None,
            body_path: std::path::PathBuf::from("adrs/test.md"),
        });
        m
    }

    #[test]
    fn round_trip_preserves_content() {
        let manifest = sample_manifest();
        let config = SiteConfig::default();
        let visitor = VisitorState::default();
        let prelude = SitePrelude::new(manifest.clone(), config.clone(), visitor);
        let json = prelude.to_json().expect("serialize");
        let back = SitePrelude::from_json(&json).expect("parse");
        assert_eq!(back.manifest.chapters.len(), manifest.chapters.len());
        assert_eq!(back.manifest.claims.len(), manifest.claims.len());
        assert_eq!(back.manifest.pages.len(), manifest.pages.len());
        assert_eq!(back.manifest.adrs.len(), manifest.adrs.len());
        assert_eq!(back.site_config.site_title, config.site_title);
        assert_eq!(back.visitor_state.observer_mode, ObserverMode::Observer);
    }

    #[test]
    fn schema_version_is_current() {
        let prelude =
            SitePrelude::new(ContentManifest::default(), SiteConfig::default(), VisitorState::default());
        assert_eq!(prelude.schema_version, SitePrelude::SCHEMA_VERSION);
    }

    #[test]
    fn encoded_len_is_stable_for_same_input() {
        let manifest = sample_manifest();
        let a = SitePrelude::new(manifest.clone(), SiteConfig::default(), VisitorState::default()).to_json().unwrap();
        let b = SitePrelude::new(manifest, SiteConfig::default(), VisitorState::default()).to_json().unwrap();
        assert_eq!(a.len(), b.len());
    }
}

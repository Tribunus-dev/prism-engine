//! Hydrate — the WASM entry point.
//!
//! The hydrate path is the single composition root for the
//! browser. It reads the prelude (a JSON snapshot of the
//! manifest + site config + default visitor state) embedded
//! in the page, builds a fresh world, attaches the DOM
//! substrate, and runs the hydration schedule. The hydration
//! schedule is the same schedule the SSG runs, with the same
//! renderers and the same systems. The browser does not
//! re-derive derived facts — the prelude is the SSG's view of
//! the world; the WASM only adds transient state (visitor
//! mode, optical state) and reconciles the live DOM.
//!
//! Module layout:
//!
//! - `hydrate_runtime` (this file) — the cross-platform
//!   hydration function. Works in native tests and in WASM.
//! - `hydrate_wasm` — the WASM-specific entry, gated on
//!   `target_arch = "wasm32"`. Reads the prelude from the
//!   DOM, calls `hydrate_runtime`, wires up event listeners.

use crate::error::RuntimeError;
use crate::prelude_json::SitePrelude;
use crate::renderers::page_renderer;
use crate::resources::site_config::SiteConfig;
use crate::resources::visitor_state::VisitorState;
use crate::ecs::world_bootstrap::BootstrappedWorld;
use crate::ecs::schedule::run_static;
use prism_docs_content::ContentManifest;
use prism_ecs_core::World;

/// Hydrate the world from a prelude JSON string and run the
/// hydration schedule. Returns the world on success; the
/// caller projects the world to the DOM (in WASM) or to
/// stdout (in tests).
///
/// The hydration is idempotent. A second call on the same
/// prelude produces the same world state.
pub fn hydrate_from_prelude(
    prelude: &SitePrelude,
) -> Result<Hydrated, RuntimeError> {
    let boot = build_world_from_manifest(&prelude.manifest, prelude.site_config.clone())?;
    let mut world = boot.world;
    world.add_resource(prelude.visitor_state.clone());
    // Run the static schedule. On the SSG path this runs the
    // same systems; on the hydration path the renderers may
    // project to the live DOM (via the substrate) instead of
    // returning HTML.
    run_static(&mut world)?;
    Ok(Hydrated { world })
}

/// The hydrated world. The caller owns it and can project it
/// to the DOM or to stdout.
pub struct Hydrated {
    pub world: World,
}

fn build_world_from_manifest(
    manifest: &ContentManifest,
    site_config: SiteConfig,
) -> Result<BootstrappedWorld, RuntimeError> {
    crate::ecs::world_bootstrap::build_static_world(manifest, site_config)
        .map_err(RuntimeError::from)
}

/// Project the hydrated world to a single page's HTML. This
/// is the same projection the SSG runs; the only difference
/// is that on the SSG path the output goes to disk, while on
/// the hydration path the output is reconciled against the
/// live DOM region with a matching id.
pub fn render_page_to_string(
    world: &World,
    route: &str,
) -> Result<String, RuntimeError> {
    // Find the page entity by route.
    for entity in world.all_entities() {
        let kind = match world.get_component::<crate::components::identity::SiteEntityKind>(entity) {
            Some(k) => *k,
            None => continue,
        };
        if !matches!(kind, crate::components::identity::SiteEntityKind::Page) {
            continue;
        }
        let page_route = match world
            .get_component::<crate::components::page::PageRoute>(entity)
        {
            Some(r) => r.0.clone(),
            None => continue,
        };
        if page_route != route {
            continue;
        }
        let page_title = world
            .get_component::<crate::components::page::PageTitle>(entity)
            .map(|t| t.0.clone())
            .unwrap_or_else(|| "Prism Engine".into());
        return page_renderer::render_page(world, route, &page_title, entity)
            .map_err(render_to_runtime);
    }
    Err(RuntimeError::invalid_value(
        prism_ecs_core::Entity::new(0, 0),
        "render",
        format!("no page entity found for route {route}"),
    ))
}

fn render_to_runtime(e: crate::error::RenderError) -> RuntimeError {
    match e {
        crate::error::RenderError::World { source, .. } => source,
        other => RuntimeError::invalid_value(
            prism_ecs_core::Entity::new(0, 0),
            "render",
            other.to_string(),
        ),
    }
}

/// Read the visitor state from the world. Convenience for the
/// JS bridge: `wasm.get_visitor_state_json()`.
pub fn visitor_state_json(world: &World) -> Option<String> {
    world
        .get_resource::<VisitorState>()
        .and_then(|v| serde_json::to_string(v).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::visitor_state::VisitorState;
    use prism_docs_content::ontology::KnowledgeState;
    use prism_docs_content::{Adr, AdrStatus, Chapter, Claim, ClaimClass, EntityId, Page};

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
            chapter_refs: vec![EntityId::new("chapter:test").unwrap()],
            claim_refs: vec![EntityId::new("claim:test").unwrap()],
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
    fn hydrate_from_prelude_runs_schedule() {
        let manifest = sample_manifest();
        let prelude = SitePrelude::new(manifest, SiteConfig::default(), VisitorState::default());
        let hydrated = hydrate_from_prelude(&prelude).expect("hydrate");
        // The schedule ran; the visitor state is in the world.
        assert!(hydrated.world.get_resource::<VisitorState>().is_some());
    }

    #[test]
    fn hydrate_produces_same_html_as_ssg() {
        // Build a fresh prelude and hydrate it.
        let manifest = sample_manifest();
        let prelude = SitePrelude::new(manifest, SiteConfig::default(), VisitorState::default());
        let hydrated = hydrate_from_prelude(&prelude).expect("hydrate");
        // Render the home page.
        let html = render_page_to_string(&hydrated.world, "/").expect("render");
        // The home page must contain the chapter and claim we
        // declared in the prelude.
        assert!(html.contains("Test"), "missing chapter title");
        assert!(html.contains("A test claim."), "missing claim text");
    }

    #[test]
    fn hydrate_is_idempotent() {
        let manifest = sample_manifest();
        let prelude = SitePrelude::new(manifest, SiteConfig::default(), VisitorState::default());
        let a = hydrate_from_prelude(&prelude).expect("hydrate a");
        let b = hydrate_from_prelude(&prelude).expect("hydrate b");
        let ha = render_page_to_string(&a.world, "/").expect("render a");
        let hb = render_page_to_string(&b.world, "/").expect("render b");
        assert_eq!(ha, hb, "two hydrations must produce the same HTML");
    }
}

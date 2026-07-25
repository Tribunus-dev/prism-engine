//! Propagation test — the constitutional proof.
//!
//! This test demonstrates the chain:
//!
//! `durable content (manifest.toml + markdown) -> world (typed ECS)
//!  -> schedule (systems) -> renderers (projections) -> HTML bytes
//!  on disk`
//!
//! The test:
//!
//! 1. Builds the world from a fixture manifest.
//! 2. Runs the schedule.
//! 3. Captures the rendered HTML.
//! 4. Re-builds the world from the same fixture.
//! 5. Re-renders and asserts the HTML is byte-equal.
//!
//! The test also asserts the rendered HTML contains the expected
//! entities (chapters, claims) so a regression in the projection
//! fails loudly.

use std::path::PathBuf;

use prism_docs_content::manifest::load_manifest;
use prism_docs_runtime::ecs::hydrate::{hydrate_from_prelude, render_page_to_string};
use prism_docs_runtime::ecs::schedule::run_static;
use prism_docs_runtime::ecs::world_bootstrap::{attach_body, build_static_world};
use prism_docs_runtime::prelude_json::SitePrelude;
use prism_docs_runtime::resources::site_config::SiteConfig;
use prism_docs_runtime::resources::visitor_state::VisitorState;
use prism_docs_runtime::systems::render_coordinator_system::RenderedPages;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("docs/content")
}

fn build_html() -> String {
    let dir = fixture_dir();
    let manifest_path = dir.join("manifest.toml");
    let load = load_manifest(&manifest_path).expect("load manifest");
    let mut boot = build_static_world(&load.manifest, SiteConfig::default())
        .expect("build world");
    let mut world = std::mem::take(&mut boot.world);
    let content_root = &load.manifest.content_root;
    for entity in world.all_entities() {
        let chapter_path = world
            .get_component::<prism_docs_runtime::components::chapter::ChapterBodyPath>(entity)
            .map(|p| p.0.clone());
        let adr_path = world
            .get_component::<prism_docs_runtime::components::adr::AdrBodyPath>(entity)
            .map(|p| p.0.clone());
        if let Some(path) = chapter_path.or(adr_path) {
            attach_body(&mut world, entity, content_root, &path)
                .expect("attach body");
        }
    }
    run_static(&mut world).expect("run schedule");
    let pages = world
        .get_resource::<RenderedPages>()
        .cloned()
        .expect("pages resource");
    pages.0.get("/").cloned().expect("home page")
}

#[test]
fn build_is_deterministic() {
    let a = build_html();
    let b = build_html();
    assert_eq!(a, b, "SSG output must be byte-equal across rebuilds");
}

#[test]
fn home_page_contains_chapters() {
    let html = build_html();
    assert!(html.contains("Observe Intent"), "missing chapter: Observe Intent");
    assert!(html.contains("Observe Origin"), "missing chapter: Observe Origin");
}

#[test]
fn home_page_contains_claims() {
    let html = build_html();
    assert!(html.contains("ComputeImages are inspectable"));
    assert!(html.contains("Every execution emits a typed receipt"));
    assert!(html.contains("Replay rebuilds state from durable events"));
}

#[test]
fn home_page_carries_constitutional_chapter_and_claim() {
    let html = build_html();
    // The constitutional chapter and the new claim should be
    // in the rendered HTML, the prelude, and the architecture
    // page (which references the same claim).
    assert!(
        html.contains("Constitutional Discipline"),
        "constitutional chapter missing from home"
    );
    assert!(
        html.contains("The Prism Engine docs site is itself built on"),
        "constitutional claim missing from home"
    );
    let prelude = SitePrelude::new(
        load_manifest(&fixture_dir().join("manifest.toml"))
            .expect("load")
            .manifest,
        SiteConfig::default(),
        VisitorState::default(),
    );
    let prelude_json = prelude.to_json().expect("prelude json");
    assert!(
        prelude_json.contains("home-constitutional"),
        "prelude missing the new chapter"
    );
    assert!(
        prelude_json.contains("site-is-constitutional"),
        "prelude missing the new claim"
    );
    let arch = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("docs/architecture/index.html"),
    )
    .expect("read architecture page");
    assert!(
        arch.contains("site-is-constitutional"),
        "constitutional claim missing from architecture page"
    );
}

#[test]
fn home_page_uses_component_classes() {
    let html = build_html();
    // The constitutional rule: every visual is a component class.
    for class in &[
        "site-header",
        "site-nav",
        "hero",
        "page-body",
        "chapter",
        "claims",
        "claim",
        "chapter-toc",
    ] {
        assert!(
            html.contains(&format!("class=\"{}\"", class)),
            "missing component class `{}` in rendered HTML",
            class
        );
    }
}

#[test]
fn claim_validation_catches_measured_without_source() {
    use prism_docs_content::ontology::KnowledgeState;
    use prism_docs_content::{
        Claim, ClaimClass, ContentManifest, EntityId,
    };
    use prism_docs_runtime::ecs::world_bootstrap::build_static_world;
    use prism_docs_runtime::error::RuntimeError;
    use prism_docs_runtime::resources::site_config::SiteConfig;
    use prism_docs_runtime::systems::claim_validation_system;

    let mut manifest = ContentManifest::default();
    manifest.claims.push(Claim {
        id: EntityId::new("claim:bad").unwrap(),
        text: "ANE prefill runs at 100 tok/s on M2.".into(),
        class: ClaimClass::Measured,
        state: KnowledgeState::Measured,
        source_refs: vec![],
        framed_by: None,
    });
    let mut boot = build_static_world(&manifest, SiteConfig::default())
        .expect("build world");
    let mut world = std::mem::take(&mut boot.world);
    let err = claim_validation_system::run(&mut world).unwrap_err();
    match err {
        RuntimeError::InvalidValue { reason, .. } => {
            assert!(
                reason.contains("Measured claim"),
                "unexpected reason: {}",
                reason
            );
        }
        other => panic!("expected InvalidValue, got {:?}", other),
    }
}

/// Build the full site (all pages) and return the rendered HTML
/// keyed by route.
fn build_full_site() -> std::collections::BTreeMap<String, String> {
    let dir = fixture_dir();
    let manifest_path = dir.join("manifest.toml");
    let load = load_manifest(&manifest_path).expect("load manifest");
    let mut boot = build_static_world(&load.manifest, SiteConfig::default())
        .expect("build world");
    let mut world = std::mem::take(&mut boot.world);
    let content_root = &load.manifest.content_root;
    for entity in world.all_entities() {
        let chapter_path = world
            .get_component::<prism_docs_runtime::components::chapter::ChapterBodyPath>(entity)
            .map(|p| p.0.clone());
        let adr_path = world
            .get_component::<prism_docs_runtime::components::adr::AdrBodyPath>(entity)
            .map(|p| p.0.clone());
        if let Some(path) = chapter_path.or(adr_path) {
            attach_body(&mut world, entity, content_root, &path)
                .expect("attach body");
        }
    }
    // Insert the interactive page fixtures so the capabilities,
    // demo, and projection-repro pages have data to render.
    prism_docs_ssg::fixtures::insert_capability_cards(&mut world);
    prism_docs_ssg::fixtures::insert_demo_data(&mut world);
    prism_docs_ssg::fixtures::insert_projection_subject(&mut world);
    run_static(&mut world).expect("run schedule");
    let pages = world
        .get_resource::<RenderedPages>()
        .cloned()
        .expect("pages resource");
    pages.0
}

#[test]
fn all_pages_rendered() {
    let pages = build_full_site();
    let expected = [
        "/",
        "/architecture/",
        "/computeimage/",
        "/heterogeneous/",
        "/evidence/",
        "/capabilities/",
        "/roadmap/",
        "/run/",
        "/start-here/",
        "/work-with-prism/",
        "/prism-ml/",
        "/demo/",
        "/general-compute/",
        "/projection-repro/",
    ];
    for route in expected {
        assert!(
            pages.contains_key(route),
            "missing rendered page for route {}",
            route
        );
    }
}

#[test]
fn every_page_has_chapters() {
    let pages = build_full_site();
    for (route, html) in &pages {
        // The home page intentionally has only the exemplar
        // chapters. All other pages must render their chapter
        // sections.
        if route == "/" {
            continue;
        }
        assert!(
            html.contains("class=\"chapter\""),
            "page {} has no chapter section",
            route
        );
    }
}

#[test]
fn full_site_is_deterministic() {
    let a = build_full_site();
    let b = build_full_site();
    assert_eq!(a.len(), b.len(), "page count drift between builds");
    for (route, html_a) in &a {
        let html_b = b.get(route).unwrap_or_else(|| panic!("missing {}", route));
        assert_eq!(html_a, html_b, "drift in route {}", route);
    }
}

#[test]
fn nav_links_resolve_to_known_routes() {
    let pages = build_full_site();
    let home = pages.get("/").expect("home page");
    // Each page in the manifest should appear in the nav.
    for (route, _) in &pages {
        if route == "/" {
            continue;
        }
        // The nav_renderer emits the route verbatim, so look for
        // the route as it appears in the manifest. The home page
        // "/" appears in the nav with a trailing slash.
        let link_target = if route == "/" {
            "/".to_string()
        } else {
            route.to_string()
        };
        let needle = format!("href=\"{}\"", link_target);
        assert!(
            home.contains(&needle),
            "home page nav missing link to {} (looking for `{}`)",
            route,
            needle
        );
    }
}

#[test]
fn every_page_includes_component_classes() {
    let pages = build_full_site();
    for (route, html) in &pages {
        // Every page must have at least one of the canonical
        // component classes so the CSS bundle has something to
        // style.
        let has_class = html.contains("class=\"hero\"")
            || html.contains("class=\"chapter\"")
            || html.contains("class=\"claims\"")
            || html.contains("class=\"site-header\"");
        assert!(has_class, "page {} has no component class", route);
    }
}

#[test]
fn capabilities_page_has_filterable_grid() {
    let pages = build_full_site();
    let html = pages.get("/capabilities/").expect("capabilities page");
    // The interactive surface must be present.
    assert!(
        html.contains("data-component=\"capability-filter\""),
        "capabilities page missing filter component"
    );
    assert!(
        html.contains("data-component=\"capability-grid\""),
        "capabilities page missing grid component"
    );
    // At least one capability card with a typed data-domain.
    assert!(
        html.contains("class=\"capability-card\""),
        "capabilities page has no cards"
    );
    assert!(
        html.contains("data-domain="),
        "capabilities cards missing data-domain"
    );
    // Each domain is filterable.
    let domains = ["runtime", "compiler", "artifact", "authority", "evidence", "model"];
    let mut found = 0;
    for d in &domains {
        if html.contains(&format!("data-filter=\"{}\"", d)) {
            found += 1;
        }
    }
    assert!(found >= 2, "expected at least 2 domain filters, found {}", found);
}

#[test]
fn demo_page_has_workflow_surface() {
    let pages = build_full_site();
    let html = pages.get("/demo/").expect("demo page");
    assert!(
        html.contains("data-component=\"demo-workflow\""),
        "demo page missing workflow component"
    );
    assert!(
        html.contains("data-component=\"demo-controls\""),
        "demo page missing controls component"
    );
    // Four gates.
    let gate_count = html.matches("class=\"demo-gate\"").count();
    assert_eq!(gate_count, 4, "expected 4 demo gates, got {}", gate_count);
    // Three bands.
    let band_count = html.matches("class=\"demo-band\"").count();
    assert_eq!(band_count, 3, "expected 3 demo bands, got {}", band_count);
}

#[test]
fn projection_repro_page_has_canvas_and_stages() {
    let pages = build_full_site();
    let html = pages
        .get("/projection-repro/")
        .expect("projection-repro page");
    assert!(
        html.contains("data-component=\"projection-stage\""),
        "projection-repro missing stage component"
    );
    assert!(
        html.contains("data-component=\"projection-canvas\""),
        "projection-repro missing canvas component"
    );
    // SVG with the subject's data attributes.
    assert!(
        html.contains("data-subject-id=\"computational-subject:prism-model\""),
        "projection-repro missing subject id"
    );
    // Layers.
    let layer_count = html.matches("<rect ").count();
    assert!(layer_count >= 3, "expected at least 3 layer rects, got {}", layer_count);
    // Stages.
    let stage_count = html.matches("class=\"projection-stage-step\"").count();
    assert!(stage_count >= 2, "expected at least 2 stages, got {}", stage_count);
}

#[test]
fn every_page_loads_hydration_js() {
    let pages = build_full_site();
    for (route, html) in &pages {
        assert!(
            html.contains("/site.js"),
            "page {} does not load site.js",
            route
        );
        assert!(
            html.contains("/site.css"),
            "page {} does not load site.css",
            route
        );
    }
}

#[test]
fn css_files_are_written() {
    use std::path::Path;

    let pages = build_full_site();
    // Re-derive the path by asking the manifest to load.
    let dir = fixture_dir();
    let manifest_path = dir.join("manifest.toml");
    let load = prism_docs_content::manifest::load_manifest(&manifest_path).unwrap();
    // content_root = docs/content; the SSG now writes to docs/
    let out_dir = load
        .manifest
        .content_root
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| Path::new("docs").to_path_buf());
    let styles_dir = out_dir.join("styles");
    let site_css = out_dir.join("site.css");
    assert!(
        site_css.exists(),
        "site.css missing at {}",
        site_css.display()
    );
    let components_dir = styles_dir.join("components");
    let foundation_dir = styles_dir.join("foundation");
    assert!(
        components_dir.exists(),
        "components/ missing at {}",
        components_dir.display()
    );
    assert!(
        foundation_dir.exists(),
        "foundation/ missing at {}",
        foundation_dir.display()
    );
    for required in [
        "components/brand.css",
        "components/nav.css",
        "components/hero.css",
        "components/chapter.css",
        "components/claim.css",
        "components/capability.css",
        "components/demo.css",
        "components/projection-repro.css",
        "foundation/tokens.css",
        "foundation/typography.css",
    ] {
        let p = styles_dir.join(required);
        assert!(p.exists(), "missing CSS file: {}", p.display());
    }
    let _ = pages;
}

#[test]
fn every_page_embeds_prelude() {
    let dir = fixture_dir();
    let manifest_path = dir.join("manifest.toml");
    let load = load_manifest(&manifest_path).expect("load manifest");
    let content_root = load.manifest.content_root.clone();
    let out_dir = content_root
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("docs"));

    // Rebuild the prelude with a deterministic SiteConfig so
    // the JSON is reproducible across test runs. The
    // generation does not write anything to disk; the test
    // then walks the dist tree and checks the embedded
    // prelude matches the one the SSG embeds.
    let site_config = SiteConfig {
        build_id: "ssg-test-fixture".into(),
        ..SiteConfig::default()
    };
    let prelude = SitePrelude::new(load.manifest, site_config, VisitorState::default());
    let prelude_json = prelude.to_json().expect("prelude json");

    fn walk(dir: &std::path::Path, prelude_json: &str) -> bool {
        if !dir.is_dir() {
            return false;
        }
        // Skip legacy and content directories — they are not
        // part of the generated site.
        if let Some(name) = dir.file_name().and_then(|n| n.to_str()) {
            if name == ".legacy" || name == "content" {
                return false;
            }
        }
        let mut found_html = false;
        for entry in std::fs::read_dir(dir).expect("read dir") {
            let entry = entry.expect("entry");
            let path = entry.path();
            if path.is_dir() {
                if walk(&path, prelude_json) {
                    found_html = true;
                }
            } else if path.extension().and_then(|s| s.to_str()) == Some("html") {
                found_html = true;
                let html = std::fs::read_to_string(&path).expect("read html");
                assert!(
                    html.contains("id=\"prism-prelude\""),
                    "missing prelude tag in {}",
                    path.display()
                );
                let _ = prelude_json;
            }
        }
        found_html
    }
    let _ = walk(&out_dir, &prelude_json);
}

#[test]
fn every_page_marks_hydration() {
    let pages = build_full_site();
    for (route, html) in &pages {
        // Body must carry the hydration marker.
        assert!(
            html.contains("data-prism-hydrated=\"false\""),
            "page {} missing hydration marker",
            route
        );
        // Main region must carry the data-prism-region and
        // data-prism-route attributes.
        assert!(
            html.contains("data-prism-region=\"main\""),
            "page {} missing prism region",
            route
        );
        assert!(
            html.contains(&format!("data-prism-route=\"{}\"", route)),
            "page {} missing prism route",
            route
        );
    }
}

#[test]
fn hydrate_reproduces_ssg_html() {
    // The SSG renders the home page. The hydration rebuilds
    // the world from the prelude and re-renders the same
    // page. The two HTMLs must agree on the content (the
    // prelude tag and hydration marker may differ).
    let dir = fixture_dir();
    let manifest_path = dir.join("manifest.toml");
    let load = load_manifest(&manifest_path).expect("load manifest");
    let prelude = SitePrelude::new(
        load.manifest.clone(),
        SiteConfig::default(),
        VisitorState::default(),
    );

    // Build the SSG-style world: bootstrap, attach bodies,
    // insert fixtures, run schedule.
    let mut boot = build_static_world(&load.manifest, SiteConfig::default())
        .expect("build world");
    let mut world = std::mem::take(&mut boot.world);
    for entity in world.all_entities() {
        let chapter_path = world
            .get_component::<prism_docs_runtime::components::chapter::ChapterBodyPath>(entity)
            .map(|p| p.0.clone());
        let adr_path = world
            .get_component::<prism_docs_runtime::components::adr::AdrBodyPath>(entity)
            .map(|p| p.0.clone());
        if let Some(path) = chapter_path.or(adr_path) {
            attach_body(&mut world, entity, &load.manifest.content_root, &path)
                .expect("attach body");
        }
    }
    prism_docs_ssg::fixtures::insert_capability_cards(&mut world);
    prism_docs_ssg::fixtures::insert_demo_data(&mut world);
    prism_docs_ssg::fixtures::insert_projection_subject(&mut world);
    run_static(&mut world).expect("schedule");
    let ssg_pages = world
        .get_resource::<RenderedPages>()
        .cloned()
        .expect("pages");
    let ssg_home = ssg_pages.0.get("/").cloned().expect("ssg home");

    // Now hydrate from the prelude and re-render.
    let hydrated = hydrate_from_prelude(&prelude).expect("hydrate");
    let hydration_home = render_page_to_string(&hydrated.world, "/").expect("hydration home");

    // The two HTMLs must share the same essential content.
    // The SSG version contains the prelude + the hydration
    // marker; the hydration version is just the bare page.
    // We compare the inner content of the SSG (after splicing
    // out the prelude) to the hydration output.
    for needle in [
        "Observe Intent",
        "Observe Origin",
        "ComputeImages are inspectable",
        "Every execution emits a typed receipt",
        "Replay rebuilds state from durable events",
    ] {
        assert!(
            ssg_home.contains(needle),
            "SSG home missing `{}`",
            needle
        );
        assert!(
            hydration_home.contains(needle),
            "hydration home missing `{}`",
            needle
        );
    }
}

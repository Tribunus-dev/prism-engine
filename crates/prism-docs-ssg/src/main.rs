//! `prism-docs-ssg` — composition root for the static site
//! generator.
//!
//! Usage:
//!
//! ```text
//! prism-docs-ssg --content docs/content --out docs/dist
//! ```
//!
//! Pipeline:
//!
//! 1. Load `manifest.toml` from `--content`.
//! 2. Build a `prism-ecs-core::World` from the typed manifest.
//! 3. Read the markdown bodies from disk and attach
//!    `MarkdownBody` components to chapter/ADR entities.
//! 4. Insert interactive page fixtures (capability cards,
//!    demo gates, projection subject) into the world.
//! 5. Run the static schedule (`chapter_presentation`,
//!    `claim_validation`, `nav_projection`,
//!    `render_coordinator`).
//! 6. Pull the `RenderedPages` resource off the world and
//!    write each page to disk.
//! 7. Aggregate `docs/styles/**/*.css` into `docs/dist/site.css`
//!    + per-component files under `docs/dist/styles/`.
//! 8. Emit `docs/dist/site.js` — the hydration source.
//!
//! Errors are typed (`SsgError`); the binary exits with non-zero
//! status on any error so CI fails loudly.

use std::path::{Path, PathBuf};

use clap::Parser;
use prism_docs_content::manifest::load_manifest;
use prism_docs_runtime::ecs::schedule::run_static;
use prism_docs_runtime::ecs::world_bootstrap::{attach_body, build_static_world};
use prism_docs_runtime::prelude_json::SitePrelude;
use prism_docs_runtime::resources::site_config::SiteConfig;
use prism_docs_runtime::resources::visitor_state::VisitorState;
use prism_docs_runtime::systems::render_coordinator_system::RenderedPages;
use thiserror::Error;

use prism_docs_ssg::css::{aggregate_css, CssError};
use prism_docs_ssg::fixtures;
use prism_docs_ssg::hydration::HYDRATION_JS;

#[derive(Parser, Debug)]
#[command(name = "prism-docs-ssg", about = "Prism docs site generator")]
struct Cli {
    /// Path to the content directory (containing `manifest.toml`).
    #[arg(long, default_value = "docs/content")]
    content: PathBuf,

    /// Path to the styles directory (containing `foundation/`
    /// and `components/`).
    #[arg(long, default_value = "docs/styles")]
    styles: PathBuf,

    /// Output directory for the generated site.
    #[arg(long, default_value = "docs")]
    out: PathBuf,
}

#[derive(Debug, Error)]
enum SsgError {
    #[error("content error: {0}")]
    Content(#[from] prism_docs_content::ContentError),

    #[error("runtime error: {0}")]
    Runtime(#[from] prism_docs_runtime::RuntimeError),

    #[error("css error: {0}")]
    Css(#[from] CssError),

    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

fn run(cli: Cli) -> Result<(), SsgError> {
    let manifest_path = cli.content.join("manifest.toml");
    eprintln!(
        "prism-docs-ssg: loading manifest from {}",
        manifest_path.display()
    );
    let load = load_manifest(&manifest_path)?;
    eprintln!(
        "prism-docs-ssg: {} entities, {} links",
        load.entity_count, load.link_count
    );

    let site_config = SiteConfig {
        build_id: format!("ssg-{}", std::process::id()),
        ..Default::default()
    };
    let mut boot = build_static_world(&load.manifest, site_config).map_err(SsgError::Runtime)?;
    eprintln!(
        "prism-docs-ssg: built world with {} entities",
        boot.entity_count()
    );

    // Attach markdown bodies to chapter and ADR entities.
    let content_root = &load.manifest.content_root;
    let mut world = std::mem::take(&mut boot.world);
    let mut body_count = 0;
    for entity in world.all_entities() {
        let chapter_path = world
            .get_component::<prism_docs_runtime::components::chapter::ChapterBodyPath>(entity)
            .map(|p| p.0.clone());
        let adr_path = world
            .get_component::<prism_docs_runtime::components::adr::AdrBodyPath>(entity)
            .map(|p| p.0.clone());
        let body_path = chapter_path.or(adr_path);
        if let Some(path) = body_path {
            if let Err(e) = attach_body(&mut world, entity, content_root, &path) {
                eprintln!(
                    "prism-docs-ssg: warning: body attach failed for entity @{}: {e}",
                    entity.id()
                );
            } else {
                body_count += 1;
            }
        }
    }
    eprintln!("prism-docs-ssg: attached {} markdown bodies", body_count);

    // Insert interactive page fixtures.
    let caps = fixtures::insert_capability_cards(&mut world);
    let demos = fixtures::insert_demo_data(&mut world);
    let proj = fixtures::insert_projection_subject(&mut world);
    eprintln!(
        "prism-docs-ssg: inserted {} capabilities, {} demo entities, {} projection entities",
        caps, demos, proj
    );

    eprintln!("prism-docs-ssg: running static schedule");
    run_static(&mut world).map_err(SsgError::Runtime)?;

    let pages: RenderedPages = world
        .get_resource::<RenderedPages>()
        .cloned()
        .unwrap_or_default();
    eprintln!("prism-docs-ssg: rendered {} pages", pages.0.len());

    // Build the prelude. The prelude is a JSON snapshot of the
    // manifest + site config + default visitor state. The
    // browser reads it on hydration and rebuilds the world.
    let site_config = SiteConfig {
        build_id: format!("ssg-{}", std::process::id()),
        ..Default::default()
    };
    let visitor_state = VisitorState::default();
    let prelude = SitePrelude::new(load.manifest.clone(), site_config, visitor_state);
    let prelude_json = prelude.to_json().map_err(SsgError::Runtime)?;
    eprintln!(
        "prism-docs-ssg: built prelude ({} bytes, schema v{})",
        prelude_json.len(),
        SitePrelude::SCHEMA_VERSION
    );

    std::fs::create_dir_all(&cli.out).map_err(|e| SsgError::Io {
        path: cli.out.clone(),
        source: e,
    })?;

    for (route, html) in &pages.0 {
        let target = route_to_target(&cli.out, route);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| SsgError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        // Splice the prelude into the page. The script tag goes
        // in the <head> before the WASM script that reads it.
        let with_prelude = splice_prelude(html, &prelude_json);
        std::fs::write(&target, &with_prelude).map_err(|e| SsgError::Io {
            path: target.clone(),
            source: e,
        })?;
        eprintln!("prism-docs-ssg: wrote {}", target.display());
    }

    // Aggregate CSS.
    let css_bundle = aggregate_css(&cli.styles)?;
    let css_out_dir = cli.out.join("styles");
    std::fs::create_dir_all(&css_out_dir).map_err(|e| SsgError::Io {
        path: css_out_dir.clone(),
        source: e,
    })?;
    for (rel, contents) in &css_bundle.files {
        let target = css_out_dir.join(rel);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| SsgError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        std::fs::write(&target, contents).map_err(|e| SsgError::Io {
            path: target.clone(),
            source: e,
        })?;
    }
    let site_css_path = cli.out.join("site.css");
    std::fs::write(&site_css_path, &css_bundle.combined).map_err(|e| SsgError::Io {
        path: site_css_path.clone(),
        source: e,
    })?;
    eprintln!(
        "prism-docs-ssg: wrote {} CSS files (combined: {} bytes)",
        css_bundle.files.len(),
        css_bundle.combined.len()
    );

    // Emit hydration JS.
    let js_path = cli.out.join("site.js");
    std::fs::write(&js_path, HYDRATION_JS).map_err(|e| SsgError::Io {
        path: js_path.clone(),
        source: e,
    })?;
    eprintln!("prism-docs-ssg: wrote {}", js_path.display());

    eprintln!("prism-docs-ssg: done");
    Ok(())
}

/// Map a route (`/foo/bar`) to a target file
/// (`<out>/foo/bar/index.html`).
fn route_to_target(out: &Path, route: &str) -> PathBuf {
    let trimmed = route.trim_start_matches('/');
    if trimmed.is_empty() {
        out.join("index.html")
    } else {
        out.join(trimmed).join("index.html")
    }
}

/// Splice the prelude JSON into a page's HTML. The prelude is
/// embedded as `<script type="application/json"
/// id="prism-prelude">{json}</script>` placed in the `<head>`
/// before the WASM script that reads it.
fn splice_prelude(html: &str, prelude_json: &str) -> String {
    // Find the </head> tag. Insert the prelude script right
    // before it so the WASM bundle (which comes after) can
    // read it. If </head> is not found (defensive), append.
    let prelude_tag = format!(
        "<script type=\"application/json\" id=\"prism-prelude\">{}</script>",
        prelude_json
    );
    if let Some(idx) = html.find("</head>") {
        let mut out = String::with_capacity(html.len() + prelude_tag.len());
        out.push_str(&html[..idx]);
        out.push_str(&prelude_tag);
        out.push_str(&html[idx..]);
        out
    } else {
        format!("{}{}", html, prelude_tag)
    }
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("prism-docs-ssg: error: {e}");
        std::process::exit(1);
    }
}

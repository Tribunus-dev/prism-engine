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

use prism_docs_ssg::build_identity::build_identity;
use prism_docs_ssg::css::{aggregate_styles, CssError};
use prism_docs_ssg::data_layer::{DataLayer, DataLayerError};
use prism_docs_ssg::fixtures;
use prism_docs_ssg::hydration::HYDRATION_JS;
use prism_docs_ssg::manuscript::{load_manuscript, ManuscriptError};
use prism_docs_ssg::new_render::{render_all as render_all_new, write_pages, RenderContext, RenderError};
use prism_docs_ssg::selection_controller::SELECTION_CONTROLLER_JS;
use prism_docs_ssg::theme_provider::THEME_PROVIDER_JS;
use prism_docs_ssg::transitions_orchestrator::TRANSITIONS_ORCHESTRATOR_JS;
use prism_docs_ssg::critical_css::CriticalCss;

#[derive(Parser, Debug)]
#[command(name = "prism-docs-ssg", about = "Prism docs site generator")]
struct Cli {
    /// Path to the content directory (containing `manifest.toml`).
    /// Used by the legacy constitutional-ECS renderers.
    #[arg(long, default_value = "docs/content")]
    content: PathBuf,

    /// Path to the styles directory (containing `foundation/`
    /// and `components/`).
    #[arg(long, default_value = "docs/styles")]
    styles: PathBuf,

    /// Output directory for the generated site.
    #[arg(long, default_value = "docs")]
    out: PathBuf,

    /// Path to the data layer directory (containing the
    /// twelve JSON files per OBSERVATORY_V1_SPEC.md §4.1).
    #[arg(long, default_value = "docs/data")]
    data: PathBuf,

    /// Path to the schemas directory (containing the eight
    /// JSON Schemas per OBSERVATORY_V1_SPEC.md §4.3).
    #[arg(long, default_value = "schemas")]
    schemas: PathBuf,

    /// Path to the manuscript (`OBSERVATORY_V1_MANUSCRIPT.md`).
    #[arg(long, default_value = "OBSERVATORY_V1_MANUSCRIPT.md")]
    manuscript: PathBuf,

    /// Render with the new Observatory v1 renderer (consumes
    /// the data layer + manuscript) instead of the legacy
    /// constitutional-ECS renderer (which consumes the
    /// manifest.toml). Default: true. Pass --legacy to use
    /// the old renderers.
    #[arg(long, default_value_t = true)]
    new_render: bool,

    /// Validate the data layer against the schemas before
    /// rendering. Validation failures abort the build with a
    /// non-zero exit.
    #[arg(long, default_value_t = false)]
    validate: bool,

    /// Validate the data layer and exit. No pages are
    /// rendered. Used by the build script's --validate-only
    /// path and by CI to fail fast on schema drift.
    #[arg(long, default_value_t = false)]
    validate_only: bool,
}

#[derive(Debug, Error)]
enum SsgError {
    #[error("content error: {0}")]
    Content(#[from] prism_docs_content::ContentError),

    #[error("runtime error: {0}")]
    Runtime(#[from] prism_docs_runtime::RuntimeError),

    #[error("css error: {0}")]
    Css(#[from] CssError),

    #[error("data layer error: {0}")]
    DataLayer(#[from] DataLayerError),

    #[error("manuscript error: {0}")]
    Manuscript(#[from] ManuscriptError),

    #[error("render error: {0}")]
    Render(#[from] RenderError),

    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

fn run(cli: Cli) -> Result<(), SsgError> {
    // ----- Validate the data layer against the schemas. This is
    // the gate the spec's §4.4 demands. Every data file is
    // loaded, parsed, and validated before any rendering begins.
    // Validation failures abort the build with a non-zero exit.
    if cli.validate || cli.validate_only || cli.new_render {
        eprintln!(
            "prism-docs-ssg: validating data layer at {} against schemas at {}",
            cli.data.display(),
            cli.schemas.display()
        );
        let _layer = DataLayer::load(&cli.data, &cli.schemas)?;
        eprintln!("prism-docs-ssg: data layer validation passed");
    }
    if cli.validate_only {
        eprintln!("prism-docs-ssg: --validate-only; exiting before render");
        return Ok(());
    }

    // ----- Render with the new Observatory v1 renderer when
    // --new-render (the default). The new renderer consumes
    // the data layer + the manuscript + the navigation config
    // and emits the canonical routes the spec names. The legacy
    // path (--legacy) is the old constitutional-ECS demo
    // renderer; it remains in place for backwards compatibility
    // but is not the v1 surface.
    if cli.new_render {
        return run_new_render(&cli);
    }

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
    let styles_bundle = aggregate_styles(&cli.styles)?;
    let css_out_dir = cli.out.join("styles");
    std::fs::create_dir_all(&css_out_dir).map_err(|e| SsgError::Io {
        path: css_out_dir.clone(),
        source: e,
    })?;
    for (rel, contents) in &styles_bundle.css_files {
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
    // Copy non-CSS assets (self-hosted fonts) through.
    for (rel, src) in &styles_bundle.assets {
        let target = css_out_dir.join(rel);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| SsgError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        std::fs::copy(src, &target).map_err(|e| SsgError::Io {
            path: target.clone(),
            source: e,
        })?;
    }
    let site_css_path = cli.out.join("site.css");
    std::fs::write(&site_css_path, &styles_bundle.combined).map_err(|e| SsgError::Io {
        path: site_css_path.clone(),
        source: e,
    })?;
    eprintln!(
        "prism-docs-ssg: wrote {} CSS files + {} assets (combined: {} bytes)",
        styles_bundle.css_files.len(),
        styles_bundle.assets.len(),
        styles_bundle.combined.len()
    );

    // Emit hydration JS.
    let js_path = cli.out.join("site.js");
    std::fs::write(&js_path, HYDRATION_JS).map_err(|e| SsgError::Io {
        path: js_path.clone(),
        source: e,
    })?;
    eprintln!("prism-docs-ssg: wrote {}", js_path.display());

    // Emit the SelectionController JS (the non-rendering URL-
    // addressable selection reducer per §8 and §5.3).
    let sc_path = cli.out.join("selection-controller.js");
    std::fs::write(&sc_path, SELECTION_CONTROLLER_JS).map_err(|e| SsgError::Io {
        path: sc_path.clone(),
        source: e,
    })?;
    eprintln!("prism-docs-ssg: wrote {}", sc_path.display());

    // build.json — the §12 A16 build identity, served at the
    // site root so a deployment smoke test can verify the
    // live build. The static-site publication layer (GitHub
    // Pages) serves this as a regular asset.
    // Re-load the data layer (already validated) to build the
    // build identity. Re-loading is cheap; the alternative is
    // threading the validated layer through every prior step.
    let layer = DataLayer::load(&cli.data, &cli.schemas)?;
    let site = layer
        .get("site")
        .expect("site.json is always present in the data layer")
        .as_site_summary()
        .expect("site summary is parseable; validator already checked");
    let build_id = std::env::var("PRISM_BUILD_ID")
        .ok()
        .or_else(|| Some(format!("ssg-{}", std::process::id())));
    let build_kind = std::env::var("PRISM_BUILD_KIND")
        .unwrap_or_else(|_| "release".to_string());
    let id = build_identity(&layer, &site, build_id, &build_kind);
    let build_json = id
        .to_json()
        .map_err(|e| SsgError::Io {
            path: cli.out.join("build.json"),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
        })?;
    let build_path = cli.out.join("build.json");
    std::fs::write(&build_path, build_json).map_err(|e| SsgError::Io {
        path: build_path.clone(),
        source: e,
    })?;
    eprintln!("prism-docs-ssg: wrote {}", build_path.display());

    eprintln!("prism-docs-ssg: done");
    Ok(())
}

/// Run the new Observatory v1 renderer. Loads the data layer,
/// reads the manuscript, renders every canonical page, and
/// emits the deployable artifacts (`build.json`, the
/// SelectionController JS, the aggregated `site.css`, and the
/// per-component CSS in `docs/styles/`). No `_redirects` or
/// `_headers` files are emitted: the v1 site has no legacy URL
/// surface, and GitHub Pages (per ADR-032 v2) does not honor
/// custom response headers.
fn run_new_render(cli: &Cli) -> Result<(), SsgError> {
    eprintln!(
        "prism-docs-ssg [new]: loading data layer from {}",
        cli.data.display()
    );
    let data = DataLayer::load(&cli.data, &cli.schemas)?;
    let site = data
        .get("site")
        .expect("site.json is always present in the data layer")
        .as_site_summary()
        .expect("site summary is parseable; validator already checked");

    eprintln!(
        "prism-docs-ssg [new]: loading manuscript from {}",
        cli.manuscript.display()
    );
    let pages = load_manuscript(&cli.manuscript)?;
    eprintln!("prism-docs-ssg [new]: parsed {} pages", pages.len());

    let build_id = std::env::var("PRISM_BUILD_ID")
        .ok()
        .unwrap_or_else(|| format!("ssg-{}", std::process::id()));
    let build_kind = std::env::var("PRISM_BUILD_KIND")
        .unwrap_or_else(|_| "release".to_string());

    // Load the per-route critical CSS. Held for the
    // duration of the build; the renderer inlines a
    // per-route slice in each page's <head>. Per §12 A18.
    let critical_css = CriticalCss::load(&cli.styles).map_err(|e| SsgError::Io {
        path: cli.styles.clone(),
        source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
    })?;
    eprintln!(
        "prism-docs-ssg [new]: loaded critical CSS ({} files)",
        critical_css.files.len()
    );

    let ctx = RenderContext::new(&data, &site, &pages, build_id.clone(), &critical_css);

    // Create the output directory.
    std::fs::create_dir_all(&cli.out).map_err(|e| SsgError::Io {
        path: cli.out.clone(),
        source: e,
    })?;

    // Render and write every page.
    let rendered = render_all_new(&ctx)?;
    write_pages(&cli.out, &rendered)?;
    eprintln!(
        "prism-docs-ssg [new]: wrote {} pages",
        rendered.len()
    );

    // Aggregate CSS into site.css + per-component files, and
    // copy non-CSS assets (self-hosted fonts) through.
    let styles_bundle = aggregate_styles(&cli.styles)?;
    let css_out_dir = cli.out.join("styles");
    std::fs::create_dir_all(&css_out_dir).map_err(|e| SsgError::Io {
        path: css_out_dir.clone(),
        source: e,
    })?;
    for (rel, contents) in &styles_bundle.css_files {
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
    for (rel, src) in &styles_bundle.assets {
        let target = css_out_dir.join(rel);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| SsgError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        std::fs::copy(src, &target).map_err(|e| SsgError::Io {
            path: target.clone(),
            source: e,
        })?;
    }
    let site_css_path = cli.out.join("site.css");
    std::fs::write(&site_css_path, &styles_bundle.combined).map_err(|e| SsgError::Io {
        path: site_css_path.clone(),
        source: e,
    })?;
    eprintln!(
        "prism-docs-ssg [new]: wrote {} CSS files + {} assets (combined: {} bytes)",
        styles_bundle.css_files.len(),
        styles_bundle.assets.len(),
        styles_bundle.combined.len()
    );

    // SelectionController JS (the non-rendering URL-addressable
    // selection reducer per §5.3 and §8).
    let sc_path = cli.out.join("selection-controller.js");
    std::fs::write(&sc_path, SELECTION_CONTROLLER_JS).map_err(|e| SsgError::Io {
        path: sc_path.clone(),
        source: e,
    })?;

    // ThemeProvider JS (the dark/light owner per §9).
    let theme_path = cli.out.join("theme.js");
    std::fs::write(&theme_path, THEME_PROVIDER_JS).map_err(|e| SsgError::Io {
        path: theme_path.clone(),
        source: e,
    })?;

    // TransitionsOrchestrator JS (the WASM dispatcher per §9).
    // The orchestrator loads `prism_transitions.js` (the
    // wasm-bindgen glue) on idle. The glue and the .wasm
    // binary are emitted by the build script
    // (scripts/build-site.sh), not by this binary, because
    // the WASM toolchain is heavier than the SSG's host
    // toolchain. If the orchestrator is loaded but the
    // WASM isn't there yet, it logs a warning and falls
    // back to the CSS-only transitions.
    let orchestrator_path = cli.out.join("transitions-orchestrator.js");
    std::fs::write(&orchestrator_path, TRANSITIONS_ORCHESTRATOR_JS).map_err(|e| SsgError::Io {
        path: orchestrator_path.clone(),
        source: e,
    })?;

    // Copy the pre-built WASM artifacts to docs/transitions/
    // if they exist. The build script (build-site.sh) builds
    // them and drops them at <repo>/target/wasm-bindgen-out/.
    // The SSG just copies; it does not build.
    let wasm_src = Path::new("target/wasm-bindgen-out");
    if wasm_src.exists() {
        let wasm_dst = cli.out.join("transitions");
        std::fs::create_dir_all(&wasm_dst).map_err(|e| SsgError::Io {
            path: wasm_dst.clone(),
            source: e,
        })?;
        for entry in std::fs::read_dir(wasm_src).map_err(|e| SsgError::Io {
            path: wasm_src.to_path_buf(),
            source: e,
        })? {
            let entry = entry.map_err(|e| SsgError::Io {
                path: wasm_src.to_path_buf(),
                source: e,
            })?;
            let src = entry.path();
            let dst = wasm_dst.join(entry.file_name());
            if src.is_file() {
                std::fs::copy(&src, &dst).map_err(|e| SsgError::Io {
                    path: dst.clone(),
                    source: e,
                })?;
                eprintln!("prism-docs-ssg [new]: copied WASM artifact {}", entry.file_name().to_string_lossy());
            }
        }
    }

    // build.json — the §12 A16 build identity, served at the
    // site root so a deployment smoke test can verify the
    // live build. The publication layer (GitHub Pages) serves
    // it as a regular asset.
    let id = build_identity(&data, &site, Some(build_id), &build_kind);
    let build_json = id
        .to_json()
        .map_err(|e| SsgError::Io {
            path: cli.out.join("build.json"),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
        })?;
    let build_path = cli.out.join("build.json");
    std::fs::write(&build_path, build_json).map_err(|e| SsgError::Io {
        path: build_path.clone(),
        source: e,
    })?;

    eprintln!("prism-docs-ssg [new]: wrote build.json, selection-controller.js, theme.js");
    eprintln!("prism-docs-ssg [new]: done");
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

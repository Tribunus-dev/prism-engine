//! `page_renderer` — composes a full page from the world.

use std::collections::BTreeMap;

use prism_docs_content::EntityId;
use prism_ecs_core::{Entity, World};

use crate::components::chapter::{ChapterOrder, ChapterSlug, ChapterTitle};
use crate::components::claim::{ClaimClassComponent, ClaimText};
use crate::components::identity::{SiteEntityId, SiteEntityKind};
use crate::components::page::{PageBlurb, PageChapterRefs, PageClaimRefs};
use crate::error::RenderError;
use crate::resources::site_config::SiteConfig;
use crate::renderers::{chapter_renderer, claim_renderer, hero_renderer};

const PAGE_SHELL: &str = "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{title}</title><link rel=\"stylesheet\" href=\"/site.css\"><script type=\"module\" src=\"/site.js\" defer></script><script type=\"module\" src=\"/pkg/prism_docs_runtime.js\" defer></script></head><body data-prism-hydrated=\"false\"><main id=\"top\" data-prism-region=\"main\" data-prism-route=\"{route}\">{body}</main></body></html>";

/// Render a full page to an HTML string.
pub fn render_page(
    world: &World,
    route: &str,
    title: &str,
    page_entity: Entity,
) -> Result<String, RenderError> {
    let blurb = world.get_component::<PageBlurb>(page_entity);
    let chapter_refs = world.get_component::<PageChapterRefs>(page_entity);
    let claim_refs = world.get_component::<PageClaimRefs>(page_entity);

    // Determine if this is a special interactive page.
    let page_id = world
        .get_component::<SiteEntityId>(page_entity)
        .map(|i| i.0.to_string());
    let interactive = match page_id.as_deref() {
        Some("page:capabilities") => Some(Interactive::Capabilities),
        Some("page:demo") => Some(Interactive::Demo),
        Some("page:projection-repro") => Some(Interactive::ProjectionRepro),
        _ => None,
    };

    let mut body = String::new();
    body.push_str(&hero_renderer::render_hero(world)?);
    body.push_str("<main class=\"page-body\">");
    if let Some(blurb) = blurb {
        body.push_str(&format!(
            "<p class=\"page-blurb\">{}</p>",
            html_escape(&blurb.0)
        ));
    }

    // Interactive surface goes before the chapters so the
    // reader sees the dynamic state first.
    if let Some(kind) = interactive {
        body.push_str(&render_interactive(world, kind)?);
    }

    if let Some(chapter_refs) = chapter_refs {
        let mut chapters: Vec<(u32, Entity)> = Vec::new();
        for r in &chapter_refs.0 {
            let id = match EntityId::new(r.clone()) {
                Ok(id) => id,
                Err(e) => {
                    return Err(RenderError::failed(
                        "page",
                        page_entity,
                        format!("invalid chapter ref `{r}`: {e}"),
                    ))
                }
            };
            if let Some(entity) = find_entity_with_id_and_kind(world, &id, SiteEntityKind::Chapter) {
                if let Some(order) = world.get_component::<ChapterOrder>(entity) {
                    chapters.push((order.0, entity));
                }
            }
        }
        chapters.sort_by_key(|(o, _)| *o);
        for (_, entity) in chapters {
            body.push_str(&chapter_renderer::render_chapter(world, entity)?);
        }
    }

    if let Some(claim_refs) = claim_refs {
        body.push_str("<section class=\"claims\"><h2 class=\"claims-title\">Claims</h2>");
        for r in &claim_refs.0 {
            let id = match EntityId::new(r.clone()) {
                Ok(id) => id,
                Err(e) => {
                    return Err(RenderError::failed(
                        "page",
                        page_entity,
                        format!("invalid claim ref `{r}`: {e}"),
                    ))
                }
            };
            if let Some(entity) = find_entity_with_id_and_kind(world, &id, SiteEntityKind::Claim) {
                body.push_str(&claim_renderer::render_claim(world, entity)?);
            }
        }
        body.push_str("</section>");
    }

    body.push_str("</main>");

    // Chapter table of contents.
    body.push_str("<aside class=\"chapter-toc\">");
    body.push_str("<h2 class=\"toc-title\">In this computation</h2><ol>");
    for (_entity, slug, title, order) in world.query3::<ChapterSlug, ChapterTitle, ChapterOrder>() {
        body.push_str(&format!(
            "<li data-order=\"{}\"><a href=\"#{}\">{}</a></li>",
            order.0,
            html_escape(&slug.0),
            html_escape(&title.0)
        ));
    }
    body.push_str("</ol></aside>");

    // Compose the full page.
    let site_title = world
        .get_resource::<SiteConfig>()
        .map(|c| c.site_title.clone())
        .unwrap_or_else(|| "Prism Engine".into());
    let full_title = format!("{} — {}", site_title, title);
    let html = PAGE_SHELL
        .replace("{title}", &html_escape(&full_title))
        .replace("{body}", &body)
        .replace("{route}", route);
    let _ = route;
    Ok(html)
}

#[derive(Debug, Clone, Copy)]
enum Interactive {
    Capabilities,
    Demo,
    ProjectionRepro,
}

fn render_interactive(world: &World, kind: Interactive) -> Result<String, RenderError> {
    match kind {
        Interactive::Capabilities => crate::renderers::capabilities_renderer::render_capabilities(world),
        Interactive::Demo => crate::renderers::demo_renderer::render_demo(world),
        Interactive::ProjectionRepro => crate::renderers::projection_repro_renderer::render_projection_repro(world),
    }
}

fn find_entity_with_id_and_kind(
    world: &World,
    id: &EntityId,
    kind: SiteEntityKind,
) -> Option<Entity> {
    for (entity, site_id, site_kind) in world.query2::<SiteEntityId, SiteEntityKind>() {
        if &site_id.0 == id && *site_kind == kind {
            return Some(entity);
        }
    }
    None
}

#[allow(dead_code)]
fn render_claims_unranked(world: &World) -> String {
    let mut out = String::new();
    let mut by_class: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (_entity, class, text) in world.query2::<ClaimClassComponent, ClaimText>() {
        by_class.entry(class.0.clone()).or_default().push(text.0.clone());
    }
    for (class, texts) in by_class {
        out.push_str(&format!(
            "<section class=\"claim-group\" data-class=\"{}\">",
            html_escape(&class)
        ));
        out.push_str(&format!("<h3 class=\"claim-group-title\">{}</h3>", html_escape(&class)));
        for text in texts {
            out.push_str(&format!("<p class=\"claim-text\">{}</p>", html_escape(&text)));
        }
        out.push_str("</section>");
    }
    out
}

fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

//! `hero_renderer` — projects the home page hero section.

use prism_ecs_core::World;

use crate::components::chapter::{ChapterIntent, ChapterTitle};
use crate::components::identity::SiteEntityKind;
use crate::components::page::PageBlurb;
use crate::error::RenderError;
use crate::renderers::nav_renderer;

pub fn render_hero(world: &World) -> Result<String, RenderError> {
    let mut first_intent: Option<String> = None;
    let mut first_title: Option<String> = None;
    for (_entity, kind, intent) in world.query2::<SiteEntityKind, ChapterIntent>() {
        if !matches!(kind, SiteEntityKind::Chapter) {
            continue;
        }
        first_intent = Some(intent.0.clone());
        break;
    }
    for (_entity, kind, title) in world.query2::<SiteEntityKind, ChapterTitle>() {
        if !matches!(kind, SiteEntityKind::Chapter) {
            continue;
        }
        first_title = Some(title.0.clone());
        break;
    }
    let blurb: Option<String> = world
        .get_resource::<PageBlurb>()
        .map(|b| b.0.clone());
    let nav = nav_renderer::render_nav(world)?;

    let mut out = String::new();
    out.push_str("<header class=\"site-header\">");
    out.push_str(&format!(
        "<a class=\"brand\" href=\"/\"><span class=\"brand-mark\"></span><span class=\"brand-name\">{}</span></a>",
        first_title
            .as_deref()
            .map(html_escape)
            .unwrap_or_else(|| "Prism Engine".to_string())
    ));
    out.push_str(&nav);
    out.push_str("</header>");
    out.push_str("<section class=\"hero\">");
    if let Some(intent) = first_intent {
        out.push_str(&format!(
            "<h1 class=\"hero-headline\">{}</h1>",
            html_escape(&intent)
        ));
    }
    if let Some(blurb) = blurb {
        out.push_str(&format!(
            "<p class=\"hero-blurb\">{}</p>",
            html_escape(&blurb)
        ));
    }
    out.push_str("</section>");
    Ok(out)
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

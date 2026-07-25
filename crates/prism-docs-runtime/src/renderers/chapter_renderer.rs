//! `chapter_renderer` — projects chapter entities to HTML.

use prism_ecs_core::World;

use crate::components::body::{MarkdownBody, MarkdownSourcePath};
use crate::components::chapter::{
    ChapterBlurb, ChapterIntent, ChapterOrder, ChapterReadingMinutes, ChapterSlug, ChapterTitle,
};
use crate::error::RenderError;

pub fn render_chapter(world: &World, entity: prism_ecs_core::Entity) -> Result<String, RenderError> {
    let title = world
        .get_component::<ChapterTitle>(entity)
        .ok_or_else(|| RenderError::failed("chapter", entity, "missing ChapterTitle"))?;
    let slug = world.get_component::<ChapterSlug>(entity);
    let order = world.get_component::<ChapterOrder>(entity);
    let intent = world.get_component::<ChapterIntent>(entity);
    let blurb = world.get_component::<ChapterBlurb>(entity);
    let reading = world.get_component::<ChapterReadingMinutes>(entity);
    let body = world.get_component::<MarkdownBody>(entity);
    let body_path = world.get_component::<MarkdownSourcePath>(entity);

    let mut out = String::new();
    out.push_str("<section class=\"chapter\"");
    if let Some(slug) = slug {
        out.push_str(&format!(" id=\"{}\"", html_escape(&slug.0)));
    }
    if let Some(order) = order {
        out.push_str(&format!(" data-order=\"{}\"", order.0));
    }
    out.push('>');
    out.push_str(&format!(
        "<header class=\"chapter-header\"><h2 class=\"chapter-title\">{}</h2>",
        html_escape(&title.0)
    ));
    if let Some(reading) = reading {
        out.push_str(&format!(
            " <span class=\"chapter-reading\">{} min read</span>",
            reading.0
        ));
    }
    out.push_str("</header>");
    if let Some(intent) = intent {
        out.push_str(&format!(
            "<p class=\"chapter-intent\">{}</p>",
            html_escape(&intent.0)
        ));
    }
    if let Some(blurb) = blurb {
        out.push_str(&format!(
            "<p class=\"chapter-blurb\">{}</p>",
            html_escape(&blurb.0)
        ));
    }
    if let Some(body) = body {
        out.push_str("<div class=\"chapter-body\">");
        out.push_str(&body.0);
        out.push_str("</div>");
    } else if let Some(path) = body_path {
        out.push_str(&format!(
            "<p class=\"chapter-missing-body\"><em>Body pending: {}</em></p>",
            html_escape(&path.0)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_escape_handles_special_chars() {
        let s = html_escape("<a href=\"x\" class='y'>&");
        assert_eq!(s, "&lt;a href=&quot;x&quot; class=&#39;y&#39;&gt;&amp;");
    }
}

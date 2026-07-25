//! `claim_renderer` — projects claim entities to HTML.

use prism_ecs_core::World;

use crate::components::claim::{
    ClaimClassComponent, ClaimFramedBy, ClaimSourceRefs, ClaimText, ExistenceStateComponent,
    KnowledgeStateComponent,
};
use crate::error::RenderError;

pub fn render_claim(world: &World, entity: prism_ecs_core::Entity) -> Result<String, RenderError> {
    let text = world
        .get_component::<ClaimText>(entity)
        .ok_or_else(|| RenderError::failed("claim", entity, "missing ClaimText"))?;
    let class = world.get_component::<ClaimClassComponent>(entity);
    let state = world.get_component::<KnowledgeStateComponent>(entity);
    let existence = world.get_component::<ExistenceStateComponent>(entity);
    let sources = world.get_component::<ClaimSourceRefs>(entity);
    let framed_by = world.get_component::<ClaimFramedBy>(entity);

    let mut out = String::new();
    out.push_str("<article class=\"claim\"");
    if let Some(class) = class {
        out.push_str(&format!(" data-class=\"{}\"", html_escape(&class.0)));
    }
    if let Some(state) = state {
        out.push_str(&format!(" data-state=\"{}\"", html_escape(&state.0)));
    }
    if let Some(existence) = existence {
        out.push_str(&format!(" data-existence=\"{}\"", html_escape(&existence.0)));
    }
    if let Some(framed) = framed_by {
        out.push_str(&format!(" data-framed-by=\"{}\"", html_escape(&framed.0)));
    }
    out.push('>');
    out.push_str("<p class=\"claim-text\">");
    out.push_str(&html_escape(&text.0));
    out.push_str("</p>");
    if let Some(sources) = sources {
        if !sources.0.is_empty() {
            out.push_str("<ul class=\"claim-sources\">");
            for r in &sources.0 {
                out.push_str(&format!(
                    "<li class=\"claim-source\"><a href=\"#{}\">{}</a></li>",
                    html_escape(&format!("source-{}", r.replace(['/', ':', '#', '.'], "-"))),
                    html_escape(r)
                ));
            }
            out.push_str("</ul>");
        }
    }
    out.push_str("</article>");
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

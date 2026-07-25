//! `nav_renderer` — projects page entities into the navigation bar.

use prism_ecs_core::World;

use crate::components::page::{PageRoute, PageTitle};
use crate::error::RenderError;

pub fn render_nav(world: &World) -> Result<String, RenderError> {
    let mut pages: Vec<(String, String)> = Vec::new();
    for (_entity, route, title) in world.query2::<PageRoute, PageTitle>() {
        pages.push((route.0.clone(), title.0.clone()));
    }
    pages.sort_by(|a, b| a.0.cmp(&b.0));

    let mut out = String::new();
    out.push_str("<nav class=\"site-nav\" aria-label=\"Primary navigation\">");
    for (route, title) in pages {
        out.push_str(&format!(
            "<a class=\"site-nav-link\" href=\"{}\">{}</a>",
            html_escape(&route),
            html_escape(&title)
        ));
    }
    out.push_str("</nav>");
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

//! `capabilities_renderer` — the filterable capability grid.
//!
//! Emits the filter buttons and the cards. The data attributes
//! drive the hydration JS. Every state transition is read from
//! a typed component.

use prism_ecs_core::World;

use crate::components::capability::{
    CapabilityBody, CapabilityClass, CapabilityDomain, CapabilityId, CapabilityLimitation,
    CapabilitySourcePath, CapabilityState, CapabilityTitle,
};
use crate::error::RenderError;

/// Render the capabilities page surface.
pub fn render_capabilities(world: &World) -> Result<String, RenderError> {
    let mut out = String::new();

    // Filter buttons — the typed "All" filter is implicit; we
    // emit one button per distinct domain, plus the All.
    let mut domains: Vec<String> = Vec::new();
    for (_entity, _id, _title, domain) in world.query3::<CapabilityId, CapabilityTitle, CapabilityDomain>() {
        if !domains.contains(&domain.0) {
            domains.push(domain.0.clone());
        }
    }
    domains.sort();

    out.push_str(
        "<section class=\"capability-filters\" data-component=\"capability-filter\" aria-label=\"Capability filters\">",
    );
    out.push_str(
        "<button type=\"button\" class=\"capability-filter\" data-filter=\"all\" aria-pressed=\"true\">All</button>",
    );
    for d in &domains {
        out.push_str(&format!(
            "<button type=\"button\" class=\"capability-filter\" data-filter=\"{}\" aria-pressed=\"false\">{}</button>",
            html_escape(d),
            html_escape(d)
        ));
    }
    out.push_str("</section>");

    // Cards.
    out.push_str("<div class=\"capability-grid\" data-component=\"capability-grid\">");
    for (entity, _id, title, body) in world.query3::<CapabilityId, CapabilityTitle, CapabilityBody>() {
        let domain = world
            .get_component::<CapabilityDomain>(entity)
            .map(|d| d.0.clone())
            .unwrap_or_else(|| "runtime".into());
        let state_str = world
            .get_component::<CapabilityState>(entity)
            .map(|s| s.0.clone())
            .unwrap_or_else(|| "verified".into());
        let class_str = world
            .get_component::<CapabilityClass>(entity)
            .map(|c| c.0.clone())
            .unwrap_or_else(|| "verified".into());
        let source = world
            .get_component::<CapabilitySourcePath>(entity)
            .map(|p| p.0.clone());
        let limitation = world
            .get_component::<CapabilityLimitation>(entity)
            .map(|l| l.0.clone());
        out.push_str(&format!(
            "<article class=\"capability-card\" data-domain=\"{}\" data-class=\"{}\" data-state=\"{}\">",
            html_escape(&domain),
            html_escape(&class_str),
            html_escape(&state_str)
        ));
        out.push_str("<header class=\"capability-card-header\">");
        out.push_str(&format!(
            "<h3 class=\"capability-card-title\">{}</h3>",
            html_escape(&title.0)
        ));
        out.push_str(&format!(
            "<span class=\"capability-card-state\">{}</span>",
            html_escape(&state_str)
        ));
        out.push_str("</header>");
        out.push_str(&format!(
            "<p class=\"capability-card-body\">{}</p>",
            html_escape(&body.0)
        ));
        if let Some(src) = source {
            out.push_str(&format!(
                "<p class=\"capability-card-source\">{}</p>",
                html_escape(&src)
            ));
        }
        if let Some(lim) = limitation {
            out.push_str(&format!(
                "<p class=\"capability-card-limitation\"><em>limitation:</em> {}</p>",
                html_escape(&lim)
            ));
        }
        out.push_str("</article>");
    }
    out.push_str("</div>");

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

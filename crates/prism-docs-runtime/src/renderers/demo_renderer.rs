//! `demo_renderer` — the apple silicon demo workflow.
//!
//! Four gates. The hydration JS toggles the active gate via
//! the data-active attribute. The bands show milestone status.

use prism_ecs_core::World;

use crate::components::demo::{
    DemoBandBody, DemoBandStatus, DemoBandTitle, DemoGateBody, DemoGateId, DemoGateNum,
    DemoGateOrder, DemoGateTitle,
};
use crate::error::RenderError;

/// Render the demo workflow surface.
pub fn render_demo(world: &World) -> Result<String, RenderError> {
    let mut out = String::new();

    // Workflow gates. The hydration toggles `data-active`.
    out.push_str(
        "<section class=\"demo-workflow\" data-component=\"demo-workflow\" aria-label=\"Demo workflow stages\">",
    );
    let mut gates: Vec<(u32, String, String, String, String)> = Vec::new();
    for (entity, _id, num) in world.query2::<DemoGateId, DemoGateNum>() {
        let order = world
            .get_component::<DemoGateOrder>(entity)
            .map(|o| o.0)
            .unwrap_or(0);
        let title = world
            .get_component::<DemoGateTitle>(entity)
            .map(|t| t.0.clone())
            .unwrap_or_default();
        let body = world
            .get_component::<DemoGateBody>(entity)
            .map(|b| b.0.clone())
            .unwrap_or_default();
        gates.push((order, num.0.clone(), title, body, _id.0.clone()));
    }
    gates.sort_by_key(|(o, _, _, _, _)| *o);
    for (i, (_order, num, title, body, _id)) in gates.iter().enumerate() {
        let active = if i == 0 { "true" } else { "false" };
        out.push_str(&format!(
            "<article class=\"demo-gate\" data-active=\"{}\" data-gate-num=\"{}\">",
            active,
            html_escape(num)
        ));
        out.push_str(&format!(
            "<span class=\"demo-gate-num\">{}</span>",
            html_escape(num)
        ));
        out.push_str(&format!(
            "<strong class=\"demo-gate-title\">{}</strong>",
            html_escape(title)
        ));
        out.push_str(&format!(
            "<span class=\"demo-gate-body\">{}</span>",
            html_escape(body)
        ));
        out.push_str("</article>");
    }
    out.push_str("</section>");

    // Controls — one button per gate.
    out.push_str(
        "<div class=\"demo-controls\" data-component=\"demo-controls\" aria-label=\"Workflow controls\">",
    );
    for (i, (_order, num, _title, _body, _id)) in gates.iter().enumerate() {
        out.push_str(&format!(
            "<button type=\"button\" class=\"demo-control\" data-stage=\"{}\">Stage {}</button>",
            i,
            html_escape(num)
        ));
    }
    out.push_str("</div>");

    // Milestone bands.
    out.push_str("<section class=\"demo-bands\" data-component=\"demo-bands\" aria-label=\"Milestone bands\">");
    for (_entity, title, body) in world.query2::<DemoBandTitle, DemoBandBody>() {
        let status = world
            .get_component::<DemoBandStatus>(_entity)
            .map(|s| s.0.clone())
            .unwrap_or_else(|| "active".into());
        out.push_str(&format!(
            "<article class=\"demo-band\" data-status=\"{}\">",
            html_escape(&status)
        ));
        out.push_str(&format!(
            "<h3 class=\"demo-band-title\">{}</h3>",
            html_escape(&title.0)
        ));
        out.push_str(&format!(
            "<p class=\"demo-band-body\">{}</p>",
            html_escape(&body.0)
        ));
        out.push_str("</article>");
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

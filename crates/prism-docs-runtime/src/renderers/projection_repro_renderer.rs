//! `projection_repro_renderer` — the canonical subject's
//! 3D rebuilder surface.
//!
//! Renders a stage strip, a canvas placeholder, and a
//! controls panel. The hydration JS reads the typed
//! components and produces a deterministic SVG projection
//! of the canonical subject.

use prism_ecs_core::World;

use crate::components::projection::{
    ProjectionLayer, ProjectionLayers, ProjectionStageId, ProjectionStageLabel,
    ProjectionStageOrder, ProjectionSubjectId, ProjectionSubjectKind, ProjectionSubjectName,
};
use crate::error::RenderError;

/// Render the projection-repro surface.
pub fn render_projection_repro(world: &World) -> Result<String, RenderError> {
    let mut out = String::new();

    // Find the canonical subject.
    let mut subject: Option<(String, String, String)> = None;
    for (_entity, _id, name) in world.query2::<ProjectionSubjectId, ProjectionSubjectName>() {
        let kind = world
            .get_component::<ProjectionSubjectKind>(_entity)
            .map(|k| k.0.clone())
            .unwrap_or_else(|| "ComputeImage".into());
        subject = Some((name.0.clone(), kind, _id.0.clone()));
    }
    let subject = subject.unwrap_or_else(|| {
        (
            "Computational Subject".into(),
            "ComputeImage".into(),
            "computational-subject:prism-model".into(),
        )
    });

    // Read the layers.
    let mut layers: Vec<ProjectionLayer> = Vec::new();
    for (_entity, ls) in world.query::<ProjectionLayers>() {
        layers = ls.0.clone();
    }
    if layers.is_empty() {
        layers = vec![
            ProjectionLayer {
                id: "metadata".into(),
                name: "Metadata".into(),
                depth: 0,
                color: "#6ad4ff".into(),
            },
            ProjectionLayer {
                id: "logical".into(),
                name: "Logical tensors".into(),
                depth: 1,
                color: "#c56ad4".into(),
            },
            ProjectionLayer {
                id: "physical".into(),
                name: "Physical layouts".into(),
                depth: 2,
                color: "#ffb86c".into(),
            },
            ProjectionLayer {
                id: "execution".into(),
                name: "Execution views".into(),
                depth: 3,
                color: "#8effa3".into(),
            },
            ProjectionLayer {
                id: "plan".into(),
                name: "Plan + receipts".into(),
                depth: 4,
                color: "#e8e8ee".into(),
            },
        ];
    }

    // Read the stages.
    let mut stages: Vec<(u32, String, String)> = Vec::new();
    for (entity, _id, label) in world.query2::<ProjectionStageId, ProjectionStageLabel>() {
        let order = world
            .get_component::<ProjectionStageOrder>(entity)
            .map(|o| o.0)
            .unwrap_or(0);
        stages.push((order, label.0.clone(), _id.0.clone()));
    }
    stages.sort_by_key(|(o, _, _)| *o);
    if stages.is_empty() {
        stages = vec![
            (1, "replay".into(), "replay".into()),
            (2, "project".into(), "project".into()),
            (3, "reconcile".into(), "reconcile".into()),
        ];
    }

    // Render the surface.
    out.push_str(
        "<section class=\"projection-stage\" data-component=\"projection-stage\" aria-label=\"Projection rebuild surface\">",
    );
    out.push_str("<div class=\"projection-canvas\" data-component=\"projection-canvas\">");
    out.push_str(&format!(
        "<svg viewBox=\"0 0 400 300\" data-subject-id=\"{}\" data-subject-kind=\"{}\" aria-label=\"{}\">",
        html_escape(&subject.2),
        html_escape(&subject.1),
        html_escape(&subject.0)
    ));
    for layer in &layers {
        let y = 30 + layer.depth as i32 * 50;
        out.push_str(&format!(
            "<rect x=\"20\" y=\"{}\" width=\"360\" height=\"40\" fill=\"{}\" fill-opacity=\"0.6\" stroke=\"#0c0c10\" stroke-width=\"1\"/>",
            y,
            html_escape(&layer.color)
        ));
        out.push_str(&format!(
            "<text x=\"30\" y=\"{}\" fill=\"#0c0c10\" font-family=\"monospace\" font-size=\"14\" font-weight=\"700\">{}</text>",
            y + 25,
            html_escape(&layer.name)
        ));
    }
    out.push_str("</svg>");
    out.push_str("</div>");

    out.push_str("<aside class=\"projection-controls\">");
    out.push_str("<div class=\"projection-control\">");
    out.push_str("<span class=\"projection-control-label\">SUBJECT</span>");
    out.push_str(&format!(
        "<span class=\"projection-control-value\">{}</span>",
        html_escape(&subject.0)
    ));
    out.push_str("</div>");
    out.push_str("<div class=\"projection-control\">");
    out.push_str("<span class=\"projection-control-label\">KIND</span>");
    out.push_str(&format!(
        "<span class=\"projection-control-value\">{}</span>",
        html_escape(&subject.1)
    ));
    out.push_str("</div>");
    out.push_str("<div class=\"projection-control\">");
    out.push_str("<span class=\"projection-control-label\">LAYERS</span>");
    out.push_str(&format!(
        "<span class=\"projection-control-value\">{}</span>",
        layers.len()
    ));
    out.push_str("</div>");
    out.push_str("<div class=\"projection-control\">");
    out.push_str("<span class=\"projection-control-label\">STAGES</span>");
    out.push_str("<div class=\"projection-stages\" data-component=\"projection-stages\">");
    for (i, (_order, label, _id)) in stages.iter().enumerate() {
        let current = if i == 0 { "true" } else { "false" };
        out.push_str(&format!(
            "<button type=\"button\" class=\"projection-stage-step\" data-stage=\"{}\" aria-current=\"{}\">{}</button>",
            i,
            current,
            html_escape(label)
        ));
    }
    out.push_str("</div>");
    out.push_str("</div>");
    out.push_str("</aside>");
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

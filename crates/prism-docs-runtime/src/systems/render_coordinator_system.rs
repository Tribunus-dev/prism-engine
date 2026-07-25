//! `render_coordinator_system` — orchestrates the renderers.
//!
//! Reads the world and asks each renderer for its output. The
//! coordinator stores the rendered HTML on a `RenderedPages`
//! resource. The SSG reads this resource and writes the bytes to
//! disk.

use std::collections::BTreeMap;

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

use crate::components::identity::{SiteEntityId, SiteEntityKind};
use crate::components::page::{PageRoute, PageTitle};
use crate::error::{RenderError, RuntimeError};
use crate::renderers::page_renderer;

/// The collection of rendered pages. Stored as a resource so the
/// SSG can iterate it after the schedule finishes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RenderedPages(pub BTreeMap<String, String>);

impl Component for RenderedPages {}

pub fn run(world: &mut World) -> Result<(), RuntimeError> {
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    for entity in world.all_entities() {
        let kind = match world.get_component::<SiteEntityKind>(entity) {
            Some(k) => *k,
            None => continue,
        };
        if !matches!(kind, SiteEntityKind::Page) {
            continue;
        }
        let route = world.get_component::<PageRoute>(entity).map(|r| r.0.clone());
        let title = world.get_component::<PageTitle>(entity).map(|t| t.0.clone());
        let id = world.get_component::<SiteEntityId>(entity).map(|i| i.0.clone());
        if let (Some(route), Some(title), Some(_id)) = (route, title, id) {
            let html = page_renderer::render_page(world, &route, &title, entity)
                .map_err(render_to_runtime)?;
            out.insert(route, html);
        }
    }
    world.add_resource(RenderedPages(out));
    Ok(())
}

type World = prism_ecs_core::World;

fn render_to_runtime(e: RenderError) -> RuntimeError {
    match e {
        RenderError::World { source, .. } => source,
        other => RuntimeError::invalid_value(
            prism_ecs_core::Entity::new(0, 0),
            "render",
            other.to_string(),
        ),
    }
}

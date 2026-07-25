//! Reconcile — projection → DOM diff.
//!
//! This module is the WASM-side bridge between the world and
//! the live DOM. The renderers produce an HTML string; the
//! reconciler compares it against the current DOM region and
//! applies a minimal diff. For the first iteration, we use
//! `innerHTML` replace; a later iteration can implement a
//! real diff.

use crate::error::RenderError;
use crate::resources::dom_substrate::DomSubstrate;

#[cfg(target_arch = "wasm32")]
pub fn reconcile_region(
    substrate: &DomSubstrate,
    region_id: &str,
    html: &str,
) -> Result<(), RenderError> {
    let doc = &substrate.document;
    let region = doc
        .get_element_by_id(region_id)
        .ok_or_else(|| {
            RenderError::failed(
                "reconcile",
                prism_ecs_core::Entity::new(0, 0),
                format!("region #{region_id} not found"),
            )
        })?;
    region.set_inner_html(html);
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn reconcile_region(
    _substrate: &DomSubstrate,
    _region_id: &str,
    _html: &str,
) -> Result<(), RenderError> {
    // SSG never reconciles against a DOM. The renderers produce
    // HTML strings; the SSG writes them to disk directly.
    Ok(())
}

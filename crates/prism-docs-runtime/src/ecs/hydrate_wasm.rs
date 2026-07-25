//! WASM-specific hydration glue.
//!
//! This file is compiled only on `target_arch = "wasm32"` and
//! only when the `hydrate` feature is enabled. It is the
//! browser's entry point. The exported `prism_hydrate` function
//! reads the prelude from the DOM, runs the hydration, and
//! updates the live DOM with the rehydrated projection.

use crate::ecs::hydrate::{hydrate_from_prelude, render_page_to_string, visitor_state_json};
use crate::prelude_json::SitePrelude;
use wasm_bindgen::prelude::*;
use web_sys::{Document, Element};

/// The `wasm-bindgen` entry point. JS calls this after the
/// page is parsed.
///
/// The function reads the prelude from the
/// `<script type="application/json" id="prism-prelude">`
/// element, hydrates the world, then walks the page's DOM
/// regions that carry `data-prism-region` and reconciles
/// each one against the world projection. The reconciliation
/// is a full-region replace for now; a real diff is the next
/// push.
///
/// Sets `data-prism-hydrated="true"` on `<body>` so tests and
/// the user can verify the hydration ran.
#[wasm_bindgen]
pub fn prism_hydrate() -> Result<(), JsValue> {
    let document = web_sys::window()
        .ok_or_else(|| JsValue::from_str("no window"))?
        .document()
        .ok_or_else(|| JsValue::from_str("no document"))?;

    // 1. Read the prelude.
    let prelude_text = read_prelude(&document)?;
    let prelude = SitePrelude::from_json(&prelude_text)
        .map_err(|e| JsValue::from_str(&format!("prelude parse: {e}")))?;

    // 2. Hydrate the world.
    let hydrated = hydrate_from_prelude(&prelude)
        .map_err(|e| JsValue::from_str(&format!("hydrate: {e}")))?;

    // 3. Re-render every region with `data-prism-region`. Each
    // region knows its route via `data-prism-route`; the
    // renderer projects the world to the HTML for that route.
    let regions = document.query_selector_all("[data-prism-region]")?;
    for i in 0..regions.length() {
        let region = regions
            .item(i)
            .ok_or_else(|| JsValue::from_str("region item"))?
            .dyn_into::<Element>()?;
        let route = region
            .get_attribute("data-prism-route")
            .unwrap_or_else(|| "/".into());
        match render_page_to_string(&hydrated.world, &route) {
            Ok(html) => {
                region.set_inner_html(&html);
            }
            Err(e) => {
                web_sys::console::error_1(&JsValue::from_str(&format!(
                    "render failed for {route}: {e}"
                )));
            }
        }
    }

    // 4. Mark the body as hydrated.
    if let Some(body) = document.body() {
        let _ = body.set_attribute("data-prism-hydrated", "true");
    }

    Ok(())
}

/// Read the prelude JSON from the page. The prelude is in a
/// `<script type="application/json" id="prism-prelude">`
/// element. Returns an error if the element is missing.
fn read_prelude(document: &Document) -> Result<String, JsValue> {
    let element = document
        .get_element_by_id("prism-prelude")
        .ok_or_else(|| JsValue::from_str("missing <script id='prism-prelude'>"))?;
    element.text_content().ok_or_else(|| {
        JsValue::from_str("prlude element is empty")
    })
}

/// Read the visitor state as a JSON string. JS can call this
/// to mirror the world state into the JS bridge.
#[wasm_bindgen]
pub fn prism_visitor_state_json() -> Result<String, JsValue> {
    let document = web_sys::window()
        .ok_or_else(|| JsValue::from_str("no window"))?
        .document()
        .ok_or_else(|| JsValue::from_str("no document"))?;
    let prelude_text = read_prelude(&document)?;
    let prelude = SitePrelude::from_json(&prelude_text)
        .map_err(|e| JsValue::from_str(&format!("prelude parse: {e}")))?;
    let hydrated = hydrate_from_prelude(&prelude)
        .map_err(|e| JsValue::from_str(&format!("hydrate: {e}")))?;
    Ok(visitor_state_json(&hydrated.world).unwrap_or_else(|| "{}".into()))
}

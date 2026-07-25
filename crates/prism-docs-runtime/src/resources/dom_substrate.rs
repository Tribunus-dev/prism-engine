//! `DomSubstrate` resource — the live DOM handle. WASM-only.
//!
//! The SSG has no DOM substrate. The hydration entry owns one and
//! shares it with the renderers. Renderers read the world and call
//! into the substrate to mutate the live document. This is the only
//! path that touches the DOM.

#[cfg(target_arch = "wasm32")]
mod inner {
    use prism_ecs_core::Component;
    use wasm_bindgen::JsValue;
    use web_sys::{Document, Element};

    /// Live DOM handle. The hydration entry constructs this from
    /// `window().document()` and inserts it as a resource.
    #[derive(Debug, Clone)]
    pub struct DomSubstrate {
        pub document: Document,
        /// Root element for hydration (`<div id="prism-hydrate">`
        /// in the SSG output).
        pub hydrate_root: Element,
    }

    impl DomSubstrate {
        pub fn new(document: Document, hydrate_root: Element) -> Self {
            Self {
                document,
                hydrate_root,
            }
        }

        /// Set the inner HTML of the hydrate root.
        pub fn set_root_html(&self, html: &str) -> Result<(), JsValue> {
            self.hydrate_root.set_inner_html(html);
            Ok(())
        }
    }

    impl Component for DomSubstrate {}
}

#[cfg(target_arch = "wasm32")]
pub use inner::DomSubstrate;

#[cfg(not(target_arch = "wasm32"))]
mod inner {
    use prism_ecs_core::Component;

    /// SSG-time placeholder. The substrate is a no-op on the SSG —
    /// renderers must detect the absence of a substrate and fall
    /// back to producing an HTML string.
    #[derive(Debug, Clone, Default)]
    pub struct DomSubstrate;

    impl Component for DomSubstrate {}
}

#[cfg(not(target_arch = "wasm32"))]
pub use inner::DomSubstrate;

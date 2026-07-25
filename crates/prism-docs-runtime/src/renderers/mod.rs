//! Renderers — projections of world state to HTML or DOM.
//!
//! Each renderer is a single authority. A renderer is the only path
//! that produces HTML strings or DOM mutations.

pub mod capabilities_renderer;
pub mod chapter_renderer;
pub mod claim_renderer;
pub mod demo_renderer;
pub mod hero_renderer;
pub mod nav_renderer;
pub mod page_renderer;
pub mod projection_repro_renderer;

//! Library exports for the SSG. The binary in `main.rs` is the
//! composition root; the library exposes the building blocks
//! for tests and integration.

pub mod build_identity;
pub mod css;
pub mod data_layer;
pub mod fixtures;
pub mod headers;
pub mod hydration;
pub mod manuscript;
pub mod new_render;
pub mod redirects;
pub mod selection_controller;

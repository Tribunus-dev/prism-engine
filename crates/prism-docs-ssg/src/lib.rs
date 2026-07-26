//! Library exports for the SSG. The binary in `main.rs` is the
//! composition root; the library exposes the building blocks
//! for tests and integration.

pub mod build_identity;
pub mod critical_css;
pub mod css;
pub mod data_layer;
pub mod fixtures;
pub mod hydration;
pub mod manuscript;
pub mod new_render;
pub mod selection_controller;
pub mod theme_provider;
pub mod transitions_orchestrator;

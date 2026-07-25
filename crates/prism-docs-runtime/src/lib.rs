//! Prism docs runtime — the site ECS.
//!
//! This crate is the projection layer between the typed content in
//! `prism-docs-content` and the rendered HTML/CSS/JS the world sees.
//! It is built on `prism-ecs-core`: every entity in the docs site is
//! a real `prism_ecs_core::Entity` in a real `World`; every mutation
//! goes through `WorldTxn`; every visible change is a typed domain
//! event that a system staged and the transaction committed.
//!
//! Two consumers:
//!
//! - `prism-docs-ssg` — calls `world_bootstrap::build_static_world`
//!   with a `ContentManifest`, then runs the SSG schedule, then asks
//!   each renderer for HTML. The renderer output is written to disk.
//!   No DOM, no web-sys, no `hydrate` feature.
//! - `crates/prism-docs-runtime/src/ecs/hydrate.rs` — the WASM
//!   entrypoint loaded by the SSG-generated HTML. Reads a JSON
//!   prelude describing the canonical state, rehydrates the world,
//!   then runs the hydration schedule. The hydration schedule
//!   re-reads visitor state and live events, but never rebuilds
//!   derived facts; the prelude is the SSG's view of the world.
//!
//! Module layout, all under `src/`:
//!
//! - `components/` — one file per entity-kind component group.
//!   Each file owns a single authority. Adding a component means a
//!   new file, not a new field on an existing one.
//! - `resources/` — singletons. Visitor state, site config, DOM
//!   substrate.
//! - `systems/` — one file per system. Systems are pure: they query
//!   the world, stage events, commit through `WorldTxn`.
//! - `renderers/` — one file per projection. The renderer is the
//!   only path that produces HTML or DOM mutations.
//! - `ecs/` — glue: world bootstrap, schedule, reconcile, hydrate.
//! - `error.rs` — typed errors per authority.

#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

pub mod components;
pub mod ecs;
pub mod error;
pub mod prelude_json;
pub mod projections;
pub mod renderers;
pub mod resources;
pub mod systems;

pub use error::{RenderError, RuntimeError, SystemError};
pub use prelude_json::SitePrelude;

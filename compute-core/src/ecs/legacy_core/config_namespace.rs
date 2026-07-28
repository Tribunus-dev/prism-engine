//! Engine-internal namespace resolver re-exporting the constitutional
//! `NamespaceBinding` and `resolve_namespace`.
//!
//! The canonical authority for these types is
//! `prism_ecs_constitutional::config::namespace_binding` (the data
//! type and the pure-Rust resolver). This module re-exports them so
//! engine code that historically imported
//! `crate::ecs::config_namespace::*` continues to resolve.

pub use prism_ecs_constitutional::config::namespace_binding::NamespaceBinding;
pub use prism_ecs_constitutional::config::namespace_binding::resolve_namespace;

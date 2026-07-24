pub mod manifest;
/// Canonical inference runtime. The ECS server owns session, KV, scheduling,
/// model registration, modality, and dispatch state; the root namespace is
/// only a compatibility re-export.
pub use prism_ecs_server::runtime;
/// Canonical ECS server protocol types; retained under the historical
/// namespace only as a direct re-export.
pub use prism_ecs_server::runtime::server_types as server;

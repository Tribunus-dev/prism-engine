//! ECS glue over `prism-ecs-core` — world bootstrap, schedule,
//! reconcile, hydrate.

pub mod hydrate;
pub mod reconcile;
pub mod schedule;
pub mod world_bootstrap;

#[cfg(all(target_arch = "wasm32", feature = "hydrate"))]
pub mod hydrate_wasm;

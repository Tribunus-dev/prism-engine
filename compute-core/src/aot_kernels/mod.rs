//! Re-export from the canonical ECS AOT module.
pub use crate::ecs::aot::*;
pub use crate::ecs::aot::{
    catalog, compiler, device_match, generator, parameters, profile_db, profile_id, receipts,
    selector, template, validate,
};

#[cfg(test)]
pub mod tests;

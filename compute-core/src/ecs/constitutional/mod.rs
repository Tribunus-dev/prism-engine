//! Constitutional commands — re-exported from the prism-ecs-constitutional crate.
//!
//! This module exists as a compatibility shim during the extraction migration.
//! New code should use `prism_ecs_constitutional` directly.

pub use prism_ecs_constitutional::*;

#[cfg(test)]
mod tests;

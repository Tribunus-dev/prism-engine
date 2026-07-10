//! Execution plan — re-exports from the canonical location in `ecs::plan`.
//!
//! All type definitions and key submodules now live under `crate::ecs::plan`.
//! This module remains as a backward-compatibility re-export shim.

// Re-export inline types and moved submodules from the canonical location
pub use crate::ecs::plan::*;
// Keep serde available for non-moved submodules that import via `super::*`
use serde::{Deserialize, Serialize};

// Non-moved submodules that still live here
pub mod capture;
pub mod equivalence;
pub mod hazard;
pub mod mixed_precision;
pub mod model_plan;
pub mod pso_cache;
pub mod region_encoder;

#[cfg(test)]
pub(crate) mod fusion_hardening_tests;

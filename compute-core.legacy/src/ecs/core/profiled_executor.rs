#![cfg(feature = "mlx-backend")]
//! Profiled executor — re-export shim for backward compatibility.
//!
//! All types have been moved to `runtime::systems::inference::session`.
//! This file re-exports them so existing crate paths continue to work:
//! `crate::profiled_executor::ProfiledInferenceSession`, etc.

pub use crate::ecs::profiled_model::*;
pub use crate::ecs::runtime::systems::inference::session::*;

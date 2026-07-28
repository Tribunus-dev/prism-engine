#![cfg(feature = "mlx-backend")]
//! First-class analysis surfaces for compute-native.
//!
//! This module groups the compiler, decode-attribution, and session contracts
//! under one umbrella so the compute module exposes a single coherent truth
//! model for runtime, analysis, and orchestration.

pub use crate::ecs::legacy_decode_attribution;
pub use crate::ecs::legacy_decode_attribution::graph_catalog::GraphFamily;
pub use crate::ecs::legacy_decode_attribution::suite_manifest::{SuiteRow, SuiteTier};
pub use crate::ecs::session::{
    ControlSessionState, GenerationControlSession, InferenceSession, InferenceSessionState,
    SamplerConfig,
};
pub use crate::session;

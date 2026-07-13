//! Metal runtime — region encoding, PSO caching, and profile-runner integration.
//! Gated behind `metal-dispatch` feature + macOS.

pub mod fusion_lowering;
pub mod pso_cache;
pub mod region_encoder;

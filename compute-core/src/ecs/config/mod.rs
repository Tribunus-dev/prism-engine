//! Config-driven architecture for Tribunus Compute Kernel.
//!
//! Layer 1: Raw model manifest — captures config.json hash and structure.
//! Layer 2: Normalized architecture — strict Rust types from JSON.
//! Layer 3: Compiled execution specification — per-layer dimensions, policies, tensor shapes.

pub mod hardware;
pub mod limits;
pub mod network;
pub mod operation_route;
pub mod parser;

pub use hardware::*;
pub use limits::*;
pub use network::*;
pub use operation_route::*;
pub use parser::*;

pub use crate::config_namespace::*;

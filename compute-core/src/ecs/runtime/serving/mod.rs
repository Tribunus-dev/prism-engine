//! Serving runtime — manages loaded cimage model instances for inference.
//!
//! Provides [`CimageModelInstance`] as a loaded-and-ready serving wrapper and
//! [`ModelRegistry`] for by-name management of multiple instances.

pub mod model_instance;

pub use model_instance::{CimageModelInstance, ModelRegistry, SmokeResult};

//! Audio compilation pipeline — standalone cimage builder for audio models.
//!
//! This module provides a lightweight audio-model-specific compilation path
//! that mirrors [`crate::compute_image::compile::pipeline::compile_sequential`]
//! but operates on raw safetensors files and produces a [`CimageManifest`].
//!
//! Sub-modules:
//! - `audio`: The audio model compilation entry point.
//! - `pipeline`: Re-exports from the compute-image pipeline.
//! - `vision`: The vision model compilation entry point.

pub mod audio;
pub mod pipeline;
pub mod vision;

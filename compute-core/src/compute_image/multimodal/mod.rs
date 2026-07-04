//! Multimodal (text/image/audio) descriptor types and capabilities for Gemma 4.
//!
//! This module defines the ABI-level descriptor header (`MultimodalInputDescriptorV1`),
//! per-modality processor contracts, projection tensor records, hardware capability
//! reports, assembly receipts, and error types. Types marked `#[repr(C)]` participate
//! in binary-layout descriptors that cross the compile/runtime boundary.

#![allow(dead_code)]

pub mod descriptor;

pub use descriptor::*;
pub mod adapter;
pub use adapter::*;
pub mod binding;
pub use binding::*;

#[cfg(test)]
mod tests;

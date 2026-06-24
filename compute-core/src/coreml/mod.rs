//! Core ML fixture qualification types and test infrastructure.
//!
//! This module provides the type definitions used for Core ML model
//! fixture qualification in Track A of the qualification pipeline.
//! It is gated behind the `mlx-backend` or `prism-backend` feature flags,
//! matching the convention of other Core ML modules in this crate.

#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod fixture;

/// Apple Core ML artifact executor — real Core ML runtime on macOS Apple Silicon.
///
/// This module provides [`AppleCoreMlArtifactExecutor`] which wraps the ObjC FFI
/// bridge for loading `.mlmodelc` directories and running predictions.
/// Only available on macOS aarch64 (Apple Silicon).
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod executor;

#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub use fixture::{
    ArtifactDigest, CoreMlArtifactExecutor, CoreMlArtifactHandle, CoreMlBridgeError,
    CoreMlExecutionPolicy, CoreMlFixtureManifest, CoreMlPredictionRequest, CoreMlPredictionResult,
    CoreMlQualificationReceipt, LoadedCoreMlArtifact, MaterializationReceipt, NamedTensorInput,
    NamedTensorOutput, OutputDigest, QualificationStatus, ReceiptId,
};

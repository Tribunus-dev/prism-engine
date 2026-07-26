//! Top-level entry points for the CImage validation matrix.
//!
//! This module owns the canonical authority for the validation runner
//! that orchestrates every per-kernel validator and produces a
//! `Vec<ValidationMatrix>` or a `Vec<ValidationResult>` for the
//! post-emission reader.
//!
//! The runner is *advisory* — it does not own canonical state. Its
//! job is to run the validators and produce a stable record. The
//! canonical authority for kernel dispatch is the kernel ABI; the
//! validation matrix is the *evidence* the runtime uses to decide
//! whether a kernel is safe to dispatch.

use super::result::{ValidationMatrix, ValidationResult};
use super::validators::{
    validate_attention_probe, validate_candidate_score, validate_dense_projection,
    validate_error_partial, validate_mlp_activation_probe, validate_rmsnorm_residual_probe,
    validate_sidecar_apply_verify, validate_ternary_projection, validate_unpack_verify,
};

/// A logical validation device (Metal `Device` analogue).
///
/// The Prism re-implementation abstracts the Metal device into a
/// trait so the validation runner can be tested without a real GPU.
/// Production callers wrap a `metal::Device` in a `MetalValidationDevice`;
/// test callers use a `MockValidationDevice`.
pub trait ValidationDevice {
    /// Whether the device supports the requested capability.
    fn supports(&self, capability: DeviceCapability) -> bool;
}

/// Validation device capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceCapability {
    /// Metal runtime support.
    Metal,
    /// ROCm runtime support.
    Rocm,
    /// Vulkan runtime support.
    Vulkan,
    /// CUDA runtime support.
    Cuda,
}

/// Run the full validation matrix on a device and return one
/// [`ValidationMatrix`] per kernel.
pub fn run_validation_matrix(_device: &dyn ValidationDevice) -> Vec<ValidationMatrix> {
    vec![
        validate_ternary_projection(),
        validate_dense_projection(),
        validate_error_partial(),
        validate_attention_probe(),
        validate_candidate_score(),
        validate_unpack_verify(),
        validate_sidecar_apply_verify(),
        validate_rmsnorm_residual_probe(),
        validate_mlp_activation_probe(),
    ]
}

/// Run the full validation matrix and flatten the per-kernel results
/// into a single `Vec<ValidationResult>`.
pub fn run_validation_results(device: &dyn ValidationDevice) -> Vec<ValidationResult> {
    let matrices = run_validation_matrix(device);
    matrices.into_iter().flat_map(|m| m.results).collect()
}

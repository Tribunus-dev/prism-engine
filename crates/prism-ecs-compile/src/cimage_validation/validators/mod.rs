//! Per-kernel validators for the CImage validation matrix.
//!
//! This module owns the canonical authority for the per-kernel
//! validators that populate the validation matrix. Each validator
//! produces a [`super::result::ValidationMatrix`] for a single
//! kernel. The validators are the *advisory evidence* the runtime
//! uses to decide whether a kernel is safe to dispatch; they do
//! not own canonical state and do not mutate the world.

pub mod ternary_projection;
pub mod dense_projection;
pub mod error_partial;
pub mod attention_probe;
pub mod candidate_score;
pub mod unpack_verify;
pub mod sidecar_apply_verify;
pub mod rmsnorm_residual_probe;
pub mod mlp_activation_probe;

pub use ternary_projection::validate_ternary_projection;
pub use dense_projection::validate_dense_projection;
pub use error_partial::validate_error_partial;
pub use attention_probe::validate_attention_probe;
pub use candidate_score::validate_candidate_score;
pub use unpack_verify::validate_unpack_verify;
pub use sidecar_apply_verify::validate_sidecar_apply_verify;
pub use rmsnorm_residual_probe::validate_rmsnorm_residual_probe;
pub use mlp_activation_probe::validate_mlp_activation_probe;

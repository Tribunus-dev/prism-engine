//! Inference state model — authoritative mutable state containers for the
//! PhaseEngine-driven execution path.
//!
//! This module defines three state domains:
//!
//! - [`execution_image_state`] — immutable image state, shareable across sessions.
//! - [`inference_session_state`] — mutable per-session state (KV caches, weights,
//!   lanes, cancellation).
//! - [`inference_step_state`] — mutable per-step state (activation, receipts,
//!   sampling).
//! - [`phase_engine_adapter`] — thin adapters that turn [`ProfiledInferenceSession`]
///   methods into PhaseEngine invocations.

#[cfg(feature = "mlx-backend")]
pub mod execution_image_state;
#[cfg(feature = "mlx-backend")]
pub mod inference_session_state;
#[cfg(feature = "mlx-backend")]
pub mod inference_step_state;
#[cfg(feature = "mlx-backend")]
pub mod phase_engine_adapter;

#[cfg(feature = "mlx-backend")]
pub use execution_image_state::ComputeImageState;
#[cfg(feature = "mlx-backend")]
pub use inference_session_state::InferenceSessionState;
#[cfg(feature = "mlx-backend")]
pub use inference_step_state::InferenceStepState;

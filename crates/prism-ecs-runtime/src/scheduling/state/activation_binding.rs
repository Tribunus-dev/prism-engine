//! Activation binding (constitutional home, state half).
//!
//! Per the inventory v2.1 row 4, the engine's `activation_binding.rs`
//! is split:
//! - State half: `ArenaBinding` (already re-exported from
//!   `state::lease`).
//! - FFI half: `ArenaBinding` (already in
//!   `prism_ecs_kernel::backend::metal::activation_binding`).
//!
//! The engine's `CurrentActivation` is a higher-level activation
//! carrier that wraps the binding with provenance and dtype
//! information. This file holds the constitutional-side
//! placeholder. The full type (with TensorId, MLX compatibility
//! view, etc.) migrates when its dependents move.

/// Placeholder for the engine's `CurrentActivation`.
///
/// The engine's `CurrentActivation` is a heavy type with
/// `TensorId`, `ArenaBinding`, `ActivationRepresentation`,
/// `TensorDType`, `TensorLayoutContract`, `ActivationGeneration`,
/// `PhaseId`, and an `mlx_rs::Array` compatibility view. The
/// constitutional home ships a minimal placeholder carrying
/// the same public-API fields; the full type migrates when the
/// engine's tensor types and MLX bindings move (separate
/// migration in `prism-ecs-compile`).
#[derive(Debug, Clone, Default)]
pub struct CurrentActivation {
    /// Opaque tensor identifier.
    pub tensor_id: u64,
    /// Opaque binding offset.
    pub arena_offset: Option<u64>,
    /// Producer phase identifier.
    pub producer_phase: u64,
}

impl CurrentActivation {
    /// Placeholder constructor matching the engine's
    /// `CurrentActivation::new` shape.
    pub fn new(
        tensor_id: u64,
        representation: u32,
        dtype: u32,
        layout: u32,
        producer_phase: u64,
    ) -> Self {
        Self {
            tensor_id,
            arena_offset: None,
            producer_phase,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_activation_carries_ids() {
        // Architectural invariant: the placeholder carries the
        // tensor id and the producer phase. The full type
        // (with representation, dtype, layout) migrates later.
        let a = CurrentActivation::new(1, 0, 0, 0, 2);
        assert_eq!(a.tensor_id, 1);
        assert_eq!(a.producer_phase, 2);
    }
}

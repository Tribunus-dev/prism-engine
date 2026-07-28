//! MoE token-routing data types.
//!
//! Authority: the pure data shapes for per-token expert routing and
//! per-core SRAM layouts.
//!
//! These types carry no engine dependencies. The MLX-coupled
//! `ExpertWeights` payload (the gate/up/down projection arrays) lives
//! in the engine's `legacy_ane/` because it depends on `mlx_rs::Array`.
//! This surface provides the routing layout the engine's `AneMoEScheduler`
//! passes to the Core ML backend.

/// Per-token routing result: which experts are active and with what weight.
///
/// The `expert_indices` and `routing_weights` slices are parallel: index `i`
/// in each is the i-th selected expert and its normalised weight. The
/// weights sum to `1.0` (within `f32` rounding).
#[derive(Debug, Clone, PartialEq)]
pub struct TokenRouting {
    /// Expert indices selected for this token (length ≤ `top_k`).
    pub expert_indices: Vec<u32>,
    /// Normalised routing weights for each selected expert (sums to 1.0).
    pub routing_weights: Vec<f32>,
}

/// Per-core SRAM layout for ANE expert residency.
///
/// Describes which experts should be loaded into a given ANE core's
/// SRAM. The actual `ExpertWeights` payload is allocated by the
/// engine's Core ML backend; this struct only describes the layout.
#[derive(Debug, Clone, PartialEq)]
pub struct AneCoreExpertLayout {
    /// Core identifier (0..num_cores).
    pub core_id: u32,
    /// Expert indices assigned to this core (at most `experts_per_core`).
    pub expert_indices: Vec<u32>,
    /// Base IOAddress of this core's SRAM region.
    pub sram_base: u64,
}

impl TokenRouting {
    /// Construct a token routing entry.
    pub fn new(expert_indices: Vec<u32>, routing_weights: Vec<f32>) -> Self {
        debug_assert_eq!(
            expert_indices.len(),
            routing_weights.len(),
            "TokenRouting: expert_indices and routing_weights must be parallel"
        );
        Self {
            expert_indices,
            routing_weights,
        }
    }
}

impl AneCoreExpertLayout {
    /// Construct a per-core expert layout.
    pub fn new(core_id: u32, expert_indices: Vec<u32>, sram_base: u64) -> Self {
        Self {
            core_id,
            expert_indices,
            sram_base,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_routing_new_stores_fields() {
        let routing = TokenRouting::new(vec![3, 7], vec![0.6, 0.4]);
        assert_eq!(routing.expert_indices, vec![3, 7]);
        assert_eq!(routing.routing_weights, vec![0.6, 0.4]);
    }

    #[test]
    fn core_layout_new_stores_fields() {
        let layout = AneCoreExpertLayout::new(2, vec![10, 11, 12], 0x4000_0000);
        assert_eq!(layout.core_id, 2);
        assert_eq!(layout.expert_indices, vec![10, 11, 12]);
        assert_eq!(layout.sram_base, 0x4000_0000);
    }
}

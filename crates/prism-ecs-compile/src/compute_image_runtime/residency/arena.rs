//! Activation arena requirements builder — helper for constructing
//! [`ActivationArenaRequirements`](super::plan::ActivationArenaRequirements).

use serde::{Deserialize, Serialize};

/// Builder for [`super::plan::ActivationArenaRequirements`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArenaRequirementsBuilder {
    total_activation_bytes: u64,
    arena_region_count: u32,
}

impl ArenaRequirementsBuilder {
    /// Set the total activation bytes.
    pub fn with_total_bytes(mut self, bytes: u64) -> Self {
        self.total_activation_bytes = bytes;
        self
    }

    /// Set the arena region count.
    pub fn with_region_count(mut self, count: u32) -> Self {
        self.arena_region_count = count;
        self
    }

    /// Build the requirements.
    pub fn build(self) -> super::plan::ActivationArenaRequirements {
        super::plan::ActivationArenaRequirements {
            total_activation_bytes: self.total_activation_bytes,
            arena_region_count: self.arena_region_count,
        }
    }
}

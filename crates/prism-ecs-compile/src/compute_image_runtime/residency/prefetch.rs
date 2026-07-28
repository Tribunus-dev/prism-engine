//! Prefetch schedule builder — pure data helper for ordering prefetch
//! actions.

use serde::{Deserialize, Serialize};

use super::plan::{PrefetchAction, PrefetchPriority, RequiredWeightObjectId};

/// Builder for an ordered list of [`PrefetchAction`]s.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrefetchScheduleBuilder {
    actions: Vec<PrefetchAction>,
}

impl PrefetchScheduleBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a prefetch action.
    pub fn add(
        mut self,
        object_id: RequiredWeightObjectId,
        prefetch_before_phase: String,
        priority: PrefetchPriority,
    ) -> Self {
        self.actions.push(PrefetchAction {
            object_id,
            prefetch_before_phase,
            priority,
        });
        self
    }

    /// Build the final schedule.
    pub fn build(self) -> Vec<PrefetchAction> {
        self.actions
    }
}

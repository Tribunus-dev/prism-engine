use serde::{Deserialize, Serialize};

/// A single epoch in the state store lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateEpoch {
    pub epoch_id: u64,
    pub parent_epoch_id: Option<u64>,
    pub phase: String,
    pub committed: bool,
}

impl StateEpoch {
    /// Create a new uncommitted epoch.
    pub fn new(epoch_id: u64, parent: Option<u64>, phase: &str) -> Self {
        Self {
            epoch_id,
            parent_epoch_id: parent,
            phase: phase.to_string(),
            committed: false,
        }
    }

    /// Mark this epoch as committed.
    pub fn commit(&mut self) {
        self.committed = true;
    }
}

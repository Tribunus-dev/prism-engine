/// Performance hint for initial World capacity.
#[derive(Debug, Clone)]
pub struct WorldCapacity {
    pub entity_capacity: u64,
    pub component_capacity_per_type: u64,
    pub resource_capacity: usize,
    pub journal_capacity: usize,
}

impl WorldCapacity {
    /// Default capacity for standard ECS usage.
    pub const fn default_ecs() -> Self {
        Self {
            entity_capacity: 1024,
            component_capacity_per_type: 1024,
            resource_capacity: 64,
            journal_capacity: 128,
        }
    }
}

impl Default for WorldCapacity {
    fn default() -> Self {
        Self::default_ecs()
    }
}

/// Capacity hint for a single ComponentStore.
#[derive(Debug, Clone)]
pub struct ComponentStoreCapacity {
    pub type_capacity: usize,
    pub entity_capacity: u64,
}

impl Default for ComponentStoreCapacity {
    fn default() -> Self {
        Self {
            type_capacity: 64,
            entity_capacity: 1024,
        }
    }
}

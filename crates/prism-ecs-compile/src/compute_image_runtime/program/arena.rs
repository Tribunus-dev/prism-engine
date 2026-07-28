//! Arena plan descriptors.

use serde::{Deserialize, Serialize};

/// Opaque arena region identifier.
pub type ArenaRegionId = String;

/// Opaque arena plan identifier.
pub type ArenaPlanId = String;

/// Descriptor for an arena plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaPlanDescriptor {
    /// Plan identifier.
    pub plan_id: ArenaPlanId,
    /// Regions in the plan.
    pub regions: Vec<ArenaRegionDescriptor>,
}

/// Descriptor for an arena region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaRegionDescriptor {
    /// Region identifier.
    pub region_id: ArenaRegionId,
    /// Region byte size.
    pub byte_size: u64,
    /// Alignment in bytes.
    pub alignment: u32,
}
